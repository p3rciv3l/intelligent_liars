"""Rolling, durable batch admission for the adaptive production study.

The timed-canary forecast is advisory.  Before each eight-trial dispatch this
module checks live committed and pending spend plus the next-batch upper bound,
then persists authorization before work begins.  It has no Optuna, W&B, GPU,
or evaluator implementation details.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .truth_editing_capacity import (
    CapacityPlanningError,
    CapacityPolicy,
    SpendSnapshot,
    select_next_batch_projection,
    validate_capacity_receipt,
)
from .truth_editing_wandb_checkpoint import (
    WandbCheckpointError,
    WandbRunCheckpoint,
    open_wandb_run_checkpoint,
)


CHECKPOINT_FORMAT = "truth_editing_adaptive_run_checkpoint_v1"
_HEX = frozenset("0123456789abcdef")
_TERMINAL_PHASES = frozenset({"finalization_reserved", "complete", "aborted"})
_STOP_REASONS = frozenset(
    {
        None,
        "search_deadline_reached",
        "total_budget_reserve_reached",
        "infrastructure_budget_reserve_reached",
        "evaluation_budget_reserve_reached",
        "maximum_trials_reached",
        "minimum_trial_guarantee_lost",
    }
)


class AdaptiveRunError(RuntimeError):
    """A batch cannot be authorized without violating the frozen run contract."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise AdaptiveRunError("adaptive checkpoint is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise AdaptiveRunError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AdaptiveRunError(f"{label} must be a nonempty trimmed string")
    return value


def _money_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AdaptiveRunError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat().replace("+00:00", "Z")


def _parse_utc(value: Any, label: str) -> datetime:
    raw = _text(value, label)
    if not raw.endswith("Z"):
        raise AdaptiveRunError(f"{label} must end in Z")
    try:
        return datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as error:
        raise AdaptiveRunError(f"{label} is invalid") from error


def _spend_mapping(value: SpendSnapshot) -> dict[str, str]:
    return {
        "actual_total_usd": _money_text(value.actual_total_usd),
        "actual_infrastructure_usd": _money_text(value.actual_infrastructure_usd),
        "actual_evaluation_usd": _money_text(value.actual_evaluation_usd),
        "pending_infrastructure_usd": _money_text(value.pending_infrastructure_usd),
        "pending_evaluation_usd": _money_text(value.pending_evaluation_usd),
    }


SpendReader = Callable[[], SpendSnapshot]
Clock = Callable[[], datetime]
CapacityReceiptReader = Callable[[], Mapping[str, Any]]


class AdaptiveBatchScheduler:
    """One fail-closed admission interface for rolling production search."""

    @property
    def has_checkpoint(self) -> bool:
        return self._checkpoint is not None

    @property
    def search_deadline(self) -> datetime:
        """Absolute search cutoff from the durable checkpoint."""

        if self._checkpoint is None:
            raise AdaptiveRunError("search deadline is unavailable before first admission")
        return _parse_utc(
            self._checkpoint["search_deadline_utc"], "search_deadline_utc"
        )

    def __init__(
        self, *, policy: CapacityPolicy, capacity_receipt: Mapping[str, Any],
        checkpoint_path: Path, study_identity_sha256: str,
        wandb_checkpoint: WandbRunCheckpoint,
        spend_reader: SpendReader, clock: Clock,
        capacity_receipt_reader: CapacityReceiptReader,
        checkpoint: dict[str, Any] | None,
        initial_started_at: datetime | None,
    ) -> None:
        self.policy = policy
        self.capacity_receipt = dict(capacity_receipt)
        self.checkpoint_path = checkpoint_path
        self.study_identity_sha256 = study_identity_sha256
        self.wandb_run_checkpoint_sha256 = wandb_checkpoint.checkpoint_sha256
        self.wandb_run_id = wandb_checkpoint.run_id
        self._spend_reader = spend_reader
        self._clock = clock
        self._capacity_receipt_reader = capacity_receipt_reader
        self._checkpoint = checkpoint
        self._initial_started_at = initial_started_at

    @classmethod
    def open(
        cls, *, policy: CapacityPolicy, capacity_receipt: Mapping[str, Any],
        checkpoint_path: Path, study_identity_sha256: str,
        wandb_checkpoint_path: Path,
        spend_reader: SpendReader, clock: Clock,
        capacity_receipt_reader: CapacityReceiptReader | None = None,
        initial_started_at: datetime | None = None,
    ) -> "AdaptiveBatchScheduler":
        try:
            receipt = validate_capacity_receipt(capacity_receipt)
        except CapacityPlanningError as error:
            raise AdaptiveRunError("capacity receipt is invalid") from error
        study_sha = _digest(study_identity_sha256, "study_identity_sha256")
        try:
            wandb_checkpoint = open_wandb_run_checkpoint(wandb_checkpoint_path)
        except WandbCheckpointError as error:
            raise AdaptiveRunError("W&B run checkpoint is invalid") from error
        wandb_sha = _digest(
            wandb_checkpoint.checkpoint_sha256, "wandb_run_checkpoint_sha256"
        )
        run_id = _text(wandb_checkpoint.run_id, "wandb_run_id")
        if receipt["policy_sha256"] != policy.self_sha256:
            raise AdaptiveRunError("capacity receipt policy identity differs")
        decision = receipt["decision"]
        if (
            decision.get("batch_size") != policy.batch_size
            or decision.get("minimum_trials") != policy.minimum_trials
            or decision.get("maximum_trials") != policy.maximum_trials
            or decision.get("search_seconds") != policy.search_seconds
            or decision.get("finalization_seconds_reserved") != policy.finalization_seconds
            or decision.get("projection_role") != "advisory_reforecast_target"
            or decision.get("reforecast_after_each_batch") is not True
            or decision.get("minimum_trial_guarantee_met") is not True
        ):
            raise AdaptiveRunError("capacity receipt decision differs from policy")
        checkpoint = None
        if checkpoint_path.exists():
            if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
                raise AdaptiveRunError("adaptive checkpoint is not a regular file")
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise AdaptiveRunError("adaptive checkpoint is unreadable") from error
            cls._validate_checkpoint(
                checkpoint, policy=policy,
                capacity_receipt_sha256=receipt["receipt_sha256"],
                study_identity_sha256=study_sha,
                wandb_run_checkpoint_sha256=wandb_sha,
                wandb_run_id=run_id,
                planned_trial_count=decision["planned_trial_limit"],
            )
        normalized_initial_started_at = (
            _utc(initial_started_at, "initial_started_at")
            if initial_started_at is not None and checkpoint is None
            else None
        )
        return cls(
            policy=policy, capacity_receipt=receipt,
            checkpoint_path=checkpoint_path, study_identity_sha256=study_sha,
            wandb_checkpoint=wandb_checkpoint,
            spend_reader=spend_reader, clock=clock,
            capacity_receipt_reader=(
                capacity_receipt_reader
                if capacity_receipt_reader is not None
                else lambda: receipt
            ),
            checkpoint=checkpoint,
            initial_started_at=normalized_initial_started_at,
        )

    @staticmethod
    def _validate_checkpoint(
        raw: Any, *, policy: CapacityPolicy, capacity_receipt_sha256: str,
        study_identity_sha256: str, wandb_run_checkpoint_sha256: str,
        wandb_run_id: str, planned_trial_count: int,
    ) -> None:
        if not isinstance(raw, Mapping):
            raise AdaptiveRunError("adaptive checkpoint must be an object")
        fields = {
            "format", "policy_sha256", "capacity_receipt_sha256",
            "study_identity_sha256", "wandb_run_checkpoint_sha256", "wandb_run_id",
            "started_at_utc", "search_deadline_utc", "hard_deadline_utc",
            "planned_trial_count", "current_advisory_trial_count",
            "current_capacity_receipt_sha256",
            "current_capacity_observation_sha256",
            "current_capacity_completed_through_trial",
            "current_judge_ledger_receipt_sha256",
            "current_next_batch_projection", "projected_search_eta_seconds",
            "accounted_infrastructure_usd", "accounted_evaluation_usd",
            "authorized_through_trial", "completed_trials",
            "coverage_complete", "phase", "stop_reason", "last_spend_snapshot",
            "checkpoint_sha256",
        }
        if set(raw) != fields:
            raise AdaptiveRunError("adaptive checkpoint fields changed")
        claimed = _digest(raw["checkpoint_sha256"], "checkpoint_sha256")
        unsigned = dict(raw)
        unsigned.pop("checkpoint_sha256")
        if claimed != _sha(unsigned):
            raise AdaptiveRunError("adaptive checkpoint identity differs")
        identities = {
            "policy_sha256": policy.self_sha256,
            "capacity_receipt_sha256": capacity_receipt_sha256,
            "study_identity_sha256": study_identity_sha256,
            "wandb_run_checkpoint_sha256": wandb_run_checkpoint_sha256,
            "wandb_run_id": wandb_run_id,
            "planned_trial_count": planned_trial_count,
        }
        if any(raw[name] != expected for name, expected in identities.items()):
            raise AdaptiveRunError("adaptive checkpoint identity binding differs")
        started = _parse_utc(raw["started_at_utc"], "started_at_utc")
        search_deadline = _parse_utc(raw["search_deadline_utc"], "search_deadline_utc")
        hard_deadline = _parse_utc(raw["hard_deadline_utc"], "hard_deadline_utc")
        if (
            search_deadline != started + timedelta(seconds=policy.search_seconds)
            or hard_deadline != search_deadline + timedelta(seconds=policy.finalization_seconds)
        ):
            raise AdaptiveRunError("adaptive checkpoint deadlines differ from policy")
        authorized = raw["authorized_through_trial"]
        completed = raw["completed_trials"]
        if (
            isinstance(authorized, bool) or not isinstance(authorized, int)
            or isinstance(completed, bool) or not isinstance(completed, int)
            or completed < 0 or authorized < completed
            or authorized > policy.maximum_trials or completed > policy.maximum_trials
            or authorized % policy.batch_size or completed % policy.batch_size
        ):
            raise AdaptiveRunError("adaptive checkpoint trial boundary is invalid")
        current_advisory = raw["current_advisory_trial_count"]
        if (
            isinstance(current_advisory, bool)
            or not isinstance(current_advisory, int)
            or not policy.minimum_trials <= current_advisory <= policy.maximum_trials
            or current_advisory % policy.batch_size
        ):
            raise AdaptiveRunError("adaptive checkpoint advisory target is invalid")
        _digest(
            raw["current_capacity_receipt_sha256"],
            "current_capacity_receipt_sha256",
        )
        observation_sha = raw["current_capacity_observation_sha256"]
        if observation_sha is not None:
            _digest(observation_sha, "current_capacity_observation_sha256")
        judge_ledger_sha = raw["current_judge_ledger_receipt_sha256"]
        if judge_ledger_sha is not None:
            _digest(judge_ledger_sha, "current_judge_ledger_receipt_sha256")
        capacity_completed = raw["current_capacity_completed_through_trial"]
        if (
            isinstance(capacity_completed, bool)
            or not isinstance(capacity_completed, int)
            or capacity_completed < 0
            or capacity_completed > completed
            or capacity_completed % policy.batch_size
        ):
            raise AdaptiveRunError("adaptive checkpoint capacity boundary is invalid")
        projection = raw["current_next_batch_projection"]
        if not isinstance(projection, Mapping) or set(projection) != {
            "tier", "batch_duration_seconds_upper_bound",
            "batch_infrastructure_cost_usd_upper_bound",
            "batch_evaluation_cost_usd_upper_bound",
            "batch_total_cost_usd_upper_bound",
        }:
            raise AdaptiveRunError("adaptive checkpoint next-batch projection is invalid")
        eta = raw["projected_search_eta_seconds"]
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or eta < 0:
            raise AdaptiveRunError("adaptive checkpoint ETA is invalid")
        for name in ("accounted_infrastructure_usd", "accounted_evaluation_usd"):
            try:
                value = Decimal(raw[name])
            except (TypeError, ValueError) as error:
                raise AdaptiveRunError("adaptive checkpoint accounted spend is invalid") from error
            if not value.is_finite() or value < 0 or _money_text(value) != raw[name]:
                raise AdaptiveRunError("adaptive checkpoint accounted spend is invalid")
        if not isinstance(raw["coverage_complete"], bool):
            raise AdaptiveRunError("adaptive checkpoint coverage_complete must be boolean")
        phase = raw["phase"]
        reason = raw["stop_reason"]
        if phase not in {
            "broad_coverage", "adaptive_search", "finalization_reserved",
            "complete", "aborted",
        } or reason not in _STOP_REASONS:
            raise AdaptiveRunError("adaptive checkpoint phase or stop reason is invalid")
        if (phase in _TERMINAL_PHASES) != (reason is not None):
            raise AdaptiveRunError("adaptive checkpoint terminal state is inconsistent")
        if (phase == "aborted") != (reason == "minimum_trial_guarantee_lost"):
            raise AdaptiveRunError("adaptive checkpoint abort reason is inconsistent")
        if phase == "aborted" and completed >= policy.minimum_trials:
            raise AdaptiveRunError("adaptive checkpoint cannot abort after the minimum")
        if phase == "adaptive_search" and raw["coverage_complete"] is not True:
            raise AdaptiveRunError("adaptive search began before broad coverage completed")
        try:
            SpendSnapshot.from_mapping(raw["last_spend_snapshot"])
        except CapacityPlanningError as error:
            raise AdaptiveRunError("adaptive checkpoint spend snapshot is invalid") from error

    def _current_capacity_receipt(self) -> dict[str, Any]:
        try:
            receipt = validate_capacity_receipt(self._capacity_receipt_reader())
        except CapacityPlanningError as error:
            raise AdaptiveRunError("rolling capacity receipt is invalid") from error
        if (
            receipt["policy_sha256"] != self.policy.self_sha256
            or receipt["timed_canary_receipt_sha256"]
            != self.capacity_receipt["timed_canary_receipt_sha256"]
        ):
            raise AdaptiveRunError("rolling capacity receipt identity differs")
        return receipt

    @staticmethod
    def _next_batch_projection(
        receipt: Mapping[str, Any], *, next_completed_trials: int, batch_size: int,
    ) -> tuple[dict[str, Any], float, Decimal, Decimal, Decimal]:
        try:
            projection = select_next_batch_projection(
                receipt, next_completed_trials=next_completed_trials
            )
            duration = float(projection["batch_duration_seconds_upper_bound"])
            infrastructure = Decimal(
                projection["batch_infrastructure_cost_usd_upper_bound"]
            )
            evaluation = Decimal(
                projection["batch_evaluation_cost_usd_upper_bound"]
            )
            total = Decimal(projection["batch_total_cost_usd_upper_bound"])
        except (CapacityPlanningError, KeyError, TypeError, ValueError) as error:
            raise AdaptiveRunError("capacity next-batch projection is malformed") from error
        if (
            projection["batch_size"] != batch_size
            or duration <= 0 or infrastructure < 0 or evaluation < 0
            or total != infrastructure + evaluation
        ):
            raise AdaptiveRunError("capacity next-batch projection is inconsistent")
        normalized = {
            "tier": _text(projection["tier"], "capacity tier name"),
            "batch_duration_seconds_upper_bound": duration,
            "batch_infrastructure_cost_usd_upper_bound": _money_text(infrastructure),
            "batch_evaluation_cost_usd_upper_bound": _money_text(evaluation),
            "batch_total_cost_usd_upper_bound": _money_text(total),
        }
        return normalized, duration, infrastructure, evaluation, total

    def _save(
        self, *, started: datetime, authorized: int, completed: int,
        coverage_complete: bool, phase: str, stop_reason: str | None,
        spend: SpendSnapshot, current_capacity_receipt: Mapping[str, Any],
        current_next_batch_projection: Mapping[str, Any],
        accounted_infrastructure: Decimal, accounted_evaluation: Decimal,
        projected_search_eta_seconds: float,
    ) -> None:
        search_deadline = started + timedelta(seconds=self.policy.search_seconds)
        hard_deadline = search_deadline + timedelta(seconds=self.policy.finalization_seconds)
        unsigned = {
            "format": CHECKPOINT_FORMAT,
            "policy_sha256": self.policy.self_sha256,
            "capacity_receipt_sha256": self.capacity_receipt["receipt_sha256"],
            "study_identity_sha256": self.study_identity_sha256,
            "wandb_run_checkpoint_sha256": self.wandb_run_checkpoint_sha256,
            "wandb_run_id": self.wandb_run_id,
            "started_at_utc": _utc_text(started),
            "search_deadline_utc": _utc_text(search_deadline),
            "hard_deadline_utc": _utc_text(hard_deadline),
            "planned_trial_count": self.capacity_receipt["decision"]["planned_trial_limit"],
            "current_advisory_trial_count": current_capacity_receipt["decision"][
                "planned_trial_limit"
            ],
            "current_capacity_receipt_sha256": current_capacity_receipt[
                "receipt_sha256"
            ],
            "current_capacity_observation_sha256": current_capacity_receipt[
                "source_batch_observation_sha256"
            ],
            "current_capacity_completed_through_trial": current_capacity_receipt[
                "completed_through_trial"
            ],
            "current_judge_ledger_receipt_sha256": current_capacity_receipt[
                "source_judge_ledger_after_receipt_sha256"
            ],
            "current_next_batch_projection": dict(current_next_batch_projection),
            "projected_search_eta_seconds": projected_search_eta_seconds,
            "accounted_infrastructure_usd": _money_text(accounted_infrastructure),
            "accounted_evaluation_usd": _money_text(accounted_evaluation),
            "authorized_through_trial": authorized,
            "completed_trials": completed,
            "coverage_complete": coverage_complete,
            "phase": phase,
            "stop_reason": stop_reason,
            "last_spend_snapshot": _spend_mapping(spend),
        }
        checkpoint = {**unsigned, "checkpoint_sha256": _sha(unsigned)}
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_name(f".{self.checkpoint_path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(_canonical(checkpoint) + b"\n")
        temporary.replace(self.checkpoint_path)
        self._checkpoint = checkpoint

    @staticmethod
    def _reject_spend_rollback(previous: SpendSnapshot, current: SpendSnapshot) -> None:
        pairs = (
            (previous.actual_total_usd, current.actual_total_usd),
            (previous.actual_infrastructure_usd, current.actual_infrastructure_usd),
            (previous.actual_evaluation_usd, current.actual_evaluation_usd),
            (previous.reserved_infrastructure_usd, current.reserved_infrastructure_usd),
            (previous.reserved_evaluation_usd, current.reserved_evaluation_usd),
            (previous.reserved_total_usd, current.reserved_total_usd),
        )
        if any(now < before for before, now in pairs):
            raise AdaptiveRunError("live spend ledger moved backwards")

    def admit_batch(
        self, *, completed_trials: int, batch_size: int, coverage_complete: bool,
        batch_started: bool = False,
    ) -> bool:
        """Authorize one complete batch, writing the checkpoint before dispatch.

        A durable authorization is revalidated until the journal proves that at
        least one trial in that batch started.  Once started, replay finishes
        the exact durable batch without applying a new-work admission decision.
        """

        if (
            batch_size != self.policy.batch_size
            or isinstance(completed_trials, bool) or not isinstance(completed_trials, int)
            or completed_trials < 0 or completed_trials % self.policy.batch_size
            or not isinstance(coverage_complete, bool)
            or not isinstance(batch_started, bool)
        ):
            raise AdaptiveRunError("adaptive execution requires exact batches of eight")
        now = _utc(self._clock(), "current time")
        spend = self._spend_reader()
        if not isinstance(spend, SpendSnapshot):
            raise AdaptiveRunError("spend reader must return SpendSnapshot")
        current_capacity_receipt = self._current_capacity_receipt()
        next_completed_trials = min(
            self.policy.maximum_trials,
            completed_trials + self.policy.batch_size,
        )
        projection, duration, next_infrastructure, next_evaluation, _next_total = (
            self._next_batch_projection(
                current_capacity_receipt,
                next_completed_trials=next_completed_trials,
                batch_size=self.policy.batch_size,
            )
        )
        if self._checkpoint is None:
            if (
                current_capacity_receipt.get("completed_through_trial")
                != completed_trials
            ):
                raise AdaptiveRunError(
                    "fresh adaptive checkpoint differs from the capacity boundary"
                )
            budget = self.capacity_receipt["budget"]
            try:
                baseline = SpendSnapshot.from_mapping({
                    "actual_total_usd": budget["actual_total_usd"],
                    "actual_infrastructure_usd": budget["actual_infrastructure_usd"],
                    "actual_evaluation_usd": budget["actual_evaluation_usd"],
                    "pending_infrastructure_usd": budget["pending_infrastructure_usd"],
                    "pending_evaluation_usd": budget["pending_evaluation_usd"],
                })
            except (KeyError, CapacityPlanningError) as error:
                raise AdaptiveRunError("capacity spend baseline is invalid") from error
            self._reject_spend_rollback(baseline, spend)
            started = self._initial_started_at or now
            if started > now:
                raise AdaptiveRunError(
                    "initial_started_at cannot be after current time"
                )
            authorized = completed_trials
            accounted_infrastructure = spend.reserved_infrastructure_usd
            accounted_evaluation = spend.reserved_evaluation_usd
        else:
            started = _parse_utc(self._checkpoint["started_at_utc"], "started_at_utc")
            authorized = int(self._checkpoint["authorized_through_trial"])
            previous_spend = SpendSnapshot.from_mapping(self._checkpoint["last_spend_snapshot"])
            self._reject_spend_rollback(previous_spend, spend)
            if self._checkpoint["coverage_complete"] is True and not coverage_complete:
                raise AdaptiveRunError("broad coverage completeness regressed")
            if (
                current_capacity_receipt["receipt_sha256"]
                != self._checkpoint["current_capacity_receipt_sha256"]
                or current_capacity_receipt["source_batch_observation_sha256"]
                != self._checkpoint["current_capacity_observation_sha256"]
                or current_capacity_receipt["completed_through_trial"]
                != self._checkpoint["current_capacity_completed_through_trial"]
                or current_capacity_receipt[
                    "source_judge_ledger_after_receipt_sha256"
                ] != self._checkpoint["current_judge_ledger_receipt_sha256"]
            ):
                raise AdaptiveRunError(
                    "rolling capacity receipt changed outside a committed batch"
                )
            if completed_trials != int(self._checkpoint["completed_trials"]):
                raise AdaptiveRunError(
                    "completed batch must be committed before another admission"
                )
            if completed_trials > authorized:
                raise AdaptiveRunError("completed journal exceeds authorized checkpoint")
            if self._checkpoint["phase"] in _TERMINAL_PHASES:
                if completed_trials != authorized:
                    raise AdaptiveRunError("terminal checkpoint differs from completed journal")
                return False
            if completed_trials < authorized and batch_started:
                return True
            accounted_infrastructure = max(
                Decimal(self._checkpoint["accounted_infrastructure_usd"]),
                spend.reserved_infrastructure_usd,
            )
            accounted_evaluation = max(
                Decimal(self._checkpoint["accounted_evaluation_usd"]),
                spend.reserved_evaluation_usd,
            )

        remaining_batches = max(
            0,
            (
                current_capacity_receipt["decision"]["planned_trial_limit"]
                - completed_trials
            ) // self.policy.batch_size,
        )
        search_seconds_remaining = max(
            0.0,
            (started + timedelta(seconds=self.policy.search_seconds) - now).total_seconds(),
        )
        projected_eta = min(
            float(remaining_batches * duration),
            search_seconds_remaining,
        )

        revalidating_unstarted = completed_trials < authorized

        def stop(reason: str) -> bool:
            terminal_reason = (
                "minimum_trial_guarantee_lost"
                if completed_trials < self.policy.minimum_trials
                else reason
            )
            self._save(
                started=started,
                authorized=(completed_trials if revalidating_unstarted else authorized),
                completed=completed_trials,
                coverage_complete=coverage_complete,
                phase=(
                    "aborted"
                    if completed_trials < self.policy.minimum_trials
                    else "finalization_reserved"
                ),
                stop_reason=terminal_reason, spend=spend,
                current_capacity_receipt=current_capacity_receipt,
                current_next_batch_projection=projection,
                accounted_infrastructure=accounted_infrastructure,
                accounted_evaluation=accounted_evaluation,
                projected_search_eta_seconds=projected_eta,
            )
            return False

        if completed_trials >= self.policy.maximum_trials:
            return stop("maximum_trials_reached")
        if now + timedelta(seconds=duration) > started + timedelta(seconds=self.policy.search_seconds):
            return stop("search_deadline_reached")

        reserve = current_capacity_receipt["budget"]
        try:
            final_infrastructure = Decimal(
                reserve["reserved_finalization_infrastructure_usd"]
            )
            final_evaluation = Decimal(reserve["reserved_evaluation_usd"])
            final_total = final_infrastructure + final_evaluation
        except (KeyError, TypeError, ValueError) as error:
            raise AdaptiveRunError("capacity finalization reserve is malformed") from error
        if final_total != final_infrastructure + final_evaluation:
            raise AdaptiveRunError("capacity finalization reserve is inconsistent")
        admitted_infrastructure = accounted_infrastructure + (
            Decimal("0") if revalidating_unstarted else next_infrastructure
        )
        admitted_evaluation = accounted_evaluation + (
            Decimal("0") if revalidating_unstarted else next_evaluation
        )
        admitted_total = admitted_infrastructure + admitted_evaluation
        if admitted_infrastructure + final_infrastructure > self.policy.infrastructure_budget_usd:
            return stop("infrastructure_budget_reserve_reached")
        if admitted_evaluation + final_evaluation > self.policy.evaluation_budget_usd:
            return stop("evaluation_budget_reserve_reached")
        if admitted_total + final_total > self.policy.total_budget_usd:
            return stop("total_budget_reserve_reached")

        next_authorized = (
            authorized
            if revalidating_unstarted
            else completed_trials + self.policy.batch_size
        )
        phase = "adaptive_search" if coverage_complete else "broad_coverage"
        self._save(
            started=started, authorized=next_authorized, completed=completed_trials,
            coverage_complete=coverage_complete, phase=phase,
            stop_reason=None, spend=spend,
            current_capacity_receipt=current_capacity_receipt,
            current_next_batch_projection=projection,
            accounted_infrastructure=admitted_infrastructure,
            accounted_evaluation=admitted_evaluation,
            projected_search_eta_seconds=projected_eta,
        )
        return True

    def commit_batch(
        self, *, completed_trials: int, coverage_complete: bool
    ) -> None:
        """Acknowledge a fully journaled batch without authorizing another one."""

        if self._checkpoint is None:
            raise AdaptiveRunError("cannot commit a batch before authorization")
        if (
            isinstance(completed_trials, bool)
            or not isinstance(completed_trials, int)
            or completed_trials % self.policy.batch_size
            or not isinstance(coverage_complete, bool)
        ):
            raise AdaptiveRunError("committed batch boundary is invalid")
        authorized = int(self._checkpoint["authorized_through_trial"])
        spend = self._spend_reader()
        if not isinstance(spend, SpendSnapshot):
            raise AdaptiveRunError("spend reader must return SpendSnapshot")
        previous_spend = SpendSnapshot.from_mapping(
            self._checkpoint["last_spend_snapshot"]
        )
        self._reject_spend_rollback(previous_spend, spend)
        current_capacity_receipt = self._current_capacity_receipt()
        checkpoint_completed = int(self._checkpoint["completed_trials"])
        if completed_trials == checkpoint_completed:
            replay_bindings = {
                "receipt_sha256": self._checkpoint[
                    "current_capacity_receipt_sha256"
                ],
                "source_batch_observation_sha256": self._checkpoint[
                    "current_capacity_observation_sha256"
                ],
                "completed_through_trial": self._checkpoint[
                    "current_capacity_completed_through_trial"
                ],
                "source_judge_ledger_after_receipt_sha256": self._checkpoint[
                    "current_judge_ledger_receipt_sha256"
                ],
            }
            if (
                coverage_complete != self._checkpoint["coverage_complete"]
                or any(
                    current_capacity_receipt[name] != expected
                    for name, expected in replay_bindings.items()
                )
            ):
                raise AdaptiveRunError(
                    "replayed batch commit differs from its durable identities"
                )
            return
        if completed_trials != authorized:
            raise AdaptiveRunError("committed batch differs from durable authorization")
        if self._checkpoint["coverage_complete"] is True and not coverage_complete:
            raise AdaptiveRunError("broad coverage completeness regressed")
        observation_sha = current_capacity_receipt[
            "source_batch_observation_sha256"
        ]
        if observation_sha is None:
            raise AdaptiveRunError("committed batch is missing its signed capacity observation")
        if observation_sha == self._checkpoint["current_capacity_observation_sha256"]:
            raise AdaptiveRunError("committed batch reused its prior capacity observation")
        if current_capacity_receipt["completed_through_trial"] != completed_trials:
            raise AdaptiveRunError("capacity observation differs from committed batch boundary")
        if current_capacity_receipt["source_judge_ledger_after_receipt_sha256"] is None:
            raise AdaptiveRunError("committed batch is missing its judge ledger identity")
        next_completed = min(
            self.policy.maximum_trials,
            completed_trials + self.policy.batch_size,
        )
        projection, duration, _infra, _eval, _total = self._next_batch_projection(
            current_capacity_receipt,
            next_completed_trials=next_completed,
            batch_size=self.policy.batch_size,
        )
        remaining_batches = max(
            0,
            (
                current_capacity_receipt["decision"]["planned_trial_limit"]
                - completed_trials
            ) // self.policy.batch_size,
        )
        terminal = completed_trials >= self.policy.maximum_trials
        started = _parse_utc(
            self._checkpoint["started_at_utc"], "started_at_utc"
        )
        now = _utc(self._clock(), "current time")
        search_seconds_remaining = max(
            0.0,
            (
                started + timedelta(seconds=self.policy.search_seconds) - now
            ).total_seconds(),
        )
        self._save(
            started=started,
            authorized=authorized,
            completed=completed_trials,
            coverage_complete=coverage_complete,
            phase=(
                "finalization_reserved"
                if terminal
                else "adaptive_search" if coverage_complete else "broad_coverage"
            ),
            stop_reason="maximum_trials_reached" if terminal else None,
            spend=spend,
            current_capacity_receipt=current_capacity_receipt,
            current_next_batch_projection=projection,
            accounted_infrastructure=Decimal(
                self._checkpoint["accounted_infrastructure_usd"]
            ),
            accounted_evaluation=Decimal(
                self._checkpoint["accounted_evaluation_usd"]
            ),
            projected_search_eta_seconds=min(
                float(remaining_batches * duration),
                search_seconds_remaining,
            ),
        )

    def abort_minimum_trial_guarantee(
        self, *, completed_trials: int, coverage_complete: bool
    ) -> None:
        """Durably stop after a completed batch invalidates the 200-trial floor."""

        if self._checkpoint is None:
            raise AdaptiveRunError("cannot abort before the first durable authorization")
        if (
            isinstance(completed_trials, bool)
            or not isinstance(completed_trials, int)
            or completed_trials % self.policy.batch_size
            or not isinstance(coverage_complete, bool)
        ):
            raise AdaptiveRunError("minimum-guarantee abort boundary is invalid")
        authorized = int(self._checkpoint["authorized_through_trial"])
        if completed_trials != authorized or completed_trials >= self.policy.minimum_trials:
            raise AdaptiveRunError(
                "minimum-guarantee abort requires an authorized pre-minimum batch"
            )
        if self._checkpoint["phase"] in _TERMINAL_PHASES:
            if (
                self._checkpoint["phase"] == "aborted"
                and self._checkpoint["stop_reason"]
                == "minimum_trial_guarantee_lost"
                and int(self._checkpoint["completed_trials"]) == completed_trials
                and self._checkpoint["coverage_complete"] == coverage_complete
            ):
                return
            raise AdaptiveRunError("adaptive run already has a different terminal state")
        spend = self._spend_reader()
        if not isinstance(spend, SpendSnapshot):
            raise AdaptiveRunError("spend reader must return SpendSnapshot")
        previous_spend = SpendSnapshot.from_mapping(
            self._checkpoint["last_spend_snapshot"]
        )
        self._reject_spend_rollback(previous_spend, spend)
        current_capacity_receipt = self._current_capacity_receipt()
        capacity_limits = current_capacity_receipt["capacity_limits"]
        if (
            current_capacity_receipt["completed_through_trial"]
            != completed_trials
            or current_capacity_receipt["source_batch_observation_sha256"] is None
            or min(
                int(capacity_limits[name])
                for name in (
                    "time_limited_trials",
                    "total_budget_limited_trials",
                    "infrastructure_budget_limited_trials",
                    "evaluation_budget_limited_trials",
                )
            )
            >= self.policy.minimum_trials
        ):
            raise AdaptiveRunError(
                "minimum-guarantee abort lacks its signed infeasible projection"
            )
        self._save(
            started=_parse_utc(
                self._checkpoint["started_at_utc"], "started_at_utc"
            ),
            authorized=authorized,
            completed=completed_trials,
            coverage_complete=coverage_complete,
            phase="aborted",
            stop_reason="minimum_trial_guarantee_lost",
            spend=spend,
            current_capacity_receipt=current_capacity_receipt,
            current_next_batch_projection=self._checkpoint[
                "current_next_batch_projection"
            ],
            accounted_infrastructure=Decimal(
                self._checkpoint["accounted_infrastructure_usd"]
            ),
            accounted_evaluation=Decimal(
                self._checkpoint["accounted_evaluation_usd"]
            ),
            projected_search_eta_seconds=0.0,
        )

    def rearm_minimum_guarantee_abort(
        self, *, started_at: datetime | None = None
    ) -> None:
        """Reopen a minimum-guarantee abort for a fresh host lease.

        The prior run remains an immutable audit record.  Only its scheduler
        phase and deadline are renewed; completed trials, spend, identities,
        and the exact replay queue owned by the study journal are untouched.
        A newly signed rolling capacity receipt must already cover the
        completed boundary and restore the 200-trial guarantee.
        """

        if self._checkpoint is None:
            raise AdaptiveRunError("cannot rearm before the first durable authorization")
        if self._checkpoint["phase"] != "aborted":
            # A controller can crash after this method durably saves the
            # renewed lease but before the first resumed batch is published.
            # Treat that already-rearmed state as idempotent so a restart can
            # still carry forward the prior lease spend and continue safely.
            if (
                self._checkpoint["phase"] in {"broad_coverage", "adaptive_search"}
                and self._checkpoint["stop_reason"] is None
            ):
                completed = int(self._checkpoint["completed_trials"])
                current_capacity_receipt = self._current_capacity_receipt()
                if (
                    current_capacity_receipt["completed_through_trial"] == completed
                    and current_capacity_receipt["decision"][
                        "minimum_trial_guarantee_met"
                    ] is True
                ):
                    return
            raise AdaptiveRunError(
                "rearm requires a minimum-guarantee abort checkpoint"
            )
        completed = int(self._checkpoint["completed_trials"])
        authorized = int(self._checkpoint["authorized_through_trial"])
        if completed != authorized or completed >= self.policy.minimum_trials:
            raise AdaptiveRunError("minimum-guarantee abort boundary is invalid")
        now = _utc(self._clock(), "current time")
        started = now if started_at is None else _utc(started_at, "started_at")
        if started > now:
            raise AdaptiveRunError("rearmed lease start cannot be in the future")
        current_capacity_receipt = self._current_capacity_receipt()
        if current_capacity_receipt["completed_through_trial"] != completed:
            raise AdaptiveRunError(
                "fresh capacity receipt must be bound to the completed trial boundary"
            )
        if current_capacity_receipt["decision"]["minimum_trial_guarantee_met"] is not True:
            raise AdaptiveRunError(
                "fresh capacity receipt does not restore the minimum-guarantee capacity"
            )
        spend = self._spend_reader()
        if not isinstance(spend, SpendSnapshot):
            raise AdaptiveRunError("spend reader must return SpendSnapshot")
        previous_spend = SpendSnapshot.from_mapping(
            self._checkpoint["last_spend_snapshot"]
        )
        self._reject_spend_rollback(previous_spend, spend)
        next_completed = min(
            self.policy.maximum_trials, completed + self.policy.batch_size
        )
        projection, _duration, _infrastructure, _evaluation, _total = (
            self._next_batch_projection(
                current_capacity_receipt,
                next_completed_trials=next_completed,
                batch_size=self.policy.batch_size,
            )
        )
        phase = (
            "adaptive_search"
            if self._checkpoint["coverage_complete"]
            else "broad_coverage"
        )
        self._save(
            started=started,
            authorized=authorized,
            completed=completed,
            coverage_complete=bool(self._checkpoint["coverage_complete"]),
            phase=phase,
            stop_reason=None,
            spend=spend,
            current_capacity_receipt=current_capacity_receipt,
            current_next_batch_projection=projection,
            accounted_infrastructure=Decimal(
                self._checkpoint["accounted_infrastructure_usd"]
            ),
            accounted_evaluation=Decimal(
                self._checkpoint["accounted_evaluation_usd"]
            ),
            projected_search_eta_seconds=float(
                self._checkpoint["projected_search_eta_seconds"]
            ),
        )


__all__ = ["AdaptiveBatchScheduler", "AdaptiveRunError", "CHECKPOINT_FORMAT"]
