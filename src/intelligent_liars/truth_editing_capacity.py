"""Fail-closed capacity planning from one timed production canary.

The module has one planning interface: a frozen policy and a fresh measurement
produce a durable receipt.  It does not launch trials, contact W&B, or mutate a
study.  Schedulers consume the receipt and continue enforcing its deadline and
budget stop conditions against live spend.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any, Mapping


POLICY_FORMAT = "truth_editing_adaptive_capacity_policy_v1"
MEASUREMENT_FORMAT = "truth_editing_capacity_measurement_v1"
RECEIPT_FORMAT = "truth_editing_adaptive_capacity_receipt_v2"
BATCH_OBSERVATION_FORMAT = "truth_editing_capacity_batch_observation_v2"
_HEX = frozenset("0123456789abcdef")
_UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")
_TIMED_CANARY_RECEIPT_FIELDS = {
    "format", "canary_config_sha256", "production_config_path",
    "production_config_sha256", "model_sha256", "observation_sha256",
    "trial_id", "trial_outcome_kind", "trial_output_sha256",
    "generated_tokens", "generation_seconds", "tokens_per_second",
    "measured_wall_seconds", "gpu_hourly_usd", "estimated_canary_cost_usd",
    "single_worker_trials_per_hour", "single_worker_200_trial_hours",
    "single_worker_200_trial_cost_usd", "judge_calls", "judge_cost_usd",
    "judge_elapsed_seconds", "persistence_kl", "software_and_live_canary_passed",
}


class CapacityPlanningError(RuntimeError):
    """The capacity input is invalid or cannot safely guarantee the minimum run."""


class MinimumTrialGuaranteeError(CapacityPlanningError):
    """A signed rolling projection proves that the 200-trial floor is infeasible."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        super().__init__(
            "measured throughput and budget cannot guarantee 200 trials with reserves"
        )
        self.receipt = dict(receipt)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CapacityPlanningError("capacity value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityPlanningError(f"{label} must be an object")
    result = dict(value)
    _canonical(result)
    return result


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != fields:
        raise CapacityPlanningError(f"{label} fields changed")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CapacityPlanningError(f"{label} must be a nonempty trimmed string")
    return value


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in _HEX for char in result):
        raise CapacityPlanningError(f"{label} must be a lowercase SHA-256")
    return result


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CapacityPlanningError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapacityPlanningError(f"{label} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CapacityPlanningError(f"{label} must be a finite number >= {minimum}")
    return result


def _money(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise CapacityPlanningError(f"{label} must be canonical nonnegative money text")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise CapacityPlanningError(f"{label} must be canonical nonnegative money text") from error
    if not result.is_finite() or result < 0 or _money_text(result) != value:
        raise CapacityPlanningError(f"{label} must be canonical nonnegative money text")
    return result


def _money_text(value: Decimal) -> str:
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result or "0"


def _utc(value: Any, label: str) -> datetime:
    raw = _text(value, label)
    if not _UTC_PATTERN.fullmatch(raw):
        raise CapacityPlanningError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as error:
        raise CapacityPlanningError(f"{label} must be a valid UTC timestamp") from error
    return parsed


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise CapacityPlanningError("planned_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CapacityPolicy:
    policy_id: str
    batch_size: int
    minimum_trials: int
    maximum_trials: int
    search_seconds: int
    finalization_seconds: int
    total_budget_usd: Decimal
    infrastructure_budget_usd: Decimal
    evaluation_budget_usd: Decimal
    evaluation_reserve_fraction: Decimal
    duration_margin: Decimal
    judge_cost_margin: Decimal
    tier_multipliers: tuple[tuple[str, int, Decimal, Decimal], ...]
    maximum_measurement_age_seconds: int
    self_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapacityPolicy":
        raw = _exact(
            value,
            {
                "format", "policy_id", "execution", "budget", "uncertainty", "projection_tiers",
                "maximum_measurement_age_seconds", "self_sha256",
            },
            "capacity policy",
        )
        claimed = _digest(raw.pop("self_sha256"), "capacity policy self_sha256")
        if _sha(raw) != claimed:
            raise CapacityPlanningError("capacity policy self_sha256 differs")
        if raw["format"] != POLICY_FORMAT:
            raise CapacityPlanningError("capacity policy format changed")
        execution = _exact(
            raw["execution"],
            {"batch_size", "minimum_trials", "maximum_trials", "search_seconds", "finalization_seconds"},
            "execution",
        )
        budget = _exact(
            raw["budget"],
            {"total_usd", "infrastructure_usd", "evaluation_usd", "evaluation_reserve_fraction"},
            "budget",
        )
        uncertainty = _exact(raw["uncertainty"], {"duration_multiplier", "judge_cost_multiplier"}, "uncertainty")
        tiers_raw = raw["projection_tiers"]
        if not isinstance(tiers_raw, list) or len(tiers_raw) != 3:
            raise CapacityPlanningError("projection_tiers must contain discovery, expanded, and concentrated")
        tiers: list[tuple[str, int, Decimal, Decimal]] = []
        for index, value in enumerate(tiers_raw):
            tier = _exact(value, {"name", "through_trial", "generation_multiplier", "judge_multiplier"}, f"projection_tiers[{index}]")
            tiers.append((_text(tier["name"], "tier name"), _integer(tier["through_trial"], "through_trial", 1), _money(tier["generation_multiplier"], "generation_multiplier"), _money(tier["judge_multiplier"], "judge_multiplier")))
        batch = _integer(execution["batch_size"], "batch_size", 1)
        minimum = _integer(execution["minimum_trials"], "minimum_trials", 1)
        maximum = _integer(execution["maximum_trials"], "maximum_trials", minimum)
        if batch != 8 or minimum != 200 or maximum != 800:
            raise CapacityPlanningError("adaptive run must preserve 8 workers and the 200-800 trial range")
        search = _integer(execution["search_seconds"], "search_seconds", 1)
        finalization = _integer(execution["finalization_seconds"], "finalization_seconds", 1)
        if search != 21 * 3600 or finalization != 3 * 3600:
            raise CapacityPlanningError("adaptive run must preserve the 21h search plus 3h finalization split")
        total = _money(budget["total_usd"], "budget.total_usd")
        infrastructure = _money(budget["infrastructure_usd"], "budget.infrastructure_usd")
        evaluation = _money(budget["evaluation_usd"], "budget.evaluation_usd")
        if total != Decimal("50") or infrastructure != Decimal("45") or evaluation != Decimal("5"):
            raise CapacityPlanningError("adaptive run must preserve the 45 + 5 = 50 USD budget split")
        if budget["evaluation_reserve_fraction"] != "0.20":
            raise CapacityPlanningError(
                "evaluation_reserve_fraction must use the frozen 0.20 spelling"
            )
        eval_fraction = Decimal("0.20")
        if eval_fraction != Decimal("0.2"):
            raise CapacityPlanningError("adaptive run must reserve exactly 20 percent of remaining evaluation budget")
        duration_margin = _money(uncertainty["duration_multiplier"], "duration_multiplier")
        judge_margin = _money(uncertainty["judge_cost_multiplier"], "judge_cost_multiplier")
        if duration_margin < 1 or judge_margin < 1:
            raise CapacityPlanningError("uncertainty multipliers must be at least one")
        if [tier[:2] for tier in tiers] != [("discovery", 80), ("expanded", 200), ("concentrated", 800)] or any(tier[2] < 1 or tier[3] < 1 for tier in tiers):
            raise CapacityPlanningError("projection tier identities or multipliers changed")
        return cls(
            policy_id=_text(raw["policy_id"], "policy_id"), batch_size=batch,
            minimum_trials=minimum, maximum_trials=maximum, search_seconds=search,
            finalization_seconds=finalization, total_budget_usd=total,
            infrastructure_budget_usd=infrastructure, evaluation_budget_usd=evaluation,
            evaluation_reserve_fraction=eval_fraction,
            duration_margin=duration_margin, judge_cost_margin=judge_margin,
            tier_multipliers=tuple(tiers),
            maximum_measurement_age_seconds=_integer(raw["maximum_measurement_age_seconds"], "maximum_measurement_age_seconds", 1),
            self_sha256=claimed,
        )


@dataclass(frozen=True)
class SpendSnapshot:
    actual_total_usd: Decimal
    actual_infrastructure_usd: Decimal
    actual_evaluation_usd: Decimal
    pending_infrastructure_usd: Decimal
    pending_evaluation_usd: Decimal

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SpendSnapshot":
        raw = _exact(
            value,
            {"actual_total_usd", "actual_infrastructure_usd", "actual_evaluation_usd", "pending_infrastructure_usd", "pending_evaluation_usd"},
            "spend snapshot",
        )
        result = cls(*(_money(raw[name], name) for name in (
            "actual_total_usd", "actual_infrastructure_usd", "actual_evaluation_usd",
            "pending_infrastructure_usd", "pending_evaluation_usd",
        )))
        if result.actual_infrastructure_usd + result.actual_evaluation_usd != result.actual_total_usd:
            raise CapacityPlanningError("actual total must equal infrastructure plus evaluation spend")
        return result

    @property
    def reserved_infrastructure_usd(self) -> Decimal:
        return self.actual_infrastructure_usd + self.pending_infrastructure_usd

    @property
    def reserved_evaluation_usd(self) -> Decimal:
        return self.actual_evaluation_usd + self.pending_evaluation_usd

    @property
    def reserved_total_usd(self) -> Decimal:
        return self.actual_total_usd + self.pending_infrastructure_usd + self.pending_evaluation_usd


def _spend_as_mapping(value: SpendSnapshot) -> dict[str, str]:
    return {
        "actual_total_usd": _money_text(value.actual_total_usd),
        "actual_infrastructure_usd": _money_text(value.actual_infrastructure_usd),
        "actual_evaluation_usd": _money_text(value.actual_evaluation_usd),
        "pending_infrastructure_usd": _money_text(value.pending_infrastructure_usd),
        "pending_evaluation_usd": _money_text(value.pending_evaluation_usd),
    }


@dataclass(frozen=True)
class CapacityMeasurement:
    measurement_id: str
    observed_at: datetime
    timed_canary_receipt_sha256: str
    generated_tokens: int
    tokens_per_second: float
    trial_wall_seconds: float
    judge_latency_seconds: float
    judge_cost_usd_per_trial: Decimal
    per_gpu_hourly_usd: Decimal
    projected_storage_network_usd: Decimal
    spend: SpendSnapshot
    self_sha256: str


def create_capacity_measurement(
    *,
    measurement_id: str,
    observed_at: datetime,
    timed_canary_receipt: Mapping[str, Any],
    spend: Mapping[str, Any],
    projected_storage_network_usd: str,
) -> dict[str, Any]:
    """Bind live timing/cost evidence and a ledger snapshot into one input receipt."""

    canary = _mapping(timed_canary_receipt, "timed canary receipt")
    claimed = _digest(canary.pop("receipt_sha256", None), "timed canary receipt_sha256")
    if canary.get("format") != "truth_editing_timed_canary_receipt_v2" or _sha(canary) != claimed:
        raise CapacityPlanningError("timed canary receipt_sha256 differs")
    if set(canary) != _TIMED_CANARY_RECEIPT_FIELDS:
        raise CapacityPlanningError("timed canary receipt fields changed")
    _integer(canary.get("judge_calls"), "timed canary judge_calls", 1)
    # The canary runs exactly one trial. Its signed cost and judge elapsed are
    # therefore per-trial totals, regardless of how many judge calls it made.
    judge_cost = Decimal(str(_number(
        canary.get("judge_cost_usd"), "timed canary judge_cost_usd", 1e-12
    )))
    judge_elapsed = _number(
        canary.get("judge_elapsed_seconds"),
        "timed canary judge_elapsed_seconds",
    )
    measured_wall = _number(
        canary.get("measured_wall_seconds"),
        "timed canary measured_wall_seconds", 1e-12,
    )
    if judge_elapsed > measured_wall:
        raise CapacityPlanningError(
            "timed canary judge_elapsed_seconds cannot exceed measured_wall_seconds"
        )
    gpu_hourly = Decimal(str(_number(canary.get("gpu_hourly_usd"), "timed canary gpu_hourly_usd", 1e-12)))
    SpendSnapshot.from_mapping(spend)
    unsigned = {
        "format": MEASUREMENT_FORMAT,
        "measurement_id": _text(measurement_id, "measurement_id"),
        "observed_at": _utc_text(observed_at),
        "timed_canary_receipt_sha256": claimed,
        "generated_tokens": _integer(canary.get("generated_tokens"), "timed canary generated_tokens", 1),
        "tokens_per_second": _number(canary.get("tokens_per_second"), "timed canary tokens_per_second", 1e-12),
        "trial_wall_seconds": measured_wall,
        "judge_latency_seconds": judge_elapsed,
        "judge_cost_usd_per_trial": _money_text(judge_cost),
        "per_gpu_hourly_usd": _money_text(gpu_hourly),
        "projected_storage_network_usd": _money_text(_money(projected_storage_network_usd, "projected_storage_network_usd")),
        "spend": dict(spend),
    }
    return {**unsigned, "self_sha256": _sha(unsigned)}


def load_capacity_measurement(
    value: Mapping[str, Any], *, now: datetime, maximum_age_seconds: int = 6 * 3600
) -> CapacityMeasurement:
    raw = _exact(
        value,
        {
            "format", "measurement_id", "observed_at", "timed_canary_receipt_sha256",
            "generated_tokens", "tokens_per_second", "trial_wall_seconds",
            "judge_latency_seconds", "judge_cost_usd_per_trial", "per_gpu_hourly_usd",
            "projected_storage_network_usd", "spend", "self_sha256",
        },
        "capacity measurement",
    )
    claimed = _digest(raw.pop("self_sha256"), "capacity measurement self_sha256")
    if _sha(raw) != claimed:
        raise CapacityPlanningError("capacity measurement self_sha256 differs")
    if raw["format"] != MEASUREMENT_FORMAT:
        raise CapacityPlanningError("capacity measurement format changed")
    observed = _utc(raw["observed_at"], "observed_at")
    if now.tzinfo is None:
        raise CapacityPlanningError("now must be timezone-aware")
    age = (now.astimezone(timezone.utc) - observed).total_seconds()
    if age < 0:
        raise CapacityPlanningError("capacity measurement is from the future")
    if age > maximum_age_seconds:
        raise CapacityPlanningError("capacity measurement is stale")
    tokens = _integer(raw["generated_tokens"], "generated_tokens", 1)
    tps = _number(raw["tokens_per_second"], "tokens_per_second", 1e-12)
    wall = _number(raw["trial_wall_seconds"], "trial_wall_seconds", 1e-12)
    latency = _number(raw["judge_latency_seconds"], "judge_latency_seconds")
    if latency > wall:
        raise CapacityPlanningError("judge_latency_seconds cannot exceed trial_wall_seconds")
    # TPS is independently consumed and must describe work that fits in the trial.
    if tokens / tps > wall * 1.01:
        raise CapacityPlanningError("tokens_per_second is inconsistent with trial_wall_seconds")
    spend = SpendSnapshot.from_mapping(raw["spend"])
    if spend.reserved_total_usd >= Decimal("50"):
        raise CapacityPlanningError("already-spent total budget leaves no launch capacity")
    if spend.reserved_evaluation_usd >= Decimal("5"):
        raise CapacityPlanningError("already-spent evaluation budget leaves no launch capacity")
    if spend.reserved_infrastructure_usd >= Decimal("45"):
        raise CapacityPlanningError("already-spent infrastructure budget leaves no launch capacity")
    judge_cost = _money(raw["judge_cost_usd_per_trial"], "judge_cost_usd_per_trial")
    gpu_hourly = _money(raw["per_gpu_hourly_usd"], "per_gpu_hourly_usd")
    if judge_cost <= 0 or gpu_hourly <= 0:
        raise CapacityPlanningError("measured judge cost and GPU hourly rate must be positive")
    return CapacityMeasurement(
        measurement_id=_text(raw["measurement_id"], "measurement_id"),
        observed_at=observed,
        timed_canary_receipt_sha256=_digest(raw["timed_canary_receipt_sha256"], "timed_canary_receipt_sha256"),
        generated_tokens=tokens, tokens_per_second=tps, trial_wall_seconds=wall,
        judge_latency_seconds=latency,
        judge_cost_usd_per_trial=judge_cost,
        per_gpu_hourly_usd=gpu_hourly,
        projected_storage_network_usd=_money(raw["projected_storage_network_usd"], "projected_storage_network_usd"),
        spend=spend,
        self_sha256=claimed,
    )


def _batch_floor(value: Decimal, batch: int, maximum: int) -> int:
    whole = int(value.to_integral_value(rounding=ROUND_FLOOR))
    return min(maximum, whole - whole % batch)


_TIER_FIELDS = {
    "name", "through_trial", "generation_multiplier", "judge_multiplier",
    "trial_wall_seconds", "judge_cost_usd_per_trial",
    "gpu_cost_usd_per_trial", "total_cost_usd_per_trial", "batch_size",
    "batch_duration_seconds_upper_bound",
    "batch_infrastructure_cost_usd_upper_bound",
    "batch_evaluation_cost_usd_upper_bound",
    "batch_total_cost_usd_upper_bound",
}


def _validated_tier_projections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 3:
        raise CapacityPlanningError("projection tiers must contain exactly three tiers")
    result: list[dict[str, Any]] = []
    expected = (("discovery", 80), ("expanded", 200), ("concentrated", 800))
    previous_duration = previous_infrastructure = previous_evaluation = Decimal(0)
    for index, ((expected_name, expected_through), candidate) in enumerate(zip(expected, value)):
        tier = _exact(candidate, _TIER_FIELDS, f"projection tier[{index}]")
        if tier["name"] != expected_name or tier["through_trial"] != expected_through:
            raise CapacityPlanningError("projection tier identities changed")
        generation_multiplier = _money(tier["generation_multiplier"], "generation_multiplier")
        judge_multiplier = _money(tier["judge_multiplier"], "judge_multiplier")
        duration = Decimal(str(_number(tier["trial_wall_seconds"], "trial_wall_seconds", 1e-12)))
        judge = _money(tier["judge_cost_usd_per_trial"], "judge_cost_usd_per_trial")
        infrastructure = _money(tier["gpu_cost_usd_per_trial"], "gpu_cost_usd_per_trial")
        total = _money(tier["total_cost_usd_per_trial"], "total_cost_usd_per_trial")
        batch = _integer(tier["batch_size"], "batch_size", 1)
        batch_duration = Decimal(str(_number(
            tier["batch_duration_seconds_upper_bound"],
            "batch_duration_seconds_upper_bound", 1e-12,
        )))
        batch_infrastructure = _money(
            tier["batch_infrastructure_cost_usd_upper_bound"],
            "batch_infrastructure_cost_usd_upper_bound",
        )
        batch_evaluation = _money(
            tier["batch_evaluation_cost_usd_upper_bound"],
            "batch_evaluation_cost_usd_upper_bound",
        )
        batch_total = _money(
            tier["batch_total_cost_usd_upper_bound"],
            "batch_total_cost_usd_upper_bound",
        )
        if generation_multiplier < 1 or judge_multiplier < 1:
            raise CapacityPlanningError("projection tier multipliers must be at least one")
        if batch != 8 or batch_duration != duration:
            raise CapacityPlanningError("projection tier batch shape is inconsistent")
        if total != infrastructure + judge:
            raise CapacityPlanningError("projection tier per-trial total is inconsistent")
        if (
            batch_infrastructure != infrastructure * batch
            or batch_evaluation != judge * batch
            or batch_total != batch_infrastructure + batch_evaluation
        ):
            raise CapacityPlanningError("projection tier batch cost is inconsistent")
        if (
            duration < previous_duration
            or batch_infrastructure < previous_infrastructure
            or batch_evaluation < previous_evaluation
        ):
            raise CapacityPlanningError("projection tiers must be conservatively nondecreasing")
        previous_duration = duration
        previous_infrastructure = batch_infrastructure
        previous_evaluation = batch_evaluation
        result.append(tier)
    return result


def _select_next_batch_projection(
    tiers: list[dict[str, Any]], *, next_completed_trials: int
) -> dict[str, Any]:
    if (
        isinstance(next_completed_trials, bool)
        or not isinstance(next_completed_trials, int)
        or next_completed_trials < 8
        or next_completed_trials > 800
        or next_completed_trials % 8
    ):
        raise CapacityPlanningError(
            "next_completed_trials must be a complete batch boundary from 8 through 800"
        )
    tier = next((row for row in tiers if next_completed_trials <= row["through_trial"]), None)
    if tier is None:  # Defensive: strict tier validation ends at trial 800.
        raise CapacityPlanningError("no projection tier covers the next batch")
    return {
        "tier": tier["name"],
        "through_trial": tier["through_trial"],
        "batch_size": tier["batch_size"],
        "batch_duration_seconds_upper_bound": tier["batch_duration_seconds_upper_bound"],
        "batch_infrastructure_cost_usd_upper_bound": tier[
            "batch_infrastructure_cost_usd_upper_bound"
        ],
        "batch_evaluation_cost_usd_upper_bound": tier[
            "batch_evaluation_cost_usd_upper_bound"
        ],
        "batch_total_cost_usd_upper_bound": tier[
            "batch_total_cost_usd_upper_bound"
        ],
    }


def build_capacity_receipt(
    *, policy: CapacityPolicy, measurement: CapacityMeasurement, planned_at: datetime,
    completed_trials: int = 0,
    source_batch_observation_sha256: str | None = None,
    source_judge_ledger_after_receipt_sha256: str | None = None,
    remaining_search_seconds: float | None = None,
) -> dict[str, Any]:
    if (
        isinstance(completed_trials, bool)
        or not isinstance(completed_trials, int)
        or completed_trials < 0
        or completed_trials > policy.maximum_trials
        or completed_trials % policy.batch_size
    ):
        raise CapacityPlanningError("completed_trials must be a complete batch boundary")
    if source_batch_observation_sha256 is not None:
        source_batch_observation_sha256 = _digest(
            source_batch_observation_sha256,
            "source_batch_observation_sha256",
        )
    if source_judge_ledger_after_receipt_sha256 is not None:
        source_judge_ledger_after_receipt_sha256 = _digest(
            source_judge_ledger_after_receipt_sha256,
            "source_judge_ledger_after_receipt_sha256",
        )
    if (
        (completed_trials == 0) != (source_batch_observation_sha256 is None)
        or (completed_trials == 0)
        != (source_judge_ledger_after_receipt_sha256 is None)
    ):
        raise CapacityPlanningError(
            "completed trials and source batch observation identity are inconsistent"
        )
    available_search_seconds = Decimal(
        str(
            policy.search_seconds
            if remaining_search_seconds is None
            else _number(
                remaining_search_seconds,
                "remaining_search_seconds",
            )
        )
    )
    if available_search_seconds > policy.search_seconds:
        raise CapacityPlanningError(
            "remaining_search_seconds cannot exceed the search window"
        )
    age = (planned_at.astimezone(timezone.utc) - measurement.observed_at).total_seconds()
    if age < 0 or age > policy.maximum_measurement_age_seconds:
        raise CapacityPlanningError("capacity measurement is stale or from the future at planning time")
    measured_wall = Decimal(str(measurement.trial_wall_seconds))
    generation_seconds = Decimal(measurement.generated_tokens) / Decimal(str(measurement.tokens_per_second))
    judge_latency = Decimal(str(measurement.judge_latency_seconds))
    fixed_seconds = measured_wall - generation_seconds - judge_latency
    if fixed_seconds < 0:
        raise CapacityPlanningError("TPS and judge latency exceed measured trial duration")
    tier_projection: list[dict[str, Any]] = []
    for name, through_trial, generation_multiplier, judge_multiplier in policy.tier_multipliers:
        duration = policy.duration_margin * (
            fixed_seconds
            + generation_seconds * generation_multiplier
            + judge_latency * judge_multiplier
        )
        judge_cost = measurement.judge_cost_usd_per_trial * policy.judge_cost_margin * judge_multiplier
        gpu_cost = measurement.per_gpu_hourly_usd * duration / Decimal(3600)
        batch_gpu_cost = gpu_cost * policy.batch_size
        batch_judge_cost = judge_cost * policy.batch_size
        tier_projection.append({
            "name": name, "through_trial": through_trial,
            "generation_multiplier": _money_text(generation_multiplier),
            "judge_multiplier": _money_text(judge_multiplier),
            "trial_wall_seconds": float(duration),
            "judge_cost_usd_per_trial": _money_text(judge_cost),
            "gpu_cost_usd_per_trial": _money_text(gpu_cost),
            "total_cost_usd_per_trial": _money_text(gpu_cost + judge_cost),
            "batch_size": policy.batch_size,
            "batch_duration_seconds_upper_bound": float(duration),
            "batch_infrastructure_cost_usd_upper_bound": _money_text(batch_gpu_cost),
            "batch_evaluation_cost_usd_upper_bound": _money_text(batch_judge_cost),
            "batch_total_cost_usd_upper_bound": _money_text(
                batch_gpu_cost + batch_judge_cost
            ),
        })
    tier_projection = _validated_tier_projections(tier_projection)
    remaining_total = policy.total_budget_usd - measurement.spend.reserved_total_usd
    remaining_infrastructure = (
        policy.infrastructure_budget_usd - measurement.spend.reserved_infrastructure_usd
    )
    remaining_eval = policy.evaluation_budget_usd - measurement.spend.reserved_evaluation_usd
    reserved_finalization_infrastructure = (
        measurement.per_gpu_hourly_usd
        * policy.batch_size
        * Decimal(policy.finalization_seconds)
        / Decimal(3600)
        + measurement.projected_storage_network_usd
    )
    search_infrastructure = remaining_infrastructure - reserved_finalization_infrastructure
    # The final-selection reserve is 20% of the full $5 evaluation envelope
    # ($1), not 20% of whatever happens to remain after search spend.  Shrinking
    # the reserve as spend rises would silently consume the final-selection pool.
    reserved_eval = policy.evaluation_budget_usd * policy.evaluation_reserve_fraction
    search_eval = remaining_eval - reserved_eval
    search_total = remaining_total - reserved_finalization_infrastructure - reserved_eval
    if search_infrastructure <= 0 or search_total <= 0:
        raise CapacityPlanningError("finalization reserve leaves no search infrastructure budget")
    cumulative_time = cumulative_total = cumulative_infrastructure = cumulative_eval = Decimal(0)
    limits = {
        "time": completed_trials, "total": completed_trials,
        "infrastructure": completed_trials, "evaluation": completed_trials,
    }
    for additional in range(
        policy.batch_size,
        policy.maximum_trials - completed_trials + 1,
        policy.batch_size,
    ):
        completed = completed_trials + additional
        tier = next(row for row in tier_projection if completed <= row["through_trial"])
        duration = Decimal(str(tier["trial_wall_seconds"]))
        gpu_cost = Decimal(tier["gpu_cost_usd_per_trial"])
        judge_cost = Decimal(tier["judge_cost_usd_per_trial"])
        cumulative_time += duration
        cumulative_infrastructure += gpu_cost * policy.batch_size
        cumulative_eval += judge_cost * policy.batch_size
        cumulative_total += (gpu_cost + judge_cost) * policy.batch_size
        if cumulative_time <= available_search_seconds:
            limits["time"] = completed
        if cumulative_total <= search_total:
            limits["total"] = completed
        if cumulative_infrastructure <= search_infrastructure:
            limits["infrastructure"] = completed
        if cumulative_eval <= search_eval:
            limits["evaluation"] = completed
    time_limit = limits["time"]
    total_limit = limits["total"]
    infrastructure_limit = limits["infrastructure"]
    eval_limit = limits["evaluation"]
    target = min(time_limit, total_limit, infrastructure_limit, eval_limit, policy.maximum_trials)
    minimum_guarantee_lost = (
        completed_trials < policy.minimum_trials and target < policy.minimum_trials
    )
    receipt_target = policy.minimum_trials if minimum_guarantee_lost else target
    next_batch_projection = _select_next_batch_projection(
        tier_projection, next_completed_trials=policy.batch_size
    )
    unsigned = {
        "format": RECEIPT_FORMAT,
        "policy_sha256": policy.self_sha256,
        "measurement_sha256": measurement.self_sha256,
        "source_batch_observation_sha256": source_batch_observation_sha256,
        "source_judge_ledger_after_receipt_sha256": (
            source_judge_ledger_after_receipt_sha256
        ),
        "completed_through_trial": completed_trials,
        "timed_canary_receipt_sha256": measurement.timed_canary_receipt_sha256,
        "planned_at": _utc_text(planned_at),
        "measured": {
            "generated_tokens": measurement.generated_tokens,
            "tokens_per_second": measurement.tokens_per_second,
            "trial_wall_seconds": measurement.trial_wall_seconds,
            "judge_latency_seconds": measurement.judge_latency_seconds,
            "judge_cost_usd_per_trial": _money_text(measurement.judge_cost_usd_per_trial),
            "per_gpu_hourly_usd": _money_text(measurement.per_gpu_hourly_usd),
            "projected_storage_network_usd": _money_text(measurement.projected_storage_network_usd),
        },
        "conservative_projection": {
            "generation_seconds_from_measured_tps": float(generation_seconds),
            "fixed_seconds": float(fixed_seconds),
            "duration_multiplier": _money_text(policy.duration_margin),
            "judge_cost_multiplier": _money_text(policy.judge_cost_margin),
            "tiers": tier_projection,
        },
        "capacity_limits": {
            "time_limited_trials": time_limit,
            "total_budget_limited_trials": total_limit,
            "infrastructure_budget_limited_trials": infrastructure_limit,
            "evaluation_budget_limited_trials": eval_limit,
        },
        "next_batch_projection": next_batch_projection,
        "budget": {
            "total_budget_usd": _money_text(policy.total_budget_usd),
            "actual_total_usd": _money_text(measurement.spend.actual_total_usd),
            "pending_total_usd": _money_text(measurement.spend.pending_infrastructure_usd + measurement.spend.pending_evaluation_usd),
            "remaining_all_in_usd": _money_text(remaining_total),
            "infrastructure_budget_usd": _money_text(policy.infrastructure_budget_usd),
            "actual_infrastructure_usd": _money_text(measurement.spend.actual_infrastructure_usd),
            "pending_infrastructure_usd": _money_text(measurement.spend.pending_infrastructure_usd),
            "remaining_infrastructure_usd": _money_text(remaining_infrastructure),
            "projected_storage_network_usd": _money_text(measurement.projected_storage_network_usd),
            "reserved_finalization_infrastructure_usd": _money_text(reserved_finalization_infrastructure),
            "search_infrastructure_usd": _money_text(search_infrastructure),
            "evaluation_budget_usd": _money_text(policy.evaluation_budget_usd),
            "actual_evaluation_usd": _money_text(measurement.spend.actual_evaluation_usd),
            "pending_evaluation_usd": _money_text(measurement.spend.pending_evaluation_usd),
            "search_evaluation_usd": _money_text(search_eval),
            "reserved_evaluation_usd": _money_text(reserved_eval),
        },
        "decision": {
            "batch_size": policy.batch_size,
            "minimum_trials": policy.minimum_trials,
            "maximum_trials": policy.maximum_trials,
            "planned_trial_limit": receipt_target,
            "planned_batch_limit": receipt_target // policy.batch_size,
            "minimum_trial_guarantee_met": not minimum_guarantee_lost,
            "search_seconds": policy.search_seconds,
            "finalization_seconds_reserved": policy.finalization_seconds,
            "projection_role": "advisory_reforecast_target",
            "reforecast_after_each_batch": True,
            "stop_conditions": [
                "search_deadline_reached", "total_budget_reserve_reached",
                "evaluation_budget_reserve_reached", "maximum_trials_reached",
            ],
        },
    }
    receipt = {**unsigned, "receipt_sha256": _sha(unsigned)}
    if minimum_guarantee_lost:
        raise MinimumTrialGuaranteeError(receipt)
    return receipt


def reforecast_capacity_receipt(
    *, policy: CapacityPolicy, previous_receipt: Mapping[str, Any],
    batch_observation: Mapping[str, Any], planned_at: datetime,
    remaining_search_seconds: float | None = None,
) -> dict[str, Any]:
    """Turn one signed eight-trial observation into the next advisory receipt.

    The batch reports conservative per-trial upper bounds.  The canary-bound GPU
    rate and storage/network reserve are inherited from the previous validated
    receipt; live spend and measured workload bounds come from the new signed
    observation.
    """

    previous = validate_capacity_receipt(previous_receipt)
    if previous["policy_sha256"] != policy.self_sha256:
        raise CapacityPlanningError("rolling receipt policy identity differs")
    observation = _exact(
        batch_observation,
        {
            "format", "observation_id", "observed_at",
            "timed_canary_receipt_sha256", "completed_through_trial",
            "batch_size", "generated_tokens_per_trial_upper_bound",
            "generation_seconds_per_trial_upper_bound",
            "trial_wall_seconds_upper_bound",
            "judge_elapsed_seconds_per_trial_upper_bound",
            "judge_cost_usd_per_trial_upper_bound",
            "judge_ledger_before_receipt_sha256",
            "judge_ledger_after_receipt_sha256", "judge_calls",
            "judge_failures", "judge_elapsed_seconds_total",
            "judge_cost_usd_total", "spend", "self_sha256",
        },
        "rolling batch observation",
    )
    claimed = _digest(
        observation.pop("self_sha256"), "rolling batch observation self_sha256"
    )
    if _sha(observation) != claimed:
        raise CapacityPlanningError("rolling batch observation self_sha256 differs")
    if observation["format"] != BATCH_OBSERVATION_FORMAT:
        raise CapacityPlanningError("rolling batch observation format changed")
    if observation["timed_canary_receipt_sha256"] != previous["timed_canary_receipt_sha256"]:
        raise CapacityPlanningError("rolling batch observation timed canary identity differs")
    completed = _integer(
        observation["completed_through_trial"], "completed_through_trial", 8
    )
    if completed > policy.maximum_trials or completed % policy.batch_size:
        raise CapacityPlanningError(
            "completed_through_trial must be a complete batch boundary"
        )
    if observation["batch_size"] != policy.batch_size:
        raise CapacityPlanningError("rolling batch observation must contain exactly eight trials")
    observed = _utc(observation["observed_at"], "observed_at")
    if not isinstance(planned_at, datetime) or planned_at.tzinfo is None:
        raise CapacityPlanningError("planned_at must be timezone-aware")
    planned = planned_at.astimezone(timezone.utc)
    age = (planned - observed).total_seconds()
    if age < 0 or age > policy.maximum_measurement_age_seconds:
        raise CapacityPlanningError("rolling batch observation is stale or from the future")
    tokens = _integer(
        observation["generated_tokens_per_trial_upper_bound"],
        "generated_tokens_per_trial_upper_bound", 1,
    )
    generation = _number(
        observation["generation_seconds_per_trial_upper_bound"],
        "generation_seconds_per_trial_upper_bound", 1e-12,
    )
    wall = _number(
        observation["trial_wall_seconds_upper_bound"],
        "trial_wall_seconds_upper_bound", 1e-12,
    )
    judge_elapsed = _number(
        observation["judge_elapsed_seconds_per_trial_upper_bound"],
        "judge_elapsed_seconds_per_trial_upper_bound",
    )
    if generation > wall or judge_elapsed > wall or generation + judge_elapsed > wall:
        raise CapacityPlanningError(
            "generation and judge elapsed upper bounds cannot exceed trial wall time"
        )
    judge_cost = _money(
        observation["judge_cost_usd_per_trial_upper_bound"],
        "judge_cost_usd_per_trial_upper_bound",
    )
    if judge_cost <= 0:
        raise CapacityPlanningError("judge_cost_usd_per_trial_upper_bound must be positive")
    ledger_before = _digest(
        observation["judge_ledger_before_receipt_sha256"],
        "judge_ledger_before_receipt_sha256",
    )
    ledger_after = _digest(
        observation["judge_ledger_after_receipt_sha256"],
        "judge_ledger_after_receipt_sha256",
    )
    judge_calls = _integer(observation["judge_calls"], "judge_calls")
    judge_failures = _integer(observation["judge_failures"], "judge_failures")
    judge_elapsed_total = Decimal(str(_number(
        observation["judge_elapsed_seconds_total"],
        "judge_elapsed_seconds_total",
    )))
    judge_cost_total = _money(
        observation["judge_cost_usd_total"], "judge_cost_usd_total"
    )
    if judge_failures > judge_calls:
        raise CapacityPlanningError("judge_failures cannot exceed judge_calls")
    if judge_calls == 0 and (
        judge_failures != 0 or judge_elapsed_total != 0 or judge_cost_total != 0
    ):
        raise CapacityPlanningError("zero judge calls require zero judge deltas")
    all_judge_deltas_zero = (
        judge_calls == 0
        and judge_failures == 0
        and judge_elapsed_total == 0
        and judge_cost_total == 0
    )
    if all_judge_deltas_zero and ledger_before != ledger_after:
        raise CapacityPlanningError("zero judge deltas require an unchanged ledger receipt")
    if judge_calls > 0 and ledger_before == ledger_after:
        raise CapacityPlanningError("nonzero judge calls require a changed ledger receipt")
    if Decimal(str(judge_elapsed)) * policy.batch_size < judge_elapsed_total:
        raise CapacityPlanningError("per-trial judge elapsed bound underprices the batch total")
    if judge_cost * policy.batch_size < judge_cost_total:
        raise CapacityPlanningError("per-trial judge cost bound underprices the batch total")
    spend = SpendSnapshot.from_mapping(observation["spend"])
    measured = _exact(
        previous["measured"],
        {
            "generated_tokens", "tokens_per_second", "trial_wall_seconds",
            "judge_latency_seconds", "judge_cost_usd_per_trial",
            "per_gpu_hourly_usd", "projected_storage_network_usd",
        },
        "previous measured capacity",
    )
    if judge_elapsed < _number(
        measured["judge_latency_seconds"], "previous judge_latency_seconds"
    ):
        raise CapacityPlanningError(
            "rolling per-trial judge elapsed bound cannot decrease"
        )
    if judge_cost < _money(
        measured["judge_cost_usd_per_trial"],
        "previous judge_cost_usd_per_trial",
    ):
        raise CapacityPlanningError("rolling per-trial judge cost bound cannot decrease")
    previous_budget = _mapping(previous["budget"], "previous capacity budget")
    previous_evaluation = _money(
        previous_budget.get("actual_evaluation_usd"),
        "previous actual_evaluation_usd",
    )
    if spend.actual_evaluation_usd < previous_evaluation:
        raise CapacityPlanningError("rolling evaluation spend moved backwards")
    if spend.actual_evaluation_usd - previous_evaluation != judge_cost_total:
        raise CapacityPlanningError(
            "rolling evaluation spend delta differs from judge ledger cost delta"
        )
    measurement_unsigned = {
        "format": MEASUREMENT_FORMAT,
        "measurement_id": _text(observation["observation_id"], "observation_id"),
        "observed_at": _utc_text(observed),
        "timed_canary_receipt_sha256": previous["timed_canary_receipt_sha256"],
        "generated_tokens": tokens,
        "tokens_per_second": tokens / generation,
        "trial_wall_seconds": wall,
        "judge_latency_seconds": judge_elapsed,
        "judge_cost_usd_per_trial": _money_text(judge_cost),
        "per_gpu_hourly_usd": measured["per_gpu_hourly_usd"],
        "projected_storage_network_usd": measured["projected_storage_network_usd"],
        "spend": _spend_as_mapping(spend),
    }
    measurement_raw = {
        **measurement_unsigned, "self_sha256": _sha(measurement_unsigned)
    }
    measurement = load_capacity_measurement(
        measurement_raw,
        now=planned,
        maximum_age_seconds=policy.maximum_measurement_age_seconds,
    )
    return build_capacity_receipt(
        policy=policy, measurement=measurement, planned_at=planned,
        completed_trials=completed,
        source_batch_observation_sha256=claimed,
        source_judge_ledger_after_receipt_sha256=ledger_after,
        remaining_search_seconds=remaining_search_seconds,
    )


def validate_capacity_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(value, "capacity receipt")
    claimed = _digest(raw.pop("receipt_sha256", None), "capacity receipt_sha256")
    if raw.get("format") != RECEIPT_FORMAT or _sha(raw) != claimed:
        raise CapacityPlanningError("capacity receipt_sha256 differs")
    if set(raw) != {
        "format", "policy_sha256", "measurement_sha256",
        "source_batch_observation_sha256", "completed_through_trial",
        "source_judge_ledger_after_receipt_sha256",
        "timed_canary_receipt_sha256",
        "planned_at", "measured", "conservative_projection", "capacity_limits",
        "next_batch_projection", "budget", "decision",
    }:
        raise CapacityPlanningError("capacity receipt fields changed")
    source_observation = raw["source_batch_observation_sha256"]
    if source_observation is not None:
        _digest(source_observation, "source_batch_observation_sha256")
    source_judge_ledger = raw["source_judge_ledger_after_receipt_sha256"]
    if source_judge_ledger is not None:
        _digest(
            source_judge_ledger,
            "source_judge_ledger_after_receipt_sha256",
        )
    completed = _integer(raw["completed_through_trial"], "completed_through_trial")
    if completed > 800 or completed % 8:
        raise CapacityPlanningError(
            "completed_through_trial must be a complete batch boundary"
        )
    if (
        (completed == 0) != (source_observation is None)
        or (completed == 0) != (source_judge_ledger is None)
    ):
        raise CapacityPlanningError(
            "completed trials and source batch observation identity are inconsistent"
        )
    projection = _exact(
        raw["conservative_projection"],
        {
            "generation_seconds_from_measured_tps", "fixed_seconds",
            "duration_multiplier", "judge_cost_multiplier", "tiers",
        },
        "conservative projection",
    )
    _number(
        projection["generation_seconds_from_measured_tps"],
        "generation_seconds_from_measured_tps",
    )
    _number(projection["fixed_seconds"], "fixed_seconds")
    if _money(projection["duration_multiplier"], "duration_multiplier") < 1:
        raise CapacityPlanningError("duration_multiplier must be at least one")
    if _money(projection["judge_cost_multiplier"], "judge_cost_multiplier") < 1:
        raise CapacityPlanningError("judge_cost_multiplier must be at least one")
    tiers = _validated_tier_projections(projection["tiers"])
    expected_next = _select_next_batch_projection(tiers, next_completed_trials=8)
    if raw["next_batch_projection"] != expected_next:
        raise CapacityPlanningError("next batch projection differs from its tier")
    decision = _exact(
        raw.get("decision"),
        {
            "batch_size", "minimum_trials", "maximum_trials",
            "planned_trial_limit", "planned_batch_limit",
            "minimum_trial_guarantee_met", "search_seconds",
            "finalization_seconds_reserved", "projection_role",
            "reforecast_after_each_batch", "stop_conditions",
        },
        "capacity receipt decision",
    )
    target = _integer(decision.get("planned_trial_limit"), "planned_trial_limit", 200)
    if target > 800 or target % 8 or target < completed:
        raise CapacityPlanningError("planned trial limit must be 200-800 in complete batches of eight")
    guarantee_met = decision["minimum_trial_guarantee_met"]
    if not isinstance(guarantee_met, bool):
        raise CapacityPlanningError("minimum_trial_guarantee_met must be boolean")
    capacity_limit = min(
        _integer(raw["capacity_limits"][name], name)
        for name in (
            "time_limited_trials", "total_budget_limited_trials",
            "infrastructure_budget_limited_trials",
            "evaluation_budget_limited_trials",
        )
    )
    expected_guarantee = completed >= 200 or capacity_limit >= 200
    if guarantee_met != expected_guarantee:
        raise CapacityPlanningError("minimum trial guarantee marker differs from capacity")
    return {**raw, "receipt_sha256": claimed}


def select_next_batch_projection(
    receipt: Mapping[str, Any], *, next_completed_trials: int
) -> dict[str, Any]:
    """Return the conservative tier bound for the exact next batch boundary.

    The receipt is identity- and schema-validated on every call so schedulers do
    not need to understand tier internals or trust a stale discovery-tier alias.
    """

    validated = validate_capacity_receipt(receipt)
    tiers = validated["conservative_projection"]["tiers"]
    return _select_next_batch_projection(tiers, next_completed_trials=next_completed_trials)


def write_capacity_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    value = validate_capacity_receipt(receipt)
    if path.exists() or path.is_symlink():
        raise CapacityPlanningError("capacity receipt path already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


__all__ = [
    "CapacityMeasurement", "CapacityPlanningError", "CapacityPolicy",
    "MinimumTrialGuaranteeError",
    "SpendSnapshot", "build_capacity_receipt", "create_capacity_measurement", "load_capacity_measurement",
    "reforecast_capacity_receipt",
    "select_next_batch_projection", "validate_capacity_receipt", "write_capacity_receipt",
]
