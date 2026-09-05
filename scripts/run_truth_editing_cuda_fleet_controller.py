#!/usr/bin/env python3
"""Adaptive Optuna controller for eight persistent CUDA-isolated workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from intelligent_liars.truth_editing_adaptive_run import AdaptiveBatchScheduler  # noqa: E402
from intelligent_liars.truth_editing_adaptive_finalization import (  # noqa: E402
    run_adaptive_finalization,
    write_adaptive_finalization_handoff,
)
from intelligent_liars.truth_editing_adaptive_causal_preparation import (  # noqa: E402
    prepare_adaptive_causal_controls,
)
from intelligent_liars.truth_editing_capacity import (  # noqa: E402
    CapacityPlanningError,
    CapacityPolicy,
    MinimumTrialGuaranteeError,
    SpendSnapshot,
    reforecast_capacity_receipt,
    validate_capacity_receipt,
)
from intelligent_liars.truth_editing_contracts import canonical_sha256  # noqa: E402
from intelligent_liars.truth_editing_directions import DirectionBank  # noqa: E402
from intelligent_liars.truth_editing_production import (  # noqa: E402
    ImmutableStudyArtifactAdapter,
    ProductionRunConfig,
    ProductionTruthEditingRun,
    configured_search_driver,
    open_finalist_export_inputs,
    open_production_run,
)
from intelligent_liars.truth_editing_production_causal_materializer import (  # noqa: E402
    ProductionCausalCandidateMaterializer,
)
from intelligent_liars.truth_editing_production_finalization import (  # noqa: E402
    ProductionAdaptiveFinalizationExecutor,
    ProductionFinalistCheckpointExporter,
)
from intelligent_liars.truth_editing_final_checkpoint_publication import (  # noqa: E402
    S3FinalCheckpointStore,
    build_final_checkpoint_target,
    open_final_checkpoint_publication_receipt,
    publish_final_checkpoint,
    retire_verified_local_checkpoint_weights,
)
from intelligent_liars.truth_editing_finalization_progress_store import (  # noqa: E402
    FinalizationProgressBinding,
    FinalizationProgressError,
    FinalizationProgressRepository,
)
from intelligent_liars.truth_editing_finalist_checkpoint import (  # noqa: E402
    open_finalist_checkpoint,
)
from intelligent_liars.truth_editing_gpu_telemetry import GpuTelemetryCollector  # noqa: E402
from intelligent_liars.truth_editing_production_judge_budget import (  # noqa: E402
    ProductionJudgeBudget,
    parse_production_judge_budget_receipt,
)
from intelligent_liars.truth_editing_qwen_causal_backend import (  # noqa: E402
    create_qwen_causal_executor_with_base_bundle,
)
from intelligent_liars.truth_editing_phase_checkpoint import (  # noqa: E402
    publish_adaptive_checkpoint,
    restore_adaptive_checkpoint,
)
from intelligent_liars.truth_editing_offhost_checkpoint import (  # noqa: E402
    OffHostCheckpointError,
    OffHostCheckpointRepository,
    OffHostCheckpointTarget,
    PartialBatchBinding,
    S3VersionedObjectStore,
    SnapshotBinding,
    hydrate_offhost_partial_snapshot,
    hydrate_offhost_snapshot,
    materialize_offhost_snapshot,
)
from intelligent_liars.truth_editing_study import (  # noqa: E402
    CompletedBatchCommit,
    PreparedStudyContext,
    TruthEditingStudy,
    TruthEditingStudyConfig,
    load_truth_editing_study_config,
)
from intelligent_liars.truth_editing_vast_fleet import (  # noqa: E402
    ADAPTIVE_FLEET_FORMAT,
    FleetBatchEvaluator,
    FleetConfig,
    SubprocessCudaWorker,
    verify_adaptive_fleet_bindings,
    verify_production_config_binding,
)
from intelligent_liars.truth_editing_wandb_checkpoint import (  # noqa: E402
    AdaptiveRunProgress,
    open_adaptive_progress_checkpoint,
    open_wandb_run_checkpoint,
)
from intelligent_liars.truth_editing_wandb_monitoring import (  # noqa: E402
    CoordinatorMonitor,
    CoordinatorTelemetryPump,
    MonitoredSearchDriver,
)


_OFFHOST_RESTORE_BOOTSTRAP_OUTPUTS = frozenset(
    {
        "model/cache-hydration-receipt.json",
        "model/model-verification-receipt.json",
    }
)


def _output_root_is_clean_for_offhost_restore(output_root: Path) -> bool:
    """Allow only model receipts created before the controller starts."""

    if not output_root.exists():
        return True
    if output_root.is_symlink() or not output_root.is_dir():
        return False
    for path in output_root.rglob("*"):
        if path.is_symlink():
            return False
        relative = path.relative_to(output_root).as_posix()
        if path.is_dir():
            if relative != "model":
                return False
        elif not path.is_file() or relative not in _OFFHOST_RESTORE_BOOTSTRAP_OUTPUTS:
            return False
    return True


def _hydrate_preserving_bootstrap_outputs(
    snapshot_root: Path,
    output_root: Path,
    *,
    binding: SnapshotBinding | PartialBatchBinding,
    hydrate: Callable[..., dict[str, object]],
) -> dict[str, object]:
    """Hydrate on a clean path, then restore verified bootstrap receipts."""

    if not _output_root_is_clean_for_offhost_restore(output_root):
        raise OffHostCheckpointError(
            "off-host hydration output contains non-bootstrap state"
        )
    preserved = {
        relative: (output_root / relative).read_bytes()
        for relative in _OFFHOST_RESTORE_BOOTSTRAP_OUTPUTS
        if (output_root / relative).is_file()
    }
    if output_root.exists():
        shutil.rmtree(output_root)
    try:
        result = hydrate(snapshot_root, output_root, binding=binding)
    finally:
        for relative, payload in preserved.items():
            destination = output_root / relative
            if destination.exists():
                if not destination.is_file() or destination.read_bytes() != payload:
                    raise OffHostCheckpointError(
                        "off-host snapshot collides with a bootstrap model receipt"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    return result


class _OrderedDurableCallback:
    """Serialize concurrent worker callbacks at the durable ordinal frontier.

    Workers are intentionally dispatched concurrently, but the off-host partial
    checkpoint is an ordered frontier.  Waiting callbacks are released when the
    preceding ordinal has published; a callback failure wakes every waiter so a
    failed publication cannot strand worker threads at a barrier.
    """

    def __init__(
        self,
        callback: Callable[[Mapping[str, object]], None],
        *,
        next_ordinal_reader: Callable[[], int],
    ) -> None:
        self._callback = callback
        self._next_ordinal_reader = next_ordinal_reader
        self._condition = threading.Condition()
        self._next_ordinal: int | None = None
        self._failure: BaseException | None = None

    def __call__(self, event: Mapping[str, object]) -> None:
        ordinal = event.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise ValueError("durable event ordinal must be an integer")
        with self._condition:
            if self._failure is not None:
                raise RuntimeError("durable callback frontier failed") from self._failure
            if self._next_ordinal is None:
                self._next_ordinal = self._next_ordinal_reader()
            while ordinal != self._next_ordinal:
                if ordinal < self._next_ordinal:
                    raise ValueError("durable event ordinal regressed")
                self._condition.wait()
                if self._failure is not None:
                    raise RuntimeError("durable callback frontier failed") from self._failure
            try:
                self._callback(event)
            except BaseException as error:
                self._failure = error
                self._condition.notify_all()
                raise
            self._next_ordinal += 1
            self._condition.notify_all()


def publish_completed_boundary_then_reconcile_judge_budget(
    *,
    publish_completed_boundary: Callable[[], None],
    judge_budget: ProductionJudgeBudget,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Durably publish a batch before resolving any uncertain paid calls."""

    publish_completed_boundary()
    return _reconcile_judge_budget(judge_budget)


def _reconcile_judge_budget(
    judge_budget: ProductionJudgeBudget,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    orphaned = judge_budget.recover_orphaned_reservations()
    reconciled = judge_budget.reconcile_ambiguous_requests()
    return orphaned, reconciled


def prepare_judge_budget_for_controller_resume(
    *,
    judge_budget: ProductionJudgeBudget,
    restored_boundary_is_durable: bool,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Reconcile restored paid-call state without mutating unbound local work."""

    if restored_boundary_is_durable:
        return _reconcile_judge_budget(judge_budget)
    receipt = judge_budget.receipt()
    if receipt["pending_call_count"] or receipt["circuit_open"]:
        raise ValueError(
            "local judge ledger has unresolved paid calls without a durable "
            "restored boundary"
        )
    return (), ()


def reconcile_restored_finalization_judge_budget(
    *,
    judge_budget: ProductionJudgeBudget,
    expected_receipt_sha256: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Validate finalization lineage before reconciling its restored ledger."""

    if judge_budget.receipt()["content_sha256"] != expected_receipt_sha256:
        raise ValueError("restored finalization judge ledger differs from its binding")
    return _reconcile_judge_budget(judge_budget)


def offhost_partial_event_is_already_published(
    *,
    repository: OffHostCheckpointRepository,
    committed_binding: SnapshotBinding,
    durable_event: Mapping[str, object],
    expected_receipt_directory: Path,
    trial_number_start: int = 0,
) -> bool:
    """Recognize one exact receipt already bound by a validated partial pointer."""

    expected_fields = {
        "format",
        "fleet_config_sha256",
        "trial_id",
        "ordinal",
        "request_sha256",
        "receipt_path",
        "receipt_sha256",
    }
    ordinal = durable_event.get("ordinal")
    receipt_path = durable_event.get("receipt_path")
    expected_trial_id = (
        f"trial-{ordinal + trial_number_start:04d}"
        if isinstance(ordinal, int) and not isinstance(ordinal, bool)
        else None
    )
    if (
        set(durable_event) != expected_fields
        or durable_event.get("format")
        != "truth_editing_vast_fleet_receipt_durable_event_v1"
        or durable_event.get("fleet_config_sha256")
        != committed_binding.fleet_config_sha256
        or isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or durable_event.get("trial_id") != expected_trial_id
        or not isinstance(receipt_path, str)
        or not receipt_path
        or Path(receipt_path).resolve()
        != (expected_receipt_directory / f"{expected_trial_id}.json").resolve()
    ):
        raise OffHostCheckpointError("restored partial event identity differs")
    existing: PartialBatchBinding | None = (
        repository.read_latest_partial_binding_if_present(committed_binding)
    )
    if existing is None:
        return False
    matched = next(
        (item for item in existing.durable_receipts if item.ordinal == ordinal), None
    )
    if matched is None:
        return False
    if (
        matched.trial_id != durable_event.get("trial_id")
        or matched.request_sha256 != durable_event.get("request_sha256")
        or matched.receipt_sha256 != durable_event.get("receipt_sha256")
    ):
        raise OffHostCheckpointError("restored partial event identity differs")
    return True


def publish_partial_event_if_needed(
    *,
    repository: OffHostCheckpointRepository,
    committed_binding: SnapshotBinding,
    durable_event: Mapping[str, object],
    expected_receipt_directory: Path,
    publish: Callable[[], object],
    trial_number_start: int = 0,
) -> object | None:
    """Publish a new frontier, but never rebuild an exact restored frontier."""

    if offhost_partial_event_is_already_published(
        repository=repository,
        committed_binding=committed_binding,
        durable_event=durable_event,
        expected_receipt_directory=expected_receipt_directory,
        trial_number_start=trial_number_start,
    ):
        return None
    return publish()


def host_hourly_usd_from_environment(environment: Mapping[str, str]) -> float:
    """Read one whole-host hourly price and reject the obsolete per-GPU name."""

    if "TRUTH_EDITING_GPU_HOURLY_USD" in environment:
        raise ValueError(
            "obsolete TRUTH_EDITING_GPU_HOURLY_USD is ambiguous for a multi-GPU "
            "host; use TRUTH_EDITING_HOST_HOURLY_USD"
        )
    raw = environment.get("TRUTH_EDITING_HOST_HOURLY_USD", "0")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "TRUTH_EDITING_HOST_HOURLY_USD must be a finite number"
        ) from error
    if not math.isfinite(value):
        raise ValueError("TRUTH_EDITING_HOST_HOURLY_USD must be a finite number")
    return value


def parse_host_lease_started_at_utc(value: str | None) -> datetime:
    """Strict-open the UTC timestamp captured when the Vast host shell began."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("host lease start must be an ISO UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("host lease start must be a valid ISO UTC timestamp") from error
    return parsed


def planned_fleet_study_identity_sha256(
    *,
    fleet: FleetConfig,
    production: ProductionRunConfig,
    study_config: TruthEditingStudyConfig,
    bank: DirectionBank,
) -> str:
    """Compute the real study identity before consulting an off-host pointer.

    The fleet's ``study_identity_sha256`` field binds the study *configuration*.
    Off-host snapshots bind the complete study identity, which additionally
    includes the direction bank, driver, evaluator, and orchestrator code.
    """

    identity_evaluator = FleetBatchEvaluator(
        fleet,
        worker_factory=lambda _slot: None,  # type: ignore[arg-type,return-value]
        receipt_directory_override=production.journal_path.parent.parent
        / "fleet-receipts",
    )
    identity_run = ProductionTruthEditingRun(
        study=TruthEditingStudy(study_config, bank.manifest),
        driver=configured_search_driver(production, study_config),
        evaluator=identity_evaluator,
        artifacts=ImmutableStudyArtifactAdapter(production.artifact_dir),
        journal_path=production.journal_path,
    )
    return identity_run.planned_study_identity_sha256


def adaptive_progress_boundary_is_already_recorded(
    path: Path,
    *,
    completed_trials: int,
    coverage: Mapping[str, tuple[int, int]],
) -> bool:
    """Make replay of a committed batch idempotent before publication.

    A crash may occur after the progress file advances but before its checkpoint
    generation is published. Replaying that batch must publish the existing
    progress node, not append a second node and break lineage continuity.
    """

    if not path.exists():
        return False
    existing = open_adaptive_progress_checkpoint(path).progress
    if existing.completed_search_trials > completed_trials:
        raise ValueError("adaptive progress is ahead of the replayed batch")
    if existing.completed_search_trials < completed_trials:
        return False
    if dict(existing.coverage) != dict(coverage):
        raise ValueError("adaptive progress coverage differs at replayed batch")
    return True


def offhost_boundary_is_already_published(
    latest_binding: object | None,
    requested_binding: object,
) -> bool:
    """Recognize an exact committed boundary during at-least-once replay."""

    return latest_binding is not None and latest_binding == requested_binding


DEFAULT_CAPACITY_POLICY = Path("configs/truth_editing_adaptive_capacity_policy_v1.json")
_OUTPUT_FIELDS = {
    "journal_path": "study/study-journal.json",
    "artifact_dir": "study/frozen",
    "runtime_output_dir": "study/runtime",
    "judge_cache_dir": "providers/judge-cache",
    "judge_budget_ledger_dir": "providers/production-judge-budget",
}


def _money_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _read_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _runtime_config(source_path: Path, output_root: Path) -> Path:
    """Materialize output-only overrides without changing frozen inputs."""

    source = _read_object(source_path, "production config")
    if source.get("format") != "truth_editing_production_config_v1":
        raise ValueError("production config format differs")
    runtime = dict(source)
    for field, relative in _OUTPUT_FIELDS.items():
        if field not in source:
            raise ValueError(f"production config is missing {field}")
        runtime[field] = str((output_root / relative).resolve())
    for field in ("model_cache_dir", "snapshot_manifest_path"):
        value = source.get(field)
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError(f"production {field} must be a portable relative path")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    runtime_identity = canonical_sha256(
        {
            "production_config_sha256": source_sha,
            "output_root": str(output_root.resolve()),
        }
    )
    target = source_path.with_name(
        f".truth_editing_production_runtime.{runtime_identity[:16]}.json"
    )
    payload = json.dumps(runtime, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.read_text() != payload:
            raise ValueError("runtime production config already differs")
        return target
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _write_json_immutable(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text() != payload:
            raise ValueError(f"immutable output already differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _batch_has_durable_receipt(
    receipt_directory: Path,
    *,
    fleet_config_sha256: str,
    completed_trials: int,
    batch_size: int,
    trial_number_start: int = 0,
) -> bool:
    """Prove that an otherwise-null journal batch has already incurred work."""

    found = False
    for ordinal in range(completed_trials, completed_trials + batch_size):
        trial_id = f"trial-{ordinal + trial_number_start:04d}"
        path = receipt_directory / f"{trial_id}.json"
        if not path.exists() and not path.is_symlink():
            continue
        raw = _read_object(path, "durable partial trial receipt")
        receipt_format = raw.get("format")
        fields = {
            "format",
            "fleet_config_sha256",
            "trial_id",
            "ordinal",
            "request_sha256",
            "worker_slot",
            "result",
            "receipt_sha256",
        }
        if receipt_format == "truth_editing_vast_fleet_trial_receipt_v2":
            fields.add("telemetry")
        elif receipt_format != "truth_editing_vast_fleet_trial_receipt_v1":
            raise ValueError("durable partial trial receipt format differs")
        if set(raw) != fields:
            raise ValueError("durable partial trial receipt fields differ")
        unsigned = dict(raw)
        claimed = unsigned.pop("receipt_sha256")
        if (
            claimed != canonical_sha256(unsigned)
            or raw["fleet_config_sha256"] != fleet_config_sha256
            or raw["trial_id"] != trial_id
            or raw["ordinal"] != ordinal
        ):
            raise ValueError("durable partial trial receipt identity differs")
        found = True
    return found


class _ControllerSpendReader:
    """Translate host time plus the judge ledger into scheduler spend."""

    def __init__(
        self,
        *,
        capacity_receipt: dict[str, object],
        judge_budget: ProductionJudgeBudget,
        host_hourly_usd: Decimal,
        host_lease_started_at: datetime,
        worker_count: int,
        prior_spend: SpendSnapshot | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        budget = capacity_receipt.get("budget")
        if not isinstance(budget, dict):
            raise ValueError("capacity receipt budget is missing")
        try:
            self._baseline = SpendSnapshot.from_mapping(
                {
                    "actual_total_usd": budget["actual_total_usd"],
                    "actual_infrastructure_usd": budget["actual_infrastructure_usd"],
                    "actual_evaluation_usd": budget["actual_evaluation_usd"],
                    "pending_infrastructure_usd": budget["pending_infrastructure_usd"],
                    "pending_evaluation_usd": budget["pending_evaluation_usd"],
                }
            )
        except KeyError as error:
            raise ValueError("capacity receipt spend baseline is incomplete") from error
        if host_hourly_usd <= 0 or worker_count != 8:
            raise ValueError(
                "adaptive production spend requires a priced eight-GPU host"
            )
        if host_lease_started_at.tzinfo is None:
            raise ValueError("host lease start must be timezone-aware")
        self._judge_budget = judge_budget
        self._host_hourly_usd = host_hourly_usd
        self._host_lease_started_at = host_lease_started_at.astimezone(timezone.utc)
        self._prior_infrastructure_carry = (
            max(
                Decimal("0"),
                prior_spend.actual_infrastructure_usd
                - self._baseline.actual_infrastructure_usd,
            )
            if prior_spend is not None
            else Decimal("0")
        )
        self._clock = clock

    def rebind_judge_budget(self, judge_budget: ProductionJudgeBudget) -> None:
        """Use the exact ledger restored by the finalization progress chain."""

        self._judge_budget = judge_budget

    def __call__(self) -> SpendSnapshot:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("controller spend clock must be timezone-aware")
        elapsed = Decimal(
            str(max(0.0, (now - self._host_lease_started_at).total_seconds()))
        )
        host_cost = self._host_hourly_usd * elapsed / Decimal(3600)
        judge = self._judge_budget.receipt()
        judge_actual = Decimal(str(judge["actual_spend_usd"]))
        judge_reserved = Decimal(str(judge["reserved_or_spent_usd"]))
        infrastructure = (
            self._baseline.actual_infrastructure_usd
            + self._prior_infrastructure_carry
            + host_cost
        )
        evaluation = self._baseline.actual_evaluation_usd + judge_actual
        return SpendSnapshot(
            actual_total_usd=infrastructure + evaluation,
            actual_infrastructure_usd=infrastructure,
            actual_evaluation_usd=evaluation,
            pending_infrastructure_usd=self._baseline.pending_infrastructure_usd,
            pending_evaluation_usd=(
                self._baseline.pending_evaluation_usd
                + judge_reserved
                - judge_actual
            ),
        )


def _prior_spend_from_checkpoint(
    checkpoint_path: Path,
) -> SpendSnapshot | None:
    if not checkpoint_path.is_file():
        return None
    checkpoint = _read_object(checkpoint_path, "adaptive checkpoint")
    try:
        return SpendSnapshot.from_mapping(checkpoint["last_spend_snapshot"])
    except KeyError as error:
        raise ValueError("adaptive checkpoint spend baseline is missing") from error


class _RollingCapacityController:
    """Accumulate one exact batch of worker telemetry and sign its reforecast."""

    def __init__(
        self,
        *,
        policy: CapacityPolicy,
        initial_receipt: dict[str, object],
        rolling_receipt_path: Path,
        spend_reader: Callable[[], SpendSnapshot],
        judge_budget: ProductionJudgeBudget,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        search_deadline_reader: Callable[[], datetime] | None = None,
    ) -> None:
        self._policy = policy
        self._initial_receipt = validate_capacity_receipt(initial_receipt)
        self._rolling_receipt_path = rolling_receipt_path
        self._spend_reader = spend_reader
        self._judge_budget = judge_budget
        self._clock = clock
        self._search_deadline_reader = search_deadline_reader
        self._telemetry: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()
        self._judge_state_path = (
            judge_budget.path / "controller-capacity-snapshot.json"
        )
        self._batch_clock_path = judge_budget.path / "controller-batch-clock.json"
        if not self._rolling_receipt_path.exists():
            self._write_rolling_receipt(self._initial_receipt)
        current_receipt = self.current_receipt()
        expected_judge_sha = current_receipt[
            "source_judge_ledger_after_receipt_sha256"
        ]
        live_judge_receipt = parse_production_judge_budget_receipt(
            judge_budget.receipt()
        )
        live_judge_snapshot = judge_budget.monitoring_snapshot()
        if self._judge_state_path.exists():
            state = _read_object(self._judge_state_path, "judge capacity snapshot")
            if set(state) != {
                "format", "receipt", "monitoring_snapshot", "self_sha256"
            } or state.get("format") != (
                "truth_editing_controller_judge_capacity_snapshot_v1"
            ):
                raise ValueError("judge capacity snapshot fields differ")
            unsigned = dict(state)
            claimed = unsigned.pop("self_sha256")
            if claimed != canonical_sha256(unsigned):
                raise ValueError("judge capacity snapshot identity differs")
            receipt = state.get("receipt")
            snapshot = state.get("monitoring_snapshot")
            if not isinstance(receipt, dict) or not isinstance(snapshot, dict):
                raise ValueError("judge capacity snapshot is malformed")
            self._judge_before = parse_production_judge_budget_receipt(receipt)
            self._judge_snapshot_before = snapshot
            saved_judge_sha = self._judge_before["content_sha256"]
            if expected_judge_sha is not None and saved_judge_sha != expected_judge_sha:
                # The rolling receipt is replaced before this local snapshot. A
                # crash in that narrow window is recoverable only when the live
                # identity-hashed ledger is exactly the receipt named by the
                # signed batch observation.
                if live_judge_receipt["content_sha256"] != expected_judge_sha:
                    raise ValueError(
                        "judge capacity snapshot is not on the rolling receipt lineage"
                    )
                self._judge_before = live_judge_receipt
                self._judge_snapshot_before = live_judge_snapshot
                self._save_judge_state()
        else:
            if (
                expected_judge_sha is not None
                and live_judge_receipt["content_sha256"] != expected_judge_sha
            ):
                raise ValueError(
                    "live judge ledger is not on the rolling receipt lineage"
                )
            self._judge_before = live_judge_receipt
            self._judge_snapshot_before = live_judge_snapshot
            self._save_judge_state()
        self._resume_batch_clock_after_controller_restart()

    def _save_judge_state(self) -> None:
        unsigned = {
            "format": "truth_editing_controller_judge_capacity_snapshot_v1",
            "receipt": self._judge_before,
            "monitoring_snapshot": self._judge_snapshot_before,
        }
        value = {**unsigned, "self_sha256": canonical_sha256(unsigned)}
        payload = json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
        self._judge_state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._judge_state_path.with_name(
            f".{self._judge_state_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(payload)
        temporary.replace(self._judge_state_path)

    def record_trial(
        self, _gpu_slot: int, trial_id: str, telemetry: dict[str, float]
    ) -> None:
        with self._lock:
            self._telemetry[trial_id] = dict(telemetry)

    def record_batch_admission(self, *, completed_trials: int) -> None:
        """Persist the start of an authorized batch before coordinator work."""

        if (
            isinstance(completed_trials, bool)
            or not isinstance(completed_trials, int)
            or completed_trials < 0
            or completed_trials % self._policy.batch_size
        ):
            raise CapacityPlanningError("batch admission boundary is invalid")
        expected_completed = completed_trials + self._policy.batch_size
        if expected_completed > self._policy.maximum_trials:
            raise CapacityPlanningError("batch admission exceeds the trial ceiling")
        if self._batch_clock_path.exists():
            current = self._load_batch_clock()
            if current["expected_completed_trials"] != expected_completed:
                raise CapacityPlanningError("another batch wall clock is active")
            if current["active_started_at_utc"] is None:
                current["active_started_at_utc"] = self._clock_text(self._clock())
                self._write_batch_clock(current)
            return
        self._write_batch_clock({
            "format": "truth_editing_controller_batch_wall_clock_v2",
            "expected_completed_trials": expected_completed,
            "active_elapsed_seconds": 0.0,
            "active_started_at_utc": self._clock_text(self._clock()),
        })

    @staticmethod
    def _clock_text(value: datetime) -> str:
        if value.tzinfo is None:
            raise CapacityPlanningError("batch wall clock must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_clock_text(value: object) -> datetime:
        if not isinstance(value, str) or not value.endswith("Z"):
            raise CapacityPlanningError("batch wall clock timestamp is invalid")
        try:
            return datetime.fromisoformat(
                value[:-1] + "+00:00"
            ).astimezone(timezone.utc)
        except (TypeError, ValueError) as error:
            raise CapacityPlanningError(
                "batch wall clock timestamp is invalid"
            ) from error

    def _write_batch_clock(self, unsigned: Mapping[str, object]) -> None:
        value = {**unsigned, "self_sha256": canonical_sha256(dict(unsigned))}
        self._batch_clock_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._batch_clock_path.with_name(
            f".{self._batch_clock_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
        )
        temporary.replace(self._batch_clock_path)

    def _load_batch_clock(self) -> dict[str, object]:
        value = _read_object(self._batch_clock_path, "batch wall clock")
        unsigned = dict(value)
        claimed = unsigned.pop("self_sha256", None)
        if claimed != canonical_sha256(unsigned):
            raise CapacityPlanningError("batch wall clock is invalid")
        if (
            unsigned.get("format")
            != "truth_editing_controller_batch_wall_clock_v2"
            or set(unsigned) != {
                "format", "expected_completed_trials",
                "active_elapsed_seconds", "active_started_at_utc",
            }
        ):
            raise CapacityPlanningError("batch wall clock is invalid")
        expected = unsigned["expected_completed_trials"]
        elapsed = unsigned["active_elapsed_seconds"]
        active_started = unsigned["active_started_at_utc"]
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected <= 0
            or expected % self._policy.batch_size
            or expected > self._policy.maximum_trials
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0
            or active_started is not None
            and not isinstance(active_started, str)
        ):
            raise CapacityPlanningError("batch wall clock is invalid")
        if active_started is not None:
            self._parse_clock_text(active_started)
        return unsigned

    def _resume_batch_clock_after_controller_restart(self) -> None:
        """Resume an existing batch without charging controller-offline time."""

        if not self._batch_clock_path.exists():
            return
        value = _read_object(self._batch_clock_path, "batch wall clock")
        unsigned = dict(value)
        claimed = unsigned.pop("self_sha256", None)
        if claimed != canonical_sha256(unsigned):
            raise CapacityPlanningError("batch wall clock is invalid")
        now_text = self._clock_text(self._clock())
        if (
            unsigned.get("format")
            == "truth_editing_controller_batch_wall_clock_v1"
            and set(unsigned)
            == {"format", "expected_completed_trials", "started_at_utc"}
        ):
            self._parse_clock_text(unsigned["started_at_utc"])
            migrated = {
                "format": "truth_editing_controller_batch_wall_clock_v2",
                "expected_completed_trials": unsigned["expected_completed_trials"],
                "active_elapsed_seconds": 0.0,
                "active_started_at_utc": now_text,
            }
            self._write_batch_clock(migrated)
            self._load_batch_clock()
            return
        current = self._load_batch_clock()
        current["active_started_at_utc"] = now_text
        self._write_batch_clock(current)

    def pause_batch_clock(self, *, completed_trials: int) -> None:
        """Stop throughput timing while preserving the paid-spend ledger."""

        if not self._batch_clock_path.exists():
            return
        current = self._load_batch_clock()
        if current["expected_completed_trials"] != (
            completed_trials + self._policy.batch_size
        ):
            raise CapacityPlanningError("batch wall clock differs from pause boundary")
        active_started = current["active_started_at_utc"]
        if active_started is None:
            return
        now = self._clock()
        self._clock_text(now)
        active_elapsed = (
            now.astimezone(timezone.utc) - self._parse_clock_text(active_started)
        ).total_seconds()
        if active_elapsed < 0:
            raise CapacityPlanningError("batch wall clock moved backwards")
        current["active_elapsed_seconds"] = (
            float(current["active_elapsed_seconds"]) + active_elapsed
        )
        current["active_started_at_utc"] = None
        self._write_batch_clock(current)

    def _batch_wall_seconds(self, *, completed_trials: int, observed: datetime) -> float | None:
        if not self._batch_clock_path.exists():
            return None
        value = self._load_batch_clock()
        if value["expected_completed_trials"] != completed_trials:
            raise CapacityPlanningError("batch wall clock differs from completion")
        elapsed = float(value["active_elapsed_seconds"])
        active_started = value["active_started_at_utc"]
        if active_started is not None:
            active_elapsed = (
                observed - self._parse_clock_text(active_started)
            ).total_seconds()
            if active_elapsed < 0:
                raise CapacityPlanningError("batch wall clock moved backwards")
            elapsed += active_elapsed
        return elapsed

    def _retire_completed_batch_clock(self, *, completed_trials: int) -> None:
        if not self._batch_clock_path.exists():
            return
        value = self._load_batch_clock()
        expected = value["expected_completed_trials"]
        if expected == completed_trials:
            self._batch_clock_path.unlink()
        elif expected != completed_trials + self._policy.batch_size:
            raise CapacityPlanningError("batch wall clock differs from replay boundary")

    def current_receipt(self) -> dict[str, object]:
        if self._rolling_receipt_path.exists():
            return validate_capacity_receipt(
                _read_object(
                    self._rolling_receipt_path, "rolling capacity receipt"
                )
            )
        return dict(self._initial_receipt)

    def reforecast(self, commit: CompletedBatchCommit) -> None:
        with self._lock:
            rows = [self._telemetry.pop(item.trial_id, None) for item in commit.trials]
        complete_telemetry = not any(
            row is None
            or not {"evaluation_seconds", "generated_tokens", "generated_tokens_per_second"}
            <= set(row)
            or row["generated_tokens"] <= 0
            or row["generated_tokens_per_second"] <= 0
            or row["evaluation_seconds"] <= 0
            for row in rows
        )
        previous = self.current_receipt()
        after_judge = self._judge_budget.receipt()
        after_snapshot = self._judge_budget.monitoring_snapshot()
        if previous.get("completed_through_trial") == commit.completed_trials:
            decision = previous.get("decision")
            if not isinstance(decision, dict):
                raise CapacityPlanningError("capacity decision is missing")
            if decision.get("minimum_trial_guarantee_met") is False:
                raise MinimumTrialGuaranteeError(previous)
            self._retire_completed_batch_clock(
                completed_trials=commit.completed_trials
            )
            self._judge_before = after_judge
            self._judge_snapshot_before = after_snapshot
            self._save_judge_state()
            return
        measured = previous.get("measured")
        if not isinstance(measured, dict):
            raise CapacityPlanningError("capacity measured values are missing")
        prior_generation = float(measured["generated_tokens"]) / float(
            measured["tokens_per_second"]
        )
        prior_judge_elapsed = float(measured["judge_latency_seconds"])
        prior_wall = float(measured["trial_wall_seconds"])
        if complete_telemetry:
            complete = [row for row in rows if row is not None]
            generated_tokens = max(int(row["generated_tokens"]) for row in complete)
            generation_seconds = max(
                row["generated_tokens"] / row["generated_tokens_per_second"]
                for row in complete
            )
            wall_seconds = max(row["evaluation_seconds"] for row in complete)
            if generation_seconds > wall_seconds:
                raise CapacityPlanningError(
                    "worker generation telemetry exceeds trial wall time"
                )
            judge_elapsed = max(
                prior_judge_elapsed, wall_seconds - generation_seconds
            )
        else:
            # Operational failures and crash-resumed receipts may not carry
            # telemetry. Preserve liveness with the already signed conservative
            # bound; never pretend missing measurements narrowed the canary.
            generated_tokens = int(measured["generated_tokens"])
            generation_seconds = prior_generation
            judge_elapsed = prior_judge_elapsed
            wall_seconds = max(prior_wall, generation_seconds + judge_elapsed)
        spend = self._spend_reader()
        observed = self._clock()
        if observed.tzinfo is None:
            raise CapacityPlanningError("rolling capacity clock must be timezone-aware")
        observed = observed.astimezone(timezone.utc)
        batch_wall_seconds = self._batch_wall_seconds(
            completed_trials=commit.completed_trials,
            observed=observed,
        )
        if batch_wall_seconds is not None:
            wall_seconds = max(wall_seconds, batch_wall_seconds)
        spend_mapping = {
            "actual_total_usd": _money_text(spend.actual_total_usd),
            "actual_infrastructure_usd": _money_text(
                spend.actual_infrastructure_usd
            ),
            "actual_evaluation_usd": _money_text(spend.actual_evaluation_usd),
            "pending_infrastructure_usd": _money_text(
                spend.pending_infrastructure_usd
            ),
            "pending_evaluation_usd": _money_text(
                spend.pending_evaluation_usd
            ),
        }
        previous_budget = previous.get("budget")
        if not isinstance(previous_budget, dict):
            raise CapacityPlanningError("capacity budget is missing")
        evaluation_delta = max(
            Decimal("0"),
            spend.actual_evaluation_usd
            - Decimal(str(previous_budget["actual_evaluation_usd"])),
        )
        before_calls = int(self._judge_snapshot_before["calls"])
        after_calls = int(after_snapshot["calls"])
        before_failures = int(self._judge_snapshot_before["failures"])
        after_failures = int(after_snapshot["failures"])
        judge_calls = after_calls - before_calls
        judge_failures = after_failures - before_failures
        if judge_calls < 0 or judge_failures < 0:
            raise CapacityPlanningError("judge ledger counters moved backwards")
        before_elapsed = float(self._judge_snapshot_before["elapsed_ms"]) / 1000.0
        after_elapsed = float(after_snapshot["elapsed_ms"]) / 1000.0
        judge_elapsed_total = max(0.0, after_elapsed - before_elapsed)
        judge_elapsed_per_trial_upper_bound = (
            math.nextafter(
                judge_elapsed_total / commit.batch_size,
                math.inf,
            )
            if judge_elapsed_total > 0
            else 0.0
        )
        judge_elapsed = max(
            judge_elapsed,
            judge_elapsed_per_trial_upper_bound,
        )
        wall_seconds = max(wall_seconds, generation_seconds + float(judge_elapsed))
        judge_cost_upper = max(
            Decimal(str(measured["judge_cost_usd_per_trial"])),
            evaluation_delta / Decimal(commit.batch_size),
        )
        unsigned = {
            "format": "truth_editing_capacity_batch_observation_v2",
            "observation_id": f"batch-{commit.batch_ordinal:04d}-{commit.batch_sha256[:12]}",
            "observed_at": observed.isoformat().replace("+00:00", "Z"),
            "timed_canary_receipt_sha256": previous[
                "timed_canary_receipt_sha256"
            ],
            "completed_through_trial": commit.completed_trials,
            "batch_size": commit.batch_size,
            "generated_tokens_per_trial_upper_bound": generated_tokens,
            "generation_seconds_per_trial_upper_bound": generation_seconds,
            "trial_wall_seconds_upper_bound": wall_seconds,
            "judge_elapsed_seconds_per_trial_upper_bound": judge_elapsed,
            "judge_cost_usd_per_trial_upper_bound": _money_text(judge_cost_upper),
            "judge_ledger_before_receipt_sha256": self._judge_before[
                "content_sha256"
            ],
            "judge_ledger_after_receipt_sha256": after_judge["content_sha256"],
            "judge_calls": judge_calls,
            "judge_failures": judge_failures,
            "judge_elapsed_seconds_total": float(judge_elapsed_total),
            "judge_cost_usd_total": _money_text(evaluation_delta),
            "spend": spend_mapping,
        }
        observation = {**unsigned, "self_sha256": canonical_sha256(unsigned)}
        remaining_search_seconds = None
        if self._search_deadline_reader is not None:
            deadline = self._search_deadline_reader()
            if deadline.tzinfo is None:
                raise CapacityPlanningError("search deadline must be timezone-aware")
            deadline = deadline.astimezone(timezone.utc)
            remaining_search_seconds = max(
                0.0, (deadline - observed).total_seconds()
            )
        try:
            receipt = reforecast_capacity_receipt(
                policy=self._policy,
                previous_receipt=previous,
                batch_observation=observation,
                planned_at=observed,
                remaining_search_seconds=remaining_search_seconds,
            )
        except MinimumTrialGuaranteeError as error:
            receipt = error.receipt
            self._write_rolling_receipt(receipt)
            raise
        self._write_rolling_receipt(receipt)
        self._retire_completed_batch_clock(
            completed_trials=commit.completed_trials
        )
        self._judge_before = after_judge
        self._judge_snapshot_before = after_snapshot
        self._save_judge_state()

    def _write_rolling_receipt(self, receipt: Mapping[str, object]) -> None:
        self._rolling_receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._rolling_receipt_path.with_name(
            f".{self._rolling_receipt_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(receipt, allow_nan=False, sort_keys=True) + "\n"
        )
        temporary.replace(self._rolling_receipt_path)


class _DurableBatchAdmission:
    """Clock an admitted batch without authorizing it before prior publication.

    ``TruthEditingStudy`` invokes this adapter only after its completed-batch
    callback returns.  The callback therefore gets a hard opportunity to
    publish the committed boundary off-host before this object can authorize
    more paid work.
    """

    def __init__(
        self,
        *,
        scheduler: AdaptiveBatchScheduler,
        rolling_capacity: _RollingCapacityController,
        batch_started_reader: Callable[[int, int], bool] | None = None,
        admission_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._rolling_capacity = rolling_capacity
        self._batch_started_reader = batch_started_reader
        self._admission_callback = admission_callback

    def admit_batch(
        self,
        *,
        completed_trials: int,
        batch_size: int,
        coverage_complete: bool,
        batch_started: bool = False,
    ) -> bool:
        durable_started = (
            self._batch_started_reader(completed_trials, batch_size)
            if self._batch_started_reader is not None
            else False
        )
        admitted = self._scheduler.admit_batch(
            completed_trials=completed_trials,
            batch_size=batch_size,
            coverage_complete=coverage_complete,
            batch_started=batch_started or durable_started,
        )
        if admitted:
            self._rolling_capacity.record_batch_admission(
                completed_trials=completed_trials
            )
            if self._admission_callback is not None:
                self._admission_callback(completed_trials, batch_size)
        return admitted


def _adaptive_progress(
    *,
    checkpoint: dict[str, object],
    completed_trials: int,
    coverage: dict[str, tuple[int, int]],
    capacity_receipt: dict[str, object],
    policy: CapacityPolicy,
    study_config_sha256: str,
    completed_repeat_trials: int = 0,
    completed_control_trials: int = 0,
    completed_final_selection_trials: int = 0,
    stage: str | None = None,
    eta_seconds: float | None = None,
) -> AdaptiveRunProgress:
    measured = capacity_receipt.get("measured")
    budget = capacity_receipt.get("budget")
    spend = checkpoint.get("last_spend_snapshot")
    if not all(isinstance(value, dict) for value in (measured, budget, spend)):
        raise ValueError("adaptive progress inputs are incomplete")
    assert isinstance(measured, dict)
    assert isinstance(budget, dict)
    assert isinstance(spend, dict)
    started = datetime.fromisoformat(
        str(checkpoint["started_at_utc"]).removesuffix("Z") + "+00:00"
    )
    now = datetime.now(timezone.utc)
    infrastructure_actual = float(spend["actual_infrastructure_usd"])
    evaluation_actual = float(spend["actual_evaluation_usd"])
    infrastructure_projected = _number(
        checkpoint["accounted_infrastructure_usd"], "accounted infrastructure spend"
    ) + float(
        budget["reserved_finalization_infrastructure_usd"]
    )
    evaluation_projected = _number(
        checkpoint["accounted_evaluation_usd"], "accounted evaluation spend"
    ) + float(
        budget["reserved_evaluation_usd"]
    )
    return AdaptiveRunProgress(
        wandb_run_checkpoint_sha256=str(
            checkpoint["wandb_run_checkpoint_sha256"]
        ),
        study_config_sha256=study_config_sha256,
        planned_floor_trials=policy.minimum_trials,
        adaptive_ceiling_trials=policy.maximum_trials,
        measured_target_trials=_integer(
            checkpoint["current_advisory_trial_count"], "advisory trial count"
        ),
        batch_size=policy.batch_size,
        search_cutoff_seconds=policy.search_seconds,
        reserve_seconds=policy.finalization_seconds,
        total_budget_usd=float(policy.total_budget_usd),
        evaluation_budget_usd=float(policy.evaluation_budget_usd),
        evaluation_budget_reserve_fraction=float(
            policy.evaluation_reserve_fraction
        ),
        completed_search_trials=completed_trials,
        completed_repeat_trials=completed_repeat_trials,
        completed_control_trials=completed_control_trials,
        completed_final_selection_trials=completed_final_selection_trials,
        current_batch=completed_trials // policy.batch_size,
        stage=stage or str(checkpoint["phase"]),
        coverage=coverage,
        elapsed_seconds=max(0.0, (now - started).total_seconds()),
        eta_seconds=(
            _number(checkpoint["projected_search_eta_seconds"], "projected ETA")
            if eta_seconds is None
            else eta_seconds
        ),
        gpu_actual_usd=infrastructure_actual,
        gpu_projected_usd=infrastructure_projected,
        judge_actual_usd=evaluation_actual,
        judge_projected_usd=evaluation_projected,
        projected_total_usd=infrastructure_projected + evaluation_projected,
        measured_trial_duration_seconds=float(measured["trial_wall_seconds"]),
        measured_tokens_per_second=float(measured["tokens_per_second"]),
        measured_judge_latency_ms=float(measured["judge_latency_seconds"]) * 1000,
        measured_judge_cost_usd_per_trial=float(
            measured["judge_cost_usd_per_trial"]
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capacity-policy", type=Path, default=DEFAULT_CAPACITY_POLICY)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--rolling-capacity-receipt", type=Path)
    parser.add_argument("--adaptive-checkpoint", type=Path)
    parser.add_argument(
        "--rearm-minimum-guarantee",
        action="store_true",
        help=(
            "reopen a minimum-guarantee abort under the current host lease; "
            "requires a fresh feasible rolling capacity receipt"
        ),
    )
    parser.add_argument("--checkpoint-publication-root", type=Path, required=True)
    parser.add_argument("--restore-study-identity-sha256")
    parser.add_argument("--restore-completed-trials", type=int)
    parser.add_argument("--restore-optuna-study-name")
    parser.add_argument("--restore-offhost-wandb-run-id")
    parser.add_argument(
        "--model-registry-config",
        type=Path,
        default=Path("configs/model_registry_v1.json"),
    )
    parser.add_argument("--offhost-key-prefix", required=True)
    parser.add_argument("--final-model-slug", required=True)
    parser.add_argument("--aws-region", default="us-east-1")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wandb-project", default="intelligent-liars")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--stop-after-trials",
        type=int,
        help="stop cleanly at this complete trial boundary before finalization",
    )
    parser.add_argument(
        "--host-hourly-usd",
        type=float,
        default=host_hourly_usd_from_environment(os.environ),
    )
    parser.add_argument(
        "--host-lease-started-at-utc",
        default=os.environ.get("TRUTH_EDITING_HOST_LEASE_STARTED_AT_UTC"),
    )
    args = parser.parse_args(argv)
    host_lease_started_at = parse_host_lease_started_at_utc(
        args.host_lease_started_at_utc
    )
    fleet = FleetConfig.from_mapping(_read_object(args.fleet_config, "fleet config"))
    if (
        fleet.format != ADAPTIVE_FLEET_FORMAT
        or fleet.worker_count != 8
        or fleet.batch_size != 8
    ):
        raise ValueError("adaptive CUDA controller requires fleet v3 with eight workers")
    production_path = verify_production_config_binding(
        fleet, repo=Path.cwd(), requested_path=args.config
    )
    runtime_production_path = _runtime_config(
        production_path, args.output_root.resolve()
    )
    production = ProductionRunConfig.open(runtime_production_path)
    if (
        production.judge_budget is None
        or production.judge_budget.identity_sha256
        != fleet.production_judge_budget_config_sha256
    ):
        raise ValueError("persistent host production judge budget identity differs")
    study_config = load_truth_editing_study_config(production.study_config)
    capacity_policy = CapacityPolicy.from_mapping(
        _read_object(args.capacity_policy, "capacity policy")
    )
    capacity_receipt = _read_object(args.capacity_receipt, "capacity receipt")
    verify_adaptive_fleet_bindings(
        fleet,
        repo=Path.cwd(),
        requested_capacity_policy_path=args.capacity_policy,
        requested_study_config_path=production.study_config,
        observed_study_identity_sha256=study_config.identity_sha256,
    )
    bank = DirectionBank.open(
        production.direction_manifest, root=production.direction_root
    )
    planned_study_identity_sha256 = planned_fleet_study_identity_sha256(
        fleet=fleet,
        production=production,
        study_config=study_config,
        bank=bank,
    )
    offhost_target = OffHostCheckpointTarget.from_model_registry_config(
        args.model_registry_config,
        key_prefix=args.offhost_key_prefix,
        region=args.aws_region,
    )
    import boto3  # type: ignore

    s3_client = boto3.client("s3", region_name=offhost_target.region)
    offhost_repository = OffHostCheckpointRepository(
        store=S3VersionedObjectStore(
            s3_client,
            bucket=offhost_target.bucket,
        ),
        target=offhost_target,
    )
    search_policy = study_config.search_policy
    if (
        study_config.batch_size != 8
        or study_config.max_trials != 800
        or production.search_driver != "optuna"
        or search_policy is None
        or search_policy.minimum_trials != capacity_policy.minimum_trials
        or search_policy.maximum_trials != capacity_policy.maximum_trials
        or search_policy.search_elapsed_limit_seconds != capacity_policy.search_seconds
        or search_policy.reserve_elapsed_seconds != capacity_policy.finalization_seconds
    ):
        raise ValueError(
            "persistent host requires the frozen adaptive 8-GPU Optuna contract"
        )
    restore_values = (
        args.restore_study_identity_sha256,
        args.restore_completed_trials,
        args.restore_optuna_study_name,
    )
    latest_binding: SnapshotBinding | None = None
    restored_judge_boundary_is_durable = False
    if any(value is not None for value in restore_values):
        if not all(value is not None for value in restore_values):
            raise ValueError("adaptive restore requires all exact resume identities")
        if args.restore_offhost_wandb_run_id is None:
            restore_adaptive_checkpoint(
                args.checkpoint_publication_root.resolve(),
                args.output_root.resolve(),
                expected_study_identity_sha256=args.restore_study_identity_sha256,
                expected_study_config_sha256=study_config.identity_sha256,
                expected_completed_trials=args.restore_completed_trials,
                expected_optuna_study_name=args.restore_optuna_study_name,
            )
            restored_judge_boundary_is_durable = True
        else:
            restore_binding = SnapshotBinding(
                study_identity_sha256=args.restore_study_identity_sha256,
                study_config_sha256=study_config.identity_sha256,
                fleet_config_sha256=fleet.identity_sha256,
                optuna_study_name=args.restore_optuna_study_name,
                wandb_run_id=args.restore_offhost_wandb_run_id,
                completed_trials=args.restore_completed_trials,
                trial_number_start=study_config.trial_number_start,
            )
            with tempfile.TemporaryDirectory(
                prefix="truth-editing-offhost-restore-"
            ) as temporary_root:
                restored_snapshot = Path(temporary_root) / "snapshot"
                offhost_repository.restore_latest(
                    restored_snapshot, restore_binding
                )
                _hydrate_preserving_bootstrap_outputs(
                    restored_snapshot,
                    args.output_root.resolve(),
                    binding=restore_binding,
                    hydrate=hydrate_offhost_snapshot,
                )
            latest_binding = restore_binding
            restored_judge_boundary_is_durable = True
    elif args.restore_offhost_wandb_run_id is not None:
        raise ValueError(
            "off-host restore requires all exact resume identities"
        )
    else:
        latest_binding = offhost_repository.read_latest_binding_if_present(
            expected_study_identity_sha256=planned_study_identity_sha256,
            expected_study_config_sha256=study_config.identity_sha256,
            expected_fleet_config_sha256=fleet.identity_sha256,
        )
        local_scheduler = (
            args.adaptive_checkpoint.resolve()
            if args.adaptive_checkpoint is not None
            else args.output_root.resolve() / "study/adaptive-run-checkpoint.json"
        )
        if latest_binding is not None and not local_scheduler.is_file():
            if not _output_root_is_clean_for_offhost_restore(
                args.output_root.resolve()
            ):
                raise ValueError(
                    "off-host resume requires a clean output root when local "
                    "adaptive state is absent"
                )
            partial_binding = (
                offhost_repository.read_latest_partial_binding_if_present(
                    latest_binding
                )
            )
            with tempfile.TemporaryDirectory(
                prefix="truth-editing-offhost-auto-restore-"
            ) as temporary_root:
                restored_snapshot = Path(temporary_root) / "snapshot"
                if partial_binding is None:
                    offhost_repository.restore_latest(
                        restored_snapshot, latest_binding
                    )
                    _hydrate_preserving_bootstrap_outputs(
                        restored_snapshot,
                        args.output_root.resolve(),
                        binding=latest_binding,
                        hydrate=hydrate_offhost_snapshot,
                    )
                else:
                    offhost_repository.restore_latest_partial(
                        restored_snapshot, partial_binding
                    )
                    _hydrate_preserving_bootstrap_outputs(
                        restored_snapshot,
                        args.output_root.resolve(),
                        binding=partial_binding,
                        hydrate=hydrate_offhost_partial_snapshot,
                    )
            restored_judge_boundary_is_durable = True
    worker_script = Path(__file__).with_name("run_truth_editing_cuda_fleet_worker.py")
    telemetry = GpuTelemetryCollector(gpu_slots=fleet.worker_count)
    monitoring_root = production.journal_path.parent.parent / "monitoring"
    wandb_checkpoint_path = monitoring_root / "wandb-run.json"
    if wandb_checkpoint_path.exists():
        wandb_checkpoint = open_wandb_run_checkpoint(wandb_checkpoint_path)
        wandb_run_id = wandb_checkpoint.run_id
        wandb_project = wandb_checkpoint.project
        wandb_entity = wandb_checkpoint.entity
    else:
        wandb_run_id = args.wandb_run_id or uuid.uuid4().hex
        wandb_project = args.wandb_project
        wandb_entity = args.wandb_entity
    monitor = CoordinatorMonitor.open(
        checkpoint_path=wandb_checkpoint_path,
        run_id=wandb_run_id,
        project=wandb_project,
        entity=wandb_entity,
        run_name="truth-editing-optuna-adaptive",
        receipt_path=monitoring_root / "wandb-events.jsonl",
        total_trials=capacity_policy.maximum_trials,
        batch_size=study_config.batch_size,
    )
    rolling_capacity: _RollingCapacityController | None = None
    scheduler: AdaptiveBatchScheduler | None = None

    def record_trial_telemetry(
        gpu_slot: int, trial_id: str, values: dict[str, float]
    ) -> None:
        monitor.record_worker_telemetry(gpu_slot, trial_id, values)
        if rolling_capacity is not None:
            rolling_capacity.record_trial(gpu_slot, trial_id, values)

    def checkpoint_partial_trial(event: Mapping[str, object]) -> None:
        """Publish each locally durable trial before its worker may continue."""

        if scheduler is None:
            raise ValueError("adaptive scheduler is unavailable for partial checkpoint")
        checkpoint = _read_object(adaptive_checkpoint_path, "adaptive checkpoint")
        completed_trials = _integer(
            checkpoint["completed_trials"], "adaptive completed trials"
        )
        committed = SnapshotBinding(
            study_identity_sha256=run.planned_study_identity_sha256,
            study_config_sha256=study_config.identity_sha256,
            fleet_config_sha256=fleet.identity_sha256,
            optuna_study_name=driver.persistent_study_name,
            wandb_run_id=monitor.run_id,
            completed_trials=completed_trials,
            trial_number_start=study_config.trial_number_start,
        )
        receipt_directory = args.output_root.resolve() / "fleet-receipts"
        published = publish_partial_event_if_needed(
            repository=offhost_repository,
            committed_binding=committed,
            durable_event=event,
            expected_receipt_directory=receipt_directory,
            trial_number_start=study_config.trial_number_start,
            publish=lambda: offhost_repository.publish_partial_from_runtime(
                args.output_root.resolve() / "checkpoint-staging/partial",
                committed_binding=committed,
                durable_event=event,
                adaptive_state_root=args.output_root.resolve(),
                fleet_receipt_dir=receipt_directory,
                runtime_output_dir=production.runtime_output_dir,
                judge_cache_dir=production.judge_cache_dir,
                judge_budget_ledger_dir=production.judge_budget_ledger_dir,
            ),
        )
        if published is None:
            return
        if not isinstance(published, tuple) or len(published) != 3:
            raise ValueError("partial publication result differs")
        _receipt, _binding, snapshot = published
        if not isinstance(snapshot, Path):
            raise ValueError("partial publication snapshot path differs")
        shutil.rmtree(snapshot, ignore_errors=True)

    adaptive_checkpoint_path = (
        args.adaptive_checkpoint
        if args.adaptive_checkpoint is not None
        else production.journal_path.parent / "adaptive-run-checkpoint.json"
    )
    ordered_checkpoint_partial_trial = _OrderedDurableCallback(
        checkpoint_partial_trial,
        next_ordinal_reader=lambda: _integer(
            _read_object(adaptive_checkpoint_path, "adaptive checkpoint")[
                "completed_trials"
            ],
            "adaptive completed trials",
        ),
    )

    evaluator = FleetBatchEvaluator(
        fleet,
        receipt_directory_override=(
            args.output_root.resolve() / "fleet-receipts"
        ),
        telemetry=telemetry,
        trial_telemetry_callback=record_trial_telemetry,
        trial_receipt_durable_callback=ordered_checkpoint_partial_trial,
        worker_factory=lambda slot: SubprocessCudaWorker(
            slot,
            (
                sys.executable,
                str(worker_script),
                "--config",
                str(runtime_production_path),
            ),
        ),
    )
    judge_budget = (
        ProductionJudgeBudget(
            production.judge_budget_ledger_dir, config=production.judge_budget
        )
        if production.judge_budget_ledger_dir is not None
        and production.judge_budget is not None
        else None
    )
    judge_projected_usd = (
        float(production.judge_budget.maximum_judge_spend_usd)
        if production.judge_budget is not None
        else 0.0
    )
    gpu_projected_usd = float(fleet.maximum_infrastructure_spend_usd)
    driver = MonitoredSearchDriver(
        configured_search_driver(production, study_config), monitor
    )
    run = ProductionTruthEditingRun(
        study=TruthEditingStudy(study_config, bank.manifest),
        driver=driver,
        evaluator=evaluator,  # type: ignore[arg-type]
        artifacts=ImmutableStudyArtifactAdapter(production.artifact_dir),
        journal_path=production.journal_path,
    )
    if judge_budget is None:
        raise ValueError("adaptive production requires the durable judge ledger")
    prepare_judge_budget_for_controller_resume(
        judge_budget=judge_budget,
        restored_boundary_is_durable=restored_judge_boundary_is_durable,
    )
    spend_reader = _ControllerSpendReader(
        capacity_receipt=capacity_receipt,
        judge_budget=judge_budget,
        host_hourly_usd=Decimal(str(args.host_hourly_usd)),
        host_lease_started_at=host_lease_started_at,
        worker_count=fleet.worker_count,
        prior_spend=_prior_spend_from_checkpoint(adaptive_checkpoint_path),
    )
    rolling_capacity_path = (
        args.rolling_capacity_receipt
        if args.rolling_capacity_receipt is not None
        else args.output_root.resolve() / "monitoring/rolling-capacity-receipt.json"
    )
    def read_search_deadline() -> datetime:
        if scheduler is None:
            raise CapacityPlanningError(
                "adaptive scheduler is unavailable for rolling reforecast"
            )
        return scheduler.search_deadline

    rolling_capacity = _RollingCapacityController(
        policy=capacity_policy,
        initial_receipt=capacity_receipt,
        rolling_receipt_path=rolling_capacity_path,
        spend_reader=spend_reader,
        judge_budget=judge_budget,
        search_deadline_reader=read_search_deadline,
    )
    scheduler = AdaptiveBatchScheduler.open(
        policy=capacity_policy,
        capacity_receipt=capacity_receipt,
        capacity_receipt_reader=rolling_capacity.current_receipt,
        checkpoint_path=adaptive_checkpoint_path,
        study_identity_sha256=run.planned_study_identity_sha256,
        wandb_checkpoint_path=wandb_checkpoint_path,
        spend_reader=spend_reader,
        clock=lambda: datetime.now(timezone.utc),
        initial_started_at=host_lease_started_at,
    )
    if restored_judge_boundary_is_durable:
        scheduler.rearm_expired_search_lease(started_at=host_lease_started_at)
    if args.rearm_minimum_guarantee:
        scheduler.rearm_minimum_guarantee_abort(started_at=host_lease_started_at)
    durable_batch_admission = _DurableBatchAdmission(
        scheduler=scheduler,
        rolling_capacity=rolling_capacity,
        batch_started_reader=lambda completed, size: _batch_has_durable_receipt(
            evaluator.receipt_directory,
            fleet_config_sha256=fleet.identity_sha256,
            completed_trials=completed,
            batch_size=size,
            trial_number_start=study_config.trial_number_start,
        ),
        admission_callback=lambda completed, size: monitor.record_execution_counts(
            active=0,
            authorized=completed + size,
        ),
    )
    pump = CoordinatorTelemetryPump(
        telemetry,
        monitor,
        host_hourly_usd=args.host_hourly_usd,
        initial_host_elapsed_seconds=max(
            0.0,
            (datetime.now(timezone.utc) - host_lease_started_at).total_seconds(),
        ),
        gpu_projected_usd=gpu_projected_usd,
        judge_projected_usd=judge_projected_usd,
        judge_budget=judge_budget,
    )
    latest_coverage: dict[str, tuple[int, int]] = {}

    def publish_boundary(
        *,
        study_identity_sha256: str,
        completed_trials: int,
        staging_identity_sha256: str,
    ) -> None:
        nonlocal latest_binding
        publish_adaptive_checkpoint(
            args.output_root.resolve(),
            args.checkpoint_publication_root.resolve(),
            expected_study_identity_sha256=study_identity_sha256,
            expected_study_config_sha256=study_config.identity_sha256,
            expected_completed_trials=completed_trials,
            expected_optuna_study_name=driver.persistent_study_name,
            tier_through_trials=tuple(
                tier.through_trial for tier in study_config.evaluation_tiers
            ),
        )
        binding = SnapshotBinding(
            study_identity_sha256=study_identity_sha256,
            study_config_sha256=study_config.identity_sha256,
            fleet_config_sha256=fleet.identity_sha256,
            optuna_study_name=driver.persistent_study_name,
            wandb_run_id=monitor.run_id,
            completed_trials=completed_trials,
            trial_number_start=study_config.trial_number_start,
        )
        if offhost_boundary_is_already_published(latest_binding, binding):
            return
        snapshot_path = (
            args.output_root.resolve()
            / "checkpoint-staging"
            / f"batch-{completed_trials:04d}-{staging_identity_sha256[:12]}"
        )
        if not snapshot_path.exists():
            materialize_offhost_snapshot(
                snapshot_path,
                binding=binding,
                adaptive_state_root=args.output_root.resolve(),
                fleet_receipt_dir=evaluator.receipt_directory,
                runtime_output_dir=production.runtime_output_dir,
                judge_cache_dir=production.judge_cache_dir,
                judge_budget_ledger_dir=production.judge_budget_ledger_dir,
            )
        offhost_repository.publish(snapshot_path, binding)
        latest_binding = binding
        shutil.rmtree(snapshot_path, ignore_errors=True)

    def record_and_publish(commit: CompletedBatchCommit) -> None:
        nonlocal latest_coverage
        latest_coverage = commit.coverage_summary
        checkpoint = _read_object(adaptive_checkpoint_path, "adaptive checkpoint")
        progress_path = monitoring_root / "adaptive-progress.json"
        if not adaptive_progress_boundary_is_already_recorded(
            progress_path,
            completed_trials=commit.completed_trials,
            coverage=commit.coverage_summary,
        ):
            monitor.record_adaptive_progress(
                _adaptive_progress(
                    checkpoint=checkpoint,
                    completed_trials=commit.completed_trials,
                    coverage=commit.coverage_summary,
                    capacity_receipt=rolling_capacity.current_receipt(),
                    policy=capacity_policy,
                    study_config_sha256=study_config.identity_sha256,
                )
            )
        publish_boundary(
            study_identity_sha256=commit.study_identity_sha256,
            completed_trials=commit.completed_trials,
            staging_identity_sha256=commit.batch_sha256,
        )

    def after_prepare_before_first_admission(context: PreparedStudyContext) -> None:
        nonlocal latest_coverage
        latest_coverage = context.coverage_summary
        if context.study_identity_sha256 != run.planned_study_identity_sha256:
            raise ValueError("prepared study identity differs from planned study")
        # Existing checkpoints are replayed through after_complete_batch below.
        # A fresh rescore lineage instead seeds immutable source batches and
        # must establish its first target-lineage authorization at that exact
        # capacity-bound boundary before dispatch.
        if context.completed_trials != 0 and scheduler.has_checkpoint:
            return
        resuming_durable_batch = _batch_has_durable_receipt(
            evaluator.receipt_directory,
            fleet_config_sha256=fleet.identity_sha256,
            completed_trials=context.completed_trials,
            batch_size=capacity_policy.batch_size,
            trial_number_start=study_config.trial_number_start,
        )
        admitted = durable_batch_admission.admit_batch(
            completed_trials=context.completed_trials,
            batch_size=capacity_policy.batch_size,
            coverage_complete=context.coverage_complete,
            batch_started=resuming_durable_batch,
        )
        if not admitted:
            raise ValueError("prepared adaptive study cannot authorize its next batch")
        # A crash after dispatch can leave an intentionally incomplete journal
        # alongside individually durable worker receipts.  That is not a valid
        # checkpoint boundary yet: let the study replay those receipts into the
        # journal first, without dispatching or calling the judge again.  The
        # normal completed-batch callback will then publish the exact boundary.
        if resuming_durable_batch:
            return
        initial_progress_path = monitoring_root / "adaptive-progress.json"
        if not initial_progress_path.exists():
            evaluator.receipt_directory.mkdir(parents=True, exist_ok=True)
            production.runtime_output_dir.mkdir(parents=True, exist_ok=True)
            production.judge_cache_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = _read_object(
                adaptive_checkpoint_path, "adaptive checkpoint"
            )
            monitor.record_adaptive_progress(
                _adaptive_progress(
                    checkpoint=checkpoint,
                    completed_trials=context.completed_trials,
                    coverage=context.coverage_summary,
                    capacity_receipt=rolling_capacity.current_receipt(),
                    policy=capacity_policy,
                    study_config_sha256=study_config.identity_sha256,
                )
            )
        publish_boundary(
            study_identity_sha256=context.study_identity_sha256,
            completed_trials=context.completed_trials,
            staging_identity_sha256=context.context_sha256,
        )

    def after_complete_batch(commit: CompletedBatchCommit) -> None:
        try:
            rolling_capacity.reforecast(commit)
        except MinimumTrialGuaranteeError:
            scheduler.abort_minimum_trial_guarantee(
                completed_trials=commit.completed_trials,
                coverage_complete=commit.coverage_complete,
            )
            publish_completed_boundary_then_reconcile_judge_budget(
                publish_completed_boundary=lambda: record_and_publish(commit),
                judge_budget=judge_budget,
            )
            return
        # Commit and publish the completed boundary before the study asks the
        # durable admission adapter to authorize any further paid work.
        scheduler.commit_batch(
            completed_trials=commit.completed_trials,
            coverage_complete=commit.coverage_complete,
        )
        # Hydrated checkpoint bytes remain immutable through replay and
        # publication. Only after that durable boundary may crashed paid calls
        # be classified and the transient global ambiguity gate be retired.
        # Exact ambiguous requests remain blocked pending separate evidence.
        # The next partial/full frontier carries those ledger events.
        publish_completed_boundary_then_reconcile_judge_budget(
            publish_completed_boundary=lambda: record_and_publish(commit),
            judge_budget=judge_budget,
        )

    with evaluator, pump:
        receipt = run.run(
            stop_after_trials=args.stop_after_trials,
            batch_admission=durable_batch_admission,
            after_complete_batch=after_complete_batch,
            after_prepare_before_first_admission=(
                after_prepare_before_first_admission
            ),
        )
    payload = receipt.to_mapping()
    payload["run_receipt_sha256"] = receipt.identity_sha256
    _write_json_immutable(args.receipt, payload)
    terminal = _read_object(adaptive_checkpoint_path, "adaptive checkpoint")
    if (
        args.stop_after_trials is not None
        and receipt.completed_trials == args.stop_after_trials
        and terminal.get("phase") not in {"finalization_reserved", "complete"}
    ):
        monitor.close()
        pause_unsigned = {
            "format": "truth_editing_adaptive_controller_pause_v1",
            "reason": "requested_complete_trial_boundary",
            "completed_trials": receipt.completed_trials,
            "study_identity_sha256": receipt.study_identity_sha256,
            "run_receipt_sha256": receipt.identity_sha256,
            "adaptive_checkpoint_sha256": terminal["checkpoint_sha256"],
        }
        _write_json_immutable(
            args.output_root.resolve() / "monitoring/controller-pause-receipt.json",
            {
                **pause_unsigned,
                "receipt_sha256": canonical_sha256(pause_unsigned),
            },
        )
        return 0
    if (
        terminal.get("phase") == "aborted"
        and terminal.get("stop_reason") == "minimum_trial_guarantee_lost"
    ):
        monitor.record_adaptive_progress(
            _adaptive_progress(
                checkpoint=terminal,
                completed_trials=receipt.completed_trials,
                coverage=latest_coverage,
                capacity_receipt=rolling_capacity.current_receipt(),
                policy=capacity_policy,
                study_config_sha256=study_config.identity_sha256,
                stage="aborted",
                eta_seconds=0.0,
            )
        )
        abort_unsigned = {
            "format": "truth_editing_adaptive_controller_abort_v1",
            "adaptive_checkpoint_sha256": terminal["checkpoint_sha256"],
            "run_receipt_sha256": receipt.identity_sha256,
            "completed_trials": terminal["completed_trials"],
            "stop_reason": terminal["stop_reason"],
        }
        _write_json_immutable(
            args.output_root.resolve() / "monitoring/controller-abort-receipt.json",
            {
                **abort_unsigned,
                "receipt_sha256": canonical_sha256(abort_unsigned),
            },
        )
        monitor.close()
        return 2
    if terminal.get("phase") != "finalization_reserved":
        raise ValueError("adaptive search did not enter the reserved finalization lane")
    judge_receipt_path = (
        args.output_root.resolve()
        / "providers/production-judge-budget/finalization-receipt.json"
    )
    _write_json_immutable(judge_receipt_path, judge_budget.receipt())
    artifact = payload.get("artifact_receipt")
    if not isinstance(artifact, dict) or not isinstance(
        artifact.get("report_path"), str
    ):
        raise ValueError("adaptive study report is missing from its artifact receipt")
    handoff_path = (
        args.output_root.resolve()
        / "finalization/adaptive-finalization-handoff.json"
    )
    handoff = write_adaptive_finalization_handoff(
        handoff_path,
        study_report_path=Path(artifact["report_path"]),
        study_artifact_receipt_path=(
            production.artifact_dir / "study-artifact-receipt.json"
        ),
        production_config_path=runtime_production_path,
        adaptive_checkpoint_path=adaptive_checkpoint_path,
        judge_budget_receipt_path=judge_receipt_path,
        output_root=args.output_root.resolve() / "finalization",
        deadline_utc=str(terminal["hard_deadline_utc"]),
        study_identity_sha256=receipt.study_identity_sha256,
        maximum_evaluation_spend_usd="1",
        strong_candidate_count=3,
        repeat_count_per_candidate=2,
    )
    artifact_receipt_path = production.artifact_dir / "study-artifact-receipt.json"
    artifact_receipt = _read_object(
        artifact_receipt_path, "study artifact receipt"
    )
    artifact_receipt_sha = artifact_receipt.get("receipt_sha256")
    if not isinstance(artifact_receipt_sha, str):
        raise ValueError("study artifact receipt identity is missing")
    finalist_compiler, finalist_bundle = open_finalist_export_inputs(
        runtime_production_path,
        study_artifact_receipt_path=artifact_receipt_path,
        expected_study_identity_sha256=receipt.study_identity_sha256,
        expected_study_artifact_receipt_sha256=artifact_receipt_sha,
    )
    exporter = ProductionFinalistCheckpointExporter(
        production_config_path=runtime_production_path,
        study_artifact_receipt_path=artifact_receipt_path,
        study_identity_sha256=receipt.study_identity_sha256,
        study_artifact_receipt_sha256=artifact_receipt_sha,
        registry_bucket=offhost_target.bucket,
        model_slug=args.final_model_slug,
        compiler=finalist_compiler,
        bundle=finalist_bundle,
    )
    def causal_executor_factory(*, config_path: Path):
        return create_qwen_causal_executor_with_base_bundle(
            config_path=config_path,
            base_bundle=finalist_bundle,
        )

    finalization_progress_target = OffHostCheckpointTarget(
        bucket=offhost_target.bucket,
        region=offhost_target.region,
        key_prefix=f"{offhost_target.key_prefix}/finalization-progress",
        registry_config_sha256=offhost_target.registry_config_sha256,
    )
    finalization_progress_repository = FinalizationProgressRepository(
        store=S3VersionedObjectStore(
            s3_client,
            bucket=finalization_progress_target.bucket,
        ),
        target=finalization_progress_target,
    )
    judge_ledger_root = str(
        _read_object(judge_receipt_path, "judge finalization receipt")[
            "content_sha256"
        ]
    )
    finalization_fixed_identity = {
        "study_identity_sha256": receipt.study_identity_sha256,
        "study_config_sha256": study_config.identity_sha256,
        "fleet_config_sha256": fleet.identity_sha256,
        "finalization_identity_sha256": handoff["self_sha256"],
        "judge_ledger_root_sha256": judge_ledger_root,
        "optuna_study_name": driver.persistent_study_name,
        "wandb_run_id": monitor.run_id,
    }
    resume_through_ordinal = -1
    finalization_ledger_cursor = judge_ledger_root
    finalization_judge_ledger_dir = production.judge_budget_ledger_dir
    if finalization_judge_ledger_dir is None:
        raise ValueError("finalization requires the production judge ledger directory")
    try:
        restored_finalization = finalization_progress_repository.restore_current(
            args.output_root.resolve(),
            finalization_fixed_identity,
            replace_existing=True,
        )
    except FinalizationProgressError as error:
        if "latest pointer is missing" not in str(error):
            raise
    else:
        restored_binding = restored_finalization["binding"]
        resume_through_ordinal = int(restored_binding["stage_ordinal"])
        finalization_ledger_cursor = str(
            restored_binding["judge_ledger_after_sha256"]
        )
        judge_budget = ProductionJudgeBudget(
            finalization_judge_ledger_dir,
            config=production.judge_budget,
        )
        reconcile_restored_finalization_judge_budget(
            judge_budget=judge_budget,
            expected_receipt_sha256=finalization_ledger_cursor,
        )
        spend_reader.rebind_judge_budget(judge_budget)
    finalization_event_ordinal = 0

    def require_finalization_capacity(_unit: str) -> None:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(
            str(terminal["hard_deadline_utc"]).removesuffix("Z") + "+00:00"
        ):
            raise ValueError("finalization hard deadline was reached")
        live_spend = spend_reader()
        if (
            live_spend.reserved_infrastructure_usd
            > capacity_policy.infrastructure_budget_usd
            or live_spend.reserved_evaluation_usd
            > capacity_policy.evaluation_budget_usd
            or live_spend.reserved_total_usd > capacity_policy.total_budget_usd
        ):
            raise ValueError("finalization spend cap was exceeded")

    def compact_finalization_evidence_paths() -> tuple[Path, ...]:
        roots = (
            production.runtime_output_dir / "causal-finalization",
            args.output_root.resolve() / "finalization",
            production.judge_cache_dir,
            finalization_judge_ledger_dir,
        )
        return tuple(
            sorted(
                path.relative_to(args.output_root.resolve())
                for root in roots
                if root.is_dir() and not root.is_symlink()
                for path in root.rglob("*.json")
                if path.is_file() and not path.is_symlink()
            )
        )

    def publish_finalization_barrier(
        *, stage_kind: str, commit_id: str, ledger_after_sha256: str
    ) -> None:
        nonlocal finalization_event_ordinal, finalization_ledger_cursor
        ordinal = finalization_event_ordinal
        finalization_event_ordinal += 1
        if ordinal <= resume_through_ordinal:
            return
        evidence_paths = compact_finalization_evidence_paths()
        if not evidence_paths:
            raise ValueError("finalization durability evidence inventory is empty")
        finalization_progress_repository.publish(
            args.output_root.resolve(),
            FinalizationProgressBinding(
                **finalization_fixed_identity,
                judge_ledger_before_sha256=finalization_ledger_cursor,
                judge_ledger_after_sha256=ledger_after_sha256,
                stage_ordinal=ordinal,
                stage_kind=stage_kind,
                commit_id=commit_id,
            ),
            evidence_paths=evidence_paths,
        )
        finalization_ledger_cursor = ledger_after_sha256

    def checkpoint_causal_candidate(event: Mapping[str, object]) -> None:
        publish_finalization_barrier(
            stage_kind="causal_candidate",
            commit_id=f"causal-{event['ordinal']}-{str(event['receipt_self_sha256'])[:16]}",
            ledger_after_sha256=str(event["judge_ledger_after_sha256"]),
        )

    causal_receipts = prepare_adaptive_causal_controls(
        handoff_path,
        compiler_identity=finalist_compiler.identity,
        materializer=ProductionCausalCandidateMaterializer(
            config=production,
            compiler=finalist_compiler,
            bundle=finalist_bundle,
        ),
        executor_factory=causal_executor_factory,
        # Keep causal state inside the runtime tree consumed by the off-host
        # snapshot contract rather than beside the final public checkpoint.
        causal_root=production.runtime_output_dir / "causal-finalization",
        before_candidate_execute=require_finalization_capacity,
        after_candidate_commit=checkpoint_causal_candidate,
    )
    finalization_production = open_production_run(runtime_production_path)
    scheduled_finalization_evaluations = handoff["strong_candidate_count"] * (
        handoff["repeat_count_per_candidate"] + 2
    )
    maximum_per_evaluation = Decimal(
        handoff["maximum_evaluation_spend_usd"]
    ) / Decimal(scheduled_finalization_evaluations)
    finalization_backend = finalization_production.build_finalization_backend(
        checkpoint_exporter=exporter,
        maximum_evaluation_cost_usd=_money_text(maximum_per_evaluation),
    )
    finalization_executor = ProductionAdaptiveFinalizationExecutor(
        Path(artifact["report_path"]),
        finalization_backend,
        causal_control_receipts=causal_receipts,
    )
    def record_finalization_progress(event: Mapping[str, object]) -> None:
        live_spend = spend_reader()
        progress_checkpoint = dict(terminal)
        progress_checkpoint["last_spend_snapshot"] = {
            "actual_total_usd": _money_text(live_spend.actual_total_usd),
            "actual_infrastructure_usd": _money_text(
                live_spend.actual_infrastructure_usd
            ),
            "actual_evaluation_usd": _money_text(
                live_spend.actual_evaluation_usd
            ),
            "pending_infrastructure_usd": _money_text(
                live_spend.pending_infrastructure_usd
            ),
            "pending_evaluation_usd": _money_text(
                live_spend.pending_evaluation_usd
            ),
        }
        event_name = str(event["phase"])
        monitor.record_adaptive_progress(
            _adaptive_progress(
                checkpoint=progress_checkpoint,
                completed_trials=receipt.completed_trials,
                coverage=latest_coverage,
                capacity_receipt=rolling_capacity.current_receipt(),
                policy=capacity_policy,
                study_config_sha256=study_config.identity_sha256,
                completed_repeat_trials=_integer(
                    event["completed_repeat_evaluations"], "completed repeats"
                ),
                completed_control_trials=_integer(
                    event["completed_control_evaluations"], "completed controls"
                ),
                completed_final_selection_trials=(
                    1
                    if event_name
                    in {"final_selection", "checkpoint_export", "complete"}
                    else 0
                ),
                stage=event_name,
                eta_seconds=0.0 if event_name == "complete" else None,
            )
        )

    def checkpoint_finalization_progress(event: Mapping[str, object]) -> None:
        phase = str(event["phase"])
        # Local checkpoint weights are intentionally absent from compact
        # progress archives. Do not mark export/complete durable until the
        # content-addressed remote checkpoint is verified below.
        if phase in {"checkpoint_export", "complete"}:
            return
        stage_kind = {
            "repeats": "repeat_evaluation",
            "controls": "matched_control",
            "final_selection": "final_selection",
        }.get(phase)
        if stage_kind is None:
            raise ValueError("finalization checkpoint phase is unsupported")
        require_finalization_capacity(stage_kind)
        current_judge = judge_budget.receipt()
        publish_finalization_barrier(
            stage_kind=stage_kind,
            commit_id=f"{stage_kind}-{canonical_sha256(dict(event))[:16]}",
            ledger_after_sha256=str(current_judge["content_sha256"]),
        )

    final_model_publication_path = (
        args.output_root.resolve()
        / "finalization/final-model-publication-receipt.json"
    )
    try:
        finalization_receipt = run_adaptive_finalization(
            handoff_path,
            finalization_executor,
            progress_callback=record_finalization_progress,
            checkpoint_callback=checkpoint_finalization_progress,
            before_unit_execute=lambda kind, _request: require_finalization_capacity(
                kind
            ),
        )
        checkpoint_publication_dir = (
            args.output_root.resolve() / "finalization/checkpoint-publication"
        )
        verified_checkpoint = open_finalist_checkpoint(checkpoint_publication_dir)
        final_target = build_final_checkpoint_target(
            args.model_registry_config,
            model_slug=args.final_model_slug,
        )
        causal_evidence_root = (
            production.runtime_output_dir / "causal-finalization"
        )
        causal_evidence_paths = tuple(
            sorted(
                path
                for path in causal_evidence_root.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix in {".json", ".pt"}
            )
        )
        if not causal_evidence_paths:
            raise ValueError("causal finalization evidence inventory is empty")
        require_finalization_capacity("final_checkpoint_s3_publication")
        publish_final_checkpoint(
            checkpoint_publication_dir,
            verified_checkpoint=verified_checkpoint,
            evidence_paths=(
                handoff_path,
                args.output_root.resolve()
                / "finalization/adaptive-finalization-audit.json",
                args.output_root.resolve()
                / "finalization/audited-selection-receipt.json",
                args.output_root.resolve()
                / "finalization/adaptive-finalization-receipt.json",
                artifact_receipt_path,
                args.receipt,
                adaptive_checkpoint_path,
                judge_receipt_path,
                *causal_evidence_paths,
            ),
            evidence_root=args.output_root.resolve(),
            target=final_target,
            store=S3FinalCheckpointStore(
                s3_client,
                bucket=final_target.bucket,
            ),
            receipt_path=final_model_publication_path,
        )
        final_model_publication = open_final_checkpoint_publication_receipt(
            final_model_publication_path
        )
        # The Vast lifecycle archives the whole output root, not merely its
        # expected-output allowlist.  Keep the hash-bound metadata receipts but
        # retire the multi-GiB local shards only after exact remote verification.
        retire_verified_local_checkpoint_weights(
            checkpoint_publication_dir,
            verified_checkpoint=verified_checkpoint,
            publication_receipt=final_model_publication,
        )
        publish_finalization_barrier(
            stage_kind="complete",
            commit_id=f"remote-complete-{final_model_publication['self_sha256'][:16]}",
            ledger_after_sha256=str(judge_budget.receipt()["content_sha256"]),
        )
    finally:
        monitor.close()
    _write_json_immutable(
        args.output_root.resolve() / "adaptive-controller-result.json",
        {
            "format": "truth_editing_adaptive_controller_result_v1",
            "run_receipt_path": str(args.receipt.resolve()),
            "run_receipt_sha256": receipt.identity_sha256,
            "finalization_status": "complete",
            "finalization_handoff_path": str(handoff_path),
            "finalization_handoff_sha256": handoff["self_sha256"],
            "finalization_receipt_path": str(
                (
                    args.output_root.resolve()
                    / "finalization/adaptive-finalization-receipt.json"
                )
            ),
            "finalization_receipt_sha256": finalization_receipt["self_sha256"],
            "final_model_publication_receipt_path": str(
                final_model_publication_path
            ),
            "final_model_publication_receipt_sha256": final_model_publication[
                "self_sha256"
            ],
            "offhost_finalization_state_sha256": final_model_publication[
                "offhost_finalization_state"
            ]["self_sha256"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
