"""Failure-isolated coordinator monitoring for the truth-editing study.

The optimizer journal and hashed fleet receipts remain authoritative.  This
module only mirrors an allowlisted view of coordinator state to one W&B run.
No monitoring error is allowed to cross the public interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .heretic_truth_editing import OBJECTIVES
from .truth_editing_batch_execution import BatchEvaluationRequest
from .truth_editing_gpu_telemetry import GpuTelemetryCollector, GpuTelemetryRecord
from .truth_editing_study import EvaluationResult, StudyTrial
from .truth_editing_wandb_checkpoint import (
    AdaptiveProgressCheckpoint,
    AdaptiveRunProgress,
    WandbCheckpointError,
    advance_adaptive_progress_checkpoint,
    create_wandb_run_checkpoint,
    open_wandb_run_checkpoint,
)


MONITORING_RECEIPT_FORMAT = "truth_editing_wandb_monitoring_event_v1"
VERIFICATION_FORMAT = "truth_editing_wandb_verification_snapshot_v2"
_ALLOWED_PROPOSAL_PARAMETERS = frozenset(
    {
        "attention_edge_strength",
        "attention_enabled",
        "attention_kernel_center",
        "attention_kernel_half_width",
        "attention_peak_strength",
        "backend_type",
        "basis_method",
        "basis_scope",
        "direction_family",
        "direction_ids",
        "edit_arm",
        "mlp_edge_strength",
        "mlp_enabled",
        "mlp_kernel_center",
        "mlp_kernel_half_width",
        "mlp_peak_strength",
        "normalization_mode",
        "proposal_origin",
        "refusal_direction_scope",
        "refusal_enabled",
        "refusal_source_layer",
        "refusal_strength",
        "refusal_writer_policy",
        "requested_rank",
        "source_layer",
        "strength",
        "truth_direction_scope",
        "writer_layers",
        "writer_policy",
        "writer_region",
        "selected_domains",
    }
)
_BOOL_PARAMETERS = frozenset(
    {"attention_enabled", "mlp_enabled", "refusal_enabled"}
)
_NUMERIC_PARAMETERS = frozenset(
    {
        "attention_edge_strength",
        "attention_kernel_center",
        "attention_kernel_half_width",
        "attention_peak_strength",
        "mlp_edge_strength",
        "mlp_kernel_center",
        "mlp_kernel_half_width",
        "mlp_peak_strength",
        "refusal_source_layer",
        "refusal_strength",
        "requested_rank",
        "source_layer",
        "strength",
    }
)
_ENUM_PARAMETERS: Mapping[str, frozenset[str]] = {
    "backend_type": frozenset({"persistent_weight"}),
    "basis_method": frozenset({"qr", "svd"}),
    "basis_scope": frozenset({"general", "domain", "mixed"}),
    "edit_arm": frozenset({"truth_only", "refusal_only", "joint"}),
    "normalization_mode": frozenset({"exact", "norm_preserving"}),
    "proposal_origin": frozenset({"coverage_anchor", "tpe_sampled"}),
    "refusal_direction_scope": frozenset({"global", "per_layer"}),
    "refusal_writer_policy": frozenset({"attention", "mlp", "both"}),
    "truth_direction_scope": frozenset({"global", "per_layer"}),
    "writer_policy": frozenset({"attention", "mlp", "both"}),
}
RUN_NAME = "truth-editing-optuna-v3"
_SAFE_OUTCOMES = frozenset(
    {"successful", "scientifically_infeasible", "operational_failure", "stopped"}
)
_SAFE_PHASES = frozenset({"discovery", "expanded", "finalist", "canary"})
_SAFE_ERROR_CATEGORIES = frozenset(
    {
        "worker_operational_failure",
        "missing_tool",
        "query_timeout",
        "malformed_output",
        "query_failed",
    }
)
_SENSITIVE_TEXT = re.compile(
    r"(?:sk-or-v1-|sk-proj-|-----BEGIN|bearer|api[_-]?key|secret|token|"
    r"prompt|response|message|(?:^|[/\\])(?:Users|home|private|workspace)(?:[/\\])|"
    r"(?:https?|s3)://)",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GPU_METRIC = re.compile(
    r"^gpu/[0-9]+/(?:utilization_pct|memory_used_mib|memory_total_mib|tps|"
    r"active_trial_ordinal)$"
)
_PARAMETER_COUNTS = frozenset(
    {"direction_ids_count", "selected_domains_count", "writer_layers_count"}
)
_PARAMETER_DIGESTS = frozenset(
    {"direction_ids_sha256", "selected_domains_sha256", "writer_layers_sha256"}
)
_FAILURE_OPERATIONS = frozenset(
    {
        "init",
        "log",
        "finish",
        "transport",
        "record_batch",
        "record_trials",
        "telemetry_poll",
    }
)
_ADAPTIVE_STAGES = frozenset(
    {
        "broad_coverage",
        "adaptive_search",
        "finalization_reserved",
        "repeats",
        "controls",
        "final_selection",
        "checkpoint_export",
        "complete",
    }
)
_COVERAGE_METRIC = re.compile(
    r"^coverage/(?:direction_family|layer_region|intervention_arm|"
    r"attention_mlp_configuration|refusal_setting|strength_range)/"
    r"(?:completed|required|fraction)$"
)
_RECEIPT_LOCKS_GUARD = threading.Lock()
_RECEIPT_LOCKS: dict[str, threading.Lock] = {}


def _receipt_lock(path: Path) -> threading.Lock:
    identity = str(path.resolve())
    with _RECEIPT_LOCKS_GUARD:
        return _RECEIPT_LOCKS.setdefault(identity, threading.Lock())


class _WandbRun(Protocol):
    def log(
        self,
        values: Mapping[str, Any],
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None: ...

    def finish(self, *, exit_code: int = 0) -> None: ...


class _WandbModule(Protocol):
    def init(self, **kwargs: Any) -> _WandbRun: ...


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _proposal_mapping(value: Any) -> Mapping[str, Any]:
    method = getattr(value, "to_dict", None)
    result = method() if callable(method) else value
    return result if isinstance(result, Mapping) else {}


def _sanitized_parameters(value: Any) -> dict[str, Any]:
    """Copy only frozen optimizer knobs from a positive field allowlist."""

    result: dict[str, Any] = {}
    proposal = _proposal_mapping(value)
    for raw_name, raw_value in proposal.items():
        name = str(raw_name)
        if name not in _ALLOWED_PROPOSAL_PARAMETERS:
            continue
        if name in {"direction_ids", "selected_domains", "writer_layers"}:
            if (
                isinstance(raw_value, Sequence)
                and not isinstance(raw_value, (str, bytes))
                and len(raw_value) <= 64
                and all(isinstance(item, (str, int)) for item in raw_value)
            ):
                result[f"{name}_count"] = len(raw_value)
                result[f"{name}_sha256"] = hashlib.sha256(
                    _canonical(list(raw_value))
                ).hexdigest()
            continue
        if name in _BOOL_PARAMETERS and isinstance(raw_value, bool):
            result[name] = raw_value
        elif name in _NUMERIC_PARAMETERS and raw_value is None and name == "refusal_source_layer":
            result[name] = -1
        elif name in _NUMERIC_PARAMETERS and not isinstance(raw_value, bool):
            number = _safe_number(raw_value)
            if number is not None and -1.0 <= number <= 10000.0:
                result[name] = number
        elif name in _ENUM_PARAMETERS and raw_value in _ENUM_PARAMETERS[name]:
            result[name] = raw_value
        elif name in {"direction_family", "writer_region"} and (
            isinstance(raw_value, str)
            and 1 <= len(raw_value) <= 64
            and all(char.isalnum() or char in "_-" for char in raw_value)
            and _SENSITIVE_TEXT.search(raw_value) is None
        ):
            result[name] = raw_value
    return result


def _parameter_suffix(metric_name: str) -> str | None:
    if metric_name.startswith("trial/params/"):
        return metric_name.removeprefix("trial/params/")
    if "/params/" in metric_name and (
        metric_name.startswith("best/") or metric_name.startswith("pareto/candidate/")
    ):
        return metric_name.rsplit("/params/", 1)[1]
    return None


def _safe_metric_value(name: object, value: Any) -> str | bool | float | None:
    """Validate both metric names and values through one positive allowlist."""

    if not isinstance(name, str):
        return None
    suffix = _parameter_suffix(name)
    if suffix is not None:
        if suffix in _BOOL_PARAMETERS and isinstance(value, bool):
            return value
        if suffix in _NUMERIC_PARAMETERS or suffix in _PARAMETER_COUNTS:
            return _safe_number(value)
        if (
            suffix in _ENUM_PARAMETERS
            and isinstance(value, str)
            and value in _ENUM_PARAMETERS[suffix]
        ):
            return value
        if suffix in _PARAMETER_DIGESTS and isinstance(value, str) and _SHA256.fullmatch(value):
            return value
        if suffix in {"direction_family", "writer_region"} and (
            isinstance(value, str)
            and 1 <= len(value) <= 64
            and all(char.isalnum() or char in "_-" for char in value)
            and _SENSITIVE_TEXT.search(value) is None
        ):
            return value
        return None
    if name == "progress/phase":
        return value if value in _SAFE_PHASES else None
    if name == "progress/stage":
        return value if value in _ADAPTIVE_STAGES else None
    if name == "trial/outcome_kind":
        return value if value in _SAFE_OUTCOMES else None
    if name == "operations/error_category":
        return value if value in _SAFE_ERROR_CATEGORIES else None
    if name == "operations/error_fingerprint":
        return value if isinstance(value, str) and _SHA256.fullmatch(value) else None
    numeric_names = {
        "progress/completed_trials",
        "progress/current_batch",
        "progress/total_batches",
        "progress/elapsed_seconds",
        "progress/eta_seconds",
        "progress/planned_floor_trials",
        "progress/adaptive_ceiling_trials",
        "progress/measured_target_trials",
        "progress/search_cutoff_seconds",
        "progress/reserve_seconds",
        "progress/completed_search_trials",
        "progress/completed_repeat_trials",
        "progress/completed_control_trials",
        "progress/completed_final_selection_trials",
        "budget/total_usd",
        "budget/evaluation_usd",
        "budget/evaluation_reserve_fraction",
        "budget/evaluation_reserve_usd",
        "budget/evaluation_pre_reserve_headroom_usd",
        "budget/total_actual_usd",
        "budget/total_projected_usd",
        "budget/total_actual_headroom_usd",
        "budget/total_projected_headroom_usd",
        "canary/trial_duration_seconds",
        "canary/tokens_per_second",
        "canary/judge_latency_ms",
        "canary/judge_cost_usd_per_trial",
        "canary/resumed_session",
        "pareto/size",
        "trial/ordinal",
        "judge/calls",
        "judge/failures",
        "judge/latency_ms",
        "judge/cost_usd",
        "cost/gpu_actual_usd",
        "cost/gpu_projected_usd",
        "cost/judge_actual_usd",
        "cost/judge_projected_usd",
        "cost/total_actual_usd",
        "cost/total_projected_usd",
        "operations/retries",
        "operations/stopped_trials",
        "operations/errors",
    }
    if name in numeric_names or _GPU_METRIC.fullmatch(name) or _COVERAGE_METRIC.fullmatch(name):
        return _safe_number(value)
    if name.startswith("trial/objectives/"):
        objective = name.removeprefix("trial/objectives/")
        return _safe_number(value) if objective in OBJECTIVES else None
    if name.startswith("best/"):
        parts = name.split("/")
        if len(parts) in {2, 3} and parts[1] in OBJECTIVES and (
            len(parts) == 2 or parts[2] == "trial_ordinal"
        ):
            return _safe_number(value)
        return None
    if name.startswith("pareto/candidate/"):
        parts = name.split("/")
        if len(parts) == 4 and parts[2].isdigit() and (
            parts[3] == "trial_ordinal" or parts[3] in OBJECTIVES
        ):
            return _safe_number(value)
    return None


def _phase(completed_trials: int) -> str:
    if completed_trials < 80:
        return "discovery"
    if completed_trials < 160:
        return "expanded"
    return "finalist"


def _duration(seconds: float) -> str:
    bounded = max(0, int(seconds))
    hours, remainder = divmod(bounded, 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}m"


def monitoring_heartbeat(
    *,
    completed_trials: int,
    total_trials: int,
    current_batch: int,
    total_batches: int,
    elapsed_seconds: float,
    eta_seconds: float,
) -> str:
    """The entire intended terminal UI: one compact status line."""

    return (
        f"{completed_trials}/{total_trials} trials | "
        f"batch {current_batch}/{total_batches} | "
        f"elapsed {_duration(elapsed_seconds)} | ETA {_duration(eta_seconds)}"
    )


class CoordinatorMonitor:
    """Mirror sanitized coordinator events to exactly one resumable W&B run.

    All methods are intentionally best-effort and never raise.  ``run_id`` is
    created and restored by the separate strict durable-checkpoint module.
    Trial/progress aggregation has one coordinator-study writer; telemetry and
    transport state cross threads only through their locked seams.
    """

    def __init__(
        self,
        *,
        run_id: str,
        project: str,
        entity: str | None,
        run_name: str,
        receipt_path: Path,
        total_trials: int,
        batch_size: int,
        wandb_module: _WandbModule | None = None,
        monotonic: Any = time.monotonic,
        run_checkpoint_sha256: str | None = None,
        adaptive_progress_path: Path | None = None,
        transport_queue_capacity: int = 1024,
        transport_close_timeout_seconds: float = 1.0,
    ) -> None:
        self.run_id = run_id
        self.project = project
        self.entity = entity
        self._receipt_path = receipt_path
        self._total_trials = total_trials
        self._display_total_trials = total_trials
        self._batch_size = batch_size
        self._total_batches = math.ceil(total_trials / batch_size)
        self._monotonic = monotonic
        self._started_at = float(monotonic())
        self._run: _WandbRun | None = None
        self._previous_receipt_sha256: str | None = None
        self._logged_metric_keys: set[str] = set()
        self._nonfatal_errors = 0
        self._heartbeat_lines: list[str] = []
        self._initialized_coordinator_count = 0
        self._init_calls: list[dict[str, Any]] = []
        self._finish_calls = 0
        self._completed_trials = 0
        self._objectives: list[dict[str, float]] = []
        self._objective_ordinals: list[int] = []
        self._objective_parameters: list[dict[str, Any]] = []
        self._loss_chart_points: dict[int, dict[str, float]] = {}
        self._attempted_logs: list[dict[str, Any]] = []
        self._state_lock = threading.Lock()
        self._transport_condition = threading.Condition(self._state_lock)
        self._receipt_lock = _receipt_lock(receipt_path)
        self._receipt_writable = True
        self._receipt_terminal = False
        self._wandb_event_step: int | None = 0
        self._closed = False
        self._close_requested = False
        self._transport_state = "starting"
        self._transport_drained = False
        self._dropped_event_count = 0
        self._coalesced_telemetry_count = 0
        self._critical_events: deque[tuple[str, Any, bool]] = deque()
        self._telemetry_events: dict[str, tuple[str, Any, bool]] = {}
        self._inflight_operation: str | None = "init"
        if (
            isinstance(transport_queue_capacity, bool)
            or not isinstance(transport_queue_capacity, int)
            or transport_queue_capacity < 1
        ):
            raise ValueError("transport_queue_capacity must be a positive integer")
        close_timeout = float(transport_close_timeout_seconds)
        if not math.isfinite(close_timeout) or close_timeout < 0.0:
            raise ValueError("transport_close_timeout_seconds must be non-negative")
        self._transport_queue_capacity = transport_queue_capacity
        self._transport_close_timeout_seconds = close_timeout
        self._wandb = wandb_module
        self._run_checkpoint_sha256 = run_checkpoint_sha256
        self._adaptive_progress_path = adaptive_progress_path
        self._restore_receipt_chain_head()
        self._transport_thread = threading.Thread(
            target=self._transport_main,
            args=(run_name,),
            name="truth-editing-wandb-transport",
            daemon=True,
        )
        self._transport_thread.start()

    @classmethod
    def open(
        cls,
        *,
        checkpoint_path: Path,
        run_id: str,
        project: str,
        entity: str | None,
        run_name: str,
        receipt_path: Path,
        total_trials: int,
        batch_size: int,
        wandb_module: _WandbModule | None = None,
        monotonic: Any = time.monotonic,
    ) -> "CoordinatorMonitor":
        """Durably bind the coordinator run ID before connecting to W&B."""

        create_wandb_run_checkpoint(
            checkpoint_path, run_id=run_id, project=project, entity=entity
        )
        checkpoint = open_wandb_run_checkpoint(checkpoint_path)
        return cls(
            run_id=checkpoint.run_id,
            project=checkpoint.project,
            entity=checkpoint.entity,
            run_name=run_name,
            receipt_path=receipt_path,
            total_trials=total_trials,
            batch_size=batch_size,
            wandb_module=wandb_module,
            monotonic=monotonic,
            run_checkpoint_sha256=checkpoint.checkpoint_sha256,
            adaptive_progress_path=checkpoint_path.with_name("adaptive-progress.json"),
        )

    @property
    def run_checkpoint_sha256(self) -> str:
        """Hash binding every adaptive progress update to this W&B identity."""

        if self._run_checkpoint_sha256 is None:
            raise WandbCheckpointError(
                "coordinator monitor was not opened through its durable run checkpoint"
            )
        return self._run_checkpoint_sha256

    def record_adaptive_progress(
        self, progress: AdaptiveRunProgress
    ) -> AdaptiveProgressCheckpoint:
        """Commit local adaptive state, then best-effort mirror it to W&B.

        The local checkpoint is authoritative and therefore fails closed.
        W&B logging remains best-effort and cannot affect the committed state.
        """

        if self._adaptive_progress_path is None:
            raise WandbCheckpointError("adaptive progress checkpoint path is unavailable")
        if progress.wandb_run_checkpoint_sha256 != self.run_checkpoint_sha256:
            raise WandbCheckpointError("adaptive progress belongs to a different W&B run")
        checkpoint = advance_adaptive_progress_checkpoint(
            self._adaptive_progress_path, progress
        )
        self._display_total_trials = progress.measured_target_trials
        self._total_batches = math.ceil(self._display_total_trials / self._batch_size)
        values: dict[str, Any] = {
            "progress/planned_floor_trials": progress.planned_floor_trials,
            "progress/adaptive_ceiling_trials": progress.adaptive_ceiling_trials,
            "progress/measured_target_trials": progress.measured_target_trials,
            "progress/search_cutoff_seconds": progress.search_cutoff_seconds,
            "progress/reserve_seconds": progress.reserve_seconds,
            "progress/completed_search_trials": progress.completed_search_trials,
            "progress/completed_repeat_trials": progress.completed_repeat_trials,
            "progress/completed_control_trials": progress.completed_control_trials,
            "progress/completed_final_selection_trials": (
                progress.completed_final_selection_trials
            ),
            "progress/current_batch": progress.current_batch,
            "progress/elapsed_seconds": progress.elapsed_seconds,
            "progress/eta_seconds": progress.eta_seconds,
            "progress/stage": progress.stage,
            "budget/total_usd": progress.total_budget_usd,
            "budget/evaluation_usd": progress.evaluation_budget_usd,
            "budget/evaluation_reserve_fraction": (
                progress.evaluation_budget_reserve_fraction
            ),
            "budget/evaluation_reserve_usd": (
                progress.evaluation_budget_usd
                * progress.evaluation_budget_reserve_fraction
            ),
            "budget/evaluation_pre_reserve_headroom_usd": max(
                0.0,
                progress.evaluation_budget_usd
                * (1.0 - progress.evaluation_budget_reserve_fraction)
                - progress.judge_actual_usd,
            ),
            "cost/gpu_actual_usd": progress.gpu_actual_usd,
            "cost/gpu_projected_usd": progress.gpu_projected_usd,
            "cost/judge_actual_usd": progress.judge_actual_usd,
            "cost/judge_projected_usd": progress.judge_projected_usd,
            "cost/total_actual_usd": (
                progress.gpu_actual_usd + progress.judge_actual_usd
            ),
            "cost/total_projected_usd": progress.projected_total_usd,
            "budget/total_actual_usd": (
                progress.gpu_actual_usd + progress.judge_actual_usd
            ),
            "budget/total_projected_usd": progress.projected_total_usd,
            "budget/total_actual_headroom_usd": max(
                0.0,
                progress.total_budget_usd
                - progress.gpu_actual_usd
                - progress.judge_actual_usd,
            ),
            "budget/total_projected_headroom_usd": max(
                0.0, progress.total_budget_usd - progress.projected_total_usd
            ),
            "canary/trial_duration_seconds": progress.measured_trial_duration_seconds,
            "canary/tokens_per_second": progress.measured_tokens_per_second,
            "canary/judge_latency_ms": progress.measured_judge_latency_ms,
            "canary/judge_cost_usd_per_trial": (
                progress.measured_judge_cost_usd_per_trial
            ),
        }
        for name, (completed, required) in progress.coverage.items():
            values[f"coverage/{name}/completed"] = completed
            values[f"coverage/{name}/required"] = required
            values[f"coverage/{name}/fraction"] = completed / required
        self._safe_log(values)
        self._receipt(
            "adaptive_progress_mirrored",
            {
                "checkpoint_revision": checkpoint.revision,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            },
        )
        return checkpoint

    def _validated_receipt_tail(self) -> str | None:
        if not self._receipt_path.exists():
            return None
        previous: str | None = None
        for raw_line in self._receipt_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if (
                not isinstance(row, Mapping)
                or row.get("format") != MONITORING_RECEIPT_FORMAT
                or row.get("run_id") != self.run_id
                or row.get("previous_receipt_sha256") != previous
            ):
                raise ValueError("monitoring receipt chain differs")
            claimed = row.get("receipt_sha256")
            unsigned = {key: value for key, value in row.items() if key != "receipt_sha256"}
            if not isinstance(claimed, str) or hashlib.sha256(
                _canonical(unsigned)
            ).hexdigest() != claimed:
                raise ValueError("monitoring receipt hash differs")
            previous = claimed
        return previous

    def _restore_receipt_chain_head(self) -> None:
        try:
            with self._receipt_lock:
                self._previous_receipt_sha256 = self._validated_receipt_tail()
        except Exception:
            self._receipt_writable = False
            self._nonfatal_errors += 1

    def _receipt(
        self, kind: str, payload: Mapping[str, Any], *, terminal: bool = False
    ) -> None:
        # Receipt construction, append, and chain-head advance are one critical
        # section shared by every monitor using this path in the process.
        with self._receipt_lock:
            if not self._receipt_writable or self._receipt_terminal:
                return
            try:
                # A sequential reopen or another same-process monitor may have
                # advanced the path after this instance restored its head.
                self._previous_receipt_sha256 = self._validated_receipt_tail()
            except Exception:
                self._receipt_writable = False
                with self._state_lock:
                    self._nonfatal_errors += 1
                return
            unsigned = {
                "format": MONITORING_RECEIPT_FORMAT,
                "kind": kind,
                "run_id": self.run_id,
                "previous_receipt_sha256": self._previous_receipt_sha256,
                "payload": dict(payload),
            }
            digest = hashlib.sha256(_canonical(unsigned)).hexdigest()
            row = {**unsigned, "receipt_sha256": digest}
            try:
                self._receipt_path.parent.mkdir(parents=True, exist_ok=True)
                with self._receipt_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._previous_receipt_sha256 = digest
                if terminal:
                    self._receipt_terminal = True
            except Exception:
                # Monitoring storage is non-authoritative too. The optimizer must
                # continue even when both W&B and the monitoring audit sink fail.
                with self._state_lock:
                    self._nonfatal_errors += 1

    def _failure(self, operation: str, error: BaseException) -> None:
        with self._state_lock:
            self._nonfatal_errors += 1
        del error  # Raw exception types and text are intentionally private.
        safe_operation = operation if operation in _FAILURE_OPERATIONS else "monitoring"
        category = f"{safe_operation}_failure"
        self._receipt(
            "wandb_failure",
            {
                "operation": safe_operation,
                "error_category": category,
                "error_fingerprint": hashlib.sha256(category.encode()).hexdigest(),
            },
        )

    def _initialize_transport(self, run_name: str) -> None:
        try:
            # These affect only W&B's coordinator process. CUDA trial workers
            # independently strip every WANDB_* variable before launch.
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_QUIET"] = "true"
            os.environ["WANDB_CONSOLE"] = "off"
            if self._wandb is None:
                import wandb  # type: ignore

                self._wandb = wandb
            kwargs: dict[str, Any] = {
                "project": self.project,
                "entity": self.entity,
                "id": self.run_id,
                "name": RUN_NAME,
                "resume": "allow",
                "reinit": False,
                "save_code": False,
                "config": {
                    "monitoring_schema": VERIFICATION_FORMAT,
                    "total_trials": self._total_trials,
                    "batch_size": self._batch_size,
                },
            }
            settings_type = getattr(self._wandb, "Settings", None)
            if callable(settings_type):
                kwargs["settings"] = settings_type(
                    console="off",
                    disable_git=True,
                    disable_job_creation=True,
                    x_disable_stats=True,
                    x_disable_meta=True,
                    x_disable_machine_info=True,
                )
            init_call = {
                "id": self.run_id,
                "project": self.project,
                "entity": self.entity,
                "resume": "allow",
                "reinit": False,
                "privacy_settings": {
                    "console": "off",
                    "disable_git": True,
                    "save_code": False,
                    "log_model": False,
                    "automatic_system_metrics": False,
                },
            }
            with self._state_lock:
                self._init_calls.append(init_call)
            self._run = self._wandb.init(**kwargs)
            try:
                resumed_step = getattr(self._run, "starting_step", None)
            except Exception:
                resumed_step = None
            if not (
                not isinstance(resumed_step, bool)
                and isinstance(resumed_step, int)
                and resumed_step >= 0
            ):
                try:
                    resumed_step = getattr(self._run, "step", None)
                except Exception:
                    # A successfully initialized SDK run remains usable even when
                    # its optional resumed-step metadata is transiently unreadable.
                    resumed_step = None
            if (
                not isinstance(resumed_step, bool)
                and isinstance(resumed_step, int)
                and resumed_step >= 0
            ):
                self._wandb_event_step = resumed_step
            else:
                self._wandb_event_step = None
            with self._state_lock:
                self._initialized_coordinator_count = 1
            self._receipt("wandb_initialized", {"coordinator_runs": 1})
        except Exception as error:
            self._run = None
            self._failure("init", error)

    def _transport_main(self, run_name: str) -> None:
        """Own every potentially blocking W&B SDK call on one daemon."""

        try:
            self._initialize_transport(run_name)
            with self._transport_condition:
                self._inflight_operation = None
                if self._transport_state == "starting":
                    self._transport_state = "running"
                self._transport_condition.notify_all()
            while True:
                with self._transport_condition:
                    while (
                        not self._critical_events
                        and not self._telemetry_events
                        and not self._close_requested
                    ):
                        self._transport_condition.wait()
                    if self._critical_events:
                        kind, payload, record_attempt = self._critical_events.popleft()
                    elif self._telemetry_events:
                        telemetry_key = min(self._telemetry_events)
                        kind, payload, record_attempt = self._telemetry_events.pop(
                            telemetry_key
                        )
                    else:
                        self._inflight_operation = "finish"
                        break
                    self._inflight_operation = kind
                try:
                    if self._run is None:
                        continue
                    if kind == "log":
                        self._commit_wandb_row(payload, record_attempt=record_attempt)
                    elif kind == "chart":
                        self._publish_loss_chart(payload)
                except Exception as error:
                    self._failure("transport", error)
                finally:
                    with self._transport_condition:
                        self._inflight_operation = None
                        self._transport_condition.notify_all()
            finish_failed = False
            if self._run is not None:
                try:
                    self._run.finish(exit_code=0)
                    with self._state_lock:
                        self._finish_calls += 1
                except Exception as error:
                    finish_failed = True
                    self._failure("finish", error)
            with self._transport_condition:
                self._inflight_operation = None
                if self._transport_state != "close_timed_out":
                    self._transport_state = (
                        "finish_failed" if finish_failed else "drained"
                    )
                    self._transport_drained = True
                self._transport_condition.notify_all()
        except BaseException as error:
            self._failure("transport", error)
            with self._transport_condition:
                self._inflight_operation = None
                self._transport_state = "failed"
                self._transport_drained = False
                pending_events = len(self._critical_events) + len(
                    self._telemetry_events
                )
                self._transport_condition.notify_all()
            self._receipt(
                "wandb_transport_terminated",
                {
                    "transport_state": "failed",
                    "transport_drained": False,
                    "pending_event_count": pending_events,
                    "in_flight_event_count": 0,
                },
                terminal=True,
            )

    def _enqueue_transport(
        self,
        kind: str,
        payload: Any,
        *,
        record_attempt: bool = False,
        telemetry_key: str | None = None,
    ) -> None:
        dropped_now = False
        with self._transport_condition:
            if self._closed:
                return
            event = (kind, payload, record_attempt)
            if telemetry_key is not None:
                if telemetry_key in self._telemetry_events:
                    self._coalesced_telemetry_count += 1
                self._telemetry_events[telemetry_key] = event
                self._transport_condition.notify()
                return
            if len(self._critical_events) >= self._transport_queue_capacity:
                self._dropped_event_count += 1
                self._nonfatal_errors += 1
                dropped_now = True
            else:
                self._critical_events.append(event)
                self._transport_condition.notify()
        if dropped_now:
            self._receipt(
                "wandb_critical_event_dropped",
                {"reason": "critical_fifo_full", "dropped_event_count": 1},
            )

    def _commit_wandb_row(
        self, values: Mapping[str, Any], *, record_attempt: bool
    ) -> None:
        """Commit one dashboard row under the coordinator's event clock.

        Trial ordinals and adaptive progress are metrics, not W&B history row
        IDs. Serializing allocation with transport prevents telemetry and the
        study thread from publishing the same or a regressing explicit step.
        The resumed SDK run supplies the first acceptable event step.
        """

        if self._run is None:
            return
        event_step = self._wandb_event_step
        # Advance before transport: a connection failure can be ambiguous, so
        # reusing its event step could overwrite a row that arrived.
        if event_step is not None:
            self._wandb_event_step = event_step + 1
        if record_attempt:
            with self._state_lock:
                if len(self._attempted_logs) < 4096:
                    self._attempted_logs.append(
                        {"step": event_step, "values": dict(values)}
                    )
        try:
            if event_step is None:
                self._run.log(values, commit=True)
                try:
                    recovered_step = getattr(self._run, "step", None)
                except Exception:
                    recovered_step = None
                if (
                    not isinstance(recovered_step, bool)
                    and isinstance(recovered_step, int)
                    and recovered_step >= 0
                ):
                    self._wandb_event_step = recovered_step
            else:
                self._run.log(values, step=event_step, commit=True)
            with self._state_lock:
                self._logged_metric_keys.update(values)
        except Exception as error:
            self._failure("log", error)

    def _safe_log(
        self, values: Mapping[str, Any], *, telemetry_key: str | None = None
    ) -> None:
        clean: dict[str, Any] = {}
        for name, value in values.items():
            checked = _safe_metric_value(name, value)
            if checked is not None:
                clean[str(name)] = checked
        if not clean:
            return
        self._enqueue_transport(
            "log", clean, record_attempt=True, telemetry_key=telemetry_key
        )

    def _log_loss_chart(self) -> None:
        """Publish one interactive, lower-is-better view of trial quality.

        W&B's line-series chart supplies the clickable legend. The aggregate
        is display-only: it never enters Optuna, the journal, or selection.
        """

        if not self._loss_chart_points:
            return
        self._enqueue_transport(
            "chart",
            {ordinal: dict(values) for ordinal, values in self._loss_chart_points.items()},
            telemetry_key="chart/loss_overview",
        )

    def _publish_loss_chart(self, points: Mapping[int, Mapping[str, float]]) -> None:
        """Create and publish the plot only from the transport daemon."""

        if self._run is None or self._wandb is None:
            return
        plot = getattr(self._wandb, "plot", None)
        line_series = getattr(plot, "line_series", None)
        if not callable(line_series) or not points:
            return
        ordinals = sorted(points)
        names = (
            "valid_false_report_rate_lcb",
            "truth_report_dissociation_lcb",
            "capability_preservation_lcb",
        )
        complete = [
            (ordinal, points[ordinal])
            for ordinal in ordinals
            if all(
                name in points[ordinal]
                and 0.0 <= points[ordinal][name] <= 1.0
                for name in names
            )
        ]
        if not complete:
            return
        trial_ordinals = [ordinal for ordinal, _ in complete]
        false_report = [row[names[0]] for _, row in complete]
        retained_truth = [row[names[1]] for _, row in complete]
        capability = [row[names[2]] for _, row in complete]
        kl_points = [
            (ordinal, -math.log(row[names[2]]))
            for ordinal, row in complete
            if row[names[2]] > 0.0
        ]
        xs = [
            trial_ordinals,
            trial_ordinals,
            trial_ordinals,
            trial_ordinals,
            [ordinal for ordinal, _ in kl_points],
        ]
        ys = [
            [
                1.0 - (false_value * truth_value * capability_value) ** (1.0 / 3.0)
                for false_value, truth_value, capability_value in zip(
                    false_report, retained_truth, capability, strict=True
                )
            ],
            [1.0 - value for value in false_report],
            [1.0 - value for value in retained_truth],
            [1.0 - value for value in capability],
            [value for _, value in kl_points],
        ]
        try:
            chart = line_series(
                xs=xs,
                ys=ys,
                keys=[
                    "Overall loss",
                    "False-report loss",
                    "Retained-truth loss",
                    "Capability loss",
                    "Worst preservation KL",
                ],
                title="Loss overview (lower is better)",
                xname="Trial",
            )
            self._commit_wandb_row({"charts/loss_overview": chart}, record_attempt=False)
        except Exception as error:
            self._failure("log", error)

    @staticmethod
    def _pareto_size(points: Sequence[Mapping[str, float]]) -> int:
        count = 0
        for index, point in enumerate(points):
            dominated = any(
                index != other_index
                and all(other.get(name, -math.inf) >= value for name, value in point.items())
                and any(other.get(name, -math.inf) > value for name, value in point.items())
                for other_index, other in enumerate(points)
            )
            if not dominated:
                count += 1
        return count

    def record_batch(
        self,
        batch_ordinal: int,
        requests: Sequence[BatchEvaluationRequest[Any]],
        results: Sequence[EvaluationResult],
    ) -> None:
        try:
            if len(requests) != len(results):
                self._receipt("monitoring_input_rejected", {"category": "batch_length"})
                return
            for request, result in zip(requests, results, strict=True):
                if (
                    isinstance(request.ordinal, bool)
                    or not isinstance(request.ordinal, int)
                    or request.ordinal < 0
                    or result.outcome_kind not in _SAFE_OUTCOMES
                    or not isinstance(result.metrics, Mapping)
                ):
                    self._receipt(
                        "monitoring_input_rejected", {"category": "trial_shape"}
                    )
                    continue
                parameters = _sanitized_parameters(request.proposal)
                values: dict[str, Any] = {
                    "trial/ordinal": request.ordinal,
                    "trial/outcome_kind": result.outcome_kind,
                }
                values.update(
                    {
                        f"trial/params/{name}": value
                        for name, value in parameters.items()
                    }
                )
                objectives = {
                    str(name): float(value)
                    for name, value in result.metrics.items()
                    if name in OBJECTIVES and _safe_number(value) is not None
                }
                values.update(
                    {f"trial/objectives/{name}": value for name, value in objectives.items()}
                )
                if objectives and result.outcome_kind != "operational_failure":
                    self._loss_chart_points[request.ordinal] = objectives
                if objectives and result.outcome_kind == "successful":
                    self._objectives.append(objectives)
                    self._objective_ordinals.append(request.ordinal)
                    self._objective_parameters.append(parameters)
                self._completed_trials = max(self._completed_trials, request.ordinal + 1)
                self._safe_log(values)
            self._log_loss_chart()
            elapsed = max(0.0, float(self._monotonic()) - self._started_at)
            eta = (
                elapsed
                / self._completed_trials
                * (self._display_total_trials - self._completed_trials)
                if self._completed_trials and elapsed
                else 0.0
            )
            summary: dict[str, Any] = {
                "progress/completed_trials": self._completed_trials,
                "progress/current_batch": batch_ordinal + 1,
                "progress/total_batches": self._total_batches,
                "progress/elapsed_seconds": elapsed,
                "progress/eta_seconds": eta,
                "progress/phase": _phase(self._completed_trials),
                "pareto/size": self._pareto_size(self._objectives),
            }
            for name in {key for row in self._objectives for key in row}:
                candidates = [
                    (row[name], ordinal)
                    for row, ordinal in zip(
                        self._objectives, self._objective_ordinals, strict=True
                    )
                    if name in row
                ]
                best_value, best_ordinal = max(candidates)
                summary[f"best/{name}"] = best_value
                summary[f"best/{name}/trial_ordinal"] = best_ordinal
                best_index = self._objective_ordinals.index(best_ordinal)
                for parameter, value in self._objective_parameters[best_index].items():
                    summary[f"best/{name}/params/{parameter}"] = value
            frontier_indexes = self._pareto_indexes(self._objectives)
            summary["pareto/size"] = len(frontier_indexes)
            for frontier_rank, index in enumerate(frontier_indexes):
                summary[f"pareto/candidate/{frontier_rank}/trial_ordinal"] = (
                    self._objective_ordinals[index]
                )
                for name, value in self._objectives[index].items():
                    summary[f"pareto/candidate/{frontier_rank}/{name}"] = value
                for parameter, value in self._objective_parameters[index].items():
                    summary[
                        f"pareto/candidate/{frontier_rank}/params/{parameter}"
                    ] = value
            self._safe_log(summary)
            heartbeat = monitoring_heartbeat(
                completed_trials=self._completed_trials,
                total_trials=self._display_total_trials,
                current_batch=batch_ordinal + 1,
                total_batches=self._total_batches,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )
            with self._state_lock:
                self._heartbeat_lines.append(heartbeat)
            print(heartbeat, flush=True)
        except Exception as error:
            self._failure("record_batch", error)

    def record_trials(self, trials: Sequence[StudyTrial]) -> None:
        """Mirror a barrier only after the study has committed it locally."""

        try:
            if not trials:
                return
            requests = tuple(
                BatchEvaluationRequest(
                    trial_id=trial.trial_id,
                    ordinal=trial.ordinal,
                    proposal=trial.proposal,
                    # Never copied to W&B; intentionally omit real record IDs.
                    record_ids=(),
                    objective_names=tuple(trial.result.metrics),
                )
                for trial in trials
            )
            self.record_batch(
                trials[0].batch_ordinal,
                requests,
                tuple(trial.result for trial in trials),
            )
            failures = sum(
                trial.result.outcome_kind == "operational_failure" for trial in trials
            )
            if failures:
                category = "worker_operational_failure"
                self.record_operational(
                    retries=0,
                    stopped_trials=0,
                    errors=failures,
                    error_category=category,
                    error_fingerprint=hashlib.sha256(category.encode()).hexdigest(),
                )
        except Exception as error:
            self._failure("record_trials", error)

    @staticmethod
    def _pareto_indexes(points: Sequence[Mapping[str, float]]) -> tuple[int, ...]:
        indexes: list[int] = []
        for index, point in enumerate(points):
            dominated = any(
                index != other_index
                and all(other.get(name, -math.inf) >= value for name, value in point.items())
                and any(other.get(name, -math.inf) > value for name, value in point.items())
                for other_index, other in enumerate(points)
            )
            if not dominated:
                indexes.append(index)
        return tuple(indexes)

    def record_gpu(self, snapshot: GpuTelemetryRecord) -> None:
        if (
            isinstance(snapshot.gpu_slot, bool)
            or not isinstance(snapshot.gpu_slot, int)
            or not 0 <= snapshot.gpu_slot <= 31
            or _safe_number(snapshot.utilization_percent) is None
            or not 0 <= float(snapshot.utilization_percent) <= 100
            or _safe_number(snapshot.memory_used_mib) is None
            or float(snapshot.memory_used_mib) < 0
            or _safe_number(snapshot.memory_total_mib) is None
            or float(snapshot.memory_total_mib) <= 0
            or float(snapshot.memory_used_mib) > float(snapshot.memory_total_mib)
            or (
                snapshot.tokens_per_second is not None
                and (
                    _safe_number(snapshot.tokens_per_second) is None
                    or float(snapshot.tokens_per_second) < 0
                )
            )
        ):
            self._receipt("monitoring_input_rejected", {"category": "gpu_shape"})
            return
        prefix = f"gpu/{snapshot.gpu_slot}"
        values: dict[str, Any] = {
            f"{prefix}/utilization_pct": snapshot.utilization_percent,
            f"{prefix}/memory_used_mib": snapshot.memory_used_mib,
            f"{prefix}/memory_total_mib": snapshot.memory_total_mib,
            f"{prefix}/tps": snapshot.tokens_per_second or 0.0,
        }
        if snapshot.active_trial_id is not None and snapshot.active_trial_id.startswith("trial-"):
            try:
                values[f"{prefix}/active_trial_ordinal"] = int(
                    snapshot.active_trial_id.removeprefix("trial-")
                )
            except ValueError:
                pass
        self._safe_log(values, telemetry_key=f"gpu/{snapshot.gpu_slot}")

    def record_worker_telemetry(
        self, gpu_slot: int, trial_id: str, telemetry: Mapping[str, float]
    ) -> None:
        """Log the worker's measured generation rate at trial completion."""

        if (
            isinstance(gpu_slot, bool)
            or not isinstance(gpu_slot, int)
            or not 0 <= gpu_slot < 32
            or not isinstance(trial_id, str)
            or not trial_id.startswith("trial-")
        ):
            return
        try:
            ordinal = int(trial_id.removeprefix("trial-"))
        except ValueError:
            return
        tps = _safe_number(telemetry.get("generated_tokens_per_second"))
        if tps is None or tps < 0:
            return
        self._safe_log(
            {
                f"gpu/{gpu_slot}/tps": tps,
                f"gpu/{gpu_slot}/active_trial_ordinal": ordinal,
            },
            telemetry_key=f"gpu/{gpu_slot}",
        )

    def record_judge(
        self, *, calls: int, failures: int, latency_ms: float, cost_usd: float
    ) -> None:
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls < 0
            or isinstance(failures, bool)
            or not isinstance(failures, int)
            or not 0 <= failures <= calls
            or any(_safe_number(item) is None or float(item) < 0 for item in (latency_ms, cost_usd))
        ):
            self._receipt("monitoring_input_rejected", {"category": "judge_shape"})
            return
        self._safe_log(
            {
                "judge/calls": calls,
                "judge/failures": failures,
                "judge/latency_ms": latency_ms,
                "judge/cost_usd": cost_usd,
            },
            telemetry_key="judge",
        )

    def record_cost(
        self,
        *,
        gpu_actual_usd: float,
        gpu_projected_usd: float,
        judge_actual_usd: float = 0.0,
        judge_projected_usd: float = 0.0,
    ) -> None:
        amounts = (
            gpu_actual_usd,
            gpu_projected_usd,
            judge_actual_usd,
            judge_projected_usd,
        )
        if any(_safe_number(item) is None or float(item) < 0 for item in amounts):
            self._receipt("monitoring_input_rejected", {"category": "cost_shape"})
            return
        self._safe_log(
            {
                "cost/gpu_actual_usd": gpu_actual_usd,
                "cost/gpu_projected_usd": gpu_projected_usd,
                "cost/judge_actual_usd": judge_actual_usd,
                "cost/judge_projected_usd": judge_projected_usd,
                "cost/total_actual_usd": gpu_actual_usd + judge_actual_usd,
                "cost/total_projected_usd": gpu_projected_usd + judge_projected_usd,
            },
            telemetry_key="cost",
        )

    def record_operational(
        self,
        *,
        retries: int,
        stopped_trials: int,
        errors: int,
        error_category: str | None = None,
        error_fingerprint: str | None = None,
    ) -> None:
        counters = (retries, stopped_trials, errors)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counters
        ):
            self._receipt(
                "monitoring_input_rejected", {"category": "operational_shape"}
            )
            return
        values: dict[str, Any] = {
            "operations/retries": retries,
            "operations/stopped_trials": stopped_trials,
            "operations/errors": errors,
        }
        if (
            error_category in _SAFE_ERROR_CATEGORIES
            and error_fingerprint
            and _SHA256.fullmatch(error_fingerprint)
        ):
            values["operations/error_category"] = error_category
            values["operations/error_fingerprint"] = error_fingerprint
        self._safe_log(values)

    def record_resume_marker(self, *, session_ordinal: int) -> None:
        """Emit the safe marker used to prove post-resume delivery."""

        if session_ordinal != 2:
            self._receipt(
                "monitoring_input_rejected", {"category": "resume_marker_shape"}
            )
            return
        self._safe_log({"canary/resumed_session": session_ordinal})

    def verification_snapshot(self) -> Mapping[str, Any]:
        with self._state_lock:
            snapshot = {
                "format": VERIFICATION_FORMAT,
                "run_id": self.run_id,
                "initialized_coordinator_count": self._initialized_coordinator_count,
                "logged_metric_keys": sorted(self._logged_metric_keys),
                "nonfatal_error_count": self._nonfatal_errors,
                "heartbeat_lines": list(self._heartbeat_lines),
                "privacy_controls": {
                    "console_capture": False,
                    "code_capture": False,
                    "git_capture": False,
                    "artifact_upload": False,
                    "model_upload": False,
                    "automatic_system_metrics": False,
                },
                "attempted_logs": list(self._attempted_logs),
                "init_calls": list(self._init_calls),
                "finish_calls": self._finish_calls,
                "transport_state": self._transport_state,
                "transport_drained": self._transport_drained,
                "pending_event_count": (
                    len(self._critical_events) + len(self._telemetry_events)
                ),
                "in_flight_event_count": int(self._inflight_operation is not None),
                "dropped_event_count": self._dropped_event_count,
                "coalesced_telemetry_count": self._coalesced_telemetry_count,
            }
        return snapshot

    def close(self) -> None:
        with self._transport_condition:
            if self._closed:
                return
            self._closed = True
            self._close_requested = True
            self._transport_condition.notify_all()
        self._transport_thread.join(timeout=self._transport_close_timeout_seconds)
        alive = self._transport_thread.is_alive()
        with self._transport_condition:
            if alive:
                self._transport_state = "close_timed_out"
                self._transport_drained = False
                self._nonfatal_errors += 1
            initialized = self._initialized_coordinator_count
            errors = self._nonfatal_errors
            dropped_events = self._dropped_event_count
            pending_events = len(self._critical_events) + len(self._telemetry_events)
            inflight_events = int(self._inflight_operation is not None)
            coalesced = self._coalesced_telemetry_count
            state = self._transport_state
            drained = self._transport_drained
        payload = {
            "initialized_coordinator_count": initialized,
            "nonfatal_error_count": errors,
            "transport_state": state,
            "transport_drained": drained,
            "pending_event_count": pending_events,
            "in_flight_event_count": inflight_events,
            "dropped_event_count": dropped_events,
            "coalesced_telemetry_count": coalesced,
        }
        kind = (
            "wandb_close_timeout"
            if alive
            else "wandb_closed"
            if drained
            else "wandb_transport_terminated"
        )
        self._receipt(kind, payload, terminal=True)


class MonitoredSearchDriver:
    """Log only committed journal barriers while preserving search identity."""

    def __init__(self, driver: Any, monitor: CoordinatorMonitor) -> None:
        self._driver = driver
        self._monitor = monitor

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._driver.identity

    @property
    def persistent_study_name(self) -> str:
        """Forward the prepared driver's durable Optuna study identity."""

        return self._driver.persistent_study_name

    def prepare(self, config: Any, directions: Any, state_path: Path) -> None:
        self._driver.prepare(config, directions, state_path)

    def suggest(self, request: Any) -> Any:
        return self._driver.suggest(request)

    def observe(self, trials: Sequence[StudyTrial]) -> None:
        self._driver.observe(trials)
        # TruthEditingStudy invokes observe only after every trial in the batch
        # was atomically committed to its authoritative journal.
        self._monitor.record_trials(trials)

    def complete_history_replay(self) -> None:
        """Forward the journal/new-work boundary without changing monitoring."""

        self._driver.complete_history_replay()


class CoordinatorTelemetryPump:
    """Poll coordinator-owned GPU telemetry without blocking optimization."""

    def __init__(
        self,
        collector: GpuTelemetryCollector,
        monitor: CoordinatorMonitor,
        *,
        interval_seconds: float = 5.0,
        host_hourly_usd: float = 0.0,
        initial_host_elapsed_seconds: float = 0.0,
        gpu_projected_usd: float = 0.0,
        judge_projected_usd: float = 0.0,
        judge_budget: Any | None = None,
        monotonic: Any = time.monotonic,
    ) -> None:
        self._collector = collector
        self._monitor = monitor
        self._interval = max(1.0, float(interval_seconds))
        self._host_hourly_usd = max(0.0, float(host_hourly_usd))
        self._initial_host_elapsed_seconds = max(
            0.0, float(initial_host_elapsed_seconds)
        )
        self._gpu_projected_usd = max(0.0, float(gpu_projected_usd))
        self._judge_projected_usd = max(0.0, float(judge_projected_usd))
        self._judge_budget = judge_budget
        self._monotonic = monotonic
        self._started_at = float(monotonic())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "CoordinatorTelemetryPump":
        self._thread = threading.Thread(
            target=self._run, name="truth-editing-gpu-monitor", daemon=True
        )
        self._thread.start()
        return self

    def estimated_host_cost_usd(self) -> float:
        """Return whole-host cost including setup before monitoring began."""

        elapsed = self._initial_host_elapsed_seconds + max(
            0.0, float(self._monotonic()) - self._started_at
        )
        return self._host_hourly_usd * elapsed / 3600.0

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for record in self._collector.poll():
                    self._monitor.record_gpu(record)
                diagnostic = self._collector.last_error
                if diagnostic is not None:
                    self._monitor.record_operational(
                        retries=0,
                        stopped_trials=0,
                        errors=1,
                        error_category=diagnostic.category,
                        error_fingerprint=diagnostic.fingerprint_sha256,
                    )
                judge = (
                    self._judge_budget.monitoring_snapshot()
                    if self._judge_budget is not None
                    else {"calls": 0, "failures": 0, "latency_ms": 0.0, "cost_usd": 0.0}
                )
                self._monitor.record_judge(
                    calls=int(judge["calls"]),
                    failures=int(judge["failures"]),
                    latency_ms=float(judge["latency_ms"]),
                    cost_usd=float(judge["cost_usd"]),
                )
                self._monitor.record_cost(
                    gpu_actual_usd=self.estimated_host_cost_usd(),
                    gpu_projected_usd=self._gpu_projected_usd,
                    judge_actual_usd=float(judge["cost_usd"]),
                    judge_projected_usd=self._judge_projected_usd,
                )
            except Exception as error:
                self._monitor._failure("telemetry_poll", error)
            self._stop.wait(self._interval)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None


__all__ = [
    "CoordinatorMonitor",
    "CoordinatorTelemetryPump",
    "MONITORING_RECEIPT_FORMAT",
    "MonitoredSearchDriver",
    "VERIFICATION_FORMAT",
    "monitoring_heartbeat",
    "RUN_NAME",
]
