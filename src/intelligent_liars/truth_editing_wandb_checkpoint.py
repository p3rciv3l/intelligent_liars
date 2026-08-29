"""Immutable coordinator-owned W&B run identity.

The sidecar is local control-plane state, not an optimization result.  It is
created before the coordinator connects to W&B and is then carried byte-for-byte
through phase checkpoints so a resumed coordinator reconnects to the same run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


FORMAT = "truth_editing_wandb_run_checkpoint_v1"
ADAPTIVE_PROGRESS_FORMAT = "truth_editing_adaptive_progress_checkpoint_v1"
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LOCATION = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COVERAGE_DIMENSIONS = frozenset(
    {
        "direction_family",
        "layer_region",
        "intervention_arm",
        "attention_mlp_configuration",
        "refusal_setting",
        "strength_range",
    }
)
_STAGES = (
    "broad_coverage",
    "adaptive_search",
    "finalization_reserved",
    "repeats",
    "controls",
    "final_selection",
    "checkpoint_export",
    "complete",
    "aborted",
)


class WandbCheckpointError(RuntimeError):
    """The local W&B run identity cannot be trusted."""


@dataclass(frozen=True)
class WandbRunCheckpoint:
    run_id: str
    project: str
    entity: str | None
    checkpoint_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "run_id": self.run_id,
            "project": self.project,
            "entity": self.entity,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class AdaptiveRunProgress:
    """Authoritative local progress for the adaptive search and reserve window."""

    wandb_run_checkpoint_sha256: str
    study_config_sha256: str
    planned_floor_trials: int
    adaptive_ceiling_trials: int
    measured_target_trials: int
    batch_size: int
    search_cutoff_seconds: int
    reserve_seconds: int
    total_budget_usd: float
    evaluation_budget_usd: float
    evaluation_budget_reserve_fraction: float
    completed_search_trials: int
    completed_repeat_trials: int
    completed_control_trials: int
    completed_final_selection_trials: int
    current_batch: int
    stage: str
    coverage: Mapping[str, tuple[int, int]]
    elapsed_seconds: float
    eta_seconds: float
    gpu_actual_usd: float
    gpu_projected_usd: float
    judge_actual_usd: float
    judge_projected_usd: float
    projected_total_usd: float
    measured_trial_duration_seconds: float
    measured_tokens_per_second: float
    measured_judge_latency_ms: float
    measured_judge_cost_usd_per_trial: float

    def __post_init__(self) -> None:
        coverage = _validated_coverage(self.coverage)
        object.__setattr__(self, "coverage", coverage)
        _validated_progress(self)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "wandb_run_checkpoint_sha256": self.wandb_run_checkpoint_sha256,
            "study_config_sha256": self.study_config_sha256,
            "planned_floor_trials": self.planned_floor_trials,
            "adaptive_ceiling_trials": self.adaptive_ceiling_trials,
            "measured_target_trials": self.measured_target_trials,
            "batch_size": self.batch_size,
            "search_cutoff_seconds": self.search_cutoff_seconds,
            "reserve_seconds": self.reserve_seconds,
            "total_budget_usd": self.total_budget_usd,
            "evaluation_budget_usd": self.evaluation_budget_usd,
            "evaluation_budget_reserve_fraction": self.evaluation_budget_reserve_fraction,
            "completed_search_trials": self.completed_search_trials,
            "completed_repeat_trials": self.completed_repeat_trials,
            "completed_control_trials": self.completed_control_trials,
            "completed_final_selection_trials": self.completed_final_selection_trials,
            "current_batch": self.current_batch,
            "stage": self.stage,
            "coverage": {
                name: {"completed": values[0], "required": values[1]}
                for name, values in sorted(self.coverage.items())
            },
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "gpu_actual_usd": self.gpu_actual_usd,
            "gpu_projected_usd": self.gpu_projected_usd,
            "judge_actual_usd": self.judge_actual_usd,
            "judge_projected_usd": self.judge_projected_usd,
            "projected_total_usd": self.projected_total_usd,
            "measured_trial_duration_seconds": self.measured_trial_duration_seconds,
            "measured_tokens_per_second": self.measured_tokens_per_second,
            "measured_judge_latency_ms": self.measured_judge_latency_ms,
            "measured_judge_cost_usd_per_trial": self.measured_judge_cost_usd_per_trial,
        }


@dataclass(frozen=True)
class AdaptiveProgressCheckpoint:
    revision: int
    previous_checkpoint_sha256: str | None
    progress: AdaptiveRunProgress
    checkpoint_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": ADAPTIVE_PROGRESS_FORMAT,
            "revision": self.revision,
            "previous_checkpoint_sha256": self.previous_checkpoint_sha256,
            "progress": self.progress.to_mapping(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def create_wandb_run_checkpoint(
    path: Path | str,
    *,
    run_id: str,
    project: str,
    entity: str | None,
) -> WandbRunCheckpoint:
    """Create one immutable identity, or reopen an identical existing one."""

    target = Path(path)
    unsigned = _validated_unsigned(run_id=run_id, project=project, entity=entity)
    checkpoint = WandbRunCheckpoint(
        run_id=run_id,
        project=project,
        entity=entity,
        checkpoint_sha256=_canonical_sha(unsigned),
    )
    _reject_symlinked_ancestors(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _encoded(checkpoint.to_mapping())
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        existing = open_wandb_run_checkpoint(target)
        if existing != checkpoint:
            raise WandbCheckpointError(
                "checkpoint already belongs to a different W&B run identity"
            )
        return existing
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(target.parent)
    return checkpoint


def open_wandb_run_checkpoint(path: Path | str) -> WandbRunCheckpoint:
    """Strictly open and verify one coordinator run identity sidecar."""

    target = Path(path)
    _reject_symlinked_ancestors(target)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise WandbCheckpointError("W&B run checkpoint is missing or unsafe") from error
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    return parse_wandb_run_checkpoint(payload)


def parse_wandb_run_checkpoint(payload: bytes) -> WandbRunCheckpoint:
    """Strictly parse checkpoint bytes already read by a durable container."""

    try:
        raw = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WandbCheckpointError("W&B run checkpoint is unreadable") from error
    if not isinstance(raw, dict) or set(raw) != {
        "format",
        "run_id",
        "project",
        "entity",
        "checkpoint_sha256",
    }:
        raise WandbCheckpointError("W&B run checkpoint fields differ")
    if raw["format"] != FORMAT:
        raise WandbCheckpointError("W&B run checkpoint format differs")
    unsigned = _validated_unsigned(
        run_id=raw["run_id"], project=raw["project"], entity=raw["entity"]
    )
    claimed = raw["checkpoint_sha256"]
    if not isinstance(claimed, str) or claimed != _canonical_sha(unsigned):
        raise WandbCheckpointError("W&B run checkpoint hash differs")
    return WandbRunCheckpoint(
        run_id=raw["run_id"],
        project=raw["project"],
        entity=raw["entity"],
        checkpoint_sha256=claimed,
    )


def advance_adaptive_progress_checkpoint(
    path: Path | str, progress: AdaptiveRunProgress
) -> AdaptiveProgressCheckpoint:
    """Atomically create or advance the hash-chained adaptive progress state.

    Repeating an identical write is idempotent. Any policy drift, identity
    switch, counter regression, or corrupt prior state fails closed before the
    file is replaced.
    """

    target = Path(path)
    _reject_symlinked_ancestors(target.parent)
    existing: AdaptiveProgressCheckpoint | None = None
    if target.exists() or target.is_symlink():
        existing = open_adaptive_progress_checkpoint(target)
        if existing.progress == progress:
            return existing
        _validate_progress_transition(existing.progress, progress)
    revision = 0 if existing is None else existing.revision + 1
    previous = None if existing is None else existing.checkpoint_sha256
    unsigned = {
        "format": ADAPTIVE_PROGRESS_FORMAT,
        "revision": revision,
        "previous_checkpoint_sha256": previous,
        "progress": progress.to_mapping(),
    }
    checkpoint = AdaptiveProgressCheckpoint(
        revision=revision,
        previous_checkpoint_sha256=previous,
        progress=progress,
        checkpoint_sha256=_canonical_sha(unsigned),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(_encoded(checkpoint.to_mapping()))
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return checkpoint


def open_adaptive_progress_checkpoint(
    path: Path | str,
) -> AdaptiveProgressCheckpoint:
    """Strictly open the authoritative adaptive progress state."""

    target = Path(path)
    _reject_symlinked_ancestors(target)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as error:
        raise WandbCheckpointError(
            "adaptive progress checkpoint is missing or unsafe"
        ) from error
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
    finally:
        os.close(descriptor)
    return parse_adaptive_progress_checkpoint(payload)


def parse_adaptive_progress_checkpoint(payload: bytes) -> AdaptiveProgressCheckpoint:
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WandbCheckpointError("adaptive progress checkpoint is unreadable") from error
    expected = {
        "format",
        "revision",
        "previous_checkpoint_sha256",
        "progress",
        "checkpoint_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise WandbCheckpointError("adaptive progress checkpoint fields differ")
    if raw["format"] != ADAPTIVE_PROGRESS_FORMAT:
        raise WandbCheckpointError("adaptive progress checkpoint format differs")
    revision = raw["revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise WandbCheckpointError("adaptive progress checkpoint revision is invalid")
    previous = raw["previous_checkpoint_sha256"]
    if previous is not None and (
        not isinstance(previous, str) or _SHA256.fullmatch(previous) is None
    ):
        raise WandbCheckpointError("adaptive progress previous hash is invalid")
    if (revision == 0) != (previous is None):
        raise WandbCheckpointError("adaptive progress hash chain is invalid")
    progress = _progress_from_mapping(raw["progress"])
    unsigned = {
        "format": ADAPTIVE_PROGRESS_FORMAT,
        "revision": revision,
        "previous_checkpoint_sha256": previous,
        "progress": progress.to_mapping(),
    }
    claimed = raw["checkpoint_sha256"]
    if not isinstance(claimed, str) or claimed != _canonical_sha(unsigned):
        raise WandbCheckpointError("adaptive progress checkpoint hash differs")
    return AdaptiveProgressCheckpoint(revision, previous, progress, claimed)


def _progress_from_mapping(value: Any) -> AdaptiveRunProgress:
    if not isinstance(value, Mapping):
        raise WandbCheckpointError("adaptive progress must be an object")
    fields = set(AdaptiveRunProgress.__dataclass_fields__)
    if set(value) != fields:
        raise WandbCheckpointError("adaptive progress fields differ")
    raw_coverage = value.get("coverage")
    if not isinstance(raw_coverage, Mapping):
        raise WandbCheckpointError("adaptive progress coverage is invalid")
    coverage: dict[str, tuple[int, int]] = {}
    for name, counts in raw_coverage.items():
        if not isinstance(counts, Mapping) or set(counts) != {"completed", "required"}:
            raise WandbCheckpointError("adaptive progress coverage fields differ")
        coverage[str(name)] = (counts["completed"], counts["required"])
    return AdaptiveRunProgress(
        **{**dict(value), "coverage": coverage}
    )


def _validated_progress(progress: AdaptiveRunProgress) -> None:
    if (
        progress.planned_floor_trials != 200
        or progress.adaptive_ceiling_trials != 800
        or progress.batch_size != 8
        or progress.search_cutoff_seconds != 21 * 3600
        or progress.reserve_seconds != 3 * 3600
        or progress.total_budget_usd != 50.0
        or progress.evaluation_budget_usd != 5.0
        or progress.evaluation_budget_reserve_fraction != 0.2
    ):
        raise WandbCheckpointError("adaptive progress frozen policy differs")
    if (
        not isinstance(progress.wandb_run_checkpoint_sha256, str)
        or _SHA256.fullmatch(progress.wandb_run_checkpoint_sha256) is None
    ):
        raise WandbCheckpointError("adaptive progress W&B identity is invalid")
    if (
        not isinstance(progress.study_config_sha256, str)
        or _SHA256.fullmatch(progress.study_config_sha256) is None
    ):
        raise WandbCheckpointError("adaptive progress study config identity is invalid")
    for name in (
        "measured_target_trials",
        "completed_search_trials",
        "completed_repeat_trials",
        "completed_control_trials",
        "completed_final_selection_trials",
        "current_batch",
    ):
        value = getattr(progress, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WandbCheckpointError(f"adaptive progress {name} is invalid")
    if (
        not progress.planned_floor_trials
        <= progress.measured_target_trials
        <= progress.adaptive_ceiling_trials
        or progress.measured_target_trials % progress.batch_size
        or progress.completed_search_trials > progress.adaptive_ceiling_trials
        or progress.completed_search_trials % progress.batch_size
        or progress.current_batch != progress.completed_search_trials // progress.batch_size
        or progress.measured_target_trials < progress.completed_search_trials
    ):
        raise WandbCheckpointError("adaptive progress trial counts are inconsistent")
    if progress.stage not in _STAGES:
        raise WandbCheckpointError("adaptive progress stage is invalid")
    if progress.stage == "adaptive_search" and any(
        completed != required for completed, required in progress.coverage.values()
    ):
        raise WandbCheckpointError(
            "adaptive progress cannot concentrate before broad coverage is complete"
        )
    for name in (
        "elapsed_seconds",
        "eta_seconds",
        "gpu_actual_usd",
        "gpu_projected_usd",
        "judge_actual_usd",
        "judge_projected_usd",
        "projected_total_usd",
        "measured_trial_duration_seconds",
        "measured_tokens_per_second",
        "measured_judge_latency_ms",
        "measured_judge_cost_usd_per_trial",
    ):
        value = getattr(progress, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WandbCheckpointError(f"adaptive progress {name} is invalid")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise WandbCheckpointError(f"adaptive progress {name} is invalid")


def _validated_coverage(value: Any) -> dict[str, tuple[int, int]]:
    if not isinstance(value, Mapping) or set(value) != _COVERAGE_DIMENSIONS:
        raise WandbCheckpointError("adaptive progress coverage dimensions differ")
    result: dict[str, tuple[int, int]] = {}
    for name, counts in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(counts, (tuple, list))
            or len(counts) != 2
        ):
            raise WandbCheckpointError("adaptive progress coverage is invalid")
        completed, required = counts
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(required, bool)
            or not isinstance(required, int)
            or required <= 0
            or not 0 <= completed <= required
        ):
            raise WandbCheckpointError("adaptive progress coverage is invalid")
        result[name] = (completed, required)
    return result


def _validate_progress_transition(
    prior: AdaptiveRunProgress, current: AdaptiveRunProgress
) -> None:
    frozen = (
        "wandb_run_checkpoint_sha256",
        "study_config_sha256",
        "planned_floor_trials",
        "adaptive_ceiling_trials",
        "batch_size",
        "search_cutoff_seconds",
        "reserve_seconds",
        "total_budget_usd",
        "evaluation_budget_usd",
        "evaluation_budget_reserve_fraction",
    )
    if any(getattr(prior, name) != getattr(current, name) for name in frozen):
        raise WandbCheckpointError("adaptive progress policy differs from prior checkpoint")
    monotonic = (
        "completed_search_trials",
        "completed_repeat_trials",
        "completed_control_trials",
        "completed_final_selection_trials",
        "current_batch",
        "elapsed_seconds",
        "gpu_actual_usd",
        "judge_actual_usd",
    )
    if any(getattr(current, name) < getattr(prior, name) for name in monotonic):
        raise WandbCheckpointError("adaptive progress regressed")
    if _STAGES.index(current.stage) < _STAGES.index(prior.stage):
        raise WandbCheckpointError("adaptive progress stage regressed")
    for name in _COVERAGE_DIMENSIONS:
        prior_completed, prior_required = prior.coverage[name]
        current_completed, current_required = current.coverage[name]
        if current_required != prior_required:
            raise WandbCheckpointError("adaptive progress coverage policy differs")
        if current_completed < prior_completed:
            raise WandbCheckpointError("adaptive progress coverage regressed")


def _validated_unsigned(
    *, run_id: Any, project: Any, entity: Any
) -> dict[str, Any]:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise WandbCheckpointError("W&B run ID is invalid")
    if not isinstance(project, str) or _LOCATION.fullmatch(project) is None:
        raise WandbCheckpointError("W&B project is invalid")
    if entity is not None and (
        not isinstance(entity, str) or _LOCATION.fullmatch(entity) is None
    ):
        raise WandbCheckpointError("W&B entity is invalid")
    return {"format": FORMAT, "run_id": run_id, "project": project, "entity": entity}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WandbCheckpointError(
                f"W&B run checkpoint contains duplicate JSON field: {key}"
            )
        result[key] = value
    return result


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _encoded(value: object) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, indent=2).encode() + b"\n"


def _reject_symlinked_ancestors(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in reversed((candidate, *candidate.parents)):
        if ancestor.is_symlink():
            raise WandbCheckpointError("W&B checkpoint path contains a symlink")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ADAPTIVE_PROGRESS_FORMAT",
    "AdaptiveProgressCheckpoint",
    "AdaptiveRunProgress",
    "WandbCheckpointError",
    "WandbRunCheckpoint",
    "advance_adaptive_progress_checkpoint",
    "create_wandb_run_checkpoint",
    "open_adaptive_progress_checkpoint",
    "open_wandb_run_checkpoint",
    "parse_adaptive_progress_checkpoint",
    "parse_wandb_run_checkpoint",
]
