"""Immutable handoff of one truth-editing study between phase instances.

Only the controller-owned study journal, its Optuna journal, and the immutable
W&B coordinator-run identity cross this seam. Publication validates a complete
batch barrier, copies all three files into an
immutable hashed generation, then atomically advances ``latest.json``. Restore
revalidates every byte before atomically making the next phase's state visible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import fcntl
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from .truth_editing_capacity import (
    CapacityPlanningError,
    SpendSnapshot,
    validate_capacity_receipt,
)
from .truth_editing_study import OBJECTIVES, SearchProposal
from .truth_editing_wandb_checkpoint import (
    WandbCheckpointError,
    parse_adaptive_progress_checkpoint,
    parse_wandb_run_checkpoint,
)


MANIFEST_FORMAT = "truth_editing_phase_checkpoint_manifest_v2"
LATEST_FORMAT = "truth_editing_phase_checkpoint_latest_v2"
RESTORE_FORMAT = "truth_editing_phase_checkpoint_restore_v2"
JOURNAL_FORMAT = "truth_editing_study_journal_v1"
STATE_FILES = (
    "study/study-journal.json",
    "study/study-journal.json.optuna.log",
    "monitoring/wandb-run.json",
)
ADAPTIVE_MANIFEST_FORMAT = "truth_editing_adaptive_checkpoint_manifest_v1"
ADAPTIVE_LATEST_FORMAT = "truth_editing_adaptive_checkpoint_latest_v1"
ADAPTIVE_RESTORE_FORMAT = "truth_editing_adaptive_checkpoint_restore_v1"
ADAPTIVE_SCHEDULER_FORMAT = "truth_editing_adaptive_run_checkpoint_v1"
ADAPTIVE_STATE_FILES = (
    "study/study-journal.json",
    "study/study-journal.json.optuna.log",
    "study/adaptive-run-checkpoint.json",
    "monitoring/wandb-run.json",
    "monitoring/adaptive-progress.json",
    "monitoring/rolling-capacity-receipt.json",
)
PHASE_TRIALS = {"discovery": 80, "expanded": 160, "finalist": 200}
NEXT_PHASE = {"discovery": "expanded", "expanded": "finalist"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_MARKERS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-or-v1-[A-Za-z0-9_-]+\b", re.IGNORECASE),
    re.compile(rb"\b(?:sk-proj-|hf_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]+\b", re.IGNORECASE),
    re.compile(
        rb"\b(?:OPENROUTER_API_KEY|WANDB_API_KEY|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN|VAST_API_KEY|INSTANCE_API_KEY|API_KEY|AUTHORIZATION)\s*[\"']?\s*[:=]",
        re.IGNORECASE,
    ),
    re.compile(rb"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}", re.IGNORECASE),
)


class PhaseCheckpointError(RuntimeError):
    """A phase checkpoint cannot be trusted for publication or resume."""


def publish_phase_checkpoint(
    source_state_dir: Path | str,
    publication_root: Path | str,
    *,
    phase: str,
    expected_study_identity_sha256: str,
    expected_completed_trials: int,
    expected_optuna_study_name: str,
) -> dict[str, Any]:
    """Validate and atomically publish one immutable phase-barrier generation."""

    source = Path(source_state_dir)
    root = Path(publication_root)
    _reject_symlinked_ancestors(source)
    _reject_symlinked_ancestors(root)
    _validate_contract(phase, expected_study_identity_sha256, expected_completed_trials)
    files, payloads = _read_and_validate_state(
        source,
        expected_study_identity_sha256=expected_study_identity_sha256,
        expected_completed_trials=expected_completed_trials,
        expected_optuna_study_name=expected_optuna_study_name,
    )
    monitoring = _monitoring_identity(payloads["monitoring/wandb-run.json"])
    with _locked_publication_root(root):
        parent_sha = _validate_phase_lineage(
            root,
            phase=phase,
            study_identity=expected_study_identity_sha256,
            optuna_study_name=expected_optuna_study_name,
            completed=expected_completed_trials,
            new_payloads=payloads,
        )
        generation_id = (
            f"{phase}-{files[0]['sha256'][:16]}-{files[1]['sha256'][:16]}-"
            f"{files[2]['sha256'][:16]}"
        )
        unsigned = {
            "format": MANIFEST_FORMAT,
            "generation_id": generation_id,
            "parent_manifest_sha256": parent_sha,
            "phase": phase,
            "study_identity_sha256": expected_study_identity_sha256,
            "optuna_study_name": expected_optuna_study_name,
            "completed_trials": expected_completed_trials,
            "monitoring": monitoring,
            "files": files,
        }
        manifest = {**unsigned, "manifest_sha256": _json_sha(unsigned)}

        generations = root / "generations"
        final = generations / generation_id
        generations.mkdir(parents=True, exist_ok=True)
        if final.exists() or final.is_symlink():
            opened = _open_generation(final)
            if opened != manifest:
                raise PhaseCheckpointError(
                    "checkpoint generation already exists with different bytes"
                )
        else:
            staging = generations / f".{generation_id}.tmp-{uuid4().hex}"
            staging.mkdir()
            try:
                for name in STATE_FILES:
                    (staging / name).parent.mkdir(parents=True, exist_ok=True)
                    _write_new_bytes(staging / name, payloads[name])
                _write_new_json(staging / "manifest.json", manifest)
                _fsync_tree(staging)
                if _open_generation(staging, require_directory_name=False) != manifest:
                    raise PhaseCheckpointError("staged checkpoint verification failed")
                os.rename(staging, final)
                _fsync_directory(generations)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        _advance_latest(root, manifest)
    return dict(manifest)


def restore_phase_checkpoint(
    publication_root: Path | str,
    target_state_dir: Path | str,
    *,
    next_phase: str,
    expected_study_identity_sha256: str,
    expected_completed_trials: int,
    expected_optuna_study_name: str,
) -> dict[str, Any]:
    """Atomically restore the latest verified generation for the next phase."""

    root = Path(publication_root)
    target = Path(target_state_dir)
    _reject_symlinked_ancestors(root)
    _reject_symlinked_ancestors(target.parent)
    _digest(expected_study_identity_sha256, "expected study identity")
    _text(expected_optuna_study_name, "expected Optuna study name")
    pointer = _read_json(root / "latest.json", "latest pointer")
    _exact(
        pointer,
        {
            "format",
            "generation_id",
            "phase",
            "manifest_sha256",
            "pointer_sha256",
        },
        "latest pointer",
    )
    if pointer["format"] != LATEST_FORMAT:
        raise PhaseCheckpointError("latest pointer format differs")
    _verify_self_hash(pointer, "pointer_sha256", "latest pointer")
    generation_id = _generation_id(pointer["generation_id"])
    generation = root / "generations" / generation_id
    manifest = _open_generation(generation)
    if manifest["generation_id"] != generation_id:
        raise PhaseCheckpointError("latest pointer generation identity differs")
    if manifest["manifest_sha256"] != pointer["manifest_sha256"]:
        raise PhaseCheckpointError("latest pointer manifest hash differs")
    source_phase = manifest["phase"]
    if NEXT_PHASE.get(source_phase) != next_phase:
        raise PhaseCheckpointError("checkpoint cannot resume the requested next phase")
    if manifest["study_identity_sha256"] != expected_study_identity_sha256:
        raise PhaseCheckpointError("checkpoint study identity differs")
    if manifest["optuna_study_name"] != expected_optuna_study_name:
        raise PhaseCheckpointError("checkpoint Optuna study identity differs")
    if manifest["completed_trials"] != expected_completed_trials:
        raise PhaseCheckpointError("checkpoint completed trial count differs")
    _validate_contract(source_phase, expected_study_identity_sha256, expected_completed_trials)
    _read_and_validate_state(
        generation,
        expected_study_identity_sha256=expected_study_identity_sha256,
        expected_completed_trials=expected_completed_trials,
        expected_optuna_study_name=expected_optuna_study_name,
    )

    unsigned_receipt = {
        "format": RESTORE_FORMAT,
        "source_generation_id": generation_id,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_phase": source_phase,
        "next_phase": next_phase,
        "study_identity_sha256": expected_study_identity_sha256,
        "optuna_study_name": expected_optuna_study_name,
        "completed_trials": expected_completed_trials,
        "monitoring": manifest["monitoring"],
    }
    receipt = {**unsigned_receipt, "receipt_sha256": _json_sha(unsigned_receipt)}
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise PhaseCheckpointError("existing resume target is not a regular directory")
        if not _target_matches(target, manifest, receipt):
            raise PhaseCheckpointError("existing resume state differs from checkpoint")
        return receipt

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        for name in STATE_FILES:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            _copy_regular_file(generation / name, staging / name)
        _write_new_json(staging / "restore-receipt.json", receipt)
        _fsync_tree(staging)
        if not _target_matches(staging, manifest, receipt):
            raise PhaseCheckpointError("staged resume verification failed")
        os.rename(staging, target)
        _fsync_directory(target.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return receipt


def publish_adaptive_checkpoint(
    source_state_dir: Path | str,
    publication_root: Path | str,
    *,
    expected_study_identity_sha256: str,
    expected_study_config_sha256: str,
    expected_completed_trials: int,
    expected_optuna_study_name: str,
    tier_through_trials: tuple[int, ...] = (80, 160, 800),
    trial_number_start: int = 0,
) -> dict[str, Any]:
    """Publish one rolling adaptive generation without legacy phase semantics."""

    source = Path(source_state_dir)
    root = Path(publication_root)
    _reject_symlinked_ancestors(source)
    _reject_symlinked_ancestors(root)
    _validate_adaptive_expectations(
        expected_study_identity_sha256,
        expected_study_config_sha256,
        expected_completed_trials,
        expected_optuna_study_name,
    )
    files, payloads, monitoring, progress, scheduler = _read_adaptive_state(
        source,
        expected_study_identity_sha256=expected_study_identity_sha256,
        expected_study_config_sha256=expected_study_config_sha256,
        expected_completed_trials=expected_completed_trials,
        expected_optuna_study_name=expected_optuna_study_name,
        tier_through_trials=tier_through_trials,
        trial_number_start=trial_number_start,
    )
    with _locked_publication_root(root):
        parent_sha = _validate_adaptive_lineage(
            root,
            expected_study_identity_sha256=expected_study_identity_sha256,
            expected_study_config_sha256=expected_study_config_sha256,
            expected_optuna_study_name=expected_optuna_study_name,
            completed=expected_completed_trials,
            new_payloads=payloads,
            new_progress=progress,
            new_scheduler=scheduler,
            tier_through_trials=tier_through_trials,
            trial_number_start=trial_number_start,
        )
        generation_id = (
            f"adaptive-{expected_completed_trials:04d}-"
            f"{files[0]['sha256'][:12]}-{files[2]['sha256'][:12]}-"
            f"{files[4]['sha256'][:12]}"
        )
        unsigned = {
            "format": ADAPTIVE_MANIFEST_FORMAT,
            "generation_id": generation_id,
            "parent_manifest_sha256": parent_sha,
            "study_identity_sha256": expected_study_identity_sha256,
            "study_config_sha256": expected_study_config_sha256,
            "optuna_study_name": expected_optuna_study_name,
            "completed_trials": expected_completed_trials,
            "monitoring": monitoring,
            "adaptive_progress_checkpoint_sha256": progress.checkpoint_sha256,
            "adaptive_scheduler_checkpoint_sha256": scheduler["checkpoint_sha256"],
            "files": files,
        }
        manifest = {**unsigned, "manifest_sha256": _json_sha(unsigned)}
        generations = root / "adaptive-generations"
        final = generations / generation_id
        generations.mkdir(parents=True, exist_ok=True)
        if final.exists() or final.is_symlink():
            if _open_adaptive_generation(final) != manifest:
                raise PhaseCheckpointError(
                    "adaptive checkpoint generation already exists with different bytes"
                )
        else:
            staging = generations / f".{generation_id}.tmp-{uuid4().hex}"
            staging.mkdir()
            try:
                for name in ADAPTIVE_STATE_FILES:
                    (staging / name).parent.mkdir(parents=True, exist_ok=True)
                    _write_new_bytes(staging / name, payloads[name])
                _write_new_json(staging / "manifest.json", manifest)
                _fsync_tree(staging)
                if _open_adaptive_generation(
                    staging, require_directory_name=False
                ) != manifest:
                    raise PhaseCheckpointError(
                        "staged adaptive checkpoint verification failed"
                    )
                os.rename(staging, final)
                _fsync_directory(generations)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        _advance_adaptive_latest(root, manifest)
    return dict(manifest)


def restore_adaptive_checkpoint(
    publication_root: Path | str,
    target_state_dir: Path | str,
    *,
    expected_study_identity_sha256: str,
    expected_study_config_sha256: str,
    expected_completed_trials: int,
    expected_optuna_study_name: str,
    trial_number_start: int = 0,
) -> dict[str, Any]:
    """Restore the latest complete adaptive generation and all control state."""

    root = Path(publication_root)
    target = Path(target_state_dir)
    _reject_symlinked_ancestors(root)
    _reject_symlinked_ancestors(target.parent)
    _validate_adaptive_expectations(
        expected_study_identity_sha256,
        expected_study_config_sha256,
        expected_completed_trials,
        expected_optuna_study_name,
    )
    pointer = _read_json(root / "adaptive-latest.json", "adaptive latest pointer")
    _validate_adaptive_latest(pointer)
    generation_id = _adaptive_generation_id(pointer["generation_id"])
    generation = root / "adaptive-generations" / generation_id
    manifest = _open_adaptive_generation(generation)
    if (
        manifest["manifest_sha256"] != pointer["manifest_sha256"]
        or manifest["generation_id"] != generation_id
    ):
        raise PhaseCheckpointError("adaptive latest pointer identity differs")
    if (
        manifest["study_identity_sha256"] != expected_study_identity_sha256
        or manifest["study_config_sha256"] != expected_study_config_sha256
        or manifest["optuna_study_name"] != expected_optuna_study_name
        or manifest["completed_trials"] != expected_completed_trials
    ):
        raise PhaseCheckpointError("adaptive checkpoint expected identity differs")
    _read_adaptive_state(
        generation,
        expected_study_identity_sha256=expected_study_identity_sha256,
        expected_study_config_sha256=expected_study_config_sha256,
        expected_completed_trials=expected_completed_trials,
        expected_optuna_study_name=expected_optuna_study_name,
        trial_number_start=trial_number_start,
    )
    unsigned = {
        "format": ADAPTIVE_RESTORE_FORMAT,
        "source_generation_id": generation_id,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "study_identity_sha256": expected_study_identity_sha256,
        "study_config_sha256": expected_study_config_sha256,
        "optuna_study_name": expected_optuna_study_name,
        "completed_trials": expected_completed_trials,
        "monitoring": manifest["monitoring"],
        "adaptive_progress_checkpoint_sha256": manifest[
            "adaptive_progress_checkpoint_sha256"
        ],
        "adaptive_scheduler_checkpoint_sha256": manifest[
            "adaptive_scheduler_checkpoint_sha256"
        ],
    }
    receipt = {**unsigned, "receipt_sha256": _json_sha(unsigned)}
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise PhaseCheckpointError("existing adaptive resume target is unsafe")
        if not _adaptive_target_matches(target, manifest, receipt):
            raise PhaseCheckpointError("existing adaptive resume state differs")
        return receipt
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        for name in ADAPTIVE_STATE_FILES:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            _copy_regular_file(generation / name, staging / name)
        _write_new_json(staging / "restore-receipt.json", receipt)
        _fsync_tree(staging)
        if not _adaptive_target_matches(staging, manifest, receipt):
            raise PhaseCheckpointError("staged adaptive resume verification failed")
        os.rename(staging, target)
        _fsync_directory(target.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return receipt


def _validate_adaptive_expectations(
    study_identity: str,
    study_config: str,
    completed: int,
    optuna_study_name: str,
) -> None:
    _digest(study_identity, "expected study identity")
    _digest(study_config, "expected study config identity")
    _text(optuna_study_name, "expected Optuna study name")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= 800
        or completed % 8
    ):
        raise PhaseCheckpointError(
            "adaptive completed trials must be a complete batch between zero and 800"
        )


def _read_adaptive_state(
    root: Path,
    *,
    expected_study_identity_sha256: str,
    expected_study_config_sha256: str,
    expected_completed_trials: int,
    expected_optuna_study_name: str,
    tier_through_trials: tuple[int, ...] = (80, 160, 800),
    trial_number_start: int = 0,
) -> tuple[
    list[dict[str, Any]],
    dict[str, bytes],
    dict[str, str],
    Any,
    dict[str, Any],
]:
    if root.is_symlink() or not root.is_dir():
        raise PhaseCheckpointError("adaptive state directory must be regular")
    files: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for name in ADAPTIVE_STATE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise PhaseCheckpointError(
                f"adaptive resume state file is missing or unsafe: {name}"
            )
        payload = _read_regular_bytes(path)
        if not payload:
            raise PhaseCheckpointError(f"adaptive resume state file is empty: {name}")
        _reject_secrets(payload)
        payloads[name] = payload
        files.append(
            {"name": name, "size_bytes": len(payload), "sha256": _bytes_sha(payload)}
        )
    trials = _validate_study_journal(
        payloads[ADAPTIVE_STATE_FILES[0]],
        expected_study_identity_sha256,
        expected_completed_trials,
        tier_through_trials=tier_through_trials,
        trial_number_start=trial_number_start,
    )
    _validate_optuna_journal(
        payloads[ADAPTIVE_STATE_FILES[1]],
        trials,
        expected_study_name=expected_optuna_study_name,
    )
    monitoring = _monitoring_identity(payloads[ADAPTIVE_STATE_FILES[3]])
    scheduler = _validate_adaptive_scheduler(
        payloads[ADAPTIVE_STATE_FILES[2]],
        expected_study_identity_sha256=expected_study_identity_sha256,
        expected_completed_trials=expected_completed_trials,
        monitoring=monitoring,
    )
    _validate_rolling_capacity_receipt(
        payloads[ADAPTIVE_STATE_FILES[5]], scheduler=scheduler
    )
    try:
        progress = parse_adaptive_progress_checkpoint(
            payloads[ADAPTIVE_STATE_FILES[4]]
        )
    except WandbCheckpointError as error:
        raise PhaseCheckpointError("adaptive progress checkpoint is invalid") from error
    if progress.progress.study_config_sha256 != expected_study_config_sha256:
        raise PhaseCheckpointError("adaptive progress study config identity differs")
    if progress.progress.wandb_run_checkpoint_sha256 != monitoring["checkpoint_sha256"]:
        raise PhaseCheckpointError("adaptive progress W&B identity differs")
    if progress.progress.completed_search_trials != expected_completed_trials:
        raise PhaseCheckpointError(
            "adaptive progress differs from authoritative journal boundary"
        )
    return files, payloads, monitoring, progress, scheduler


def _validate_rolling_capacity_receipt(
    payload: bytes, *, scheduler: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_json_object)
        receipt = validate_capacity_receipt(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, CapacityPlanningError) as error:
        raise PhaseCheckpointError("rolling-capacity-receipt is invalid") from error
    bindings = {
        "policy_sha256": scheduler["policy_sha256"],
        "receipt_sha256": scheduler["current_capacity_receipt_sha256"],
        "source_batch_observation_sha256": scheduler[
            "current_capacity_observation_sha256"
        ],
        "completed_through_trial": scheduler[
            "current_capacity_completed_through_trial"
        ],
        "source_judge_ledger_after_receipt_sha256": scheduler[
            "current_judge_ledger_receipt_sha256"
        ],
    }
    if any(receipt[name] != expected for name, expected in bindings.items()):
        raise PhaseCheckpointError(
            "rolling-capacity-receipt scheduler binding differs"
        )
    guarantee_met = receipt["decision"]["minimum_trial_guarantee_met"]
    minimum_abort = scheduler["stop_reason"] == "minimum_trial_guarantee_lost"
    if guarantee_met is False and not minimum_abort:
        raise PhaseCheckpointError(
            "minimum-trial abort differs from rolling capacity guarantee"
        )
    return receipt


def _validate_adaptive_scheduler(
    payload: bytes,
    *,
    expected_study_identity_sha256: str,
    expected_completed_trials: int,
    monitoring: Mapping[str, str],
) -> dict[str, Any]:
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCheckpointError("adaptive scheduler checkpoint is unreadable") from error
    expected = {
        "format",
        "policy_sha256",
        "capacity_receipt_sha256",
        "study_identity_sha256",
        "wandb_run_checkpoint_sha256",
        "wandb_run_id",
        "started_at_utc",
        "search_deadline_utc",
        "hard_deadline_utc",
        "planned_trial_count",
        "current_advisory_trial_count",
        "current_capacity_receipt_sha256",
        "current_capacity_observation_sha256",
        "current_capacity_completed_through_trial",
        "current_judge_ledger_receipt_sha256",
        "current_next_batch_projection",
        "projected_search_eta_seconds",
        "accounted_infrastructure_usd",
        "accounted_evaluation_usd",
        "authorized_through_trial",
        "completed_trials",
        "coverage_complete",
        "phase",
        "stop_reason",
        "last_spend_snapshot",
        "checkpoint_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise PhaseCheckpointError("adaptive scheduler checkpoint fields differ")
    if raw["format"] != ADAPTIVE_SCHEDULER_FORMAT:
        raise PhaseCheckpointError("adaptive scheduler checkpoint format differs")
    claimed = _digest(raw["checkpoint_sha256"], "adaptive scheduler self hash")
    unsigned = dict(raw)
    unsigned.pop("checkpoint_sha256")
    if claimed != _json_sha(unsigned):
        raise PhaseCheckpointError("adaptive scheduler checkpoint hash differs")
    for name in ("policy_sha256", "capacity_receipt_sha256"):
        _digest(raw[name], f"adaptive scheduler {name}")
    if raw["study_identity_sha256"] != expected_study_identity_sha256:
        raise PhaseCheckpointError("adaptive scheduler study identity differs")
    if (
        raw["wandb_run_checkpoint_sha256"] != monitoring["checkpoint_sha256"]
        or raw["wandb_run_id"] != monitoring["wandb_run_id"]
    ):
        raise PhaseCheckpointError("adaptive scheduler W&B identity differs")
    for name in (
        "planned_trial_count",
        "current_advisory_trial_count",
        "authorized_through_trial",
        "completed_trials",
    ):
        value = raw[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 800
            or value % 8
        ):
            raise PhaseCheckpointError("adaptive scheduler trial boundary is invalid")
    if not 200 <= raw["planned_trial_count"] <= 800:
        raise PhaseCheckpointError("adaptive scheduler initial plan is invalid")
    if not 200 <= raw["current_advisory_trial_count"] <= 800:
        raise PhaseCheckpointError("adaptive scheduler advisory target is invalid")
    _digest(
        raw["current_capacity_receipt_sha256"],
        "adaptive scheduler current capacity receipt",
    )
    observation_sha = raw["current_capacity_observation_sha256"]
    if observation_sha is not None:
        _digest(
            observation_sha,
            "adaptive scheduler current capacity observation",
        )
    judge_ledger_sha = raw["current_judge_ledger_receipt_sha256"]
    if judge_ledger_sha is not None:
        _digest(
            judge_ledger_sha,
            "adaptive scheduler current judge ledger receipt",
        )
    capacity_completed = raw["current_capacity_completed_through_trial"]
    if (
        isinstance(capacity_completed, bool)
        or not isinstance(capacity_completed, int)
        or capacity_completed < 0
        or capacity_completed != raw["completed_trials"]
        or capacity_completed % 8
    ):
        raise PhaseCheckpointError(
            "adaptive scheduler capacity boundary is invalid or lineage regressed"
        )
    projection = raw["current_next_batch_projection"]
    if not isinstance(projection, Mapping) or set(projection) != {
        "tier",
        "batch_duration_seconds_upper_bound",
        "batch_infrastructure_cost_usd_upper_bound",
        "batch_evaluation_cost_usd_upper_bound",
        "batch_total_cost_usd_upper_bound",
    }:
        raise PhaseCheckpointError("adaptive scheduler next-batch projection is invalid")
    if projection["tier"] not in {"discovery", "expanded", "concentrated"}:
        raise PhaseCheckpointError("adaptive scheduler projection tier is invalid")
    duration = projection["batch_duration_seconds_upper_bound"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise PhaseCheckpointError("adaptive scheduler projection duration is invalid")
    projected_infrastructure = _canonical_nonnegative_money(
        projection["batch_infrastructure_cost_usd_upper_bound"],
        "adaptive scheduler projected infrastructure cost",
    )
    projected_evaluation = _canonical_nonnegative_money(
        projection["batch_evaluation_cost_usd_upper_bound"],
        "adaptive scheduler projected evaluation cost",
    )
    projected_total = _canonical_nonnegative_money(
        projection["batch_total_cost_usd_upper_bound"],
        "adaptive scheduler projected total cost",
    )
    if projected_total != projected_infrastructure + projected_evaluation:
        raise PhaseCheckpointError("adaptive scheduler projection total differs")
    eta = raw["projected_search_eta_seconds"]
    if (
        isinstance(eta, bool)
        or not isinstance(eta, (int, float))
        or not math.isfinite(float(eta))
        or eta < 0
    ):
        raise PhaseCheckpointError("adaptive scheduler ETA is invalid")
    for name in ("accounted_infrastructure_usd", "accounted_evaluation_usd"):
        _canonical_nonnegative_money(
            raw[name], "adaptive scheduler accounted spend"
        )
    if (
        raw["completed_trials"] != expected_completed_trials
        or raw["authorized_through_trial"] < raw["completed_trials"]
        or raw["authorized_through_trial"] - raw["completed_trials"] > 8
    ):
        raise PhaseCheckpointError("adaptive scheduler authorization differs from journal")
    try:
        timestamps = tuple(
            datetime.fromisoformat(str(raw[name]).removesuffix("Z") + "+00:00")
            for name in ("started_at_utc", "search_deadline_utc", "hard_deadline_utc")
        )
    except ValueError as error:
        raise PhaseCheckpointError("adaptive scheduler deadlines are invalid") from error
    if (
        any(
            not isinstance(raw[name], str) or not raw[name].endswith("Z")
            for name in ("started_at_utc", "search_deadline_utc", "hard_deadline_utc")
        )
        or timestamps[1] != timestamps[0] + timedelta(hours=21)
        or timestamps[2] != timestamps[1] + timedelta(hours=3)
    ):
        raise PhaseCheckpointError("adaptive scheduler deadlines are invalid")
    if raw["phase"] not in {
        "broad_coverage",
        "adaptive_search",
        "finalization_reserved",
        "complete",
        "aborted",
    }:
        raise PhaseCheckpointError("adaptive scheduler stage is invalid")
    if raw["stop_reason"] not in {
        None,
        "search_deadline_reached",
        "total_budget_reserve_reached",
        "infrastructure_budget_reserve_reached",
        "evaluation_budget_reserve_reached",
        "maximum_trials_reached",
        "minimum_trial_guarantee_lost",
    }:
        raise PhaseCheckpointError("adaptive scheduler stop reason is invalid")
    if (raw["phase"] in {"finalization_reserved", "complete", "aborted"}) != (
        raw["stop_reason"] is not None
    ):
        raise PhaseCheckpointError("adaptive scheduler terminal state is inconsistent")
    if (raw["phase"] == "aborted") != (
        raw["stop_reason"] == "minimum_trial_guarantee_lost"
    ):
        raise PhaseCheckpointError("adaptive scheduler abort reason is inconsistent")
    if raw["phase"] == "aborted" and raw["completed_trials"] >= 200:
        raise PhaseCheckpointError("adaptive scheduler cannot abort after the minimum")
    if not isinstance(raw["coverage_complete"], bool):
        raise PhaseCheckpointError("adaptive scheduler coverage state is invalid")
    if raw["phase"] == "adaptive_search" and not raw["coverage_complete"]:
        raise PhaseCheckpointError("adaptive scheduler began search before coverage")
    try:
        SpendSnapshot.from_mapping(raw["last_spend_snapshot"])
    except CapacityPlanningError as error:
        raise PhaseCheckpointError("adaptive scheduler spend snapshot is invalid") from error
    return raw


def _canonical_nonnegative_money(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise PhaseCheckpointError(f"{label} is invalid")
    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise PhaseCheckpointError(f"{label} is invalid") from error
    rendered = format(amount, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if not amount.is_finite() or amount < 0 or (rendered or "0") != value:
        raise PhaseCheckpointError(f"{label} is invalid")
    return amount


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseCheckpointError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _open_adaptive_generation(
    path: Path, *, require_directory_name: bool = True
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise PhaseCheckpointError("adaptive checkpoint generation is unsafe")
    manifest = _read_json(path / "manifest.json", "adaptive checkpoint manifest")
    expected = {
        "format",
        "generation_id",
        "parent_manifest_sha256",
        "study_identity_sha256",
        "study_config_sha256",
        "optuna_study_name",
        "completed_trials",
        "monitoring",
        "adaptive_progress_checkpoint_sha256",
        "adaptive_scheduler_checkpoint_sha256",
        "files",
        "manifest_sha256",
    }
    _exact(manifest, expected, "adaptive checkpoint manifest")
    if manifest["format"] != ADAPTIVE_MANIFEST_FORMAT:
        raise PhaseCheckpointError("adaptive checkpoint manifest format differs")
    _verify_self_hash(manifest, "manifest_sha256", "adaptive checkpoint manifest")
    generation_id = _adaptive_generation_id(manifest["generation_id"])
    if require_directory_name and path.name != generation_id:
        raise PhaseCheckpointError("adaptive generation directory identity differs")
    _validate_adaptive_expectations(
        manifest["study_identity_sha256"],
        manifest["study_config_sha256"],
        manifest["completed_trials"],
        manifest["optuna_study_name"],
    )
    parent = manifest["parent_manifest_sha256"]
    if parent is not None:
        _digest(parent, "adaptive parent manifest identity")
    files = manifest["files"]
    if (
        not isinstance(files, list)
        or [row.get("name") for row in files if isinstance(row, Mapping)]
        != list(ADAPTIVE_STATE_FILES)
    ):
        raise PhaseCheckpointError("adaptive checkpoint file inventory differs")
    actual = {
        child.relative_to(path).as_posix()
        for child in path.rglob("*")
        if child.is_file() or child.is_symlink()
    }
    if actual != {*ADAPTIVE_STATE_FILES, "manifest.json"}:
        raise PhaseCheckpointError("adaptive checkpoint contains undeclared files")
    payloads: dict[str, bytes] = {}
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"name", "size_bytes", "sha256"}:
            raise PhaseCheckpointError("adaptive checkpoint inventory is malformed")
        name = str(row["name"])
        artifact = path / name
        if artifact.is_symlink() or not artifact.is_file():
            raise PhaseCheckpointError("adaptive checkpoint artifact is unsafe")
        payload = _read_regular_bytes(artifact)
        if len(payload) != row["size_bytes"] or _bytes_sha(payload) != row["sha256"]:
            raise PhaseCheckpointError("adaptive checkpoint artifact identity differs")
        _reject_secrets(payload)
        payloads[name] = payload
    monitoring = _monitoring_identity(payloads[ADAPTIVE_STATE_FILES[3]])
    try:
        progress = parse_adaptive_progress_checkpoint(payloads[ADAPTIVE_STATE_FILES[4]])
    except WandbCheckpointError as error:
        raise PhaseCheckpointError("adaptive progress checkpoint is invalid") from error
    scheduler = _validate_adaptive_scheduler(
        payloads[ADAPTIVE_STATE_FILES[2]],
        expected_study_identity_sha256=manifest["study_identity_sha256"],
        expected_completed_trials=manifest["completed_trials"],
        monitoring=monitoring,
    )
    _validate_rolling_capacity_receipt(
        payloads[ADAPTIVE_STATE_FILES[5]], scheduler=scheduler
    )
    if (
        monitoring != manifest["monitoring"]
        or progress.checkpoint_sha256
        != manifest["adaptive_progress_checkpoint_sha256"]
        or scheduler["checkpoint_sha256"]
        != manifest["adaptive_scheduler_checkpoint_sha256"]
        or progress.progress.study_config_sha256 != manifest["study_config_sha256"]
        or progress.progress.completed_search_trials != manifest["completed_trials"]
    ):
        raise PhaseCheckpointError("adaptive checkpoint manifest binding differs")
    return dict(manifest)


def _validate_adaptive_lineage(
    root: Path,
    *,
    expected_study_identity_sha256: str,
    expected_study_config_sha256: str,
    expected_optuna_study_name: str,
    completed: int,
    new_payloads: Mapping[str, bytes],
    new_progress: Any,
    new_scheduler: Mapping[str, Any],
    tier_through_trials: tuple[int, ...] = (80, 160, 800),
    trial_number_start: int = 0,
) -> str | None:
    latest_path = root / "adaptive-latest.json"
    if not latest_path.exists() and not latest_path.is_symlink():
        return None
    pointer = _read_json(latest_path, "adaptive latest pointer")
    _validate_adaptive_latest(pointer)
    prior = _open_adaptive_generation(
        root
        / "adaptive-generations"
        / _adaptive_generation_id(pointer["generation_id"])
    )
    if prior["manifest_sha256"] != pointer["manifest_sha256"]:
        raise PhaseCheckpointError("adaptive latest pointer manifest differs")
    if (
        prior["study_identity_sha256"] != expected_study_identity_sha256
        or prior["study_config_sha256"] != expected_study_config_sha256
        or prior["optuna_study_name"] != expected_optuna_study_name
    ):
        raise PhaseCheckpointError("adaptive checkpoint lineage identity differs")
    prior_completed = int(prior["completed_trials"])
    if completed < prior_completed:
        raise PhaseCheckpointError("adaptive checkpoint completed trials regressed")
    prior_root = root / "adaptive-generations" / str(prior["generation_id"])
    _files, prior_payloads, _monitoring, prior_progress, prior_scheduler = (
        _read_adaptive_state(
            prior_root,
            expected_study_identity_sha256=expected_study_identity_sha256,
            expected_study_config_sha256=expected_study_config_sha256,
            expected_completed_trials=prior_completed,
            expected_optuna_study_name=expected_optuna_study_name,
            tier_through_trials=tier_through_trials,
            trial_number_start=trial_number_start,
        )
    )
    prior_trials = _journal_trial_rows(prior_payloads[ADAPTIVE_STATE_FILES[0]])
    new_trials = _journal_trial_rows(new_payloads[ADAPTIVE_STATE_FILES[0]])
    if new_trials[:prior_completed] != prior_trials or len(new_trials) != completed:
        raise PhaseCheckpointError("adaptive journal is not an exact prior extension")
    if completed == prior_completed and all(
        new_payloads[name] == prior_payloads[name] for name in ADAPTIVE_STATE_FILES
    ):
        parent = prior["parent_manifest_sha256"]
        if parent is not None and (
            not isinstance(parent, str) or _SHA256.fullmatch(parent) is None
        ):
            raise PhaseCheckpointError("adaptive checkpoint parent identity is invalid")
        return parent
    prior_optuna = prior_payloads[ADAPTIVE_STATE_FILES[1]]
    new_optuna = new_payloads[ADAPTIVE_STATE_FILES[1]]
    if completed == prior_completed:
        if new_optuna != prior_optuna:
            raise PhaseCheckpointError("adaptive Optuna journal changed without trials")
    elif new_optuna == prior_optuna and _adaptive_optuna_omission_is_audit_only(
        new_trials[prior_completed:]
    ):
        # Operational failures and matched controls are authoritative in the
        # controller study journal but intentionally omitted from the TPE
        # journal.  Their checkpoint advance is valid even though Optuna's
        # bytes do not grow; any scientific/TPE-observed addition still must
        # satisfy the strict append-only rule below.
        pass
    elif len(new_optuna) <= len(prior_optuna) or not new_optuna.startswith(prior_optuna):
        raise PhaseCheckpointError("adaptive Optuna journal is not append-only")
    if new_payloads[ADAPTIVE_STATE_FILES[3]] != prior_payloads[ADAPTIVE_STATE_FILES[3]]:
        raise PhaseCheckpointError("adaptive checkpoint changed the W&B run identity")
    if (
        new_progress.checkpoint_sha256 != prior_progress.checkpoint_sha256
        and new_progress.previous_checkpoint_sha256 != prior_progress.checkpoint_sha256
    ):
        raise PhaseCheckpointError("adaptive progress hash-chain continuity differs")
    prior_capacity = _validate_rolling_capacity_receipt(
        prior_payloads[ADAPTIVE_STATE_FILES[5]], scheduler=prior_scheduler
    )
    new_capacity = _validate_rolling_capacity_receipt(
        new_payloads[ADAPTIVE_STATE_FILES[5]], scheduler=new_scheduler
    )
    prior_capacity_completed = prior_capacity["completed_through_trial"]
    new_capacity_completed = new_capacity["completed_through_trial"]
    if (
        new_capacity_completed < prior_capacity_completed
        or new_capacity["timed_canary_receipt_sha256"]
        != prior_capacity["timed_canary_receipt_sha256"]
    ):
        raise PhaseCheckpointError("adaptive rolling capacity lineage regressed")
    if (
        new_capacity_completed == prior_capacity_completed
        and new_capacity["receipt_sha256"] != prior_capacity["receipt_sha256"]
    ):
        raise PhaseCheckpointError(
            "adaptive rolling capacity changed without a committed batch"
        )
    if (
        new_capacity_completed > prior_capacity_completed
        and new_capacity["source_batch_observation_sha256"]
        == prior_capacity["source_batch_observation_sha256"]
    ):
        raise PhaseCheckpointError(
            "adaptive rolling capacity reused a prior batch observation"
        )
    immutable_scheduler = {
        "policy_sha256",
        "capacity_receipt_sha256",
        "study_identity_sha256",
        "wandb_run_checkpoint_sha256",
        "wandb_run_id",
        "started_at_utc",
        "search_deadline_utc",
        "hard_deadline_utc",
        "planned_trial_count",
    }
    if any(prior_scheduler[name] != new_scheduler[name] for name in immutable_scheduler):
        raise PhaseCheckpointError("adaptive scheduler lineage identity differs")
    if (
        new_scheduler["authorized_through_trial"]
        < prior_scheduler["authorized_through_trial"]
        or new_scheduler["completed_trials"] < prior_scheduler["completed_trials"]
        or (
            prior_scheduler["coverage_complete"]
            and not new_scheduler["coverage_complete"]
        )
    ):
        raise PhaseCheckpointError("adaptive scheduler lineage regressed")
    prior_spend = SpendSnapshot.from_mapping(prior_scheduler["last_spend_snapshot"])
    new_spend = SpendSnapshot.from_mapping(new_scheduler["last_spend_snapshot"])
    spend_pairs = (
        (prior_spend.actual_total_usd, new_spend.actual_total_usd),
        (
            prior_spend.actual_infrastructure_usd,
            new_spend.actual_infrastructure_usd,
        ),
        (prior_spend.actual_evaluation_usd, new_spend.actual_evaluation_usd),
        (
            prior_spend.reserved_infrastructure_usd,
            new_spend.reserved_infrastructure_usd,
        ),
        (
            prior_spend.reserved_evaluation_usd,
            new_spend.reserved_evaluation_usd,
        ),
        (prior_spend.reserved_total_usd, new_spend.reserved_total_usd),
        (
            _canonical_nonnegative_money(
                prior_scheduler["accounted_infrastructure_usd"],
                "prior adaptive scheduler accounted infrastructure spend",
            ),
            _canonical_nonnegative_money(
                new_scheduler["accounted_infrastructure_usd"],
                "adaptive scheduler accounted infrastructure spend",
            ),
        ),
        (
            _canonical_nonnegative_money(
                prior_scheduler["accounted_evaluation_usd"],
                "prior adaptive scheduler accounted evaluation spend",
            ),
            _canonical_nonnegative_money(
                new_scheduler["accounted_evaluation_usd"],
                "adaptive scheduler accounted evaluation spend",
            ),
        ),
    )
    if any(current < previous for previous, current in spend_pairs):
        raise PhaseCheckpointError("adaptive scheduler spend lineage regressed")
    return str(prior["manifest_sha256"])


def _adaptive_optuna_omission_is_audit_only(
    trials: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether newly completed rows may be absent from Optuna state.

    The controller study journal records every completed attempt.  Optuna is a
    learning journal, so it deliberately omits operational failures and
    matched-basis controls.  This helper keeps that policy explicit at the
    publication seam: an unchanged Optuna file is acceptable only when *every*
    newly appended controller row is one of those non-learning records.
    """

    if not trials:
        return False
    for trial in trials:
        proposal = trial.get("proposal")
        result = trial.get("result")
        if not isinstance(proposal, Mapping) or not isinstance(result, Mapping):
            return False
        if proposal.get("matched_basis_control") == "orthogonal":
            continue
        if result.get("outcome_kind") != "operational_failure":
            return False
    return True


def _advance_adaptive_latest(root: Path, manifest: Mapping[str, Any]) -> None:
    unsigned = {
        "format": ADAPTIVE_LATEST_FORMAT,
        "generation_id": manifest["generation_id"],
        "completed_trials": manifest["completed_trials"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    pointer = {**unsigned, "pointer_sha256": _json_sha(unsigned)}
    _atomic_json(root / "adaptive-latest.json", pointer)


def _adaptive_generation_id(value: Any) -> str:
    result = _text(value, "adaptive generation ID")
    if (
        Path(result).is_absolute()
        or "/" in result
        or "\\" in result
        or re.fullmatch(
            r"adaptive-[0-9]{4}-[0-9a-f]{12}-[0-9a-f]{12}-[0-9a-f]{12}",
            result,
        )
        is None
    ):
        raise PhaseCheckpointError("adaptive checkpoint generation ID is unsafe")
    return result


def _validate_adaptive_latest(pointer: Mapping[str, Any]) -> None:
    _exact(
        pointer,
        {
            "format",
            "generation_id",
            "completed_trials",
            "manifest_sha256",
            "pointer_sha256",
        },
        "adaptive latest pointer",
    )
    if pointer["format"] != ADAPTIVE_LATEST_FORMAT:
        raise PhaseCheckpointError("adaptive latest pointer format differs")
    _verify_self_hash(pointer, "pointer_sha256", "adaptive latest pointer")


def _adaptive_target_matches(
    target: Path, manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    actual = {
        child.relative_to(target).as_posix()
        for child in target.rglob("*")
        if child.is_file() or child.is_symlink()
    }
    if actual != {*ADAPTIVE_STATE_FILES, "restore-receipt.json"}:
        return False
    expected = {row["name"]: row for row in manifest["files"]}
    for name in ADAPTIVE_STATE_FILES:
        path = target / name
        if path.is_symlink() or not path.is_file():
            return False
        payload = path.read_bytes()
        if (
            len(payload) != expected[name]["size_bytes"]
            or _bytes_sha(payload) != expected[name]["sha256"]
        ):
            return False
    try:
        return _read_json(target / "restore-receipt.json", "restore receipt") == receipt
    except PhaseCheckpointError:
        return False


def _validate_contract(phase: str, study_identity: str, completed: int) -> None:
    if phase not in PHASE_TRIALS:
        raise PhaseCheckpointError("phase must be discovery, expanded, or finalist")
    _digest(study_identity, "expected study identity")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise PhaseCheckpointError("expected completed trials must be an integer")
    if PHASE_TRIALS[phase] != completed:
        raise PhaseCheckpointError("completed trial count does not match the phase barrier")


def _read_and_validate_state(
    root: Path,
    *,
    expected_study_identity_sha256: str,
    expected_completed_trials: int,
    expected_optuna_study_name: str,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    if root.is_symlink() or not root.is_dir():
        raise PhaseCheckpointError("state directory must be a regular directory")
    file_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for name in STATE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise PhaseCheckpointError(f"resume state file is missing or unsafe: {name}")
        payload = _read_regular_bytes(path)
        if not payload:
            raise PhaseCheckpointError(f"resume state file is empty: {name}")
        _reject_secrets(payload)
        payloads[name] = payload
        file_rows.append(
            {"name": name, "size_bytes": len(payload), "sha256": _bytes_sha(payload)}
        )
    trials = _validate_study_journal(
        payloads["study/study-journal.json"],
        expected_study_identity_sha256,
        expected_completed_trials,
    )
    _validate_optuna_journal(
        payloads["study/study-journal.json.optuna.log"],
        trials,
        expected_study_name=expected_optuna_study_name,
    )
    _monitoring_identity(payloads["monitoring/wandb-run.json"])
    return file_rows, payloads


def _monitoring_identity(payload: bytes) -> dict[str, str]:
    try:
        checkpoint = parse_wandb_run_checkpoint(payload)
    except WandbCheckpointError as error:
        raise PhaseCheckpointError("W&B run checkpoint is invalid") from error
    return {
        "wandb_run_id": checkpoint.run_id,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }


def _validate_study_journal(
    payload: bytes,
    study_identity: str,
    completed: int,
    *,
    tier_through_trials: tuple[int, ...] = (80, 160, 800),
    trial_number_start: int = 0,
) -> list[Mapping[str, Any]]:
    if trial_number_start not in {0, 1}:
        raise PhaseCheckpointError("study journal trial number start is invalid")
    if (
        len(tier_through_trials) != 3
        or any(
            isinstance(boundary, bool)
            or not isinstance(boundary, int)
            or not 0 < boundary <= 800
            for boundary in tier_through_trials
        )
        or not (
            tier_through_trials[0]
            < tier_through_trials[1]
            < tier_through_trials[2]
        )
    ):
        raise PhaseCheckpointError("study journal tier boundaries are invalid")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCheckpointError("study journal is unreadable") from error
    if not isinstance(raw, Mapping):
        raise PhaseCheckpointError("study journal must be an object")
    _exact(
        raw,
        {"format", "study_identity_sha256", "identity_inputs", "batches", "journal_sha256"},
        "study journal",
    )
    if raw["format"] != JOURNAL_FORMAT:
        raise PhaseCheckpointError("study journal format differs")
    if raw["study_identity_sha256"] != study_identity:
        raise PhaseCheckpointError("study journal identity differs")
    _verify_self_hash(raw, "journal_sha256", "study journal")
    batches = raw["batches"]
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise PhaseCheckpointError("study journal batches must be an array")
    if len(batches) != completed // 8:
        raise PhaseCheckpointError("study journal must contain exact eight-trial batches")
    ordinals: list[int] = []
    trial_ids: list[str] = []
    validated_trials: list[Mapping[str, Any]] = []
    for batch_ordinal, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise PhaseCheckpointError("study journal batch must be an object")
        _exact(batch, {"ordinal", "trials"}, "study journal batch")
        if batch["ordinal"] != batch_ordinal:
            raise PhaseCheckpointError("study journal batch ordering differs")
        trials = batch["trials"]
        if isinstance(trials, (str, bytes)) or not isinstance(trials, Sequence):
            raise PhaseCheckpointError("study journal trials must be an array")
        if len(trials) != 8:
            raise PhaseCheckpointError("study journal must contain exact eight-trial batches")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise PhaseCheckpointError("study journal trial must be an object")
            _exact(
                trial,
                {
                    "trial_id",
                    "ordinal",
                    "tier_name",
                    "evaluation_record_ids",
                    "proposal",
                    "result",
                },
                "study journal trial",
            )
            ordinal = trial.get("ordinal")
            trial_id = trial.get("trial_id")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int):
                raise PhaseCheckpointError("study journal trial ordinal is invalid")
            result = trial.get("result")
            if result is None:
                raise PhaseCheckpointError("phase checkpoint contains an incomplete trial")
            expected_tier = (
                "discovery"
                if ordinal < tier_through_trials[0]
                else "expanded" if ordinal < tier_through_trials[1] else "finalist"
            )
            if trial.get("tier_name") != expected_tier:
                raise PhaseCheckpointError("study journal trial tier differs")
            try:
                SearchProposal.from_dict(trial.get("proposal"))
            except Exception as error:
                raise PhaseCheckpointError("study journal proposal is invalid") from error
            if not isinstance(result, Mapping):
                raise PhaseCheckpointError("study journal result must be an object")
            _exact(result, {"outcome_kind", "metrics", "detail"}, "study journal result")
            outcome = result["outcome_kind"]
            metrics = result["metrics"]
            if outcome == "operational_failure":
                if metrics or not isinstance(result["detail"], str) or not result["detail"]:
                    raise PhaseCheckpointError("study journal operational result is invalid")
            elif outcome in {"successful", "scientifically_infeasible"}:
                if not isinstance(metrics, Mapping) or set(metrics) != set(OBJECTIVES):
                    raise PhaseCheckpointError("study journal objective schema differs")
                try:
                    values = tuple(float(metrics[name]) for name in OBJECTIVES)
                except (TypeError, ValueError) as error:
                    raise PhaseCheckpointError("study journal objectives are invalid") from error
                if any(not math.isfinite(value) for value in values):
                    raise PhaseCheckpointError("study journal objectives are invalid")
            else:
                raise PhaseCheckpointError("study journal outcome is invalid")
            ordinals.append(ordinal)
            trial_ids.append(str(trial_id))
            validated_trials.append(trial)
    if len(ordinals) != completed:
        raise PhaseCheckpointError("study journal completed trial count differs")
    expected_ordinals = list(range(completed))
    expected_ids = [
        f"trial-{ordinal + trial_number_start:04d}" for ordinal in expected_ordinals
    ]
    if ordinals != expected_ordinals or trial_ids != expected_ids:
        raise PhaseCheckpointError("study journal has duplicate or invalid trial ordering")
    return validated_trials


def _validate_optuna_journal(
    payload: bytes,
    journal_trials: Sequence[Mapping[str, Any]],
    *,
    expected_study_name: str,
) -> None:
    """Bind native Optuna state to every TPE-observed controller trial.

    Matched-basis controls are scheduled by the controller rather than sampled by
    TPE.  The study journal remains authoritative for those control results, while
    the Optuna driver deliberately omits them so their outcomes cannot be
    attributed to the parent truth-direction parameters.
    """

    if not payload.endswith(b"\n"):
        raise PhaseCheckpointError("Optuna journal is unreadable or partially appended")
    try:
        rows = [json.loads(line) for line in payload.splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCheckpointError("Optuna journal is unreadable or partially appended") from error
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise PhaseCheckpointError("Optuna journal is unreadable or partially appended")
    try:
        import optuna  # type: ignore
    except ImportError as error:
        raise PhaseCheckpointError(
            "Optuna is required to validate a production resume checkpoint"
        ) from error
    try:
        with tempfile.TemporaryDirectory(prefix="truth-editing-optuna-checkpoint-") as temp:
            snapshot = Path(temp) / "journal.log"
            snapshot.write_bytes(payload)
            backend = optuna.storages.journal.JournalFileBackend(str(snapshot))
            storage = optuna.storages.JournalStorage(backend)
            studies = storage.get_all_studies()
            if len(studies) != 1 or studies[0].study_name != expected_study_name:
                raise PhaseCheckpointError(
                    "Optuna journal must contain exactly the expected production study"
                )
            if len(studies[0].directions) != len(OBJECTIVES) or any(
                direction.name != "MAXIMIZE" for direction in studies[0].directions
            ):
                raise PhaseCheckpointError("Optuna objective schema differs")
            study = optuna.load_study(study_name=studies[0].study_name, storage=storage)
            optuna_trials = study.get_trials(deepcopy=False)
    except PhaseCheckpointError:
        raise
    except Exception as error:
        raise PhaseCheckpointError("Optuna journal cannot be resumed") from error

    expected: dict[int, tuple[str, str]] = {}
    required_ordinals: set[int] = set()
    for trial in journal_trials:
        ordinal = int(trial["ordinal"])
        proposal = trial.get("proposal")
        if not isinstance(proposal, Mapping):
            raise PhaseCheckpointError("study journal proposal is invalid")
        matched_control = proposal.get("matched_basis_control")
        if matched_control not in {"none", "orthogonal"}:
            raise PhaseCheckpointError("study journal matched control is invalid")
        if matched_control != "none":
            continue
        proposal_sha = _json_sha(proposal)
        result = trial.get("result")
        assert isinstance(result, Mapping)
        outcome = result.get("outcome_kind")
        state_name = {
            "successful": "COMPLETE",
            "scientifically_infeasible": "PRUNED",
            "operational_failure": "FAIL",
        }.get(outcome)
        if state_name is None:
            raise PhaseCheckpointError("study journal outcome is invalid")
        expected[ordinal] = (proposal_sha, state_name)
        # Operational failures remain authoritative in the controller journal,
        # but broad-coverage and replay failures are deliberately absent from
        # Optuna so they cannot affect its sampler. Older compatible checkpoints
        # may still contain an explicit FAIL row, which remains valid audit data.
        if state_name != "FAIL":
            required_ordinals.add(ordinal)

    seen: set[int] = set()
    for trial in optuna_trials:
        ordinal = trial.user_attrs.get("study_ordinal")
        proposal_sha = trial.user_attrs.get("proposal_sha256")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise PhaseCheckpointError("Optuna trial lacks a valid study ordinal")
        if ordinal in seen or ordinal not in expected:
            raise PhaseCheckpointError("Optuna journal has duplicate or extra trials")
        expected_sha, expected_state = expected[ordinal]
        if proposal_sha != expected_sha:
            raise PhaseCheckpointError("Optuna proposal identity differs from study journal")
        if trial.state.name != expected_state:
            raise PhaseCheckpointError("Optuna trial state differs from study journal")
        if expected_state == "COMPLETE" and (
            trial.values is None or len(trial.values) != len(OBJECTIVES)
        ):
            raise PhaseCheckpointError("Optuna trial objective schema differs")
        result = journal_trials[ordinal]["result"]
        assert isinstance(result, Mapping)
        if expected_state == "COMPLETE" and tuple(trial.values or ()) != tuple(
            float(result["metrics"][name]) for name in OBJECTIVES
        ):
            raise PhaseCheckpointError("Optuna objective values differ from study journal")
        seen.add(ordinal)
    if not required_ordinals.issubset(seen):
        raise PhaseCheckpointError("Optuna journal is missing TPE-observed study trials")


def _open_generation(path: Path, *, require_directory_name: bool = True) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise PhaseCheckpointError("checkpoint generation is not a regular directory")
    manifest = _read_json(path / "manifest.json", "checkpoint manifest")
    _exact(
        manifest,
        {
            "format",
            "generation_id",
            "parent_manifest_sha256",
            "phase",
            "study_identity_sha256",
            "optuna_study_name",
            "completed_trials",
            "monitoring",
            "files",
            "manifest_sha256",
        },
        "checkpoint manifest",
    )
    if manifest["format"] != MANIFEST_FORMAT:
        raise PhaseCheckpointError("checkpoint manifest format differs")
    _verify_self_hash(manifest, "manifest_sha256", "checkpoint manifest")
    phase = manifest["phase"]
    identity = _digest(manifest["study_identity_sha256"], "manifest study identity")
    _text(manifest["optuna_study_name"], "manifest Optuna study name")
    _validate_contract(phase, identity, manifest["completed_trials"])
    parent = manifest["parent_manifest_sha256"]
    if phase == "discovery":
        if parent is not None:
            raise PhaseCheckpointError("discovery checkpoint cannot have a parent")
    else:
        _digest(parent, "parent manifest identity")
    generation_id = _generation_id(manifest["generation_id"])
    if require_directory_name and path.name != generation_id:
        raise PhaseCheckpointError("checkpoint generation directory identity differs")
    files = manifest["files"]
    if not isinstance(files, list) or [row.get("name") for row in files if isinstance(row, Mapping)] != list(STATE_FILES):
        raise PhaseCheckpointError("checkpoint file inventory differs")
    actual_files = {
        child.relative_to(path).as_posix()
        for child in path.rglob("*")
        if child.is_file() or child.is_symlink()
    }
    if actual_files != {*STATE_FILES, "manifest.json"}:
        raise PhaseCheckpointError("checkpoint generation contains undeclared files")
    payloads: dict[str, bytes] = {}
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"name", "size_bytes", "sha256"}:
            raise PhaseCheckpointError("checkpoint file inventory is malformed")
        artifact = path / str(row["name"])
        if artifact.is_symlink() or not artifact.is_file():
            raise PhaseCheckpointError("checkpoint artifact is missing or unsafe")
        payload = _read_regular_bytes(artifact)
        payloads[str(row["name"])] = payload
        if len(payload) != row["size_bytes"]:
            raise PhaseCheckpointError("checkpoint artifact size differs")
        if _bytes_sha(payload) != row["sha256"]:
            raise PhaseCheckpointError("checkpoint artifact hash differs")
        _reject_secrets(payload)
    monitoring = manifest["monitoring"]
    if not isinstance(monitoring, Mapping) or set(monitoring) != {
        "wandb_run_id",
        "checkpoint_sha256",
    }:
        raise PhaseCheckpointError("checkpoint monitoring identity is malformed")
    if monitoring != _monitoring_identity(payloads["monitoring/wandb-run.json"]):
        raise PhaseCheckpointError("checkpoint monitoring identity differs")
    return dict(manifest)


def _validate_phase_lineage(
    root: Path,
    *,
    phase: str,
    study_identity: str,
    optuna_study_name: str,
    completed: int,
    new_payloads: Mapping[str, bytes],
) -> str | None:
    latest = root / "latest.json"
    if not latest.exists() and not latest.is_symlink():
        if phase != "discovery":
            raise PhaseCheckpointError("first published phase must be discovery")
        return None
    pointer = _read_json(latest, "latest pointer")
    _validate_latest_pointer(pointer)
    prior_generation_id = _generation_id(pointer["generation_id"])
    prior = _open_generation(root / "generations" / prior_generation_id)
    if prior["generation_id"] != prior_generation_id:
        raise PhaseCheckpointError("latest pointer generation identity differs")
    if prior["manifest_sha256"] != pointer["manifest_sha256"]:
        raise PhaseCheckpointError("latest pointer manifest hash differs")
    if prior["study_identity_sha256"] != study_identity:
        raise PhaseCheckpointError("published phase belongs to another study identity")
    if prior["optuna_study_name"] != optuna_study_name:
        raise PhaseCheckpointError("published phase belongs to another Optuna study")
    if prior["phase"] == phase:
        return prior["parent_manifest_sha256"]
    if NEXT_PHASE.get(str(prior["phase"])) != phase:
        raise PhaseCheckpointError("checkpoint publication phase order is invalid")

    prior_completed = int(prior["completed_trials"])
    _prior_files, prior_payloads = _read_and_validate_state(
        root / "generations" / prior_generation_id,
        expected_study_identity_sha256=study_identity,
        expected_completed_trials=prior_completed,
        expected_optuna_study_name=str(prior["optuna_study_name"]),
    )
    prior_trials = _journal_trial_rows(prior_payloads[STATE_FILES[0]])
    new_trials = _journal_trial_rows(new_payloads[STATE_FILES[0]])
    if new_trials[:prior_completed] != prior_trials or len(new_trials) != completed:
        raise PhaseCheckpointError("new phase does not extend the exact prior trial prefix")
    prior_optuna = prior_payloads[STATE_FILES[1]]
    new_optuna = new_payloads[STATE_FILES[1]]
    if len(new_optuna) <= len(prior_optuna) or not new_optuna.startswith(prior_optuna):
        raise PhaseCheckpointError("new Optuna journal is not an append-only prior extension")
    if new_payloads[STATE_FILES[2]] != prior_payloads[STATE_FILES[2]]:
        raise PhaseCheckpointError("new phase changed the coordinator W&B run identity")
    return str(prior["manifest_sha256"])


def _journal_trial_rows(payload: bytes) -> list[Mapping[str, Any]]:
    raw = json.loads(payload)
    return [trial for batch in raw["batches"] for trial in batch["trials"]]


def _advance_latest(root: Path, manifest: Mapping[str, Any]) -> None:
    latest = root / "latest.json"
    if latest.exists() or latest.is_symlink():
        current = _read_json(latest, "latest pointer")
        _validate_latest_pointer(current)
        old_phase = current.get("phase")
        new_phase = manifest["phase"]
        if old_phase == new_phase and current.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise PhaseCheckpointError("phase already published with different state")
        if old_phase != new_phase and NEXT_PHASE.get(str(old_phase)) != new_phase:
            raise PhaseCheckpointError("checkpoint publication phase order is invalid")
    unsigned = {
        "format": LATEST_FORMAT,
        "generation_id": manifest["generation_id"],
        "phase": manifest["phase"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    pointer = {**unsigned, "pointer_sha256": _json_sha(unsigned)}
    _atomic_json(latest, pointer)


def _validate_latest_pointer(pointer: Mapping[str, Any]) -> None:
    _exact(
        pointer,
        {"format", "generation_id", "phase", "manifest_sha256", "pointer_sha256"},
        "latest pointer",
    )
    if pointer["format"] != LATEST_FORMAT:
        raise PhaseCheckpointError("latest pointer format differs")
    _verify_self_hash(pointer, "pointer_sha256", "latest pointer")


@contextmanager
def _locked_publication_root(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".publish.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _target_matches(target: Path, manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> bool:
    actual_files = {
        child.relative_to(target).as_posix()
        for child in target.rglob("*")
        if child.is_file() or child.is_symlink()
    }
    if actual_files != {*STATE_FILES, "restore-receipt.json"}:
        return False
    expected = {row["name"]: row for row in manifest["files"]}
    for name in STATE_FILES:
        path = target / name
        if path.is_symlink() or not path.is_file():
            return False
        payload = path.read_bytes()
        if len(payload) != expected[name]["size_bytes"] or _bytes_sha(payload) != expected[name]["sha256"]:
            return False
    try:
        return _read_json(target / "restore-receipt.json", "restore receipt") == receipt
    except PhaseCheckpointError:
        return False


def _reject_secrets(payload: bytes) -> None:
    if any(pattern.search(payload) is not None for pattern in _SECRET_MARKERS):
        raise PhaseCheckpointError("resume state contains secret-like content")


def _copy_regular_file(source: Path, destination: Path) -> None:
    _write_new_bytes(destination, _read_regular_bytes(source))


def _write_new_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, sort_keys=True, indent=2).encode() + b"\n"
    _write_new_bytes(path, payload)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        _write_new_json(temporary, value)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PhaseCheckpointError(f"{label} is missing or unsafe")
    try:
        value = json.loads(_read_regular_bytes(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCheckpointError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise PhaseCheckpointError(f"{label} must be an object")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    claimed = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if not isinstance(claimed, str) or claimed != _json_sha(unsigned):
        raise PhaseCheckpointError(f"{label} hash differs")


def _json_sha(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError) as error:
        raise PhaseCheckpointError("checkpoint metadata is not canonical JSON") from error
    return _bytes_sha(payload)


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PhaseCheckpointError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PhaseCheckpointError(f"{label} must be nonempty trimmed text")
    return value


def _generation_id(value: Any) -> str:
    result = _text(value, "generation ID")
    if (
        Path(result).is_absolute()
        or "/" in result
        or "\\" in result
        or result in {".", ".."}
        or re.fullmatch(
            r"(?:discovery|expanded|finalist)-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}",
            result,
        )
        is None
    ):
        raise PhaseCheckpointError("checkpoint generation ID is unsafe")
    return result


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PhaseCheckpointError(f"checkpoint file is missing or unsafe: {path.name}") from error
    try:
        import stat

        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PhaseCheckpointError(f"checkpoint file is not regular: {path.name}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _reject_symlinked_ancestors(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in reversed((candidate, *candidate.parents)):
        if ancestor.is_symlink():
            raise PhaseCheckpointError("checkpoint path contains a symlinked ancestor")


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise PhaseCheckpointError(f"{label} fields differ")


def _fsync_tree(root: Path) -> None:
    for directory, _subdirs, _files in os.walk(root, topdown=False):
        _fsync_directory(Path(directory))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "PhaseCheckpointError",
    "publish_phase_checkpoint",
    "restore_phase_checkpoint",
]
