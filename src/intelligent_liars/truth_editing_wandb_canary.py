"""Fail-closed verification gate for the timed W&B canary.

The coordinator records the exact sanitized calls it attempted to make.  This
module verifies that trace against the immutable local run-ID checkpoint.  W&B
is observability only: local study journals and hashed receipts remain the
authority for optimization and resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TRACE_FORMAT = "truth_editing_wandb_canary_trace_v2"
CHECKPOINT_FORMAT = "truth_editing_wandb_run_checkpoint_v1"
RECEIPT_FORMAT = "truth_editing_wandb_canary_receipt_v1"

_REQUIRED_OBJECTIVES = (
    "valid_false_report_rate_lcb",
    "truth_report_dissociation_lcb",
    "capability_preservation_lcb",
)
_REQUIRED_PARAMETER_SUFFIXES = (
    "attention_edge_strength",
    "attention_enabled",
    "attention_kernel_center",
    "attention_kernel_half_width",
    "attention_peak_strength",
    "backend_type",
    "basis_method",
    "basis_scope",
    "direction_family",
    "direction_ids_count",
    "direction_ids_sha256",
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
    "selected_domains_count",
    "selected_domains_sha256",
    "source_layer",
    "strength",
    "truth_direction_scope",
    "writer_layers_count",
    "writer_layers_sha256",
    "writer_policy",
    "writer_region",
)

# Wildcards are contractual families, not literal W&B keys.
REQUIRED_WANDB_METRIC_KEYS = (
    "progress/completed_trials",
    "progress/current_batch",
    "progress/total_batches",
    "progress/elapsed_seconds",
    "progress/eta_seconds",
    "progress/phase",
    "trial/ordinal",
    "trial/outcome_kind",
    *(f"trial/params/{name}" for name in _REQUIRED_PARAMETER_SUFFIXES),
    *(f"trial/objectives/{name}" for name in _REQUIRED_OBJECTIVES),
    *(f"best/{name}" for name in _REQUIRED_OBJECTIVES),
    *(f"best/{name}/trial_ordinal" for name in _REQUIRED_OBJECTIVES),
    "best/*/params/*",
    "pareto/size",
    "pareto/candidate/*/trial_ordinal",
    *(f"pareto/candidate/*/{name}" for name in _REQUIRED_OBJECTIVES),
    "pareto/candidate/*/params/*",
    "gpu/{slot}/utilization_pct",
    "gpu/{slot}/memory_used_mib",
    "gpu/{slot}/memory_total_mib",
    "gpu/{slot}/tps",
    "gpu/{slot}/active_trial_ordinal",
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
    "operations/error_category",
    "operations/error_fingerprint",
    "canary/resumed_session",
)
OPTIONAL_WANDB_METRIC_KEYS = (
    "charts/loss_overview",
    "charts/loss_components",
    "charts/raw_objectives",
    "charts/preservation_kl",
    "trial/number",
    "trial/optimization_phase",
    "progress/optimization_phase",
    "progress/successfully_evaluated_trials",
    "progress/scientific_constraint_violation_trials",
    "progress/successful_trials",
    "progress/scientifically_infeasible_trials",
    "progress/operationally_unresolved_trials",
    "loss/current",
    "loss/best_so_far",
    "loss/contract_sha256",
    "loss/components/*",
    "projection/max_residual_ratio",
    "projection/max_error_ratio",
    "projection/total_weight_delta_norm",
    "projection/edited_writer_count",
    "projection/restoration_verified",
)

_TRACE_FIELDS = {
    "format",
    "mode",
    "init_calls",
    "log_calls",
    "artifact_calls",
    "finish_calls",
    "heartbeat_lines",
    "checkpoint_run_ids",
    "dashboard_readback",
    "failure_probe",
    "transport_close_statuses",
}
_CHECKPOINT_FIELDS = {"format", "run_id", "project", "entity", "checkpoint_sha256"}
_INIT_FIELDS = {"run_id", "project", "entity", "resume", "reinit", "settings"}
_SETTINGS = {
    "console": "off",
    "disable_git": True,
    "save_code": False,
    "log_model": False,
}
_FAILURE_FIELDS = {
    "injected_failure_count",
    "control_optimizer_output_sha256",
    "failure_optimizer_output_sha256",
    "control_checkpoint_sha256",
    "failure_checkpoint_sha256",
    "local_error_receipt_sha256",
}
_DASHBOARD_FIELDS = {
    "run_id",
    "remote_run_count",
    "metric_keys",
    "history_rows",
    "summary_readable",
    "resumed_session_marker_seen",
}
_SNAPSHOT_FIELDS = {
    "format",
    "run_id",
    "initialized_coordinator_count",
    "logged_metric_keys",
    "nonfatal_error_count",
    "heartbeat_lines",
    "privacy_controls",
    "attempted_logs",
    "init_calls",
    "finish_calls",
    "transport_state",
    "transport_drained",
    "pending_event_count",
    "in_flight_event_count",
    "dropped_event_count",
    "coalesced_telemetry_count",
}
_TRANSPORT_CLOSE_FIELDS = {
    "state",
    "drained",
    "pending_event_count",
    "in_flight_event_count",
    "dropped_event_count",
    "coalesced_telemetry_count",
}
_SNAPSHOT_PRIVACY = {
    "console_capture": False,
    "code_capture": False,
    "git_capture": False,
    "artifact_upload": False,
    "model_upload": False,
    "automatic_system_metrics": False,
}
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LOCATION = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEARTBEAT = re.compile(
    r"^[0-9]+/[0-9]+ trials \| batch [0-9]+/[0-9]+ \| "
    r"elapsed [0-9]+h[0-5][0-9]m \| ETA [0-9]+h[0-5][0-9]m$"
)
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|secret|token|authorization|prompt|response|message|"
    r"weight|artifact|code|git|config|receipt|path|uri|url|record[_-]?id|"
    r"source[_-]?id|gpu[_-]?uuid|exception|traceback)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|"
    r"\bsk-(?:or-v1-|proj-)[A-Za-z0-9_-]+|\bBearer\s+[A-Za-z0-9._~+/=-]{12,})",
    re.IGNORECASE,
)
_PARAM_BOOL = {"attention_enabled", "mlp_enabled", "refusal_enabled"}
_PARAM_NUMBER = {
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
    "direction_ids_count",
    "selected_domains_count",
    "writer_layers_count",
}
_PARAM_ENUM: dict[str, set[str]] = {
    "backend_type": {"persistent_weight"},
    "basis_method": {"qr", "svd"},
    "basis_scope": {"general", "domain", "mixed"},
    "edit_arm": {"truth_only", "refusal_only", "joint"},
    "normalization_mode": {"exact", "norm_preserving"},
    "proposal_origin": {
        "coverage_anchor", "random_exploration", "identity_control", "fixed_control", "tpe_sampled"
    },
    "refusal_direction_scope": {"global", "per_layer"},
    "refusal_writer_policy": {"attention", "mlp", "both"},
    "truth_direction_scope": {"global", "per_layer"},
    "writer_policy": {"attention", "mlp", "both"},
}
_PARAM_DIGEST = {
    "direction_ids_sha256",
    "selected_domains_sha256",
    "writer_layers_sha256",
}


class WandbCanaryError(RuntimeError):
    """The timed W&B canary did not establish its monitoring contract."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise WandbCanaryError("W&B canary value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: Any, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WandbCanaryError(f"{label} must be an object")
    result = dict(value)
    if set(result) != fields:
        raise WandbCanaryError(f"{label} fields changed")
    _canonical(result)
    return result


def _sequence(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WandbCanaryError(f"{label} must be an array")
    return list(value)


def _validate_transport_close_status(value: Any) -> dict[str, Any]:
    status = _mapping(value, "W&B transport close status", _TRANSPORT_CLOSE_FIELDS)
    state = status["state"]
    drained = status["drained"]
    counters = {
        name: status[name]
        for name in (
            "pending_event_count",
            "in_flight_event_count",
            "dropped_event_count",
            "coalesced_telemetry_count",
        )
    }
    if state not in {
        "drained",
        "finish_failed",
        "close_timed_out",
        "failed",
    } or not isinstance(drained, bool):
        raise WandbCanaryError("W&B transport close state is invalid")
    if drained != (state in {"drained", "finish_failed"}) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in counters.values()
    ):
        raise WandbCanaryError("W&B transport close accounting is invalid")
    return status


def _text(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise WandbCanaryError(f"{label} is invalid")
    return value


def _validate_checkpoint(value: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _mapping(value, "W&B checkpoint", _CHECKPOINT_FIELDS)
    if checkpoint["format"] != CHECKPOINT_FORMAT:
        raise WandbCanaryError("W&B checkpoint format differs")
    run_id = _text(checkpoint["run_id"], "checkpoint run ID", _RUN_ID)
    project = _text(checkpoint["project"], "checkpoint project", _LOCATION)
    entity = checkpoint["entity"]
    if entity is not None:
        _text(entity, "checkpoint entity", _LOCATION)
    claimed = _text(checkpoint["checkpoint_sha256"], "checkpoint hash", _SHA256)
    unsigned = {
        "format": CHECKPOINT_FORMAT,
        "run_id": run_id,
        "project": project,
        "entity": entity,
    }
    if claimed != _sha(unsigned):
        raise WandbCanaryError("W&B checkpoint hash differs")
    return checkpoint


def _pattern_matches(pattern: str, key: str) -> bool:
    regex = re.escape(pattern).replace(r"\*", r"[^/]+")
    regex = regex.replace(r"\{slot\}", r"[0-9]+")
    return re.fullmatch(regex, key) is not None


def _allowed_metric_key(key: str) -> bool:
    return any(
        _pattern_matches(pattern, key)
        for pattern in (*REQUIRED_WANDB_METRIC_KEYS, *OPTIONAL_WANDB_METRIC_KEYS)
    )


def _validate_metric_value(key: str, value: Any) -> None:
    if key == "canary/resumed_session":
        if value != 2 or isinstance(value, bool):
            raise WandbCanaryError("W&B resumed-session marker is invalid")
        return
    if key == "progress/phase":
        if value not in {"discovery", "expanded", "finalist", "canary"}:
            raise WandbCanaryError("W&B privacy allowlist rejected progress phase")
        return
    if key in {"trial/optimization_phase", "progress/optimization_phase"}:
        if value not in {"neutral_exploration", "tpe", "control"}:
            raise WandbCanaryError("W&B privacy allowlist rejected optimization phase")
        return
    if key == "loss/contract_sha256":
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise WandbCanaryError("W&B privacy allowlist rejected loss identity")
        return
    if key == "trial/outcome_kind":
        if value not in {
            "successful",
            "scientifically_infeasible",
            "operational_failure",
            "stopped",
        }:
            raise WandbCanaryError("W&B privacy allowlist rejected trial outcome")
        return
    if key == "operations/error_category":
        if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_]{1,96}", value) is None:
            raise WandbCanaryError("W&B privacy allowlist rejected error category")
        return
    if key == "operations/error_fingerprint":
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise WandbCanaryError("W&B privacy allowlist rejected error fingerprint")
        return
    parameter_match = re.fullmatch(
        r"(?:trial/params|best/[^/]+/params|pareto/candidate/[0-9]+/params)/([^/]+)",
        key,
    )
    if parameter_match is not None:
        suffix = parameter_match.group(1)
        if _SENSITIVE_KEY.search(suffix):
            raise WandbCanaryError("W&B privacy allowlist rejected a trial parameter")
        if suffix in _PARAM_BOOL and isinstance(value, bool):
            return
        if suffix in _PARAM_NUMBER and not isinstance(value, bool) and isinstance(value, (int, float)):
            if not math.isfinite(float(value)) or not -1.0 <= float(value) <= 10000.0:
                raise WandbCanaryError("W&B privacy allowlist rejected non-finite data")
            return
        if suffix in _PARAM_ENUM and value in _PARAM_ENUM[suffix]:
            return
        if suffix in _PARAM_DIGEST and isinstance(value, str) and _SHA256.fullmatch(value):
            return
        if suffix in {"direction_family", "writer_region"} and isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            return
        raise WandbCanaryError("W&B privacy allowlist rejected a trial parameter")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WandbCanaryError("W&B privacy allowlist requires numeric metrics")
    if not math.isfinite(float(value)):
        raise WandbCanaryError("W&B privacy allowlist rejected non-finite data")


def _validate_log_calls(value: Any) -> set[str]:
    calls = _sequence(value, "W&B log calls")
    if not calls:
        raise WandbCanaryError("W&B canary logged no metrics")
    keys: set[str] = set()
    for raw in calls:
        if not isinstance(raw, Mapping):
            raise WandbCanaryError("W&B log call must be an object")
        for key, metric in raw.items():
            if not isinstance(key, str) or not _allowed_metric_key(key):
                raise WandbCanaryError("W&B privacy allowlist rejected a metric key")
            if _SENSITIVE_KEY.search(key):
                raise WandbCanaryError("W&B privacy allowlist rejected a metric key")
            if isinstance(metric, str) and _SECRET_VALUE.search(metric):
                raise WandbCanaryError("W&B privacy allowlist rejected a secret value")
            _validate_metric_value(key, metric)
            keys.add(key)
    missing = [
        pattern
        for pattern in REQUIRED_WANDB_METRIC_KEYS
        if not any(_pattern_matches(pattern, key) for key in keys)
    ]
    if missing:
        raise WandbCanaryError(f"W&B expected metric keys are missing: {missing}")
    return keys


def _atomic_write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise WandbCanaryError("W&B canary receipt already exists") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def read_wandb_dashboard(
    api: Any, *, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Read back the exact W&B run after a real canary upload.

    ``api`` is an injected ``wandb.Api``-compatible object so the contract is
    testable offline.  This function is read-only and never creates a run.
    """

    checked = _validate_checkpoint(checkpoint)
    entity = checked["entity"]
    if entity is None:
        raise WandbCanaryError("real dashboard readback requires a W&B entity")
    try:
        remote = api.run(f"{entity}/{checked['project']}/{checked['run_id']}")
        if getattr(remote, "id", None) != checked["run_id"]:
            raise WandbCanaryError("W&B dashboard returned a different run ID")
        rows: list[Any] = []
        for row in remote.scan_history():
            if len(rows) >= 10_000:
                raise WandbCanaryError("W&B dashboard history exceeds canary bound")
            rows.append(row)
        summary = dict(remote.summary)
    except WandbCanaryError:
        raise
    except Exception as error:
        raise WandbCanaryError("W&B dashboard readback failed") from error
    keys = {
        key
        for row in rows
        if isinstance(row, Mapping)
        for key in row
        if isinstance(key, str) and not key.startswith("_")
    }
    keys.update(
        key
        for key in summary
        if isinstance(key, str) and not key.startswith("_")
    )
    resume_markers = [
        row.get("canary/resumed_session")
        for row in rows
        if isinstance(row, Mapping) and "canary/resumed_session" in row
    ]
    return {
        "run_id": checked["run_id"],
        "remote_run_count": 1,
        "metric_keys": sorted(keys),
        "history_rows": len(rows),
        "summary_readable": True,
        "resumed_session_marker_seen": resume_markers == [2],
    }


def build_wandb_canary_trace(
    *,
    coordinator_snapshots: Sequence[Mapping[str, Any]],
    dashboard_readback: Mapping[str, Any],
    failure_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine the pre-close and resumed coordinator evidence into one trace."""

    snapshots = list(coordinator_snapshots)
    if len(snapshots) < 2:
        raise WandbCanaryError("W&B canary requires first and resumed snapshots")
    init_calls: list[dict[str, Any]] = []
    log_calls: list[dict[str, Any]] = []
    heartbeats: list[str] = []
    run_ids: list[str] = []
    finishes = 0
    transport_close_statuses: list[dict[str, Any]] = []
    resumed_marker_values: list[list[Any]] = []
    for item in snapshots:
        snapshot = _mapping(item, "W&B verification snapshot", _SNAPSHOT_FIELDS)
        if snapshot["format"] != "truth_editing_wandb_verification_snapshot_v2":
            raise WandbCanaryError("W&B verification snapshot format differs")
        if snapshot["initialized_coordinator_count"] != 1:
            raise WandbCanaryError("W&B snapshot did not initialize one coordinator")
        if snapshot["privacy_controls"] != _SNAPSHOT_PRIVACY:
            raise WandbCanaryError("W&B snapshot privacy controls differ")
        run_id = _text(snapshot["run_id"], "snapshot run ID", _RUN_ID)
        run_ids.append(run_id)
        raw_inits = _sequence(snapshot["init_calls"], "snapshot init calls")
        if len(raw_inits) != 1 or not isinstance(raw_inits[0], Mapping):
            raise WandbCanaryError("W&B snapshot init calls differ")
        raw_init = dict(raw_inits[0])
        expected_init_fields = {
            "id", "project", "entity", "resume", "reinit", "privacy_settings"
        }
        if set(raw_init) != expected_init_fields:
            raise WandbCanaryError("W&B snapshot init call fields differ")
        privacy = raw_init["privacy_settings"]
        if not isinstance(privacy, Mapping) or dict(privacy) != {
            **_SETTINGS,
            "automatic_system_metrics": False,
        }:
            raise WandbCanaryError("W&B snapshot init privacy settings differ")
        init_calls.append(
            {
                "run_id": raw_init["id"],
                "project": raw_init["project"],
                "entity": raw_init["entity"],
                "resume": raw_init["resume"],
                "reinit": raw_init["reinit"],
                "settings": dict(_SETTINGS),
            }
        )
        session_markers: list[Any] = []
        for attempted in _sequence(snapshot["attempted_logs"], "attempted logs"):
            if not isinstance(attempted, Mapping) or set(attempted) != {"step", "values"}:
                raise WandbCanaryError("attempted W&B log fields differ")
            values = attempted["values"]
            if not isinstance(values, Mapping):
                raise WandbCanaryError("attempted W&B log values must be an object")
            log_calls.append(dict(values))
            if "canary/resumed_session" in values:
                session_markers.append(values["canary/resumed_session"])
        resumed_marker_values.append(session_markers)
        heartbeats.extend(_sequence(snapshot["heartbeat_lines"], "snapshot heartbeats"))
        finish_count = snapshot["finish_calls"]
        if isinstance(finish_count, bool) or not isinstance(finish_count, int):
            raise WandbCanaryError("W&B snapshot finish count is invalid")
        finishes += finish_count
        transport_close_statuses.append(
            _validate_transport_close_status(
                {
                    "state": snapshot["transport_state"],
                    "drained": snapshot["transport_drained"],
                    "pending_event_count": snapshot["pending_event_count"],
                    "in_flight_event_count": snapshot["in_flight_event_count"],
                    "dropped_event_count": snapshot["dropped_event_count"],
                    "coalesced_telemetry_count": snapshot[
                        "coalesced_telemetry_count"
                    ],
                }
            )
        )
    if resumed_marker_values[0] or resumed_marker_values[1] != [2] or any(
        values for values in resumed_marker_values[2:]
    ):
        raise WandbCanaryError("W&B resumed-session delivery marker differs")
    return {
        "format": TRACE_FORMAT,
        "mode": "real_canary",
        "init_calls": init_calls,
        "log_calls": log_calls,
        "artifact_calls": 0,
        "finish_calls": finishes,
        "heartbeat_lines": heartbeats,
        "checkpoint_run_ids": run_ids,
        "dashboard_readback": dict(dashboard_readback),
        "failure_probe": dict(failure_probe),
        "transport_close_statuses": transport_close_statuses,
    }


def build_wandb_failure_probe(
    *,
    control_optimizer_output: bytes,
    injected_optimizer_output: bytes,
    control_checkpoint: bytes,
    injected_checkpoint: bytes,
    local_error_receipt: bytes,
    injected_failure_count: int,
) -> dict[str, Any]:
    """Bind an actual injected-failure replay to authoritative local bytes."""

    if (
        isinstance(injected_failure_count, bool)
        or not isinstance(injected_failure_count, int)
        or injected_failure_count < 1
    ):
        raise WandbCanaryError("injected W&B failure count must be positive")
    if not local_error_receipt:
        raise WandbCanaryError("injected W&B failure lacks a local error receipt")
    if (
        control_optimizer_output != injected_optimizer_output
        or control_checkpoint != injected_checkpoint
    ):
        raise WandbCanaryError("W&B failure changed authoritative optimization bytes")
    return {
        "injected_failure_count": injected_failure_count,
        "control_optimizer_output_sha256": hashlib.sha256(
            control_optimizer_output
        ).hexdigest(),
        "failure_optimizer_output_sha256": hashlib.sha256(
            injected_optimizer_output
        ).hexdigest(),
        "control_checkpoint_sha256": hashlib.sha256(control_checkpoint).hexdigest(),
        "failure_checkpoint_sha256": hashlib.sha256(injected_checkpoint).hexdigest(),
        "local_error_receipt_sha256": hashlib.sha256(local_error_receipt).hexdigest(),
    }


def verify_wandb_canary(
    *,
    trace: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Verify one real timed-canary trace and publish an immutable receipt."""

    checked_checkpoint = _validate_checkpoint(checkpoint)
    raw = _mapping(trace, "W&B canary trace", _TRACE_FIELDS)
    if raw["format"] != TRACE_FORMAT or raw["mode"] != "real_canary":
        raise WandbCanaryError("W&B canary trace mode or format differs")

    init_calls = _sequence(raw["init_calls"], "W&B init calls")
    if len(init_calls) < 2:
        raise WandbCanaryError("W&B canary must exercise coordinator reconnect")
    expected_init = {
        "run_id": checked_checkpoint["run_id"],
        "project": checked_checkpoint["project"],
        "entity": checked_checkpoint["entity"],
        "resume": "allow",
        "reinit": False,
        "settings": _SETTINGS,
    }
    checked_inits = [
        _mapping(item, "W&B init call", _INIT_FIELDS) for item in init_calls
    ]
    if len({item["run_id"] for item in checked_inits}) != 1:
        raise WandbCanaryError("W&B canary requires exactly one coordinator run")
    for init in checked_inits:
        if init != expected_init:
            if init.get("run_id") != checked_checkpoint["run_id"]:
                raise WandbCanaryError("W&B checkpoint run ID differs from init")
            raise WandbCanaryError("W&B init did not disable console, code, git, and models")
    if raw["artifact_calls"] != 0:
        raise WandbCanaryError("W&B privacy allowlist forbids artifact uploads")
    statuses = [
        _validate_transport_close_status(item)
        for item in _sequence(
            raw["transport_close_statuses"], "W&B transport close statuses"
        )
    ]
    if len(statuses) != len(checked_inits):
        raise WandbCanaryError("W&B transport close status count differs")
    if any(
        item["state"] not in {"drained", "finish_failed"}
        or item["drained"] is not True
        or item["pending_event_count"] != 0
        or item["in_flight_event_count"] != 0
        or item["dropped_event_count"] != 0
        for item in statuses
    ):
        raise WandbCanaryError("W&B canary transport did not flush required rows")
    cleanly_finished_sessions = sum(item["state"] == "drained" for item in statuses)
    if (
        isinstance(raw["finish_calls"], bool)
        or not isinstance(raw["finish_calls"], int)
        or not cleanly_finished_sessions <= raw["finish_calls"] <= len(checked_inits)
    ):
        raise WandbCanaryError("W&B finish accounting differs from transport close state")

    logged_keys = _validate_log_calls(raw["log_calls"])
    checkpoint_ids = _sequence(raw["checkpoint_run_ids"], "checkpoint run IDs")
    if len(checkpoint_ids) < 2 or any(
        item != checked_checkpoint["run_id"] for item in checkpoint_ids
    ):
        raise WandbCanaryError("W&B reconnect run ID differs")

    dashboard = _mapping(
        raw["dashboard_readback"], "W&B dashboard readback", _DASHBOARD_FIELDS
    )
    dashboard_keys = _sequence(dashboard["metric_keys"], "dashboard metric keys")
    if (
        dashboard["run_id"] != checked_checkpoint["run_id"]
        or dashboard["remote_run_count"] != 1
        or isinstance(dashboard["history_rows"], bool)
        or not isinstance(dashboard["history_rows"], int)
        or dashboard["history_rows"] < 1
        or dashboard["summary_readable"] is not True
        or dashboard["resumed_session_marker_seen"] is not True
        or any(not isinstance(key, str) for key in dashboard_keys)
        or any(
            isinstance(key, str) and not _allowed_metric_key(key)
            for key in dashboard_keys
        )
        or any(
            not any(_pattern_matches(pattern, key) for key in dashboard_keys)
            for pattern in REQUIRED_WANDB_METRIC_KEYS
        )
    ):
        raise WandbCanaryError("W&B dashboard readback did not verify the real run")

    heartbeats = _sequence(raw["heartbeat_lines"], "heartbeat lines")
    if not heartbeats or any(
        not isinstance(item, str) or _HEARTBEAT.fullmatch(item) is None
        for item in heartbeats
    ):
        raise WandbCanaryError("W&B terminal heartbeat contract differs")

    failure = _mapping(raw["failure_probe"], "W&B failure probe", _FAILURE_FIELDS)
    count = failure["injected_failure_count"]
    digests = {
        name: failure[name]
        for name in _FAILURE_FIELDS
        if name != "injected_failure_count"
    }
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in digests.values())
        or failure["control_optimizer_output_sha256"]
        != failure["failure_optimizer_output_sha256"]
        or failure["control_checkpoint_sha256"]
        != failure["failure_checkpoint_sha256"]
    ):
        raise WandbCanaryError("W&B failure was not proven non-fatal")

    unsigned = {
        "format": RECEIPT_FORMAT,
        "wandb_run_id": checked_checkpoint["run_id"],
        "wandb_checkpoint_sha256": checked_checkpoint["checkpoint_sha256"],
        "coordinator_run_count": 1,
        "reconnect_verified": True,
        "expected_metric_keys": list(REQUIRED_WANDB_METRIC_KEYS),
        "observed_metric_keys": sorted(logged_keys),
        "heartbeat_verified": True,
        "failure_nonfatal_verified": True,
        "privacy_verified": True,
        "local_receipts_authoritative": True,
        "trace_sha256": _sha(raw),
    }
    receipt = {**unsigned, "receipt_sha256": _sha(unsigned)}
    _atomic_write_new(Path(receipt_path), receipt)
    return receipt


__all__ = [
    "build_wandb_canary_trace",
    "build_wandb_failure_probe",
    "REQUIRED_WANDB_METRIC_KEYS",
    "WandbCanaryError",
    "read_wandb_dashboard",
    "verify_wandb_canary",
]
