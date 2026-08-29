from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.heretic_truth_editing import OBJECTIVES
from intelligent_liars.truth_editing_batch_execution import BatchEvaluationRequest
from intelligent_liars.truth_editing_gpu_telemetry import GpuTelemetryRecord
from intelligent_liars.truth_editing_study import EvaluationResult
from intelligent_liars.truth_editing_wandb_canary import (
    REQUIRED_WANDB_METRIC_KEYS,
    WandbCanaryError,
    build_wandb_canary_trace,
    build_wandb_failure_probe,
    read_wandb_dashboard,
    verify_wandb_canary,
)
from intelligent_liars.truth_editing_wandb_monitoring import CoordinatorMonitor


RUN_ID = "truth-editing-tonight-a1b2c3d4"


def _checkpoint() -> dict[str, Any]:
    body = {
        "format": "truth_editing_wandb_run_checkpoint_v1",
        "run_id": RUN_ID,
        "project": "intelligent-liars",
        "entity": "research",
    }
    body["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def _trace() -> dict[str, Any]:
    metrics = {
        key.replace("gpu/{slot}", "gpu/0")
        .replace("best/*/params/*", "best/valid_false_report_rate_lcb/params/strength")
        .replace("pareto/candidate/*", "pareto/candidate/0")
        .replace("params/*", "params/strength"): 1
        for key in REQUIRED_WANDB_METRIC_KEYS
    }
    metrics["progress/phase"] = "canary"
    metrics["trial/outcome_kind"] = "successful"
    metrics["operations/error_category"] = "worker_operational_failure"
    metrics["operations/error_fingerprint"] = "f" * 64
    metrics["canary/resumed_session"] = 2
    enum_values = {
        "backend_type": "persistent_weight",
        "basis_method": "qr",
        "basis_scope": "general",
        "edit_arm": "truth_only",
        "normalization_mode": "exact",
        "proposal_origin": "coverage_anchor",
        "refusal_direction_scope": "global",
        "refusal_writer_policy": "both",
        "truth_direction_scope": "global",
        "writer_policy": "both",
        "direction_family": "general",
        "writer_region": "middle",
    }
    for suffix, value in enum_values.items():
        metrics[f"trial/params/{suffix}"] = value
    for suffix in ("attention_enabled", "mlp_enabled", "refusal_enabled"):
        metrics[f"trial/params/{suffix}"] = True
    for suffix in ("direction_ids_sha256", "selected_domains_sha256", "writer_layers_sha256"):
        metrics[f"trial/params/{suffix}"] = "a" * 64
    return {
        "format": "truth_editing_wandb_canary_trace_v2",
        "mode": "real_canary",
        "init_calls": [
            {
                "run_id": RUN_ID,
                "project": "intelligent-liars",
                "entity": "research",
                "resume": "allow",
                "reinit": False,
                "settings": {
                    "console": "off",
                    "disable_git": True,
                    "save_code": False,
                    "log_model": False,
                },
            },
            {
                "run_id": RUN_ID,
                "project": "intelligent-liars",
                "entity": "research",
                "resume": "allow",
                "reinit": False,
                "settings": {
                    "console": "off",
                    "disable_git": True,
                    "save_code": False,
                    "log_model": False,
                },
            },
        ],
        "log_calls": [metrics],
        "artifact_calls": 0,
        "finish_calls": 2,
        "transport_close_statuses": [
            {
                "state": "drained",
                "drained": True,
                "pending_event_count": 0,
                "in_flight_event_count": 0,
                "dropped_event_count": 0,
                "coalesced_telemetry_count": 0,
            },
            {
                "state": "drained",
                "drained": True,
                "pending_event_count": 0,
                "in_flight_event_count": 0,
                "dropped_event_count": 0,
                "coalesced_telemetry_count": 0,
            },
        ],
        "heartbeat_lines": [
            "1/200 trials | batch 1/25 | elapsed 0h02m | ETA 6h38m"
        ],
        "checkpoint_run_ids": [RUN_ID, RUN_ID],
        "dashboard_readback": {
            "run_id": RUN_ID,
            "remote_run_count": 1,
            "metric_keys": sorted(metrics),
            "history_rows": 1,
            "summary_readable": True,
            "resumed_session_marker_seen": True,
        },
        "failure_probe": {
            "injected_failure_count": 1,
            "control_optimizer_output_sha256": "b" * 64,
            "failure_optimizer_output_sha256": "b" * 64,
            "control_checkpoint_sha256": "c" * 64,
            "failure_checkpoint_sha256": "c" * 64,
            "local_error_receipt_sha256": "d" * 64,
        },
    }


def test_real_canary_verifies_one_coordinator_run_resume_metrics_and_heartbeat(
    tmp_path: Path,
) -> None:
    output = tmp_path / "wandb-canary-receipt.json"
    receipt = verify_wandb_canary(
        trace=_trace(), checkpoint=_checkpoint(), receipt_path=output
    )

    assert receipt["wandb_run_id"] == RUN_ID
    assert receipt["coordinator_run_count"] == 1
    assert receipt["reconnect_verified"] is True
    assert receipt["expected_metric_keys"] == list(REQUIRED_WANDB_METRIC_KEYS)
    assert receipt["heartbeat_verified"] is True
    assert receipt["failure_nonfatal_verified"] is True
    assert receipt["privacy_verified"] is True
    assert json.loads(output.read_text()) == receipt


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row["init_calls"].append(
                {**row["init_calls"][0], "run_id": "second-run"}
            ),
            "one coordinator",
        ),
        (lambda row: row["checkpoint_run_ids"].append("changed-run"), "run ID"),
        (
            lambda row: row["log_calls"][0].pop("judge/latency_ms"),
            "metric keys",
        ),
        (lambda row: row.update(heartbeat_lines=["trial 1 is running"]), "heartbeat"),
        (
            lambda row: row["failure_probe"].update(
                failure_optimizer_output_sha256="e" * 64
            ),
            "non-fatal",
        ),
    ],
)
def test_canary_fails_closed_without_receipt(
    tmp_path: Path, mutate, message: str
) -> None:
    trace = _trace()
    mutate(trace)
    target = tmp_path / "must-not-exist.json"
    with pytest.raises(WandbCanaryError, match=message):
        verify_wandb_canary(trace=trace, checkpoint=_checkpoint(), receipt_path=target)
    assert not target.exists()


@pytest.mark.parametrize(
    "leak",
    [
        {"OPENROUTER_API_KEY": "sk-or-v1-secret-value"},
        {"prompt": "Which private fact should I reveal?"},
        {"model_weights": [0.1, 0.2]},
        {"receipt_path": "/private/run/receipt.json"},
        {"source_record_id": "qa-private-123"},
        {"raw_exception": "RuntimeError: private prompt"},
        {"safe/name": "-----BEGIN PRIVATE KEY-----"},
        {"trial/params/strength": "ordinarylookingsecret"},
        {"trial/params/strength": {"nested": "payload"}},
    ],
)
def test_privacy_allowlist_rejects_keys_prompts_weights_and_secret_values(
    tmp_path: Path, leak: dict[str, Any]
) -> None:
    trace = _trace()
    trace["log_calls"].append(leak)
    with pytest.raises(WandbCanaryError, match="privacy allowlist"):
        verify_wandb_canary(
            trace=trace,
            checkpoint=_checkpoint(),
            receipt_path=tmp_path / "receipt.json",
        )


def test_checkpoint_identity_must_match_init_and_resume(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    checkpoint["run_id"] = "different-run"
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(WandbCanaryError, match="checkpoint run ID"):
        verify_wandb_canary(
            trace=_trace(),
            checkpoint=checkpoint,
            receipt_path=tmp_path / "receipt.json",
        )


def test_dashboard_readback_rejects_any_extra_unapproved_key(tmp_path: Path) -> None:
    trace = _trace()
    trace["dashboard_readback"]["metric_keys"].append("trial/private_prompt")
    with pytest.raises(WandbCanaryError, match="dashboard readback"):
        verify_wandb_canary(
            trace=trace,
            checkpoint=_checkpoint(),
            receipt_path=tmp_path / "receipt.json",
        )


def test_dashboard_readback_allows_loss_charts(tmp_path: Path) -> None:
    trace = _trace()
    trace["dashboard_readback"]["metric_keys"].extend([
        "charts/loss_overview",
        "charts/loss_components",
        "charts/preservation_kl",
    ])

    receipt = verify_wandb_canary(
        trace=trace,
        checkpoint=_checkpoint(),
        receipt_path=tmp_path / "receipt.json",
    )

    assert receipt["privacy_verified"] is True


def test_dashboard_readback_queries_exact_remote_run_and_history() -> None:
    class Run:
        id = RUN_ID
        summary = {"progress/completed_trials": 1}

        def scan_history(self):
            row = {
                    key.replace("trial/params/*", "trial/params/strength")
                    .replace("trial/objectives/*", "trial/objectives/deception")
                    .replace("best/*", "best/deception")
                    .replace("gpu/{slot}", "gpu/0")
                    .replace(
                        "best/*/params/*",
                        "best/valid_false_report_rate_lcb/params/strength",
                    )
                    .replace("pareto/candidate/*", "pareto/candidate/0")
                    .replace("params/*", "params/strength"): 1
                    for key in REQUIRED_WANDB_METRIC_KEYS
                }
            row["canary/resumed_session"] = 2
            return [row]

    class Api:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def run(self, path: str):
            self.paths.append(path)
            return Run()

    api = Api()
    readback = read_wandb_dashboard(api, checkpoint=_checkpoint())
    assert api.paths == [f"research/intelligent-liars/{RUN_ID}"]
    assert readback["run_id"] == RUN_ID
    assert readback["remote_run_count"] == 1
    assert readback["history_rows"] == 1
    assert readback["resumed_session_marker_seen"] is True


def test_trace_builder_combines_first_and_resumed_coordinator_snapshots() -> None:
    raw = _trace()
    init = raw["init_calls"][0]
    monitor_init = {
        "id": init["run_id"],
        "project": init["project"],
        "entity": init["entity"],
        "resume": init["resume"],
        "reinit": init["reinit"],
        "privacy_settings": {**init["settings"], "automatic_system_metrics": False},
    }

    def snapshot(logs, heartbeats):
        return {
            "format": "truth_editing_wandb_verification_snapshot_v2",
            "run_id": RUN_ID,
            "initialized_coordinator_count": 1,
            "logged_metric_keys": sorted({key for row in logs for key in row}),
            "nonfatal_error_count": 0,
            "heartbeat_lines": heartbeats,
            "privacy_controls": {
                "console_capture": False,
                "code_capture": False,
                "git_capture": False,
                "artifact_upload": False,
                "model_upload": False,
                "automatic_system_metrics": False,
            },
            "attempted_logs": [
                {"step": 0, "values": row} for row in logs
            ],
            "init_calls": [monitor_init],
            "finish_calls": 1,
            "transport_state": "drained",
            "transport_drained": True,
            "pending_event_count": 0,
            "in_flight_event_count": 0,
            "dropped_event_count": 0,
            "coalesced_telemetry_count": 0,
        }

    first_logs = [dict(row) for row in raw["log_calls"]]
    first_logs[0].pop("canary/resumed_session")
    resumed_logs = [{"canary/resumed_session": 2}]
    built = build_wandb_canary_trace(
        coordinator_snapshots=(
            snapshot(first_logs, raw["heartbeat_lines"]),
            snapshot(resumed_logs, []),
        ),
        dashboard_readback=raw["dashboard_readback"],
        failure_probe=raw["failure_probe"],
    )
    assert built["init_calls"] == raw["init_calls"]
    assert built["checkpoint_run_ids"] == [RUN_ID, RUN_ID]
    assert built["log_calls"] == [*first_logs, *resumed_logs]


def test_canary_rejects_truthful_timeout_with_unflushed_required_rows(
    tmp_path: Path,
) -> None:
    trace = _trace()
    trace["finish_calls"] = 0
    trace["transport_close_statuses"] = [
        {
            "state": "close_timed_out",
            "drained": False,
            "pending_event_count": 0,
            "in_flight_event_count": 1,
            "dropped_event_count": 0,
            "coalesced_telemetry_count": 0,
        },
        {
            "state": "close_timed_out",
            "drained": False,
            "pending_event_count": 1,
            "in_flight_event_count": 1,
            "dropped_event_count": 0,
            "coalesced_telemetry_count": 0,
        },
    ]

    with pytest.raises(WandbCanaryError, match="did not flush required rows"):
        verify_wandb_canary(
            trace=trace,
            checkpoint=_checkpoint(),
            receipt_path=tmp_path / "receipt.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pending_event_count", 999),
        ("in_flight_event_count", 1),
        ("dropped_event_count", 999),
    ],
)
def test_canary_rejects_drained_status_with_unflushed_or_dropped_rows(
    tmp_path: Path, field: str, value: int
) -> None:
    trace = _trace()
    trace["transport_close_statuses"][0][field] = value

    with pytest.raises(WandbCanaryError, match="did not flush required rows"):
        verify_wandb_canary(
            trace=trace,
            checkpoint=_checkpoint(),
            receipt_path=tmp_path / "receipt.json",
        )


def test_canary_rejects_dashboard_without_resumed_session_marker(
    tmp_path: Path,
) -> None:
    trace = _trace()
    trace["dashboard_readback"]["resumed_session_marker_seen"] = False

    with pytest.raises(WandbCanaryError, match="did not verify the real run"):
        verify_wandb_canary(
            trace=trace,
            checkpoint=_checkpoint(),
            receipt_path=tmp_path / "receipt.json",
        )


def test_failure_probe_hashes_actual_authoritative_bytes() -> None:
    authoritative_output = b'{"trials":[{"ordinal":0,"value":0.9}]}'
    authoritative_checkpoint = b'{"completed_trials":1}'
    probe = build_wandb_failure_probe(
        control_optimizer_output=authoritative_output,
        injected_optimizer_output=authoritative_output,
        control_checkpoint=authoritative_checkpoint,
        injected_checkpoint=authoritative_checkpoint,
        local_error_receipt=b'{"kind":"wandb_failure"}',
        injected_failure_count=3,
    )
    assert probe["injected_failure_count"] == 3
    assert (
        probe["control_optimizer_output_sha256"]
        == probe["failure_optimizer_output_sha256"]
    )


def test_failure_probe_rejects_changed_optimizer_or_checkpoint_bytes() -> None:
    with pytest.raises(WandbCanaryError, match="changed authoritative"):
        build_wandb_failure_probe(
            control_optimizer_output=b"same",
            injected_optimizer_output=b"changed",
            control_checkpoint=b"checkpoint",
            injected_checkpoint=b"checkpoint",
            local_error_receipt=b"receipt",
            injected_failure_count=1,
        )


def test_gate_accepts_actual_coordinator_snapshot_surface(tmp_path: Path) -> None:
    class Run:
        def log(self, _values, *, step=None, commit=None):
            del step, commit

        def finish(self, *, exit_code=0):
            pass

    class Wandb:
        @staticmethod
        def Settings(**kwargs):
            return kwargs

        @staticmethod
        def init(**kwargs):
            return Run()

    proposal = {
        "attention_edge_strength": 0.5,
        "attention_enabled": True,
        "attention_kernel_center": 21.0,
        "attention_kernel_half_width": 3.0,
        "attention_peak_strength": 1.0,
        "backend_type": "persistent_weight",
        "basis_method": "qr",
        "basis_scope": "general",
        "direction_family": "general",
        "direction_ids": ["private-direction-id"],
        "edit_arm": "truth_only",
        "mlp_edge_strength": 0.5,
        "mlp_enabled": True,
        "mlp_kernel_center": 21.0,
        "mlp_kernel_half_width": 3.0,
        "mlp_peak_strength": 1.0,
        "normalization_mode": "exact",
        "proposal_origin": "coverage_anchor",
        "refusal_direction_scope": "global",
        "refusal_enabled": False,
        "refusal_source_layer": None,
        "refusal_strength": 0.0,
        "refusal_writer_policy": "both",
        "requested_rank": 1,
        "selected_domains": [],
        "source_layer": 21,
        "strength": 1.0,
        "truth_direction_scope": "global",
        "writer_layers": [20, 21, 22],
        "writer_policy": "both",
        "writer_region": "middle",
    }
    snapshots = []
    for index in range(2):
        monitor = CoordinatorMonitor(
            run_id=RUN_ID,
            project="intelligent-liars",
            entity="research",
            run_name="canary",
            receipt_path=tmp_path / f"monitor-{index}.jsonl",
            total_trials=200,
            batch_size=8,
            wandb_module=Wandb(),
            monotonic=lambda: 100.0,
        )
        if index == 0:
            monitor.record_batch(
                0,
                (
                    BatchEvaluationRequest(
                        trial_id="trial-0000",
                        ordinal=0,
                        proposal=proposal,
                        record_ids=("private-record-id",),
                        objective_names=OBJECTIVES,
                    ),
                ),
                (EvaluationResult.successful({name: 0.5 for name in OBJECTIVES}),),
            )
            monitor.record_gpu(
                GpuTelemetryRecord(
                    gpu_slot=0,
                    utilization_percent=90.0,
                    memory_used_mib=20_000.0,
                    memory_total_mib=24_576.0,
                    tokens_per_second=40.0,
                    active_trial_id="trial-0000",
                    observed_at="2026-08-28T00:00:00Z",
                )
            )
            monitor.record_judge(calls=1, failures=0, latency_ms=100.0, cost_usd=0.01)
            monitor.record_cost(
                gpu_actual_usd=0.01,
                gpu_projected_usd=1.0,
                judge_actual_usd=0.01,
                judge_projected_usd=1.0,
            )
            monitor.record_operational(
                retries=1,
                stopped_trials=0,
                errors=1,
                error_category="worker_operational_failure",
                error_fingerprint="e" * 64,
            )
        if index == 1:
            monitor.record_resume_marker(session_ordinal=2)
        monitor.close()
        snapshots.append(monitor.verification_snapshot())

    trace = build_wandb_canary_trace(
        coordinator_snapshots=snapshots,
        dashboard_readback={
            "run_id": RUN_ID,
            "remote_run_count": 1,
            "metric_keys": sorted(
                set(snapshots[0]["logged_metric_keys"])
                | set(snapshots[1]["logged_metric_keys"])
            ),
            "history_rows": 7,
            "summary_readable": True,
            "resumed_session_marker_seen": True,
        },
        failure_probe=_trace()["failure_probe"],
    )
    receipt = verify_wandb_canary(
        trace=trace,
        checkpoint=_checkpoint(),
        receipt_path=tmp_path / "receipt.json",
    )
    assert receipt["privacy_verified"] is True
    attempted = json.dumps(snapshots[0]["attempted_logs"])
    assert "private-direction-id" not in attempted
    assert "private-record-id" not in attempted
