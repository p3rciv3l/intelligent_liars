from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from intelligent_liars.truth_editing_batch_execution import BatchEvaluationRequest
from intelligent_liars.truth_editing_gpu_telemetry import GpuTelemetryRecord
from intelligent_liars.truth_editing_study import EvaluationResult
from intelligent_liars.truth_editing_wandb_monitoring import (
    CoordinatorMonitor,
    CoordinatorTelemetryPump,
    MonitoredSearchDriver,
    monitoring_heartbeat,
)
from intelligent_liars.truth_editing_wandb_checkpoint import (
    AdaptiveRunProgress,
    open_adaptive_progress_checkpoint,
)


class _FakeRun:
    def __init__(self, *, fail_log: bool = False) -> None:
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.finished = 0
        self.fail_log = fail_log

    def log(self, values: dict[str, Any], *, step: int | None = None) -> None:
        if self.fail_log:
            raise RuntimeError("dashboard unavailable")
        self.logged.append((values, step))

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished += 1


class _FakeWandb:
    def __init__(self, run: _FakeRun) -> None:
        self.run = run
        self.init_calls: list[dict[str, Any]] = []

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        return self.run


class _FakePlot:
    def __init__(self) -> None:
        self.line_series_calls: list[dict[str, Any]] = []

    def line_series(self, **kwargs: Any) -> dict[str, Any]:
        self.line_series_calls.append(kwargs)
        return {"kind": "toggleable-line-series", **kwargs}


class _FakeWandbWithPlots(_FakeWandb):
    def __init__(self, run: _FakeRun) -> None:
        super().__init__(run)
        self.plot = _FakePlot()


def test_wandb_host_cost_includes_elapsed_setup_before_monitor_started() -> None:
    pump = CoordinatorTelemetryPump(
        SimpleNamespace(poll=lambda: (), last_error=None),
        SimpleNamespace(),
        host_hourly_usd=2.4,
        initial_host_elapsed_seconds=2 * 3600,
        monotonic=lambda: 100.0,
    )

    assert pump.estimated_host_cost_usd() == pytest.approx(4.8)


def _request(ordinal: int) -> BatchEvaluationRequest[dict[str, Any]]:
    return BatchEvaluationRequest(
        trial_id=f"trial-{ordinal:04d}",
        ordinal=ordinal,
        proposal={
            "strength": 0.75,
            "direction_ids": ["general-l21"],
            "prompt": "must never leave the coordinator",
            "api_key": "secret",
        },
        record_ids=("validation-1",),
        objective_names=("deception", "retained_truth"),
    )


def test_one_resumable_coordinator_run_logs_sanitized_progress_and_operations(
    tmp_path: Path,
) -> None:
    run = _FakeRun()
    wandb = _FakeWandb(run)
    monitor = CoordinatorMonitor(
        run_id="run-abc123",
        project="intelligent-liars",
        entity="research",
        run_name="truth-editing-v3",
        receipt_path=tmp_path / "monitoring.jsonl",
        total_trials=200,
        batch_size=8,
        wandb_module=wandb,
        monotonic=lambda: 100.0,
    )
    monitor.record_batch(
        0,
        (_request(0),),
        (EvaluationResult.successful({
            "valid_false_report_rate_lcb": 0.9,
            "truth_report_dissociation_lcb": 0.8,
            "capability_preservation_lcb": 0.7,
        }),),
    )
    monitor.record_gpu(
        GpuTelemetryRecord(0, 91.0, 20100.0, 24576.0, 42.5, "trial-0000", "2026-08-28T00:00:00Z")
    )
    monitor.record_judge(calls=3, failures=1, latency_ms=1200.0, cost_usd=0.03)
    monitor.record_cost(gpu_actual_usd=0.4, gpu_projected_usd=8.2)
    monitor.record_operational(retries=2, stopped_trials=1, errors=1)
    monitor.close()

    assert len(wandb.init_calls) == 1
    assert wandb.init_calls[0]["id"] == "run-abc123"
    assert wandb.init_calls[0]["resume"] == "allow"
    assert wandb.init_calls[0]["reinit"] is False
    logged = json.dumps(run.logged)
    assert "prompt" not in logged
    assert "must never leave" not in logged
    assert "api_key" not in logged
    assert "trial/params/strength" in logged
    assert "trial/params/direction_ids_sha256" in logged
    assert "general-l21" not in logged
    assert "progress/completed_trials" in logged
    assert "pareto/size" in logged
    assert "gpu/0/utilization_pct" in logged
    assert "judge/calls" in logged
    assert "cost/gpu_actual_usd" in logged
    assert "operations/retries" in logged
    assert run.finished == 1


def test_successful_trials_publish_one_toggleable_loss_chart(tmp_path: Path) -> None:
    run = _FakeRun()
    wandb = _FakeWandbWithPlots(run)
    monitor = CoordinatorMonitor(
        run_id="loss-chart",
        project="intelligent-liars",
        entity=None,
        run_name="loss-chart-test",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
        monotonic=lambda: 10.0,
    )

    monitor.record_batch(
        0,
        (_request(0),),
        (EvaluationResult.successful({
            "valid_false_report_rate_lcb": 0.8,
            "truth_report_dissociation_lcb": 0.6,
            "capability_preservation_lcb": 0.9,
        }),),
    )

    assert len(wandb.plot.line_series_calls) == 1
    chart = wandb.plot.line_series_calls[0]
    assert chart["keys"] == [
        "Overall loss",
        "False-report loss",
        "Retained-truth loss",
        "Capability loss",
        "Worst preservation KL",
    ]
    assert chart["xs"] == [[0], [0], [0], [0], [0]]
    assert [series[0] for series in chart["ys"][1:]] == pytest.approx([
        0.2,
        0.4,
        0.1,
        -math.log(0.9),
    ])
    assert chart["ys"][0][0] == pytest.approx(1.0 - (0.8 * 0.6 * 0.9) ** (1 / 3))
    assert any("charts/loss_overview" in values for values, _step in run.logged)


def test_sdk_metadata_and_job_creation_are_disabled_in_coordinator_environment(
    tmp_path: Path, monkeypatch
) -> None:
    for name in ("WANDB_SILENT", "WANDB_QUIET", "WANDB_CONSOLE"):
        monkeypatch.delenv(name, raising=False)

    class Wandb(_FakeWandb):
        def __init__(self) -> None:
            super().__init__(_FakeRun())
            self.settings_calls: list[dict[str, Any]] = []

        def Settings(self, **kwargs: Any) -> dict[str, Any]:
            self.settings_calls.append(dict(kwargs))
            return dict(kwargs)

    wandb = Wandb()
    monitor = CoordinatorMonitor(
        run_id="metadata-disabled",
        project="intelligent-liars",
        entity=None,
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
    )
    monitor.close()

    assert wandb.settings_calls == [{
        "console": "off",
        "disable_git": True,
        "disable_job_creation": True,
        "x_disable_stats": True,
        "x_disable_meta": True,
        "x_disable_machine_info": True,
    }]
    assert os.environ["WANDB_SILENT"] == "true"
    assert os.environ["WANDB_QUIET"] == "true"
    assert os.environ["WANDB_CONSOLE"] == "off"


def test_dashboard_failure_is_swallowed_and_hash_receipted(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.jsonl"
    monitor = CoordinatorMonitor(
        run_id="run-failure",
        project="intelligent-liars",
        entity=None,
        run_name="failure-test",
        receipt_path=path,
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(_FakeRun(fail_log=True)),
        monotonic=lambda: 10.0,
    )

    # Monitoring must never raise into or alter optimization.
    monitor.record_batch(
        0,
        (_request(0),),
        (EvaluationResult.successful({
            "valid_false_report_rate_lcb": 0.9,
            "truth_report_dissociation_lcb": 0.8,
            "capability_preservation_lcb": 0.7,
        }),),
    )
    monitor.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row["kind"] == "wandb_failure" for row in rows)
    assert all(len(row["receipt_sha256"]) == 64 for row in rows)
    assert rows[1]["previous_receipt_sha256"] == rows[0]["receipt_sha256"]


def test_monitoring_wrapper_preserves_search_identity_and_logs_after_observation(
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class Driver:
        identity = {"adapter": "frozen-search-v1"}

        def prepare(self, config, directions, state_path):
            pass

        def suggest(self, request):
            return request

        def observe(self, trials):
            order.append("authoritative_observe")

    run = _FakeRun()
    monitor = CoordinatorMonitor(
        run_id="run-wrapper",
        project="intelligent-liars",
        entity=None,
        run_name="wrapper-test",
        receipt_path=tmp_path / "monitoring.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(run),
        monotonic=lambda: 10.0,
    )
    original_record = monitor.record_trials

    def record(trials):
        order.append("monitor")
        original_record(trials)

    monitor.record_trials = record  # type: ignore[method-assign]
    wrapper = MonitoredSearchDriver(Driver(), monitor)
    trial = SimpleNamespace(
        trial_id="trial-0000",
        ordinal=0,
        batch_ordinal=0,
        proposal=_request(0).proposal,
        result=EvaluationResult.successful(
            {
                "valid_false_report_rate_lcb": 0.9,
                "truth_report_dissociation_lcb": 0.8,
                "capability_preservation_lcb": 0.7,
            }
        ),
    )
    wrapper.observe((trial,))

    assert wrapper.identity == Driver.identity
    assert order == ["authoritative_observe", "monitor"]
    assert [step for _values, step in run.logged if step is not None][0] == 0


def test_monitoring_wrapper_forwards_read_only_persistent_study_name_after_prepare(
    tmp_path: Path,
) -> None:
    class Driver:
        identity = {"adapter": "persistent-search-v1"}

        def __init__(self) -> None:
            self._persistent_study_name: str | None = None

        @property
        def persistent_study_name(self) -> str:
            if self._persistent_study_name is None:
                raise RuntimeError("persistent study unavailable before prepare")
            return self._persistent_study_name

        def prepare(self, config, directions, state_path):
            self._persistent_study_name = "truth-editing-study-immutable"

    monitor = CoordinatorMonitor(
        run_id="run-persistent-name",
        project="intelligent-liars",
        entity=None,
        run_name="persistent-name-test",
        receipt_path=tmp_path / "monitoring.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(_FakeRun()),
        monotonic=lambda: 10.0,
    )
    wrapper = MonitoredSearchDriver(Driver(), monitor)

    with pytest.raises(RuntimeError, match="unavailable before prepare"):
        _ = wrapper.persistent_study_name

    wrapper.prepare(object(), (), tmp_path / "study-state.json")

    assert wrapper.persistent_study_name == "truth-editing-study-immutable"
    with pytest.raises(AttributeError):
        wrapper.persistent_study_name = "replacement"  # type: ignore[misc]


def test_plain_terminal_heartbeat() -> None:
    assert monitoring_heartbeat(
        completed_trials=48,
        total_trials=200,
        current_batch=6,
        total_batches=25,
        elapsed_seconds=3600,
        eta_seconds=11400,
    ) == "48/200 trials | batch 6/25 | elapsed 1h00m | ETA 3h10m"


def _adaptive_progress(checkpoint_sha256: str) -> AdaptiveRunProgress:
    return AdaptiveRunProgress(
        wandb_run_checkpoint_sha256=checkpoint_sha256,
        study_config_sha256="d" * 64,
        planned_floor_trials=200,
        adaptive_ceiling_trials=800,
        measured_target_trials=536,
        batch_size=8,
        search_cutoff_seconds=75600,
        reserve_seconds=10800,
        total_budget_usd=50.0,
        evaluation_budget_usd=5.0,
        evaluation_budget_reserve_fraction=0.2,
        completed_search_trials=24,
        completed_repeat_trials=8,
        completed_control_trials=8,
        completed_final_selection_trials=0,
        current_batch=3,
        stage="broad_coverage",
        coverage={
            "direction_family": (6, 18),
            "layer_region": (3, 3),
            "intervention_arm": (3, 3),
            "attention_mlp_configuration": (4, 4),
            "refusal_setting": (2, 2),
            "strength_range": (4, 4),
        },
        elapsed_seconds=1200.0,
        eta_seconds=70000.0,
        gpu_actual_usd=1.25,
        gpu_projected_usd=35.0,
        judge_actual_usd=0.08,
        judge_projected_usd=4.0,
        projected_total_usd=39.0,
        measured_trial_duration_seconds=150.0,
        measured_tokens_per_second=36.0,
        measured_judge_latency_ms=900.0,
        measured_judge_cost_usd_per_trial=0.003,
    )


def test_adaptive_progress_is_checkpointed_before_allowlisted_wandb_mirror(
    tmp_path: Path,
) -> None:
    run = _FakeRun()
    monitoring_root = tmp_path / "monitoring"
    monitor = CoordinatorMonitor.open(
        checkpoint_path=monitoring_root / "wandb-run.json",
        run_id="adaptive-monitor",
        project="intelligent-liars",
        entity="centipawn",
        run_name="truth-editing-optuna-adaptive",
        receipt_path=monitoring_root / "wandb-events.jsonl",
        total_trials=800,
        batch_size=8,
        wandb_module=_FakeWandb(run),
    )
    progress = _adaptive_progress(monitor.run_checkpoint_sha256)

    checkpoint = monitor.record_adaptive_progress(progress)

    assert open_adaptive_progress_checkpoint(
        monitoring_root / "adaptive-progress.json"
    ) == checkpoint
    logged = {key for values, _step in run.logged for key in values}
    assert {
        "progress/planned_floor_trials",
        "progress/adaptive_ceiling_trials",
        "progress/measured_target_trials",
        "progress/search_cutoff_seconds",
        "progress/reserve_seconds",
        "progress/completed_repeat_trials",
        "progress/completed_control_trials",
        "progress/stage",
        "budget/evaluation_reserve_fraction",
        "coverage/direction_family/completed",
        "coverage/direction_family/required",
        "canary/trial_duration_seconds",
        "canary/tokens_per_second",
        "canary/judge_latency_ms",
        "canary/judge_cost_usd_per_trial",
    } <= logged


def test_wandb_failure_does_not_prevent_authoritative_adaptive_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "monitoring"
    monitor = CoordinatorMonitor.open(
        checkpoint_path=root / "wandb-run.json",
        run_id="adaptive-offline",
        project="intelligent-liars",
        entity=None,
        run_name="ignored",
        receipt_path=root / "events.jsonl",
        total_trials=800,
        batch_size=8,
        wandb_module=_FakeWandb(_FakeRun(fail_log=True)),
    )

    checkpoint = monitor.record_adaptive_progress(
        _adaptive_progress(monitor.run_checkpoint_sha256)
    )

    assert open_adaptive_progress_checkpoint(root / "adaptive-progress.json") == checkpoint
    assert any(
        json.loads(line)["kind"] == "wandb_failure"
        for line in (root / "events.jsonl").read_text().splitlines()
    )


def test_adaptive_resume_reuses_same_wandb_run_and_extends_progress_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "monitoring"
    first_wandb = _FakeWandb(_FakeRun())
    first = CoordinatorMonitor.open(
        checkpoint_path=root / "wandb-run.json",
        run_id="same-run-on-resume",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=root / "events.jsonl",
        total_trials=800,
        batch_size=8,
        wandb_module=first_wandb,
    )
    progress = _adaptive_progress(first.run_checkpoint_sha256)
    initial = first.record_adaptive_progress(progress)
    first.close()

    resumed_wandb = _FakeWandb(_FakeRun())
    resumed = CoordinatorMonitor.open(
        checkpoint_path=root / "wandb-run.json",
        run_id="same-run-on-resume",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=root / "events.jsonl",
        total_trials=800,
        batch_size=8,
        wandb_module=resumed_wandb,
    )
    advanced = resumed.record_adaptive_progress(
        replace(
            progress,
            completed_search_trials=32,
            current_batch=4,
            elapsed_seconds=1500.0,
            gpu_actual_usd=1.5,
            judge_actual_usd=0.1,
        )
    )

    assert first_wandb.init_calls[0]["id"] == "same-run-on-resume"
    assert resumed_wandb.init_calls[0]["id"] == "same-run-on-resume"
    assert resumed_wandb.init_calls[0]["resume"] == "allow"
    assert advanced.previous_checkpoint_sha256 == initial.checkpoint_sha256
