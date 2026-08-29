from __future__ import annotations

import json
import math
import os
import threading
import time
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
        self.commits: list[bool | None] = []
        self.finished = 0
        self.fail_log = fail_log

    @property
    def step(self) -> int:
        return len(self.logged)

    def log(
        self,
        values: dict[str, Any],
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        if self.fail_log:
            raise RuntimeError("dashboard unavailable")
        self.logged.append((values, step))
        self.commits.append(commit)

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished += 1


class _StrictMonotonicFakeRun(_FakeRun):
    """Model W&B's committed-history rule for explicit steps.

    Every call commits its row, so a later call at the same or an earlier step
    is stale and rejected.  ``step`` exposes the next server-accepted row on a
    resumed run, matching the SDK seam used by the coordinator.
    """

    def __init__(self, *, resumed_step: int = 0) -> None:
        super().__init__()
        self._next_step = resumed_step

    @property
    def step(self) -> int:
        return self._next_step

    def log(
        self,
        values: dict[str, Any],
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        committed_step = self._next_step if step is None else step
        if committed_step < self._next_step:
            raise RuntimeError("W&B rejected a stale explicit step")
        super().log(values, step=committed_step, commit=commit)
        self._next_step = committed_step + 1


class _ResumedServerCursorFakeRun(_StrictMonotonicFakeRun):
    """Model W&B resume before the local ``step`` property catches up."""

    def __init__(self, *, starting_step: int) -> None:
        super().__init__(resumed_step=starting_step)
        self.starting_step = starting_step

    @property
    def step(self) -> int:
        return 0


class _UnreadableStepFakeRun(_FakeRun):
    @property
    def step(self) -> int:
        raise RuntimeError("resumed step metadata unavailable")


class _BlockingFakeRun(_FakeRun):
    def __init__(self) -> None:
        super().__init__()
        self.log_started = threading.Event()
        self.release_log = threading.Event()

    def log(
        self,
        values: dict[str, Any],
        *,
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        self.log_started.set()
        assert self.release_log.wait(timeout=5.0)
        super().log(values, step=step, commit=commit)


class _BlockingFinishRun(_FakeRun):
    def __init__(self) -> None:
        super().__init__()
        self.finish_started = threading.Event()
        self.release_finish = threading.Event()

    def finish(self, *, exit_code: int = 0) -> None:
        self.finish_started.set()
        assert self.release_finish.wait(timeout=5.0)
        super().finish(exit_code=exit_code)


class _FakeWandb:
    def __init__(self, run: _FakeRun) -> None:
        self.run = run
        self.init_calls: list[dict[str, Any]] = []

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        return self.run


class _BlockingInitWandb(_FakeWandb):
    def __init__(self, run: _FakeRun) -> None:
        super().__init__(run)
        self.init_started = threading.Event()
        self.release_init = threading.Event()

    def init(self, **kwargs: Any) -> _FakeRun:
        self.init_calls.append(kwargs)
        self.init_started.set()
        assert self.release_init.wait(timeout=5.0)
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


def _wait_until(predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for monitoring transport")
        time.sleep(0.001)


def test_wandb_host_cost_includes_elapsed_setup_before_monitor_started() -> None:
    pump = CoordinatorTelemetryPump(
        SimpleNamespace(poll=lambda: (), last_error=None),
        SimpleNamespace(),
        host_hourly_usd=2.4,
        initial_host_elapsed_seconds=2 * 3600,
        monotonic=lambda: 100.0,
    )

    assert pump.estimated_host_cost_usd() == pytest.approx(4.8)


def test_blocked_wandb_init_log_and_finish_never_block_optimizer_callers(
    tmp_path: Path,
) -> None:
    run = _BlockingFakeRun()
    wandb = _BlockingInitWandb(run)

    started = time.monotonic()
    monitor = CoordinatorMonitor(
        run_id="nonblocking-transport",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
        transport_close_timeout_seconds=0.05,
    )
    assert time.monotonic() - started < 0.25
    assert wandb.init_started.wait(timeout=1.0)

    started = time.monotonic()
    monitor.record_judge(calls=1, failures=0, latency_ms=20.0, cost_usd=0.001)
    assert time.monotonic() - started < 0.25

    started = time.monotonic()
    monitor.close()
    assert time.monotonic() - started < 0.25
    assert run.finished == 0
    snapshot = monitor.verification_snapshot()
    assert snapshot["transport_state"] == "close_timed_out"
    assert snapshot["transport_drained"] is False
    assert snapshot["in_flight_event_count"] == 1
    receipt_rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert receipt_rows[-1]["kind"] == "wandb_close_timeout"
    assert receipt_rows[-1]["payload"]["transport_drained"] is False
    terminal_receipts = (tmp_path / "events.jsonl").read_bytes()

    # The daemon may remain stuck in an external SDK call, but optimizer
    # shutdown is bounded and no caller is held hostage by it.
    wandb.release_init.set()
    run.release_log.set()
    _wait_until(lambda: run.finished == 1)
    assert (tmp_path / "events.jsonl").read_bytes() == terminal_receipts
    assert monitor.verification_snapshot()["transport_state"] == "close_timed_out"


def test_blocked_wandb_log_does_not_block_more_metrics_or_bounded_close(
    tmp_path: Path,
) -> None:
    run = _BlockingFakeRun()
    monitor = CoordinatorMonitor(
        run_id="blocked-log",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(run),
        transport_close_timeout_seconds=0.05,
    )
    monitor.record_judge(calls=1, failures=0, latency_ms=20.0, cost_usd=0.001)
    assert run.log_started.wait(timeout=1.0)

    started = time.monotonic()
    monitor.record_judge(calls=2, failures=0, latency_ms=40.0, cost_usd=0.002)
    assert time.monotonic() - started < 0.25
    started = time.monotonic()
    monitor.close()
    assert time.monotonic() - started < 0.25
    assert run.finished == 0
    assert monitor.verification_snapshot()["transport_state"] == "close_timed_out"
    run.release_log.set()


def test_blocked_wandb_finish_keeps_close_bounded(tmp_path: Path) -> None:
    run = _BlockingFinishRun()
    monitor = CoordinatorMonitor(
        run_id="blocked-finish",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(run),
        transport_close_timeout_seconds=0.05,
    )
    _wait_until(lambda: bool(monitor.verification_snapshot()["init_calls"]))

    started = time.monotonic()
    monitor.close()
    assert time.monotonic() - started < 0.25
    assert run.finish_started.wait(timeout=1.0)
    assert monitor.verification_snapshot()["transport_state"] == "close_timed_out"
    run.release_finish.set()


def test_critical_fifo_saturation_accounts_for_drop_then_recovers(tmp_path: Path) -> None:
    run = _FakeRun()
    wandb = _BlockingInitWandb(run)
    monitor = CoordinatorMonitor(
        run_id="bounded-fifo",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
        transport_queue_capacity=1,
        transport_close_timeout_seconds=0.05,
    )
    assert wandb.init_started.wait(timeout=1.0)

    monitor.record_operational(retries=1, stopped_trials=0, errors=0)
    monitor.record_operational(retries=2, stopped_trials=0, errors=0)

    snapshot = monitor.verification_snapshot()
    assert snapshot["nonfatal_error_count"] == 1
    assert snapshot["dropped_event_count"] == 1
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    dropped = next(row for row in rows if row["kind"] == "wandb_critical_event_dropped")
    assert dropped["payload"] == {
        "reason": "critical_fifo_full",
        "dropped_event_count": 1,
    }
    wandb.release_init.set()
    _wait_until(lambda: len(run.logged) == 1)
    monitor.record_operational(retries=3, stopped_trials=0, errors=0)
    _wait_until(lambda: len(run.logged) == 2)
    monitor.close()

    assert [row["operations/retries"] for row, _step in run.logged] == [1.0, 3.0]


def test_telemetry_flood_is_coalesced_without_crowding_out_critical_rows(
    tmp_path: Path,
) -> None:
    run = _FakeRun()
    wandb = _BlockingInitWandb(run)
    monitor = CoordinatorMonitor(
        run_id="telemetry-coalescing",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
        transport_queue_capacity=4,
    )
    assert wandb.init_started.wait(timeout=1.0)
    for index in range(1000):
        monitor.record_gpu(
            GpuTelemetryRecord(
                0,
                float(index % 100),
                20_000.0,
                24_576.0,
                40.0,
                "trial-0000",
                "2026-08-28T00:00:00Z",
            )
        )
        monitor.record_judge(
            calls=index, failures=0, latency_ms=100.0, cost_usd=0.01
        )
        monitor.record_cost(gpu_actual_usd=1.0, gpu_projected_usd=2.0)
    monitor.record_batch(
        0,
        (_request(0),),
        (EvaluationResult.successful({
            "valid_false_report_rate_lcb": 0.8,
            "truth_report_dissociation_lcb": 0.7,
            "capability_preservation_lcb": 0.9,
        }),),
    )

    before = monitor.verification_snapshot()
    assert before["pending_event_count"] <= 7  # 4 critical + 3 telemetry keys
    assert before["coalesced_telemetry_count"] >= 2997
    assert before["dropped_event_count"] == 0
    wandb.release_init.set()
    monitor.close()

    logged_keys = {key for values, _step in run.logged for key in values}
    assert "trial/ordinal" in logged_keys
    assert "progress/completed_trials" in logged_keys
    assert "judge/calls" in logged_keys
    assert "cost/gpu_actual_usd" in logged_keys
    assert "gpu/0/utilization_pct" in logged_keys


def test_concurrent_monitoring_receipts_form_one_linear_hash_chain(tmp_path: Path) -> None:
    monitor = CoordinatorMonitor(
        run_id="linear-receipts",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(_FakeRun()),
    )
    threads = [
        threading.Thread(
            target=lambda: monitor.record_judge(
                calls=-1, failures=0, latency_ms=0.0, cost_usd=0.0
            )
        )
        for _ in range(32)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
    monitor.close()

    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(rows) == 34  # initialized + 32 rejected inputs + closed
    assert rows[0]["previous_receipt_sha256"] is None
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_receipt_sha256"] == previous["receipt_sha256"]


def test_reopened_monitor_restores_receipt_tail_and_continues_same_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    for _ in range(2):
        monitor = CoordinatorMonitor(
            run_id="continuous-receipts",
            project="intelligent-liars",
            entity="centipawn",
            run_name="ignored",
            receipt_path=path,
            total_trials=8,
            batch_size=8,
            wandb_module=_FakeWandb(_FakeRun()),
        )
        monitor.record_judge(calls=-1, failures=0, latency_ms=0.0, cost_usd=0.0)
        monitor.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 6
    assert rows[0]["previous_receipt_sha256"] is None
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_receipt_sha256"] == previous["receipt_sha256"]


def test_same_process_monitors_share_receipt_path_without_forking_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    monitors = [
        CoordinatorMonitor(
            run_id="shared-receipts",
            project="intelligent-liars",
            entity="centipawn",
            run_name="ignored",
            receipt_path=path,
            total_trials=8,
            batch_size=8,
            wandb_module=_FakeWandb(_FakeRun()),
        )
        for _ in range(2)
    ]
    threads = [threading.Thread(target=monitor.close) for monitor in monitors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 4
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_receipt_sha256"] == previous["receipt_sha256"]


def test_one_daemon_owns_init_log_plot_and_finish_calls(tmp_path: Path) -> None:
    owner_threads: list[int] = []

    class Run(_FakeRun):
        def log(self, values, *, step=None, commit=None):
            owner_threads.append(threading.get_ident())
            super().log(values, step=step, commit=commit)

        def finish(self, *, exit_code=0):
            owner_threads.append(threading.get_ident())
            super().finish(exit_code=exit_code)

    class Plot(_FakePlot):
        def line_series(self, **kwargs):
            owner_threads.append(threading.get_ident())
            return super().line_series(**kwargs)

    class Wandb(_FakeWandb):
        def __init__(self):
            super().__init__(Run())
            self.plot = Plot()

        def init(self, **kwargs):
            owner_threads.append(threading.get_ident())
            return super().init(**kwargs)

    wandb = Wandb()
    caller_thread = threading.get_ident()
    monitor = CoordinatorMonitor(
        run_id="one-transport-owner",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
    )
    monitor.record_batch(
        0,
        (_request(0),),
        (EvaluationResult.successful({name: 0.5 for name in (
            "valid_false_report_rate_lcb",
            "truth_report_dissociation_lcb",
            "capability_preservation_lcb",
        )}),),
    )
    monitor.close()

    assert len(owner_threads) >= 5
    assert len(set(owner_threads)) == 1
    assert owner_threads[0] != caller_thread


def test_unexpected_daemon_termination_is_stateful_and_terminally_receipted(
    tmp_path: Path,
) -> None:
    class FatalPlot(_FakePlot):
        def line_series(self, **kwargs):
            raise SystemExit("unexpected plot termination")

    wandb = _FakeWandb(_FakeRun())
    wandb.plot = FatalPlot()  # type: ignore[attr-defined]
    path = tmp_path / "events.jsonl"
    monitor = CoordinatorMonitor(
        run_id="guarded-daemon",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=path,
        total_trials=8,
        batch_size=8,
        wandb_module=wandb,
    )
    monitor.record_batch(
        0,
        (_request(0),),
        (EvaluationResult.successful({
            "valid_false_report_rate_lcb": 0.8,
            "truth_report_dissociation_lcb": 0.7,
            "capability_preservation_lcb": 0.9,
        }),),
    )
    _wait_until(
        lambda: monitor.verification_snapshot()["transport_state"] == "failed"
    )
    before_close = path.read_bytes()

    monitor.close()

    rows = [json.loads(line) for line in before_close.decode().splitlines()]
    assert rows[-1]["kind"] == "wandb_transport_terminated"
    assert rows[-1]["payload"]["transport_state"] == "failed"
    assert path.read_bytes() == before_close


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


def test_successful_trials_publish_readable_live_loss_charts(tmp_path: Path) -> None:
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
        (_request(0), _request(1)),
        (
            EvaluationResult.successful({
                "valid_false_report_rate_lcb": 0.8,
                "truth_report_dissociation_lcb": 0.6,
                "capability_preservation_lcb": 0.9,
            }),
            EvaluationResult.successful({
                "valid_false_report_rate_lcb": 0.5,
                "truth_report_dissociation_lcb": 0.5,
                "capability_preservation_lcb": 0.8,
            }),
        ),
    )

    _wait_until(lambda: len(wandb.plot.line_series_calls) == 3)
    assert len(wandb.plot.line_series_calls) == 3
    overview, components, preservation = wandb.plot.line_series_calls
    assert overview["keys"] == [
        "Trial overall loss",
        "Best overall loss so far",
    ]
    assert overview["xs"] == [[0, 1], [0, 1]]
    first_loss = 1.0 - (0.8 * 0.6 * 0.9) ** (1 / 3)
    second_loss = 1.0 - (0.5 * 0.5 * 0.8) ** (1 / 3)
    assert overview["ys"][0] == pytest.approx([first_loss, second_loss])
    assert overview["ys"][1] == pytest.approx([first_loss, first_loss])
    assert overview["title"] == "Optimization loss (live; lower is better)"

    assert components["keys"] == [
        "False-report loss",
        "Retained-truth loss",
        "Capability loss",
    ]
    assert components["xs"] == [[0, 1], [0, 1], [0, 1]]
    assert components["ys"][0] == pytest.approx([0.2, 0.5])
    assert components["ys"][1] == pytest.approx([0.4, 0.5])
    assert components["ys"][2] == pytest.approx([0.1, 0.2])
    assert components["title"] == "Loss components by trial"

    assert preservation["keys"] == ["Worst preservation KL"]
    assert preservation["xs"] == [[0, 1]]
    assert preservation["ys"][0] == pytest.approx([
        -math.log(0.9),
        -math.log(0.8),
    ])
    assert preservation["title"] == "Preservation KL by trial"

    chart_rows = [values for values, _step in run.logged if "charts/loss_overview" in values]
    assert len(chart_rows) == 1
    assert set(chart_rows[0]) == {
        "charts/loss_overview",
        "charts/loss_components",
        "charts/preservation_kl",
    }


def test_every_dashboard_row_uses_one_monotonic_coordinator_event_step(
    tmp_path: Path,
) -> None:
    run = _StrictMonotonicFakeRun(resumed_step=41)
    wandb = _FakeWandbWithPlots(run)
    monitor = CoordinatorMonitor(
        run_id="monotonic-events",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=800,
        batch_size=8,
        wandb_module=wandb,
        monotonic=lambda: 10.0,
    )

    # Logical trial ordinals deliberately have no relationship to the resumed
    # W&B history step. Trial identity belongs in metrics, not in W&B's row ID.
    monitor.record_batch(
        7,
        (_request(56), _request(57)),
        (
            EvaluationResult.successful({
                "valid_false_report_rate_lcb": 0.8,
                "truth_report_dissociation_lcb": 0.7,
                "capability_preservation_lcb": 0.9,
            }),
            EvaluationResult.successful({
                "valid_false_report_rate_lcb": 0.9,
                "truth_report_dissociation_lcb": 0.6,
                "capability_preservation_lcb": 0.8,
            }),
        ),
    )

    _wait_until(lambda: len(wandb.plot.line_series_calls) == 3)
    monitor.record_gpu(
        GpuTelemetryRecord(
            0, 91.0, 20100.0, 24576.0, 42.5, "trial-0056", "2026-08-28T00:00:00Z"
        )
    )

    _wait_until(lambda: len(run.logged) == 5)

    snapshot = monitor.verification_snapshot()
    assert snapshot["nonfatal_error_count"] == 0
    assert [step for _values, step in run.logged] == list(
        range(41, 41 + len(run.logged))
    )
    assert run.commits == [True] * len(run.logged)
    trial_rows = [values for values, _step in run.logged if "trial/ordinal" in values]
    assert [row["trial/ordinal"] for row in trial_rows] == [56.0, 57.0]
    assert any("progress/completed_trials" in values for values, _step in run.logged)
    assert any("charts/loss_overview" in values for values, _step in run.logged)


def test_resumed_server_cursor_precedes_stale_local_step(tmp_path: Path) -> None:
    run = _ResumedServerCursorFakeRun(starting_step=64)
    monitor = CoordinatorMonitor(
        run_id="server-resume-cursor",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=800,
        batch_size=8,
        wandb_module=_FakeWandb(run),
    )

    monitor.record_judge(calls=1, failures=0, latency_ms=20.0, cost_usd=0.001)
    monitor.record_operational(retries=0, stopped_trials=0, errors=0)

    _wait_until(lambda: len(run.logged) == 2)

    assert [step for _values, step in run.logged] == [64, 65]
    assert run.commits == [True, True]
    assert monitor.verification_snapshot()["nonfatal_error_count"] == 0


def test_unreadable_resumed_step_does_not_discard_initialized_run(
    tmp_path: Path,
) -> None:
    run = _UnreadableStepFakeRun()
    monitor = CoordinatorMonitor(
        run_id="step-read-failure",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(run),
    )

    monitor.record_judge(calls=1, failures=0, latency_ms=20.0, cost_usd=0.001)

    _wait_until(lambda: len(run.logged) == 1)

    snapshot = monitor.verification_snapshot()
    assert snapshot["initialized_coordinator_count"] == 1
    assert run.logged == [({
        "judge/calls": 1.0,
        "judge/failures": 0.0,
        "judge/latency_ms": 20.0,
        "judge/cost_usd": 0.001,
    }, None)]
    assert run.commits == [True]


def test_close_waits_for_inflight_log_and_blocks_later_rows(tmp_path: Path) -> None:
    run = _BlockingFakeRun()
    monitor = CoordinatorMonitor(
        run_id="close-serialization",
        project="intelligent-liars",
        entity="centipawn",
        run_name="ignored",
        receipt_path=tmp_path / "events.jsonl",
        total_trials=8,
        batch_size=8,
        wandb_module=_FakeWandb(run),
    )

    logger = threading.Thread(
        target=lambda: monitor.record_judge(
            calls=1, failures=0, latency_ms=20.0, cost_usd=0.001
        )
    )
    closer = threading.Thread(target=monitor.close)
    logger.start()
    assert run.log_started.wait(timeout=5.0)
    closer.start()
    assert closer.is_alive()
    assert run.finished == 0
    run.release_log.set()
    logger.join(timeout=5.0)
    closer.join(timeout=5.0)
    assert not logger.is_alive()
    assert not closer.is_alive()

    monitor.record_judge(calls=2, failures=0, latency_ms=40.0, cost_usd=0.002)

    assert len(run.logged) == 1
    assert run.commits == [True]
    assert run.finished == 1


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

        def complete_history_replay(self):
            order.append("history_replay_complete")

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
    wrapper.complete_history_replay()
    wrapper.observe((trial,))

    _wait_until(lambda: bool(run.logged))

    assert wrapper.identity == Driver.identity
    assert order == [
        "history_replay_complete",
        "authoritative_observe",
        "monitor",
    ]
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
    _wait_until(
        lambda: any(
            json.loads(line)["kind"] == "wandb_failure"
            for line in (root / "events.jsonl").read_text().splitlines()
        )
    )
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
