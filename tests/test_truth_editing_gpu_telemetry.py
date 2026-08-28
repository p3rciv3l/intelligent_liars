from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intelligent_liars.truth_editing_gpu_telemetry import GpuTelemetryCollector


class _Clock:
    def __init__(self) -> None:
        self.monotonic_seconds = 100.0

    def monotonic(self) -> float:
        return self.monotonic_seconds

    def utcnow(self) -> datetime:
        return datetime(2026, 8, 28, 8, 30, tzinfo=timezone.utc)


def test_collector_returns_normalized_gpu_records_with_scheduler_state() -> None:
    clock = _Clock()
    collector = GpuTelemetryCollector(
        gpu_slots=2,
        query=lambda: "0, 87, 12000, 24576\n1, 4, 1024, 24576\n",
        monotonic=clock.monotonic,
        utcnow=clock.utcnow,
    )
    collector.begin_trial(0, "trial-0042")
    clock.monotonic_seconds += 2.0
    collector.record_token_progress(0, "trial-0042", total_generated_tokens=50)

    records = collector.poll()

    assert [record.gpu_slot for record in records] == [0, 1]
    assert records[0].to_mapping() == {
        "active_trial_id": "trial-0042",
        "gpu_slot": 0,
        "memory_total_mib": 24576.0,
        "memory_used_mib": 12000.0,
        "observed_at": "2026-08-28T08:30:00+00:00",
        "tokens_per_second": 25.0,
        "utilization_percent": 87.0,
    }
    assert records[1].active_trial_id is None
    assert records[1].tokens_per_second is None
    assert collector.last_error is None


def test_polling_is_rate_limited_and_reuses_the_last_clean_snapshot() -> None:
    clock = _Clock()
    calls = 0

    def query() -> str:
        nonlocal calls
        calls += 1
        return "0, 10, 100, 1000\n"

    collector = GpuTelemetryCollector(
        gpu_slots=1,
        poll_interval_seconds=5.0,
        query=query,
        monotonic=clock.monotonic,
        utcnow=clock.utcnow,
    )
    first = collector.poll()
    clock.monotonic_seconds += 4.9
    assert collector.poll() is first
    assert calls == 1

    clock.monotonic_seconds += 0.1
    assert collector.poll() is not first
    assert calls == 2


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("nvidia-smi is unavailable"),
        TimeoutError("nvidia-smi timed out"),
        RuntimeError("driver query failed"),
    ],
)
def test_gpu_query_failure_is_nonfatal_and_exposes_a_sanitized_diagnostic(
    failure: Exception,
) -> None:
    def query() -> str:
        raise failure

    collector = GpuTelemetryCollector(gpu_slots=8, query=query)

    assert collector.poll() == ()
    diagnostic = collector.last_error
    assert diagnostic is not None
    assert diagnostic.category in {"missing_tool", "query_timeout", "query_failed"}
    assert len(diagnostic.fingerprint_sha256) == 64
    assert str(failure) not in str(diagnostic.to_mapping())


def test_malformed_or_incomplete_gpu_output_is_discarded_instead_of_misreported() -> None:
    collector = GpuTelemetryCollector(
        gpu_slots=2,
        query=lambda: "0, 10, 100, 1000\n0, 20, 200, 1000\n",
    )

    assert collector.poll() == ()
    assert collector.last_error is not None
    assert collector.last_error.category == "malformed_output"


def test_trial_progress_is_slot_scoped_and_stale_updates_cannot_cross_trials() -> None:
    clock = _Clock()
    collector = GpuTelemetryCollector(
        gpu_slots=2,
        query=lambda: "0, 10, 100, 1000\n1, 20, 200, 1000\n",
        monotonic=clock.monotonic,
    )
    collector.begin_trial(1, "trial-0001")
    clock.monotonic_seconds += 4.0
    collector.record_token_progress(1, "trial-0001", total_generated_tokens=80)
    assert collector.snapshot_scheduler_state()[1] == ("trial-0001", 20.0)

    collector.begin_trial(1, "trial-0002")
    collector.set_tokens_per_second(1, "trial-0002", 12.5)
    assert collector.snapshot_scheduler_state()[1] == ("trial-0002", 12.5)
    with pytest.raises(ValueError, match="active trial"):
        collector.record_token_progress(1, "trial-0001", total_generated_tokens=100)
    collector.end_trial(1, "trial-0002")
    assert collector.snapshot_scheduler_state()[1] == (None, None)


def test_invalid_metrics_are_rejected_before_they_reach_the_monitor() -> None:
    collector = GpuTelemetryCollector(gpu_slots=1, query=lambda: "")

    with pytest.raises(ValueError, match="CUDA slot"):
        collector.begin_trial(1, "trial-0001")
    with pytest.raises(ValueError, match="trimmed"):
        collector.begin_trial(0, " trial-0001 ")
    with pytest.raises(ValueError, match="safe identifier"):
        collector.begin_trial(0, "question: secret prompt text")
    collector.begin_trial(0, "trial-0001")
    with pytest.raises(ValueError, match="nonnegative integer"):
        collector.record_token_progress(0, "trial-0001", total_generated_tokens=-1)


def test_error_fingerprint_cannot_reveal_raw_error_text() -> None:
    errors = iter(
        [RuntimeError("secret-one"), RuntimeError("different-secret-two")]
    )

    def query() -> str:
        raise next(errors)

    clock = _Clock()
    collector = GpuTelemetryCollector(
        gpu_slots=1,
        poll_interval_seconds=1.0,
        query=query,
        monotonic=clock.monotonic,
    )
    collector.poll()
    first = collector.last_error
    clock.monotonic_seconds += 1.0
    collector.poll()
    second = collector.last_error

    assert first == second
    assert "secret" not in str(first.to_mapping())
