from __future__ import annotations

import argparse
import importlib.util
import io
import time
from pathlib import Path

import pytest

from intelligent_liars.vast_host_gate import (
    DownloadTrial,
    FailureDomain,
    HostGateThresholds,
    MachineAction,
    decide_machine_action,
    download_url_origin,
    evaluate_host_gate,
    measure_download_url,
    measure_download_stream,
    qualification_failure_domain,
    validate_protected_download_url,
    validate_public_download_url,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "qualify_vast_step5_host.py"
SPEC = importlib.util.spec_from_file_location("qualify_vast_step5_host", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFIER)


def _trial(
    *,
    speed: float = 120.0,
    ttfb: float | None = 0.2,
    stall: float = 0.3,
    completed: bool = True,
    sample_bytes: int = 64 * 1024**2,
    error: str | None = None,
) -> DownloadTrial:
    return DownloadTrial(
        bytes_downloaded=sample_bytes,
        elapsed_seconds=4.0,
        time_to_first_byte_seconds=ttfb,
        effective_mbps=speed,
        transfer_mbps=speed + 5,
        longest_stall_seconds=stall,
        stalled=stall > 10.0,
        completed=completed,
        error=error,
    )


def _thresholds() -> HostGateThresholds:
    return HostGateThresholds(
        min_download_mbps=100.0,
        min_sample_bytes=64 * 1024**2,
        max_time_to_first_byte_seconds=5.0,
        max_stall_seconds=10.0,
    )


class _Clock:
    def __init__(self, values: list[float]):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def test_measure_download_stream_records_latency_speed_and_stall():
    stream = io.BytesIO(b"x" * 12)
    clock = _Clock([1.5, 2.5, 3.0, 3.0])

    trial = measure_download_stream(
        stream,
        sample_bytes=12,
        chunk_bytes=6,
        max_stall_seconds=2.0,
        request_started_at=1.0,
        clock=clock,
    )

    assert trial.completed is True
    assert trial.bytes_downloaded == 12
    assert trial.time_to_first_byte_seconds == pytest.approx(0.5)
    assert trial.elapsed_seconds == pytest.approx(2.0)
    assert trial.longest_stall_seconds == pytest.approx(1.0)
    assert trial.stalled is False
    assert trial.effective_mbps == pytest.approx(48e-6)


def test_measure_download_stream_stops_after_stall_threshold():
    trial = measure_download_stream(
        io.BytesIO(b"x" * 12),
        sample_bytes=12,
        chunk_bytes=6,
        max_stall_seconds=0.75,
        request_started_at=1.0,
        clock=_Clock([1.5, 2.5, 3.0]),
    )

    assert trial.completed is False
    assert trial.stalled is True
    assert trial.error == "StallThresholdExceeded"


def test_partial_read_timeout_is_normalized_to_host_stall():
    class PartialTimeoutStream:
        reads = 0

        def read(self, _: int) -> bytes:
            self.reads += 1
            if self.reads == 1:
                return b"x"
            raise TimeoutError("signed-url-secret-must-not-appear")

    trial = measure_download_stream(
        PartialTimeoutStream(),
        sample_bytes=2,
        max_stall_seconds=10.0,
        request_started_at=0.0,
        clock=_Clock([1.0, 12.0, 12.0]),
    )
    gate = evaluate_host_gate(
        [trial],
        HostGateThresholds(
            min_download_mbps=1.0,
            min_sample_bytes=2,
            max_time_to_first_byte_seconds=5.0,
            max_stall_seconds=10.0,
        ),
    )

    assert trial.error == "StallThresholdExceeded"
    assert trial.stalled is True
    assert "secret" not in str(trial.to_dict())
    assert qualification_failure_domain(gate) is FailureDomain.HOST


def test_direct_stream_rejects_nan_stall_threshold():
    with pytest.raises(ValueError, match="finite and positive"):
        measure_download_stream(
            io.BytesIO(b"x"),
            sample_bytes=1,
            max_stall_seconds=float("nan"),
        )


def test_total_trial_deadline_stops_cumulative_subthreshold_reads():
    trial = measure_download_stream(
        io.BytesIO(b"x" * 4),
        sample_bytes=4,
        chunk_bytes=1,
        max_stall_seconds=1.0,
        max_elapsed_seconds=2.0,
        request_started_at=0.0,
        clock=_Clock([0.5, 1.4, 2.3, 2.3]),
    )

    assert trial.completed is False
    assert trial.error == "TrialDeadlineExceeded"
    gate = evaluate_host_gate(
        [trial],
        HostGateThresholds(
            min_download_mbps=1.0,
            min_sample_bytes=4,
            max_time_to_first_byte_seconds=5.0,
            max_stall_seconds=1.0,
        ),
    )
    assert qualification_failure_domain(gate) is FailureDomain.HOST


def test_url_trial_deadline_returns_without_waiting_for_blocked_read():
    class SlowResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def read(self, _: int) -> bytes:
            time.sleep(0.2)
            return b"x"

    started = time.monotonic()
    trial = measure_download_url(
        "https://example.test/model",
        sample_bytes=2,
        max_stall_seconds=1.0,
        timeout_seconds=0.02,
        opener=lambda *_args, **_kwargs: SlowResponse(),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert trial.completed is False
    assert trial.error == "TrialDeadlineExceeded"


def test_public_argv_url_rejects_credentials_and_persisted_origin_has_no_path():
    with pytest.raises(ValueError, match="credential-bearing"):
        validate_public_download_url("https://example.test/model?token=secret")
    with pytest.raises(ValueError, match="credential-bearing"):
        validate_public_download_url("https://user:secret@example.test/model")

    url = validate_public_download_url("https://example.test/public/model.bin")
    assert download_url_origin(url) == "https://example.test"


def test_protected_url_allows_credentials_but_requires_absolute_http_transport():
    assert validate_protected_download_url(
        "https://example.test/model?token=secret"
    ).endswith("token=secret")
    with pytest.raises(ValueError, match="absolute HTTP"):
        validate_protected_download_url("ftp://example.test/model")
    with pytest.raises(ValueError, match="absolute HTTP"):
        validate_protected_download_url("/relative/model")
    assert download_url_origin("https://[2001:db8::1]:8443/private") == (
        "https://[2001:db8::1]:8443"
    )


def test_signed_url_file_must_have_private_permissions(tmp_path: Path):
    path = tmp_path / "download-url"
    path.write_text("https://example.test/model?token=secret\n")
    args = argparse.Namespace(
        download_url=None,
        download_url_env=None,
        download_url_file=path,
    )

    path.chmod(0o644)
    with pytest.raises(ValueError, match="group/world accessible"):
        QUALIFIER._load_download_url(args)

    path.chmod(0o600)
    assert QUALIFIER._load_download_url(args).endswith("token=secret")


def test_thresholds_reject_nan_and_infinity():
    with pytest.raises(ValueError, match="finite and positive"):
        HostGateThresholds(
            min_download_mbps=float("nan"),
            min_sample_bytes=1,
            max_time_to_first_byte_seconds=1.0,
            max_stall_seconds=1.0,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        HostGateThresholds(
            min_download_mbps=1.0,
            min_sample_bytes=1,
            max_time_to_first_byte_seconds=float("inf"),
            max_stall_seconds=1.0,
        )


def test_gate_accepts_three_complete_fast_trials():
    decision = evaluate_host_gate(
        [_trial(), _trial(speed=140), _trial(speed=110)], _thresholds()
    )

    assert decision.accepted is True
    assert decision.median_effective_mbps == 120.0
    assert decision.minimum_effective_mbps == 110.0
    assert decision.reasons == ()


def test_gate_rejects_host_below_minimum_measured_download_speed():
    decision = evaluate_host_gate(
        [_trial(speed=60), _trial(speed=120), _trial(speed=140)], _thresholds()
    )

    assert decision.accepted is False
    assert decision.median_effective_mbps == 120.0
    assert decision.minimum_effective_mbps == 60.0
    assert any("below 100.00 Mbps" in reason for reason in decision.reasons)


def test_gate_rejects_latency_stall_and_incomplete_sample():
    bad = _trial(
        ttfb=8.0,
        stall=12.0,
        completed=False,
        sample_bytes=1024,
        error="TimeoutError: stalled",
    )
    decision = evaluate_host_gate([bad], _thresholds())

    assert decision.accepted is False
    assert any("incomplete" in reason for reason in decision.reasons)
    assert any("minimum sample size" in reason for reason in decision.reasons)
    assert any("time to first byte" in reason for reason in decision.reasons)
    assert any("stall reached or exceeded" in reason for reason in decision.reasons)
    assert any("TimeoutError" in reason for reason in decision.reasons)


def test_gate_rejects_one_high_latency_trial_even_when_median_is_fast():
    decision = evaluate_host_gate(
        [_trial(ttfb=0.2), _trial(ttfb=0.3), _trial(ttfb=6.0)], _thresholds()
    )

    assert decision.accepted is False
    assert any("trials: [3]" in reason for reason in decision.reasons)


def test_rejected_unused_host_can_be_replaced():
    decision = decide_machine_action(
        host_accepted=False,
        workload_started=False,
        workload_succeeded=False,
        artifacts_durable=False,
        failure_domain=FailureDomain.HOST,
        diagnosis_complete=True,
        resume_possible=False,
    )

    assert decision.action is MachineAction.DESTROY_AND_REPLACE
    assert decision.replacement_allowed is True


def test_endpoint_failure_does_not_authorize_host_replacement():
    failed_trial = _trial(
        speed=0.0,
        ttfb=None,
        completed=False,
        sample_bytes=0,
        error="HTTPError",
    )
    gate = evaluate_host_gate([failed_trial], _thresholds())
    domain = qualification_failure_domain(gate)
    decision = decide_machine_action(
        host_accepted=False,
        workload_started=False,
        workload_succeeded=False,
        artifacts_durable=False,
        failure_domain=domain,
        diagnosis_complete=False,
        resume_possible=True,
    )

    assert domain is FailureDomain.SOFTWARE
    assert decision.action is MachineAction.DIAGNOSE_AND_RESUME
    assert decision.replacement_allowed is False


def test_measured_stall_is_classified_as_host_performance():
    stalled = _trial(speed=120.0, stall=12.0, completed=False)
    stalled = DownloadTrial(**{**stalled.to_dict(), "error": "StallThresholdExceeded"})
    gate = evaluate_host_gate([stalled], _thresholds())

    assert qualification_failure_domain(gate) is FailureDomain.HOST


def test_software_failure_must_be_diagnosed_and_resumed_on_same_machine():
    decision = decide_machine_action(
        host_accepted=True,
        workload_started=True,
        workload_succeeded=False,
        artifacts_durable=True,
        failure_domain=FailureDomain.SOFTWARE,
        diagnosis_complete=False,
        resume_possible=True,
    )

    assert decision.action is MachineAction.DIAGNOSE_AND_RESUME
    assert decision.replacement_allowed is False


def test_software_failure_allows_replacement_only_after_diagnosis_rules_out_resume():
    decision = decide_machine_action(
        host_accepted=True,
        workload_started=True,
        workload_succeeded=False,
        artifacts_durable=True,
        failure_domain=FailureDomain.SOFTWARE,
        diagnosis_complete=True,
        resume_possible=False,
    )

    assert decision.action is MachineAction.DESTROY_AND_REPLACE
    assert decision.replacement_allowed is True


def test_undurable_partial_checkpoint_blocks_replacement_after_software_diagnosis():
    decision = decide_machine_action(
        host_accepted=True,
        workload_started=True,
        workload_succeeded=False,
        artifacts_durable=False,
        failure_domain=FailureDomain.SOFTWARE,
        diagnosis_complete=True,
        resume_possible=False,
    )

    assert decision.action is MachineAction.STOP_FOR_RECOVERY
    assert decision.replacement_allowed is False


def test_suspected_host_failure_must_be_diagnosed_before_replacement():
    decision = decide_machine_action(
        host_accepted=True,
        workload_started=True,
        workload_succeeded=False,
        artifacts_durable=True,
        failure_domain=FailureDomain.HOST,
        diagnosis_complete=False,
        resume_possible=True,
    )

    assert decision.action is MachineAction.DIAGNOSE_AND_RESUME
    assert decision.replacement_allowed is False


def test_missing_durable_artifacts_stops_machine_for_recovery():
    decision = decide_machine_action(
        host_accepted=True,
        workload_started=True,
        workload_succeeded=True,
        artifacts_durable=False,
        failure_domain=FailureDomain.NONE,
        diagnosis_complete=True,
        resume_possible=False,
    )

    assert decision.action is MachineAction.STOP_FOR_RECOVERY
    assert decision.replacement_allowed is False


def test_success_with_durable_artifacts_destroys_machine():
    decision = decide_machine_action(
        host_accepted=True,
        workload_started=True,
        workload_succeeded=True,
        artifacts_durable=True,
        failure_domain=FailureDomain.NONE,
        diagnosis_complete=True,
        resume_possible=False,
    )

    assert decision.action is MachineAction.DESTROY
    assert decision.replacement_allowed is False


def test_unclassified_workload_failure_is_rejected():
    with pytest.raises(ValueError, match="requires a failure domain"):
        decide_machine_action(
            host_accepted=True,
            workload_started=True,
            workload_succeeded=False,
            artifacts_durable=False,
            failure_domain=FailureDomain.NONE,
            diagnosis_complete=False,
            resume_possible=True,
        )
