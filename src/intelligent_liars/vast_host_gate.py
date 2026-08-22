"""Measured Vast host qualification and recovery-first lifecycle policy.

The network gate is intentionally independent from the Vast API.  It runs on an
already-created host, measures a bounded download, and emits a deterministic
accept/reject decision.  The lifecycle policy prevents callers from replacing a
machine merely because the workload had a software failure.
"""

from __future__ import annotations

import statistics
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, BinaryIO


MEBIBYTE = 1024**2


@dataclass(frozen=True)
class DownloadTrial:
    """One bounded download measurement, including failure diagnostics."""

    bytes_downloaded: int
    elapsed_seconds: float
    time_to_first_byte_seconds: float | None
    effective_mbps: float
    transfer_mbps: float
    longest_stall_seconds: float
    stalled: bool
    completed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HostGateThresholds:
    """Explicit thresholds used to accept or reject a host."""

    min_download_mbps: float
    min_sample_bytes: int
    max_time_to_first_byte_seconds: float
    max_stall_seconds: float

    def __post_init__(self) -> None:
        if self.min_download_mbps <= 0:
            raise ValueError("min_download_mbps must be positive")
        if self.min_sample_bytes <= 0:
            raise ValueError("min_sample_bytes must be positive")
        if self.max_time_to_first_byte_seconds <= 0:
            raise ValueError("max_time_to_first_byte_seconds must be positive")
        if self.max_stall_seconds <= 0:
            raise ValueError("max_stall_seconds must be positive")


@dataclass(frozen=True)
class HostGateDecision:
    """Aggregate decision across one or more download trials."""

    accepted: bool
    median_effective_mbps: float
    minimum_effective_mbps: float
    median_time_to_first_byte_seconds: float | None
    longest_stall_seconds: float
    reasons: tuple[str, ...]
    trials: tuple[DownloadTrial, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trials"] = [trial.to_dict() for trial in self.trials]
        return payload


def _safe_rate_mbps(byte_count: int, seconds: float) -> float:
    if byte_count <= 0 or seconds <= 0:
        return 0.0
    return byte_count * 8 / seconds / 1_000_000


def measure_download_stream(
    stream: BinaryIO,
    *,
    sample_bytes: int,
    chunk_bytes: int = MEBIBYTE,
    max_stall_seconds: float,
    request_started_at: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> DownloadTrial:
    """Measure a bounded stream without retaining its contents.

    ``request_started_at`` should be captured before connection setup so the
    effective rate and time-to-first-byte include DNS, TCP, and TLS latency.
    """

    if sample_bytes <= 0:
        raise ValueError("sample_bytes must be positive")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if max_stall_seconds <= 0:
        raise ValueError("max_stall_seconds must be positive")

    started_at = clock() if request_started_at is None else request_started_at
    previous_at = started_at
    first_byte_at: float | None = None
    longest_stall = 0.0
    downloaded = 0
    error: str | None = None

    try:
        while downloaded < sample_bytes:
            wanted = min(chunk_bytes, sample_bytes - downloaded)
            chunk = stream.read(wanted)
            observed_at = clock()
            if not chunk:
                break
            if first_byte_at is None:
                first_byte_at = observed_at
            else:
                gap = max(0.0, observed_at - previous_at)
                longest_stall = max(longest_stall, gap)
            previous_at = observed_at
            downloaded += len(chunk)
    except Exception as exc:  # Network streams expose many transport exceptions.
        observed_at = clock()
        if first_byte_at is not None:
            longest_stall = max(longest_stall, max(0.0, observed_at - previous_at))
        previous_at = observed_at
        error = f"{type(exc).__name__}: {exc}"

    ended_at = max(previous_at, clock())
    elapsed = max(0.0, ended_at - started_at)
    ttfb = None if first_byte_at is None else max(0.0, first_byte_at - started_at)
    transfer_elapsed = (
        0.0 if first_byte_at is None else max(0.0, ended_at - first_byte_at)
    )
    return DownloadTrial(
        bytes_downloaded=downloaded,
        elapsed_seconds=elapsed,
        time_to_first_byte_seconds=ttfb,
        effective_mbps=_safe_rate_mbps(downloaded, elapsed),
        transfer_mbps=_safe_rate_mbps(downloaded, transfer_elapsed),
        longest_stall_seconds=longest_stall,
        stalled=longest_stall > max_stall_seconds,
        completed=downloaded >= sample_bytes and error is None,
        error=error,
    )


def measure_download_url(
    url: str,
    *,
    sample_bytes: int,
    chunk_bytes: int = MEBIBYTE,
    max_stall_seconds: float,
    timeout_seconds: float,
    user_agent: str = "intelligent-liars-vast-host-gate/1",
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
) -> DownloadTrial:
    """Open ``url`` and measure at most ``sample_bytes`` from it."""

    started_at = clock()
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=0-{sample_bytes - 1}",
            "User-Agent": user_agent,
            "Accept-Encoding": "identity",
        },
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            return measure_download_stream(
                response,
                sample_bytes=sample_bytes,
                chunk_bytes=chunk_bytes,
                max_stall_seconds=max_stall_seconds,
                request_started_at=started_at,
                clock=clock,
            )
    except Exception as exc:
        ended_at = clock()
        return DownloadTrial(
            bytes_downloaded=0,
            elapsed_seconds=max(0.0, ended_at - started_at),
            time_to_first_byte_seconds=None,
            effective_mbps=0.0,
            transfer_mbps=0.0,
            longest_stall_seconds=max(0.0, ended_at - started_at),
            stalled=max(0.0, ended_at - started_at) > max_stall_seconds,
            completed=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_host_gate(
    trials: Iterable[DownloadTrial], thresholds: HostGateThresholds
) -> HostGateDecision:
    """Accept only when every trial clears every configured threshold."""

    frozen_trials = tuple(trials)
    if not frozen_trials:
        raise ValueError("at least one download trial is required")

    reasons: list[str] = []
    incomplete = [
        index + 1 for index, trial in enumerate(frozen_trials) if not trial.completed
    ]
    if incomplete:
        reasons.append(f"incomplete download trials: {incomplete}")

    undersized = [
        index + 1
        for index, trial in enumerate(frozen_trials)
        if trial.bytes_downloaded < thresholds.min_sample_bytes
    ]
    if undersized:
        reasons.append(f"trials below minimum sample size: {undersized}")

    speeds = [trial.effective_mbps for trial in frozen_trials]
    median_speed = float(statistics.median(speeds))
    minimum_speed = min(speeds)
    slow = [
        index + 1
        for index, trial in enumerate(frozen_trials)
        if trial.effective_mbps < thresholds.min_download_mbps
    ]
    if slow:
        reasons.append(
            f"effective download fell below {thresholds.min_download_mbps:.2f} Mbps "
            f"in trials: {slow} (minimum {minimum_speed:.2f} Mbps)"
        )

    ttfbs = [
        trial.time_to_first_byte_seconds
        for trial in frozen_trials
        if trial.time_to_first_byte_seconds is not None
    ]
    median_ttfb = float(statistics.median(ttfbs)) if ttfbs else None
    missing_ttfb = [
        index + 1
        for index, trial in enumerate(frozen_trials)
        if trial.time_to_first_byte_seconds is None
    ]
    if missing_ttfb:
        reasons.append(f"trials received no first byte: {missing_ttfb}")
    late_ttfb = [
        index + 1
        for index, trial in enumerate(frozen_trials)
        if trial.time_to_first_byte_seconds is not None
        and trial.time_to_first_byte_seconds > thresholds.max_time_to_first_byte_seconds
    ]
    if late_ttfb:
        reasons.append(
            f"time to first byte exceeded {thresholds.max_time_to_first_byte_seconds:.2f}s "
            f"in trials: {late_ttfb}"
        )

    longest_stall = max(trial.longest_stall_seconds for trial in frozen_trials)
    stalled = [
        index + 1
        for index, trial in enumerate(frozen_trials)
        if trial.stalled or trial.longest_stall_seconds > thresholds.max_stall_seconds
    ]
    if stalled:
        reasons.append(
            f"download stall exceeded {thresholds.max_stall_seconds:.2f}s "
            f"in trials: {stalled}"
        )

    errors = [
        f"trial {index + 1}: {trial.error}"
        for index, trial in enumerate(frozen_trials)
        if trial.error
    ]
    reasons.extend(errors)
    return HostGateDecision(
        accepted=not reasons,
        median_effective_mbps=median_speed,
        minimum_effective_mbps=minimum_speed,
        median_time_to_first_byte_seconds=median_ttfb,
        longest_stall_seconds=longest_stall,
        reasons=tuple(reasons),
        trials=frozen_trials,
    )


class FailureDomain(str, Enum):
    NONE = "none"
    SOFTWARE = "software"
    HOST = "host"
    TRANSFER = "transfer"
    INTERRUPTED = "interrupted"


class MachineAction(str, Enum):
    REUSE = "reuse"
    STOP_FOR_RECOVERY = "stop_for_recovery"
    DIAGNOSE_AND_RESUME = "diagnose_and_resume"
    DESTROY = "destroy"
    DESTROY_AND_REPLACE = "destroy_and_replace"


@dataclass(frozen=True)
class MachinePolicyDecision:
    action: MachineAction
    replacement_allowed: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


def decide_machine_action(
    *,
    host_accepted: bool,
    workload_started: bool,
    workload_succeeded: bool,
    artifacts_durable: bool,
    failure_domain: FailureDomain,
    diagnosis_complete: bool,
    resume_possible: bool,
) -> MachinePolicyDecision:
    """Choose a recovery-first action without treating software as host failure."""

    if workload_succeeded and failure_domain is not FailureDomain.NONE:
        raise ValueError("a successful workload cannot have a failure domain")
    if not workload_started and workload_succeeded:
        raise ValueError("workload_succeeded requires workload_started")
    if (
        workload_started
        and not workload_succeeded
        and failure_domain is FailureDomain.NONE
    ):
        raise ValueError("an unsuccessful started workload requires a failure domain")

    if not host_accepted and not workload_started:
        return MachinePolicyDecision(
            MachineAction.DESTROY_AND_REPLACE,
            True,
            "host failed qualification before workload start",
        )

    if (
        workload_started
        and not artifacts_durable
        and (
            workload_succeeded
            or failure_domain in {FailureDomain.TRANSFER, FailureDomain.INTERRUPTED}
        )
    ):
        return MachinePolicyDecision(
            MachineAction.STOP_FOR_RECOVERY,
            False,
            "machine may contain the only checkpoint or artifact copy",
        )

    if workload_succeeded:
        return MachinePolicyDecision(
            MachineAction.DESTROY,
            False,
            "workload succeeded and artifacts are durable",
        )

    if failure_domain is FailureDomain.SOFTWARE:
        if not diagnosis_complete or resume_possible:
            return MachinePolicyDecision(
                MachineAction.DIAGNOSE_AND_RESUME,
                False,
                "diagnose the software failure and resume this machine when possible",
            )
        return MachinePolicyDecision(
            MachineAction.DESTROY_AND_REPLACE,
            True,
            "diagnosis is complete and the existing machine cannot resume",
        )

    if failure_domain is FailureDomain.HOST:
        if workload_started and not artifacts_durable:
            return MachinePolicyDecision(
                MachineAction.STOP_FOR_RECOVERY,
                False,
                "recover checkpoint evidence before replacing the failed host",
            )
        return MachinePolicyDecision(
            MachineAction.DESTROY_AND_REPLACE,
            True,
            "diagnosed host failure permits replacement",
        )

    if failure_domain in {FailureDomain.TRANSFER, FailureDomain.INTERRUPTED}:
        return MachinePolicyDecision(
            MachineAction.STOP_FOR_RECOVERY,
            False,
            "preserve the existing machine until recovery is complete",
        )

    return MachinePolicyDecision(
        MachineAction.REUSE,
        False,
        "host is qualified and no terminal workload outcome was supplied",
    )
