"""Non-fatal, bounded GPU telemetry for the persistent CUDA fleet.

The coordinator owns this collector.  It combines read-only ``nvidia-smi``
samples with scheduler-owned trial and token-progress state, then exposes plain
records to any monitor.  It deliberately has no dependency on W&B and never
raises GPU-query failures into the optimization path.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone


GpuQuery = Callable[[], str]


@dataclass(frozen=True)
class GpuTelemetryRecord:
    """One coordinator-ready observation for one physical CUDA slot."""

    gpu_slot: int
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    tokens_per_second: float | None
    active_trial_id: str | None
    observed_at: str

    def to_mapping(self) -> dict[str, int | float | str | None]:
        return {
            "active_trial_id": self.active_trial_id,
            "gpu_slot": self.gpu_slot,
            "memory_total_mib": self.memory_total_mib,
            "memory_used_mib": self.memory_used_mib,
            "observed_at": self.observed_at,
            "tokens_per_second": self.tokens_per_second,
            "utilization_percent": self.utilization_percent,
        }


@dataclass
class _TrialState:
    trial_id: str | None = None
    tokens_per_second: float | None = None
    last_total_tokens: int = 0
    last_observed_monotonic: float | None = None


@dataclass(frozen=True)
class GpuTelemetryDiagnostic:
    """Safe operational diagnostic suitable for external monitoring."""

    category: str
    fingerprint_sha256: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "category": self.category,
            "fingerprint_sha256": self.fingerprint_sha256,
        }


class GpuTelemetryCollector:
    """Rate-limited GPU sampler plus authoritative CUDA-slot trial mapping.

    ``poll`` is safe to call more frequently than ``poll_interval_seconds``;
    calls inside the interval return the exact previous immutable snapshot.
    A missing binary, timeout, driver error, or malformed response returns an
    empty tuple and records ``last_error`` instead of disrupting the study.
    """

    def __init__(
        self,
        *,
        gpu_slots: int,
        poll_interval_seconds: float = 5.0,
        query_timeout_seconds: float = 3.0,
        query: GpuQuery | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        if isinstance(gpu_slots, bool) or not isinstance(gpu_slots, int) or not 1 <= gpu_slots <= 32:
            raise ValueError("gpu_slots must be an integer from 1 through 32")
        if not _finite_between(poll_interval_seconds, 1.0, 60.0):
            raise ValueError("poll_interval_seconds must be from 1 through 60")
        if not _finite_between(query_timeout_seconds, 0.1, 10.0):
            raise ValueError("query_timeout_seconds must be from 0.1 through 10")
        self.gpu_slots = gpu_slots
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.query_timeout_seconds = float(query_timeout_seconds)
        self._monotonic = monotonic
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._query = query or self._query_nvidia_smi
        self._states = [_TrialState() for _ in range(gpu_slots)]
        self._state_lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._next_poll_at: float | None = None
        self._last_snapshot: tuple[GpuTelemetryRecord, ...] = ()
        self._last_error: GpuTelemetryDiagnostic | None = None

    @property
    def last_error(self) -> GpuTelemetryDiagnostic | None:
        with self._poll_lock:
            return self._last_error

    def begin_trial(self, gpu_slot: int, trial_id: str) -> None:
        slot = self._validate_slot(gpu_slot)
        normalized_id = _trial_id(trial_id)
        with self._state_lock:
            state = self._states[slot]
            state.trial_id = normalized_id
            state.tokens_per_second = None
            state.last_total_tokens = 0
            state.last_observed_monotonic = self._monotonic()

    def record_token_progress(
        self,
        gpu_slot: int,
        trial_id: str,
        *,
        total_generated_tokens: int,
    ) -> None:
        """Update live TPS from a trial's monotonically increasing token count."""

        slot = self._validate_slot(gpu_slot)
        normalized_id = _trial_id(trial_id)
        if (
            isinstance(total_generated_tokens, bool)
            or not isinstance(total_generated_tokens, int)
            or total_generated_tokens < 0
        ):
            raise ValueError("total_generated_tokens must be a nonnegative integer")
        observed = self._monotonic()
        with self._state_lock:
            state = self._states[slot]
            if state.trial_id != normalized_id:
                raise ValueError("token progress does not match the active trial")
            if total_generated_tokens < state.last_total_tokens:
                raise ValueError("total_generated_tokens must not decrease")
            previous_at = state.last_observed_monotonic
            elapsed = observed - previous_at if previous_at is not None else 0.0
            if elapsed > 0:
                state.tokens_per_second = (
                    total_generated_tokens - state.last_total_tokens
                ) / elapsed
            state.last_total_tokens = total_generated_tokens
            state.last_observed_monotonic = observed

    def set_tokens_per_second(
        self, gpu_slot: int, trial_id: str, tokens_per_second: float
    ) -> None:
        """Accept an already measured worker TPS value when counters are unavailable."""

        slot = self._validate_slot(gpu_slot)
        normalized_id = _trial_id(trial_id)
        if not _finite_between(tokens_per_second, 0.0, math.inf):
            raise ValueError("tokens_per_second must be a finite nonnegative number")
        with self._state_lock:
            state = self._states[slot]
            if state.trial_id != normalized_id:
                raise ValueError("TPS does not match the active trial")
            state.tokens_per_second = float(tokens_per_second)

    def end_trial(self, gpu_slot: int, trial_id: str) -> None:
        slot = self._validate_slot(gpu_slot)
        normalized_id = _trial_id(trial_id)
        with self._state_lock:
            state = self._states[slot]
            if state.trial_id != normalized_id:
                raise ValueError("trial completion does not match the active trial")
            self._states[slot] = _TrialState()

    def snapshot_scheduler_state(self) -> Mapping[int, tuple[str | None, float | None]]:
        with self._state_lock:
            return {
                slot: (state.trial_id, state.tokens_per_second)
                for slot, state in enumerate(self._states)
            }

    def poll(self) -> tuple[GpuTelemetryRecord, ...]:
        """Return a fresh or rate-limited immutable telemetry snapshot."""

        now = self._monotonic()
        with self._poll_lock:
            if self._next_poll_at is not None and now < self._next_poll_at:
                return self._last_snapshot
            self._next_poll_at = now + self.poll_interval_seconds
            try:
                raw = self._query()
                observed_at = _utc_iso(self._utcnow())
                gpu_rows = self._parse_gpu_rows(raw)
                scheduler = self.snapshot_scheduler_state()
                snapshot = tuple(
                    GpuTelemetryRecord(
                        gpu_slot=slot,
                        utilization_percent=utilization,
                        memory_used_mib=memory_used,
                        memory_total_mib=memory_total,
                        tokens_per_second=scheduler[slot][1],
                        active_trial_id=scheduler[slot][0],
                        observed_at=observed_at,
                    )
                    for slot, utilization, memory_used, memory_total in gpu_rows
                )
            except Exception as error:  # monitoring is explicitly non-fatal
                self._last_snapshot = ()
                self._last_error = _diagnostic(error)
                return self._last_snapshot
            self._last_snapshot = snapshot
            self._last_error = None
            return snapshot

    def _validate_slot(self, gpu_slot: int) -> int:
        if (
            isinstance(gpu_slot, bool)
            or not isinstance(gpu_slot, int)
            or not 0 <= gpu_slot < self.gpu_slots
        ):
            raise ValueError("CUDA slot is outside the configured GPU range")
        return gpu_slot

    def _query_nvidia_smi(self) -> str:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.query_timeout_seconds,
        )
        return completed.stdout

    def _parse_gpu_rows(
        self, raw: str
    ) -> tuple[tuple[int, float, float, float], ...]:
        if not isinstance(raw, str):
            raise ValueError("nvidia-smi output must be text")
        rows: dict[int, tuple[int, float, float, float]] = {}
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = tuple(part.strip() for part in line.split(","))
            if len(parts) != 4:
                raise ValueError("nvidia-smi row must contain exactly four fields")
            try:
                slot = int(parts[0])
                utilization = float(parts[1])
                memory_used = float(parts[2])
                memory_total = float(parts[3])
            except ValueError as error:
                raise ValueError("nvidia-smi row contains a nonnumeric metric") from error
            if slot in rows or not 0 <= slot < self.gpu_slots:
                raise ValueError("nvidia-smi must return exactly one row per configured GPU slot")
            if not _finite_between(utilization, 0.0, 100.0):
                raise ValueError("GPU utilization is outside 0 through 100")
            if (
                not _finite_between(memory_used, 0.0, math.inf)
                or not _finite_between(memory_total, 0.0, math.inf)
                or memory_total <= 0
                or memory_used > memory_total
            ):
                raise ValueError("GPU memory metrics are invalid")
            rows[slot] = (slot, utilization, memory_used, memory_total)
        if set(rows) != set(range(self.gpu_slots)):
            raise ValueError("nvidia-smi must return exactly one row per configured GPU slot")
        return tuple(rows[slot] for slot in range(self.gpu_slots))


def _finite_between(value: object, minimum: float, maximum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and minimum <= float(value) <= maximum
    )


def _trial_id(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("trial_id must be a nonempty trimmed string")
    if len(value) != 10 or not value.startswith("trial-") or not value[6:].isdigit():
        raise ValueError("trial_id must be the safe identifier trial-####")
    return value


def _utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("telemetry timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _diagnostic(error: Exception) -> GpuTelemetryDiagnostic:
    if isinstance(error, FileNotFoundError):
        category = "missing_tool"
    elif isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        category = "query_timeout"
    elif isinstance(error, (ValueError, TypeError)):
        category = "malformed_output"
    else:
        category = "query_failed"
    # Raw exception text can contain paths, commands, or secrets.  The external
    # fingerprint intentionally identifies only the bounded failure class.
    private_detail = f"{category}:{type(error).__name__}".encode("ascii")
    return GpuTelemetryDiagnostic(
        category=category,
        fingerprint_sha256=hashlib.sha256(private_detail).hexdigest(),
    )


__all__ = [
    "GpuTelemetryCollector",
    "GpuTelemetryDiagnostic",
    "GpuTelemetryRecord",
]
