"""Durable all-worker spend circuit for the production semantic judge.

The interface is deliberately small: open one identity-bound ledger and wrap
the existing judge transport.  Reservations, reconciliation, replay, locking,
and circuit persistence stay behind that seam.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Protocol

from .truth_editing_failure_policy import PaidJudgeCircuitOpen
from .truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256


CONFIG_FORMAT = "truth_editing_production_judge_budget_config_v1"
MANIFEST_FORMAT = "truth_editing_production_judge_budget_manifest_v1"
EVENT_FORMAT = "truth_editing_production_judge_budget_event_v1"
RECEIPT_FORMAT = "truth_editing_production_judge_budget_receipt_v1"
MAXIMUM_ALL_IN_SPEND_USD = Decimal("50")
MINIMUM_CALL_RESERVATION_USD = Decimal("0.025")
_INPUT_USD_PER_TOKEN = Decimal("0.000000075")
_OUTPUT_USD_PER_TOKEN = Decimal("0.00000025")
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_SECRET_FIELD_NAMES = frozenset(
    {"api_key", "authorization", "openrouter_api_key", "x-api-key", "cookie", "set-cookie"}
)


class ProductionJudgeBudgetError(RuntimeError):
    """The production judge budget contract or ledger is invalid."""


class ProductionJudgeBudgetCircuitOpen(ProductionJudgeBudgetError, PaidJudgeCircuitOpen):
    """No new paid judge request may cross the transport seam."""


class JudgeTransport(Protocol):
    def complete(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProductionJudgeBudgetError("judge budget value is not canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _money(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ProductionJudgeBudgetError(f"{label} must be finite nonnegative money")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ProductionJudgeBudgetError(
            f"{label} must be finite nonnegative money"
        ) from error
    if not result.is_finite() or result < 0:
        raise ProductionJudgeBudgetError(f"{label} must be finite nonnegative money")
    return result


def _money_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


@dataclass(frozen=True)
class ProductionJudgeBudgetConfig:
    all_in_maximum_spend_usd: Decimal
    non_judge_reserved_spend_usd: Decimal
    maximum_judge_spend_usd: Decimal
    per_call_reservation_usd: Decimal
    judge_config_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProductionJudgeBudgetConfig":
        if not isinstance(value, Mapping) or set(value) != {
            "format",
            "all_in_maximum_spend_usd",
            "non_judge_reserved_spend_usd",
            "maximum_judge_spend_usd",
            "per_call_reservation_usd",
            "judge_config_sha256",
        }:
            raise ProductionJudgeBudgetError("production judge budget config fields differ")
        if value["format"] != CONFIG_FORMAT:
            raise ProductionJudgeBudgetError("production judge budget config format differs")
        all_in = _money(value["all_in_maximum_spend_usd"], "all-in maximum")
        non_judge = _money(
            value["non_judge_reserved_spend_usd"], "non-judge reserved spend"
        )
        judge = _money(value["maximum_judge_spend_usd"], "maximum judge spend")
        per_call = _money(value["per_call_reservation_usd"], "per-call reservation")
        if all_in <= 0 or all_in > MAXIMUM_ALL_IN_SPEND_USD:
            raise ProductionJudgeBudgetError("all-in maximum must be in (0, 50] USD")
        if judge <= 0 or non_judge + judge > all_in:
            raise ProductionJudgeBudgetError(
                "maximum judge spend exceeds the configured all-in remainder"
            )
        if per_call < MINIMUM_CALL_RESERVATION_USD or per_call > judge:
            raise ProductionJudgeBudgetError(
                "per-call reservation must be at least 0.025 USD and within judge spend"
            )
        judge_config = value["judge_config_sha256"]
        if judge_config != FROZEN_JUDGE_CONFIG_SHA256:
            raise ProductionJudgeBudgetError(
                "production judge budget differs from the frozen judge configuration"
            )
        return cls(all_in, non_judge, judge, per_call, str(judge_config))

    def to_mapping(self) -> dict[str, str]:
        return {
            "format": CONFIG_FORMAT,
            "all_in_maximum_spend_usd": _money_text(self.all_in_maximum_spend_usd),
            "non_judge_reserved_spend_usd": _money_text(
                self.non_judge_reserved_spend_usd
            ),
            "maximum_judge_spend_usd": _money_text(self.maximum_judge_spend_usd),
            "per_call_reservation_usd": _money_text(self.per_call_reservation_usd),
            "judge_config_sha256": self.judge_config_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha(self.to_mapping())


class ProductionJudgeBudget:
    """One append-only ledger shared by every production judge worker."""

    def __init__(
        self, path: str | Path, *, config: ProductionJudgeBudgetConfig
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_dir()):
            raise ProductionJudgeBudgetError("judge budget ledger must be a real directory")
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "calls").mkdir(exist_ok=True)
        self.config = config
        with self._locked():
            self._open_manifest()

    def transport(self, downstream: JudgeTransport) -> JudgeTransport:
        ledger = self

        class BudgetedTransport:
            def complete(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
                request_copy = copy.deepcopy(dict(request))
                request_sha = _sha(request_copy)
                replay = ledger._reserve_or_replay(request_sha, request_copy)
                if replay is not None:
                    return replay
                try:
                    response = downstream.complete(request_copy)
                except Exception as error:
                    ledger._record_ambiguous(request_sha, error.__class__.__name__)
                    raise ProductionJudgeBudgetCircuitOpen(
                        "production judge transport outcome is ambiguous; circuit is open"
                    ) from error
                try:
                    return ledger._reconcile(request_sha, response)
                except ProductionJudgeBudgetCircuitOpen:
                    raise
                except Exception as error:
                    ledger._record_ambiguous(request_sha, error.__class__.__name__)
                    raise ProductionJudgeBudgetCircuitOpen(
                        "production judge response cost is ambiguous; circuit is open"
                    ) from error

        return BudgetedTransport()

    def receipt(self) -> dict[str, Any]:
        with self._locked():
            state = self._scan_state()
            circuit = self._read_optional_event(self.path / "circuit.json", "circuit")
            unsigned = {
                "format": RECEIPT_FORMAT,
                "budget_config_sha256": self.config.identity_sha256,
                "judge_config_sha256": self.config.judge_config_sha256,
                "maximum_judge_spend_usd": _money_text(
                    self.config.maximum_judge_spend_usd
                ),
                "actual_spend_usd": _money_text(state["actual"]),
                "reserved_or_spent_usd": _money_text(state["reserved"]),
                "completed_call_count": state["completed"],
                "pending_call_count": state["pending"],
                "ambiguous_call_count": state["ambiguous"],
                "circuit_open": circuit is not None,
                "circuit_event_sha256": (
                    None if circuit is None else str(circuit["content_sha256"])
                ),
            }
            return parse_production_judge_budget_receipt(
                {**unsigned, "content_sha256": _sha(unsigned)}
            )

    def monitoring_snapshot(self) -> dict[str, int | float]:
        """Return aggregate numeric telemetry without prompts or responses."""

        with self._locked():
            state = self._scan_state()
            latencies: list[float] = []
            for call_dir in sorted((self.path / "calls").iterdir()):
                completed = self._read_optional_event(
                    call_dir / "completed.json", "completed"
                )
                if completed is None:
                    continue
                response = completed.get("response")
                latency = response.get("latency_ms") if isinstance(response, Mapping) else None
                if (
                    not isinstance(latency, bool)
                    and isinstance(latency, (int, float))
                    and math.isfinite(float(latency))
                    and float(latency) >= 0
                ):
                    latencies.append(float(latency))
            return {
                "calls": int(state["completed"] + state["ambiguous"]),
                "failures": int(state["ambiguous"]),
                "latency_ms": (
                    sum(latencies) / len(latencies) if latencies else 0.0
                ),
                # Calls are made synchronously by the evaluator, so their summed
                # provider latency is the judge contribution to trial critical path.
                "elapsed_ms": sum(latencies),
                "cost_usd": float(state["actual"]),
            }

    def _reserve_or_replay(
        self, request_sha: str, request: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        estimate = _maximum_request_cost(request)
        if estimate > self.config.per_call_reservation_usd:
            raise ProductionJudgeBudgetError(
                "request exceeds the configured per-call reservation"
            )
        with self._locked():
            call_dir = self.path / "calls" / request_sha
            completed = (
                self._read_optional_event(call_dir / "completed.json", "completed")
                if call_dir.exists()
                else None
            )
            if completed is not None:
                response = completed.get("response")
                if not isinstance(response, Mapping):
                    raise ProductionJudgeBudgetError(
                        "completed judge budget event has no response"
                    )
                return copy.deepcopy(dict(response))
            ambiguous = (
                self._read_optional_event(call_dir / "ambiguous.json", "ambiguous")
                if call_dir.exists()
                else None
            )
            if ambiguous is not None:
                raise ProductionJudgeBudgetCircuitOpen(
                    "production judge request has an ambiguous prior outcome; circuit is open"
                )
            if self._read_optional_event(self.path / "circuit.json", "circuit") is not None:
                raise ProductionJudgeBudgetCircuitOpen(
                    "production judge budget circuit is open"
                )
            reservation_path = call_dir / "reservation.json"
            if reservation_path.exists():
                self._read_event(reservation_path, "reservation")
                raise ProductionJudgeBudgetCircuitOpen(
                    "production judge request is already in flight; no duplicate call was made"
                )
            state = self._scan_state()
            prospective = state["reserved"] + self.config.per_call_reservation_usd
            if prospective > self.config.maximum_judge_spend_usd:
                self._open_circuit("cap_exhausted", request_sha)
                raise ProductionJudgeBudgetCircuitOpen(
                    "production judge spend cap is exhausted; circuit is open"
                )
            call_dir.mkdir(exist_ok=False)
            unsigned = self._event(
                "reservation",
                request_sha,
                authorized_usd=_money_text(self.config.per_call_reservation_usd),
                request_sha256=request_sha,
            )
            self._write_once(reservation_path, unsigned)
            return None

    def _record_ambiguous(self, request_sha: str, error_class: str) -> None:
        with self._locked():
            call_dir = self.path / "calls" / request_sha
            self._write_once(
                call_dir / "ambiguous.json",
                self._event(
                    "ambiguous",
                    request_sha,
                    authorized_usd=_money_text(self.config.per_call_reservation_usd),
                    error_class=error_class,
                ),
            )
            self._open_circuit("ambiguous_transport", request_sha)

    def _reconcile(
        self, request_sha: str, response: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(response, Mapping):
            raise ProductionJudgeBudgetError("paid judge response must be an object")
        response_copy = copy.deepcopy(dict(response))
        if _contains_secret_field(response_copy):
            raise ProductionJudgeBudgetError("paid judge response contains a secret field")
        price = _money(response_copy.get("price_usd"), "paid judge actual price")
        with self._locked():
            call_dir = self.path / "calls" / request_sha
            self._read_event(call_dir / "reservation.json", "reservation")
            if price > self.config.per_call_reservation_usd:
                self._write_once(
                    call_dir / "overrun.json",
                    self._event(
                        "overrun",
                        request_sha,
                        actual_usd=_money_text(price),
                    ),
                )
                self._open_circuit("price_over_reservation", request_sha)
                raise ProductionJudgeBudgetCircuitOpen(
                    "production judge price exceeded its reservation; circuit is open"
                )
            self._write_once(
                call_dir / "completed.json",
                self._event(
                    "completed",
                    request_sha,
                    actual_usd=_money_text(price),
                    response=response_copy,
                ),
            )
        return response_copy

    def _scan_state(self) -> dict[str, Any]:
        actual = Decimal("0")
        reserved = Decimal("0")
        completed_count = pending_count = ambiguous_count = 0
        calls = self.path / "calls"
        for call_dir in sorted(calls.iterdir()):
            if call_dir.is_symlink() or not call_dir.is_dir():
                raise ProductionJudgeBudgetError("judge budget call entry is invalid")
            request_sha = call_dir.name
            if len(request_sha) != 64 or any(c not in "0123456789abcdef" for c in request_sha):
                raise ProductionJudgeBudgetError("judge budget call identity is invalid")
            reservation = self._read_event(call_dir / "reservation.json", "reservation")
            if reservation["request_sha256"] != request_sha:
                raise ProductionJudgeBudgetError("judge budget event identity differs")
            completed = self._read_optional_event(call_dir / "completed.json", "completed")
            ambiguous = self._read_optional_event(call_dir / "ambiguous.json", "ambiguous")
            overrun = self._read_optional_event(call_dir / "overrun.json", "overrun")
            terminal_count = sum(item is not None for item in (completed, ambiguous, overrun))
            if terminal_count > 1:
                raise ProductionJudgeBudgetError("judge budget call has conflicting outcomes")
            if completed is not None:
                amount = _money(completed.get("actual_usd"), "completed actual price")
                actual += amount
                reserved += amount
                completed_count += 1
            elif overrun is not None:
                amount = _money(overrun.get("actual_usd"), "overrun actual price")
                actual += amount
                reserved += amount
                ambiguous_count += 1
            else:
                amount = _money(reservation.get("authorized_usd"), "authorized price")
                reserved += amount
                if ambiguous is not None:
                    ambiguous_count += 1
                else:
                    pending_count += 1
        if reserved > self.config.maximum_judge_spend_usd:
            raise ProductionJudgeBudgetError("judge budget ledger exceeds its hard cap")
        return {
            "actual": actual,
            "reserved": reserved,
            "completed": completed_count,
            "pending": pending_count,
            "ambiguous": ambiguous_count,
        }

    def _open_manifest(self) -> None:
        unsigned = {
            "format": MANIFEST_FORMAT,
            "budget_config": self.config.to_mapping(),
            "budget_config_sha256": self.config.identity_sha256,
            "judge_config_sha256": self.config.judge_config_sha256,
        }
        path = self.path / "manifest.json"
        if path.exists():
            if self._read_bound(path, MANIFEST_FORMAT) != {
                **unsigned,
                "content_sha256": _sha(unsigned),
            }:
                raise ProductionJudgeBudgetError("judge budget manifest identity differs")
        else:
            self._write_payload_once(path, {**unsigned, "content_sha256": _sha(unsigned)})

    def _open_circuit(self, reason: str, request_sha: str) -> None:
        path = self.path / "circuit.json"
        if path.exists():
            self._read_event(path, "circuit")
            return
        self._write_once(path, self._event("circuit", request_sha, reason=reason))

    def _event(self, status: str, request_sha: str, **fields: Any) -> dict[str, Any]:
        return {
            "format": EVENT_FORMAT,
            "status": status,
            "budget_config_sha256": self.config.identity_sha256,
            "judge_config_sha256": self.config.judge_config_sha256,
            "request_sha256": request_sha,
            **fields,
        }

    def _read_optional_event(self, path: Path, status: str) -> dict[str, Any] | None:
        return None if not path.exists() else self._read_event(path, status)

    def _read_event(self, path: Path, status: str) -> dict[str, Any]:
        payload = self._read_bound(path, EVENT_FORMAT)
        if (
            payload.get("status") != status
            or payload.get("budget_config_sha256") != self.config.identity_sha256
            or payload.get("judge_config_sha256") != self.config.judge_config_sha256
        ):
            raise ProductionJudgeBudgetError("judge budget event identity differs")
        return payload

    @staticmethod
    def _read_bound(path: Path, format_name: str) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ProductionJudgeBudgetError("judge budget event is not a regular file")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProductionJudgeBudgetError("judge budget event is unreadable") from error
        if not isinstance(payload, dict):
            raise ProductionJudgeBudgetError("judge budget event must be an object")
        unsigned = dict(payload)
        claimed = unsigned.pop("content_sha256", None)
        if payload.get("format") != format_name or claimed != _sha(unsigned):
            raise ProductionJudgeBudgetError("judge budget event identity differs")
        return payload

    def _write_once(self, path: Path, unsigned: Mapping[str, Any]) -> None:
        self._write_payload_once(path, {**unsigned, "content_sha256": _sha(unsigned)})

    @staticmethod
    def _write_payload_once(path: Path, payload: Mapping[str, Any]) -> None:
        rendered = _canonical(payload) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != rendered:
                    raise ProductionJudgeBudgetError("judge budget event identity differs")
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        key = str(self.path.resolve())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
        with thread_lock:
            lock_path = self.path / ".ledger.lock"
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def parse_production_judge_budget_receipt(value: Any) -> dict[str, Any]:
    expected = {
        "format",
        "budget_config_sha256",
        "judge_config_sha256",
        "maximum_judge_spend_usd",
        "actual_spend_usd",
        "reserved_or_spent_usd",
        "completed_call_count",
        "pending_call_count",
        "ambiguous_call_count",
        "circuit_open",
        "circuit_event_sha256",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProductionJudgeBudgetError("production judge budget receipt fields differ")
    raw = dict(value)
    if raw["format"] != RECEIPT_FORMAT:
        raise ProductionJudgeBudgetError("production judge budget receipt format differs")
    for field in ("budget_config_sha256", "judge_config_sha256"):
        digest = raw[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ProductionJudgeBudgetError("production judge budget receipt identity differs")
    maximum = _money(raw["maximum_judge_spend_usd"], "maximum judge spend")
    actual = _money(raw["actual_spend_usd"], "actual judge spend")
    reserved = _money(raw["reserved_or_spent_usd"], "reserved judge spend")
    if actual > reserved or reserved > maximum:
        raise ProductionJudgeBudgetError("production judge budget receipt exceeds its cap")
    for field in ("completed_call_count", "pending_call_count", "ambiguous_call_count"):
        count = raw[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ProductionJudgeBudgetError("production judge budget receipt count differs")
    if not isinstance(raw["circuit_open"], bool):
        raise ProductionJudgeBudgetError("production judge budget circuit state differs")
    circuit_sha = raw["circuit_event_sha256"]
    if raw["circuit_open"] != (circuit_sha is not None):
        raise ProductionJudgeBudgetError("production judge budget circuit receipt differs")
    if circuit_sha is not None and (
        not isinstance(circuit_sha, str)
        or len(circuit_sha) != 64
        or any(c not in "0123456789abcdef" for c in circuit_sha)
    ):
        raise ProductionJudgeBudgetError("production judge budget circuit identity differs")
    unsigned = dict(raw)
    claimed = unsigned.pop("content_sha256")
    if claimed != _sha(unsigned):
        raise ProductionJudgeBudgetError("production judge budget receipt identity differs")
    return copy.deepcopy(raw)


def _maximum_request_cost(request: Mapping[str, Any]) -> Decimal:
    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ProductionJudgeBudgetError("paid judge request max_tokens differs")
    # UTF-8 bytes conservatively upper-bound text token count.
    input_tokens = len(_canonical(request))
    estimate = (
        Decimal(input_tokens) * _INPUT_USD_PER_TOKEN
        + Decimal(max_tokens) * _OUTPUT_USD_PER_TOKEN
    ) * Decimal("2")
    return max(MINIMUM_CALL_RESERVATION_USD, estimate)


def _contains_secret_field(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _SECRET_FIELD_NAMES or _contains_secret_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_secret_field(item) for item in value)
    return False


__all__ = [
    "ProductionJudgeBudget",
    "ProductionJudgeBudgetCircuitOpen",
    "ProductionJudgeBudgetConfig",
    "ProductionJudgeBudgetError",
    "parse_production_judge_budget_receipt",
]
