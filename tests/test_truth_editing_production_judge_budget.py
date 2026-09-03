from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from intelligent_liars.clients.openrouter_client import OpenRouterAPIError
from intelligent_liars.truth_editing_failure_policy import PaidJudgeCircuitOpen
from intelligent_liars.truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256
from intelligent_liars.truth_editing_production_judge_budget import (
    BILLING_RESOLUTION_EVIDENCE_FORMAT,
    ProductionJudgeBudget,
    ProductionJudgeBudgetConfig,
    ProductionJudgeBudgetError,
    ProductionJudgeRequestAmbiguous,
    TRUSTED_OPENROUTER_LOOKUP_VERIFIER_SHA256,
    TRUSTED_PRE_SEND_VERIFIER_SHA256,
    parse_production_judge_budget_receipt,
)


def _request(index: int = 0) -> dict[str, object]:
    return {
        "model": "z-ai/glm-5.3-flash",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": f"request-{index}"}],
    }


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _billing_evidence(
    request: object,
    *,
    kind: str,
    response: object | None = None,
    verifier_identity_sha256: str | None = None,
) -> dict[str, object]:
    recovered = kind == "recovered_response"
    unsigned: dict[str, object] = {
        "format": BILLING_RESOLUTION_EVIDENCE_FORMAT,
        "resolution_kind": kind,
        "canonical_request_sha256": _canonical_sha(request),
        "evidence_source": (
            "openrouter_generation_lookup"
            if recovered
            else "pre_send_transport_verifier"
        ),
        "verifier_identity_sha256": verifier_identity_sha256
        or (
            TRUSTED_OPENROUTER_LOOKUP_VERIFIER_SHA256
            if recovered
            else TRUSTED_PRE_SEND_VERIFIER_SHA256
        ),
        "verifier_receipt_sha256": "9" * 64,
        "provider_request_id": "generation-123" if recovered else None,
        "response_sha256": _canonical_sha(response) if recovered else None,
    }
    return {**unsigned, "content_sha256": _canonical_sha(unsigned)}


def _config(**overrides: object) -> ProductionJudgeBudgetConfig:
    raw: dict[str, object] = {
        "format": "truth_editing_production_judge_budget_config_v1",
        "all_in_maximum_spend_usd": "50.00",
        "non_judge_reserved_spend_usd": "49.00",
        "maximum_judge_spend_usd": "1.00",
        "per_call_reservation_usd": "0.025",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
    }
    raw.update(overrides)
    return ProductionJudgeBudgetConfig.from_mapping(raw)


def test_budget_money_serialization_preserves_integer_trailing_zeroes() -> None:
    mapping = _config(
        all_in_maximum_spend_usd="50",
        non_judge_reserved_spend_usd="45",
        maximum_judge_spend_usd="5",
        per_call_reservation_usd="0.0250",
    ).to_mapping()

    assert mapping["all_in_maximum_spend_usd"] == "50"
    assert mapping["non_judge_reserved_spend_usd"] == "45"
    assert mapping["maximum_judge_spend_usd"] == "5"
    assert mapping["per_call_reservation_usd"] == "0.025"


def test_judge_concurrency_is_bounded_and_configurable(tmp_path: Path) -> None:
    for value in (0, 9, True):
        with pytest.raises(ProductionJudgeBudgetError, match="concurrency"):
            ProductionJudgeBudget(
                tmp_path / f"invalid-{value}",
                config=_config(),
                max_concurrency=value,  # type: ignore[arg-type]
            )
    budget = ProductionJudgeBudget(
        tmp_path / "eight", config=_config(), max_concurrency=8
    )
    assert budget.max_concurrency == 8


class _Transport:
    def __init__(
        self,
        *,
        price: float = 0.01,
        latency_ms: float = 1.0,
        error: Exception | None = None,
    ) -> None:
        self.price = price
        self.latency_ms = latency_ms
        self.error = error
        self.calls = 0

    def complete(self, request: object) -> dict[str, object]:
        del request
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {
            "content": "{}",
            "model": "z-ai/glm-5.3-flash",
            "provider_route": "z-ai/fp8",
            "price_usd": self.price,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "latency_ms": self.latency_ms,
        }


def _process_budget_call(root: str, index: int, start: object, output: object) -> None:
    start.wait()
    try:
        ProductionJudgeBudget(
            root,
            config=_config(
                non_judge_reserved_spend_usd="49.95",
                maximum_judge_spend_usd="0.05",
            ),
        ).transport(_Transport(price=0.025)).complete(_request(index))
    except PaidJudgeCircuitOpen:
        output.put("blocked")
    except Exception as error:  # pragma: no cover - returned to the parent assertion
        output.put(f"error:{type(error).__name__}:{error}")
    else:
        output.put("completed")


def _process_measured_transport_call(
    root: str,
    index: int,
    start: object,
    active: object,
    maximum_active: object,
    guard: object,
    output: object,
) -> None:
    class MeasuredTransport(_Transport):
        def complete(self, request: object) -> dict[str, object]:
            with guard:
                active.value += 1
                maximum_active.value = max(maximum_active.value, active.value)
            try:
                time.sleep(0.03)
                return super().complete(request)
            finally:
                with guard:
                    active.value -= 1

    start.wait()
    try:
        ProductionJudgeBudget(root, config=_config()).transport(
            MeasuredTransport()
        ).complete(_request(index))
    except Exception as error:  # pragma: no cover - returned to the parent assertion
        output.put(f"error:{type(error).__name__}:{error}")
    else:
        output.put("completed")


def _process_crash_inside_transport(root: str, entered: object) -> None:
    class CrashTransport(_Transport):
        def complete(self, request: object) -> dict[str, object]:
            del request
            entered.set()
            os._exit(7)

    ProductionJudgeBudget(root, config=_config()).transport(CrashTransport()).complete(
        _request(0)
    )


def _process_same_budget_call(
    root: str,
    start: object,
    calls: object,
    guard: object,
    output: object,
) -> None:
    class SharedTransport(_Transport):
        def complete(self, request: object) -> dict[str, object]:
            with guard:
                calls.value += 1
            time.sleep(0.03)
            return super().complete(request)

    start.wait()
    try:
        ProductionJudgeBudget(root, config=_config()).transport(
            SharedTransport(price=0.01)
        ).complete(_request(0))
    except PaidJudgeCircuitOpen:
        output.put("blocked")
    except Exception as error:  # pragma: no cover - returned to parent assertion
        output.put(f"error:{type(error).__name__}:{error}")
    else:
        output.put("completed")


def test_reserves_before_call_reconciles_actual_and_replays_without_second_call(
    tmp_path: Path,
) -> None:
    class RecordingTransport(_Transport):
        def __init__(self) -> None:
            super().__init__(price=0.01)
            self.request: object | None = None

        def complete(self, request: object) -> dict[str, object]:
            self.request = request
            return super().complete(request)

    transport = RecordingTransport()
    budget = ProductionJudgeBudget(tmp_path / "ledger", config=_config())
    wrapped = budget.transport(transport)

    first = wrapped.complete(_request())
    second = wrapped.complete(_request())

    assert first == second
    assert transport.calls == 1
    assert transport.request == _request(0)
    receipt = parse_production_judge_budget_receipt(budget.receipt())
    assert receipt["actual_spend_usd"] == "0.01"
    assert receipt["reserved_or_spent_usd"] == "0.01"
    assert receipt["completed_call_count"] == 1
    assert receipt["pending_call_count"] == 0
    assert receipt["circuit_open"] is False
    monitoring = budget.monitoring_snapshot()
    assert monitoring["latency_ms"] == pytest.approx(1.0)
    assert monitoring["elapsed_ms"] == pytest.approx(1.0)


def test_monitoring_snapshot_sums_completed_call_latency_for_critical_path(
    tmp_path: Path,
) -> None:
    budget = ProductionJudgeBudget(tmp_path / "ledger", config=_config())
    budget.transport(_Transport(latency_ms=12.5)).complete(_request(1))
    budget.transport(_Transport(latency_ms=7.25)).complete(_request(2))

    monitoring = budget.monitoring_snapshot()
    assert monitoring["calls"] == 2
    assert monitoring["latency_ms"] == pytest.approx(9.875)
    assert monitoring["elapsed_ms"] == pytest.approx(19.75)


def test_ambiguous_transport_failure_is_durable_and_never_retried(tmp_path: Path) -> None:
    transport = _Transport(error=TimeoutError("possibly charged"))
    budget = ProductionJudgeBudget(tmp_path / "ledger", config=_config())

    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        budget.transport(transport).complete(_request())
    reopened = ProductionJudgeBudget(tmp_path / "ledger", config=_config())
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome"):
        reopened.transport(transport).complete(_request())

    assert transport.calls == 1
    receipt = parse_production_judge_budget_receipt(reopened.receipt())
    assert receipt["actual_spend_usd"] == "0"
    assert receipt["reserved_or_spent_usd"] == "0.025"
    assert receipt["ambiguous_call_count"] == 1
    assert receipt["circuit_open"] is False
    rendered = json.dumps(receipt)
    assert "possibly charged" not in rendered


def test_openrouter_failure_persists_only_safe_structured_metadata(
    tmp_path: Path,
) -> None:
    error = OpenRouterAPIError(
        "secret-message sk-or-v1-must-not-be-written",
        status_code=429,
        error_type="rate_limit_exceeded",
        provider_name="Z.AI",
        payload={
            "authorization": "Bearer must-not-be-written",
            "prompt": "private prompt body",
        },
    )
    root = tmp_path / "ledger"

    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        ProductionJudgeBudget(root, config=_config()).transport(
            _Transport(error=error)
        ).complete(_request())

    event = json.loads(next(root.glob("calls/*/ambiguous.json")).read_text())
    assert event["transport_failure"] == {
        "error_class": "OpenRouterAPIError",
        "status_code": 429,
        "error_type": "rate_limit_exceeded",
        "provider": "Z.AI",
        "retryable": True,
    }
    rendered = "".join(
        path.read_text(encoding="utf-8") for path in root.rglob("*.json")
    ).casefold()
    for forbidden in (
        "secret-message",
        "sk-or-v1-must-not-be-written",
        "authorization",
        "bearer must-not-be-written",
        "private prompt body",
    ):
        assert forbidden not in rendered


def test_timeout_failure_metadata_has_no_untrusted_text(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        ProductionJudgeBudget(root, config=_config()).transport(
            _Transport(error=TimeoutError("private timeout detail"))
        ).complete(_request())

    event = json.loads(next(root.glob("calls/*/ambiguous.json")).read_text())
    assert event["transport_failure"] == {
        "error_class": "TimeoutError",
        "status_code": None,
        "error_type": None,
        "provider": None,
        "retryable": True,
    }
    assert "private timeout detail" not in json.dumps(event)


def test_ambiguous_transport_immediately_keeps_exact_request_blocked_but_allows_fresh_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    failed = _Transport(error=TimeoutError("possibly charged"))
    budget = ProductionJudgeBudget(root, config=_config())
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        budget.transport(failed).complete(_request(1))

    assert budget.acknowledge_ambiguous_transport_circuit() is None
    assert budget.receipt()["circuit_open"] is False

    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome"):
        budget.transport(_Transport()).complete(_request(1))
    fresh = _Transport(price=0.01)
    budget.transport(fresh).complete(_request(2))
    assert failed.calls == 1
    assert fresh.calls == 1
    assert budget.receipt()["reserved_or_spent_usd"] == "0.035"

    assert budget.acknowledge_ambiguous_transport_circuit() is None


def test_ambiguous_request_needs_external_unbilled_evidence_for_one_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    failed = _Transport(error=TimeoutError("possibly charged"))
    budget = ProductionJudgeBudget(root, config=_config())

    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        budget.transport(failed).complete(_request(1))
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome|open"):
        budget.transport(_Transport()).complete(_request(1))
    distinct = _Transport(price=0.01)
    budget.transport(distinct).complete(_request(2))
    assert distinct.calls == 1

    first = budget.reconcile_ambiguous_requests()
    second = budget.reconcile_ambiguous_requests()

    assert first == ()
    assert second == ()

    still_blocked = _Transport()
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome"):
        budget.transport(still_blocked).complete(_request(1))
    assert still_blocked.calls == 0

    evidence = _billing_evidence(_request(1), kind="definitely_unbilled")
    resolution = budget.authorize_definitely_unbilled_retry(
        _request(1), evidence=evidence
    )
    assert (
        budget.authorize_definitely_unbilled_retry(
            _request(1), evidence=evidence
        )
        == resolution
    )
    assert resolution["status"] == "ambiguity_resolution"
    assert resolution["resolution_kind"] == "definitely_unbilled"
    retry_sha = resolution["retry_request_sha256"]
    assert retry_sha != resolution["canonical_request_sha256"]

    retry = _Transport(price=0.01)
    first_response = budget.transport(retry).complete(_request(1))
    second_response = budget.transport(retry).complete(_request(1))
    independent = _Transport(price=0.01)
    budget.transport(independent).complete(_request(3))

    assert first_response == second_response
    assert retry.calls == 1
    assert independent.calls == 1
    assert (root / "calls" / retry_sha / "completed.json").is_file()
    assert next(root.glob("calls/*/ambiguous.json")).is_file()
    assert len(tuple(root.glob("calls/*/ambiguity-resolution.json"))) == 1
    assert budget.receipt()["reserved_or_spent_usd"] == "0.03"


def test_recovered_provider_result_resolves_exact_ambiguity_without_rebilling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    budget = ProductionJudgeBudget(root, config=_config())
    failed = _Transport(error=TimeoutError("possibly charged"))
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        budget.transport(failed).complete(_request(3))
    budget.reconcile_ambiguous_requests()

    response = _Transport(price=0.01, latency_ms=7.0).complete(_request(3))
    evidence = _billing_evidence(
        _request(3), kind="recovered_response", response=response
    )
    first = budget.record_recovered_ambiguous_response(
        _request(3), response, evidence=evidence
    )
    second = budget.record_recovered_ambiguous_response(
        _request(3), response, evidence=evidence
    )

    replay_transport = _Transport()
    assert budget.transport(replay_transport).complete(_request(3)) == response
    assert replay_transport.calls == 0
    assert first == second
    assert first["resolution_kind"] == "recovered_response"
    receipt = budget.receipt()
    assert receipt["actual_spend_usd"] == "0.01"
    assert receipt["reserved_or_spent_usd"] == "0.01"
    assert receipt["completed_call_count"] == 1
    assert receipt["ambiguous_call_count"] == 0

    with pytest.raises(ProductionJudgeBudgetError, match="identity differs"):
        budget.record_recovered_ambiguous_response(
            _request(3),
            response,
            evidence={**evidence, "verifier_receipt_sha256": "8" * 64},
        )


def test_billing_resolution_evidence_rejects_cross_wired_or_untrusted_proof(
    tmp_path: Path,
) -> None:
    budget = ProductionJudgeBudget(tmp_path / "ledger", config=_config())
    with pytest.raises(ProductionJudgeRequestAmbiguous):
        budget.transport(_Transport(error=TimeoutError())).complete(_request(6))
    budget.reconcile_ambiguous_requests()

    with pytest.raises(ProductionJudgeBudgetError, match="cross-wired"):
        budget.authorize_definitely_unbilled_retry(
            _request(6),
            evidence=_billing_evidence(_request(7), kind="definitely_unbilled"),
        )
    with pytest.raises(ProductionJudgeBudgetError, match="not trusted"):
        budget.authorize_definitely_unbilled_retry(
            _request(6),
            evidence=_billing_evidence(
                _request(6),
                kind="definitely_unbilled",
                verifier_identity_sha256="f" * 64,
            ),
        )

    response = _Transport(price=0.01).complete(_request(6))
    different_response = {**response, "content": '{"different":true}'}
    with pytest.raises(ProductionJudgeBudgetError, match="differs from billing evidence"):
        budget.record_recovered_ambiguous_response(
            _request(6),
            different_response,
            evidence=_billing_evidence(
                _request(6), kind="recovered_response", response=response
            ),
        )


def test_non_ambiguous_budget_circuit_cannot_be_acknowledged(tmp_path: Path) -> None:
    budget = ProductionJudgeBudget(
        tmp_path / "ledger",
        config=_config(
            non_judge_reserved_spend_usd="49.975",
            maximum_judge_spend_usd="0.025",
        ),
    )
    budget.transport(_Transport(price=0.025)).complete(_request(1))
    with pytest.raises(PaidJudgeCircuitOpen, match="exhausted"):
        budget.transport(_Transport()).complete(_request(2))

    with pytest.raises(ProductionJudgeBudgetError, match="cannot be acknowledged"):
        budget.acknowledge_ambiguous_transport_circuit()
    assert budget.receipt()["circuit_open"] is True


def test_price_over_reservation_is_a_restart_durable_terminal_circuit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    budget = ProductionJudgeBudget(root, config=_config())
    with pytest.raises(PaidJudgeCircuitOpen, match="price exceeded"):
        budget.transport(_Transport(price=0.03)).complete(_request(4))

    reopened = ProductionJudgeBudget(root, config=_config())
    blocked = _Transport()
    with pytest.raises(PaidJudgeCircuitOpen, match="circuit is open"):
        reopened.transport(blocked).complete(_request(5))
    with pytest.raises(ProductionJudgeBudgetError, match="cannot be reconciled"):
        reopened.reconcile_ambiguous_requests()

    assert blocked.calls == 0
    circuit = json.loads((root / "circuit.json").read_text())
    assert circuit["reason"] == "price_over_reservation"
    assert next(root.glob("calls/*/overrun.json")).is_file()
    receipt = reopened.receipt()
    assert receipt["actual_spend_usd"] == "0.03"
    assert receipt["reserved_or_spent_usd"] == "0.03"
    assert receipt["circuit_open"] is True


def test_eight_workers_share_one_cap_and_open_one_durable_circuit(tmp_path: Path) -> None:
    config = _config(
        non_judge_reserved_spend_usd="49.95",
        maximum_judge_spend_usd="0.05",
    )
    barrier = threading.Barrier(8)
    transports = [_Transport(price=0.025) for _ in range(8)]

    def call(index: int) -> str:
        barrier.wait()
        try:
            ProductionJudgeBudget(tmp_path / "ledger", config=config).transport(
                transports[index]
            ).complete(_request(index))
        except PaidJudgeCircuitOpen:
            return "blocked"
        return "completed"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(call, range(8)))

    assert outcomes.count("completed") <= 2
    assert sum(item.calls for item in transports) <= 2
    receipt = parse_production_judge_budget_receipt(
        ProductionJudgeBudget(tmp_path / "ledger", config=config).receipt()
    )
    assert receipt["reserved_or_spent_usd"] == "0.05"
    assert receipt["circuit_open"] is True


def test_eight_process_workers_cannot_double_spend_the_same_exact_request(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    calls = context.Value("i", 0)
    guard = context.Lock()
    output = context.Queue()
    root = str(tmp_path / "same-request-ledger")
    processes = [
        context.Process(
            target=_process_same_budget_call,
            args=(root, start, calls, guard, output),
        )
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert calls.value == 1
    assert outcomes == ["completed"] * 8
    assert not any(item.startswith("error:") for item in outcomes)
    receipt = ProductionJudgeBudget(root, config=_config()).receipt()
    assert receipt["completed_call_count"] == 1
    assert receipt["reserved_or_spent_usd"] == "0.01"
    assert len(tuple((Path(root) / "calls").iterdir())) == 1


def test_paid_transport_gate_allows_four_threads_in_flight_per_ledger(
    tmp_path: Path,
) -> None:
    class MeasuringTransport(_Transport):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.maximum_active = 0
            self.lock = threading.Lock()

        def complete(self, request: object) -> dict[str, object]:
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                time.sleep(0.02)
                return super().complete(request)
            finally:
                with self.lock:
                    self.active -= 1

    root = tmp_path / "ledger"
    transport = MeasuringTransport()
    barrier = threading.Barrier(8)

    def call(index: int) -> None:
        barrier.wait()
        ProductionJudgeBudget(root, config=_config()).transport(transport).complete(
            _request(index)
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(call, range(8)))

    assert transport.calls == 8
    assert transport.maximum_active == 4


def test_distinct_paid_transport_waiters_proceed_after_request_local_ambiguity(
    tmp_path: Path,
) -> None:
    class FirstCallFails(_Transport):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def complete(self, request: object) -> dict[str, object]:
            self.calls += 1
            self.entered.set()
            assert self.release.wait(timeout=5)
            raise TimeoutError("private possibly charged detail")

    root = tmp_path / "ledger"
    transport = FirstCallFails()

    def call(index: int) -> str:
        try:
            ProductionJudgeBudget(
                root, config=_config(), max_concurrency=1
            ).transport(transport).complete(
                _request(index)
            )
        except ProductionJudgeRequestAmbiguous:
            return "blocked"
        return "completed"

    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(call, 0)
        assert transport.entered.wait(timeout=5)
        waiters = [pool.submit(call, index) for index in range(1, 8)]
        # The single transport slot keeps waiters outside during the first
        # in-flight call. Once its ambiguous result is durable, each distinct
        # request proceeds in turn; request-local ambiguity cannot poison them.
        time.sleep(0.05)
        transport.release.set()
        outcomes = [first.result(timeout=5)] + [
            waiter.result(timeout=5) for waiter in waiters
        ]

    assert outcomes == ["blocked"] * 8
    assert transport.calls == 8
    receipt = parse_production_judge_budget_receipt(
        ProductionJudgeBudget(root, config=_config()).receipt()
    )
    assert receipt["ambiguous_call_count"] == 8
    assert receipt["reserved_or_spent_usd"] == "0.2"
    assert receipt["circuit_open"] is False


def test_paid_transport_gate_allows_four_processes_in_flight(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    active = context.Value("i", 0)
    maximum_active = context.Value("i", 0)
    guard = context.Lock()
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_measured_transport_call,
            args=(
                str(tmp_path / "ledger"),
                index,
                start,
                active,
                maximum_active,
                guard,
                output,
            ),
        )
        for index in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert outcomes == ["completed"] * 4
    assert maximum_active.value == 4


def test_paid_transport_gate_is_released_when_worker_crashes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    root = tmp_path / "ledger"
    entered = context.Event()
    process = context.Process(
        target=_process_crash_inside_transport,
        args=(str(root), entered),
    )
    process.start()
    assert entered.wait(timeout=5)
    process.join(timeout=5)
    assert process.exitcode == 7

    # The exact crashed request remains conservatively reserved, but the OS
    # releases the transport gate and a distinct request can still complete.
    result = ProductionJudgeBudget(root, config=_config()).transport(
        _Transport()
    ).complete(_request(1))
    assert result["price_usd"] == 0.01


def test_orphaned_reservation_stays_blocked_until_explicit_reconciliation(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    root = tmp_path / "ledger"
    entered = context.Event()
    process = context.Process(
        target=_process_crash_inside_transport,
        args=(str(root), entered),
    )
    process.start()
    assert entered.wait(timeout=5)
    process.join(timeout=5)
    assert process.exitcode == 7

    budget = ProductionJudgeBudget(root, config=_config())
    before_recovery = _Transport()
    with pytest.raises(PaidJudgeCircuitOpen, match="already in flight"):
        budget.transport(before_recovery).complete(_request(0))
    assert before_recovery.calls == 0

    recovery = budget.recover_orphaned_reservations()

    before_reconciliation = _Transport()
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome"):
        budget.transport(before_reconciliation).complete(_request(0))
    assert before_reconciliation.calls == 0
    assert budget.reconcile_ambiguous_requests() == ()
    still_blocked = _Transport()
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome"):
        budget.transport(still_blocked).complete(_request(0))
    assert still_blocked.calls == 0

    resolution = budget.authorize_definitely_unbilled_retry(
        _request(0),
        evidence=_billing_evidence(_request(0), kind="definitely_unbilled"),
    )

    class RecordingRetryTransport(_Transport):
        def __init__(self) -> None:
            super().__init__(price=0.01)
            self.request: object | None = None

        def complete(self, request: object) -> dict[str, object]:
            self.request = request
            return super().complete(request)

    transport = RecordingRetryTransport()
    first = budget.transport(transport).complete(_request(0))
    reopened = ProductionJudgeBudget(root, config=_config())
    assert reopened.recover_orphaned_reservations() == ()
    assert reopened.reconcile_ambiguous_requests() == ()
    second = reopened.transport(transport).complete(_request(0))

    assert len(recovery) == 1
    assert recovery[0]["status"] == "ambiguous"
    assert recovery[0]["ambiguity_origin"] == "orphaned_reservation"
    assert resolution["status"] == "ambiguity_resolution"
    assert resolution["resolution_kind"] == "definitely_unbilled"
    assert first == second
    assert transport.calls == 1
    assert transport.request == _request(0)
    receipt = reopened.receipt()
    assert receipt["actual_spend_usd"] == "0.01"
    assert receipt["reserved_or_spent_usd"] == "0.01"
    assert receipt["completed_call_count"] == 1
    assert receipt["ambiguous_call_count"] == 0
    assert receipt["pending_call_count"] == 0
    assert not (root / "circuit.json").exists()

    original = next(root.glob("calls/*/ambiguity-resolution.json"))
    event = json.loads(original.read_text())
    assert event["canonical_request_sha256"] == original.parent.name
    retry_dir = root / "calls" / event["retry_request_sha256"]
    assert (retry_dir / "completed.json").exists()
    assert (original.parent / "reservation.json").exists()
    assert (original.parent / "ambiguous.json").exists()
    assert not tuple(root.glob("calls/*/orphan-recovery.json"))


def test_ambiguous_recovery_attempt_never_derives_a_second_paid_retry(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    root = tmp_path / "ledger"

    budget = ProductionJudgeBudget(root, config=_config())
    for expected_attempt in (0, 1):
        entered = context.Event()
        process = context.Process(
            target=_process_crash_inside_transport,
            args=(str(root), entered),
        )
        process.start()
        assert entered.wait(timeout=5)
        process.join(timeout=5)
        assert process.exitcode == 7

        recovered = budget.recover_orphaned_reservations()
        assert len(recovered) == 1
        assert recovered[0]["attempt_number"] == expected_attempt
        assert budget.reconcile_ambiguous_requests() == ()
        if expected_attempt == 0:
            resolution = budget.authorize_definitely_unbilled_retry(
                _request(0),
                evidence=_billing_evidence(_request(0), kind="definitely_unbilled"),
            )

    blocked = _Transport(price=0.01)
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous prior outcome"):
        budget.transport(blocked).complete(_request(0))

    assert blocked.calls == 0
    assert (
        budget.authorize_definitely_unbilled_retry(
            _request(0),
            evidence=_billing_evidence(_request(0), kind="definitely_unbilled"),
        )
        == resolution
    )
    receipt = budget.receipt()
    assert receipt["actual_spend_usd"] == "0"
    assert receipt["reserved_or_spent_usd"] == "0.025"
    assert receipt["ambiguous_call_count"] == 1
    assert receipt["completed_call_count"] == 0
    assert len(tuple(root.glob("calls/*/ambiguity-resolution.json"))) == 1


def test_ambiguity_resolution_identity_fails_closed_on_tampering(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("fork")
    root = tmp_path / "ledger"
    entered = context.Event()
    process = context.Process(
        target=_process_crash_inside_transport,
        args=(str(root), entered),
    )
    process.start()
    assert entered.wait(timeout=5)
    process.join(timeout=5)
    assert process.exitcode == 7
    budget = ProductionJudgeBudget(root, config=_config())
    budget.recover_orphaned_reservations()
    budget.reconcile_ambiguous_requests()
    budget.authorize_definitely_unbilled_retry(
        _request(0),
        evidence=_billing_evidence(_request(0), kind="definitely_unbilled"),
    )

    recovery_path = next(root.glob("calls/*/ambiguity-resolution.json"))
    recovery = json.loads(recovery_path.read_text())
    recovery["retry_request_sha256"] = "a" * 64
    unsigned = dict(recovery)
    unsigned.pop("content_sha256")
    recovery["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    recovery_path.write_text(
        json.dumps(recovery, separators=(",", ":"), sort_keys=True) + "\n"
    )

    with pytest.raises(ProductionJudgeBudgetError, match="retry identity"):
        budget.transport(_Transport()).complete(_request(0))


def test_eight_process_workers_share_the_same_locked_ledger(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_budget_call,
            args=(str(tmp_path / "ledger"), index, start, output),
        )
        for index in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert outcomes.count("completed") <= 2
    assert not any(item.startswith("error:") for item in outcomes)
    receipt = ProductionJudgeBudget(
        tmp_path / "ledger",
        config=_config(
            non_judge_reserved_spend_usd="49.95",
            maximum_judge_spend_usd="0.05",
        ),
    ).receipt()
    assert receipt["reserved_or_spent_usd"] == "0.05"
    assert receipt["circuit_open"] is True


def test_config_is_bound_to_frozen_judge_and_fifty_dollar_all_in_remainder() -> None:
    with pytest.raises(ProductionJudgeBudgetError, match="remainder"):
        _config(maximum_judge_spend_usd="1.01")
    with pytest.raises(ProductionJudgeBudgetError, match="50"):
        _config(all_in_maximum_spend_usd="50.01")
    with pytest.raises(ProductionJudgeBudgetError, match="judge configuration"):
        _config(judge_config_sha256="a" * 64)


def test_ledger_identity_and_events_fail_closed_on_tampering(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    budget = ProductionJudgeBudget(root, config=_config())
    budget.transport(_Transport()).complete(_request())
    completed = next(root.glob("calls/*/completed.json"))
    raw = json.loads(completed.read_text())
    raw["actual_usd"] = "0"
    completed.write_text(json.dumps(raw))

    with pytest.raises(ProductionJudgeBudgetError, match="identity"):
        ProductionJudgeBudget(root, config=_config()).receipt()


def test_provider_response_with_secret_shaped_field_is_not_persisted(tmp_path: Path) -> None:
    class SecretTransport(_Transport):
        def complete(self, request: object) -> dict[str, object]:
            response = super().complete(request)
            response["authorization"] = "Bearer must-not-be-written"
            return response

    root = tmp_path / "ledger"
    with pytest.raises(ProductionJudgeRequestAmbiguous, match="ambiguous"):
        ProductionJudgeBudget(root, config=_config()).transport(
            SecretTransport()
        ).complete(_request())

    assert "must-not-be-written" not in "".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.json")
    )
