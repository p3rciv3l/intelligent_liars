from __future__ import annotations

import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_failure_policy import PaidJudgeCircuitOpen
from intelligent_liars.truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256
from intelligent_liars.truth_editing_production_judge_budget import (
    ProductionJudgeBudget,
    ProductionJudgeBudgetConfig,
    ProductionJudgeBudgetError,
    parse_production_judge_budget_receipt,
)


def _request(index: int = 0) -> dict[str, object]:
    return {
        "model": "z-ai/glm-5.3-flash",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": f"request-{index}"}],
    }


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


def test_reserves_before_call_reconciles_actual_and_replays_without_second_call(
    tmp_path: Path,
) -> None:
    transport = _Transport(price=0.01)
    budget = ProductionJudgeBudget(tmp_path / "ledger", config=_config())
    wrapped = budget.transport(transport)

    first = wrapped.complete(_request())
    second = wrapped.complete(_request())

    assert first == second
    assert transport.calls == 1
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

    with pytest.raises(PaidJudgeCircuitOpen, match="ambiguous"):
        budget.transport(transport).complete(_request())
    reopened = ProductionJudgeBudget(tmp_path / "ledger", config=_config())
    with pytest.raises(PaidJudgeCircuitOpen, match="open"):
        reopened.transport(transport).complete(_request())

    assert transport.calls == 1
    receipt = parse_production_judge_budget_receipt(reopened.receipt())
    assert receipt["actual_spend_usd"] == "0"
    assert receipt["reserved_or_spent_usd"] == "0.025"
    assert receipt["ambiguous_call_count"] == 1
    assert receipt["circuit_open"] is True
    rendered = json.dumps(receipt)
    assert "possibly charged" not in rendered


def test_acknowledged_ambiguous_transport_keeps_exact_request_blocked_but_allows_fresh_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    failed = _Transport(error=TimeoutError("possibly charged"))
    budget = ProductionJudgeBudget(root, config=_config())
    with pytest.raises(PaidJudgeCircuitOpen, match="ambiguous"):
        budget.transport(failed).complete(_request(1))

    resolution = budget.acknowledge_ambiguous_transport_circuit()
    assert resolution["status"] == "circuit_resolution"
    assert resolution["reason"] == "ambiguous_transport_accounted_at_reservation"
    assert budget.receipt()["circuit_open"] is False

    with pytest.raises(PaidJudgeCircuitOpen, match="ambiguous prior outcome"):
        budget.transport(_Transport()).complete(_request(1))
    fresh = _Transport(price=0.01)
    budget.transport(fresh).complete(_request(2))
    assert failed.calls == 1
    assert fresh.calls == 1
    assert budget.receipt()["reserved_or_spent_usd"] == "0.035"

    assert budget.acknowledge_ambiguous_transport_circuit() is None


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
    with pytest.raises(PaidJudgeCircuitOpen, match="ambiguous"):
        ProductionJudgeBudget(root, config=_config()).transport(
            SecretTransport()
        ).complete(_request())

    assert "must-not-be-written" not in "".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.json")
    )
