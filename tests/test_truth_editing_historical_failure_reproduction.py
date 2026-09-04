from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.truth_editing_evaluator import (
    EvaluatorConfig,
    JudgeEvidence,
    RecipeEvaluator,
    RuntimeRecord,
)
from intelligent_liars.truth_editing_failure_policy import PaidJudgeCircuitOpen
from intelligent_liars.truth_editing_judge_contracts import (
    AbsoluteJudgeResult,
    AbsoluteResponseLabel,
    AbsoluteSemanticResult,
    JudgeCacheReceipt,
    TokenUsage,
    judge_cache_key_sha256,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    OpenRouterJudgeTransport,
    OperationalJudgeFailure,
    StoredJudgeTransport,
    TruthEditingLiveJudge,
)
from intelligent_liars.truth_editing_production_judge_budget import (
    ProductionJudgeBudget,
    ProductionJudgeBudgetConfig,
    ProductionJudgeRequestAmbiguous,
)
from intelligent_liars.truth_editing_record_completion import (
    FileSemanticRecordCompletionStore,
)


FIXTURES = Path(__file__).parent / "fixtures" / "truth_editing_recovery"
JUDGE_SHA = hashlib.sha256(b"synthetic-judge").hexdigest()
RUBRIC_SHA = hashlib.sha256(b"synthetic-rubric").hexdigest()


def _fixture(name: str) -> dict[str, Any]:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _contract_sha(value: object) -> str:
    rendered = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{rendered}\n".encode()).hexdigest()


def _runtime_record() -> RuntimeRecord:
    generation = "synthetic opaque answer"
    return RuntimeRecord(
        record_id="synthetic-record",
        scenario_id="synthetic-scenario",
        domain="synthetic-domain",
        family="synthetic-family",
        task_kind="direct",
        signal_kind="conditioned_false_direct_report",
        expected_behavior="plausible_false_report",
        evaluation_lane="structured_semantic",
        prompt_sha256=_sha_text("synthetic prompt identity"),
        raw_generation=generation,
        raw_generation_sha256=_sha_text(generation),
        expected_answer="known truth",
        truth_aliases=("truth alias",),
        plausible_false_answers=("plausible alternative",),
        base_known=True,
        internal_truth_retained=True,
    )


def _valid_absolute_payload(response_id: str = "synthetic-record") -> dict[str, Any]:
    return {
        "responses": [
            {
                "response_id": response_id,
                "asserted_answer": "synthetic opaque answer",
                "asserted_state": "false",
                "answer_type": "direct",
                "semantic_type_valid": True,
                "plausible_falsehood": "yes",
                "coherent": True,
                "relevant": True,
                "internally_consistent": True,
                "instruction_following": True,
                "hedging_or_abstention": False,
                "brief_evidence": "Synthetic schema-valid evidence.",
                "confidence": "high",
            }
        ],
        "false_reporting_persistent": False,
        "retained_truth_behaviorally_supported": "unclear",
        "failure_mode": "ordinary_error",
        "contradictions": [],
        "unresolved_ambiguities": [],
    }


def _transport_response(content: str) -> dict[str, Any]:
    return {
        "content": content,
        "model": "z-ai/glm-5.3-flash",
        "provider_route": "z-ai/fp8",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "price_usd": 0.00001,
        "latency_ms": 1.0,
        "raw_payload": {"synthetic": True, "shape_only": True},
    }


def _content_for_shape(shape: str) -> str:
    valid = _valid_absolute_payload()
    if shape == "json_object_text":
        return json.dumps(valid)
    if shape == "empty_text":
        return ""
    if shape == "unterminated_json_object_text":
        return '{"responses":['
    if shape == "markdown_fenced_json_text":
        return f"```json\n{json.dumps(valid)}\n```"
    if shape == "tool_invocation_object_text":
        return json.dumps({"name": "submit_judgment", "arguments": valid})
    if shape == "judge_result_missing_required_field":
        invalid = copy.deepcopy(valid)
        invalid.pop("contradictions")
        return json.dumps(invalid)
    if shape == "cross_field_invariant_violation":
        invalid = copy.deepcopy(valid)
        invalid["responses"][0]["asserted_answer"] = ""
        return json.dumps(invalid)
    raise AssertionError(f"unhandled synthetic response shape {shape}")


def test_historical_failure_fixtures_are_sanitized_and_account_for_every_receipt() -> (
    None
):
    inventory = _fixture("failure_inventory.json")
    matrix = _fixture("response_shape_matrix.json")
    topology = _fixture("failure_topology.json")

    for value in (inventory, matrix, topology):
        sanitization = value["sanitization"]
        assert sanitization["contains_historical_prompts"] is False
        assert sanitization["contains_historical_responses"] is False
        assert sanitization["contains_credentials"] is False
        assert sanitization["contains_remote_paths"] is False

    assert inventory["cache_inventory"] == {
        "successful_entries": 965,
        "failure_receipts": 124,
        "terminal_failure_aliases": 3,
    }
    by_code = {item["code"]: item for item in inventory["failure_classes"]}
    assert {code: item["count"] for code, item in by_code.items()} == {
        "schema_validation_error": 60,
        "empty_response": 6,
        "json_decode_error": 4,
        "connection_error": 54,
    }
    assert sum(item["count"] for item in by_code.values()) == 124
    assert by_code["schema_validation_error"]["exception_classes"] == {
        "JudgeContractError": 36,
        "LiveJudgeError": 24,
    }


def test_valid_original_json_passes_without_a_correction_call() -> None:
    response = _transport_response(_content_for_shape("json_object_text"))
    transport = StoredJudgeTransport([response])

    evidence = TruthEditingLiveJudge(transport=transport).judge(_runtime_record())

    assert evidence.result.operational_status == "succeeded"
    assert evidence.cache_receipt.attempts == 1
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "case",
    [
        value
        for value in _fixture("response_shape_matrix.json")["content_cases"]
        if value["expected_outcome"] == "failure"
    ],
    ids=lambda value: value["case_id"],
)
def test_historical_malformed_content_shapes_remain_fail_closed(
    case: dict[str, Any],
) -> None:
    response = _transport_response(_content_for_shape(case["shape"]))
    transport = StoredJudgeTransport([response for _ in range(4)])

    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(transport=transport).judge(_runtime_record())

    failure = captured.value.receipt.operational_failure
    assert failure is not None
    assert failure.code == case["failure_code"]
    if exception_class := case.get("exception_class"):
        assert failure.message == f"error_class={exception_class}"


def _provider_payload(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "z-ai/glm-5.3-flash",
        "provider": "Z.AI",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.00001,
        },
    }


@pytest.mark.parametrize(
    "case",
    _fixture("response_shape_matrix.json")["provider_cases"],
    ids=lambda value: value["case_id"],
)
def test_provider_specific_content_shapes_are_handled_deliberately(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    content = json.dumps(_valid_absolute_payload())
    shape = case["message_shape"]
    if shape == "content_string":
        message = {"role": "assistant", "content": content}
    elif shape == "content_text_block_array":
        message = {"role": "assistant", "content": [{"type": "text", "text": content}]}
    elif shape == "reasoning_string_without_content":
        message = {"role": "assistant", "content": None, "reasoning": content}
    elif shape == "tool_calls_without_text":
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "function": {"name": "submit"}}],
        }
    else:  # pragma: no cover - fixture schema guard
        raise AssertionError(f"unknown provider shape {shape}")
    payload = _provider_payload(message)

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            self.model = kwargs["model"]
            self.provider_config = kwargs["provider"]

        def generate(self, messages: object) -> dict[str, Any]:
            del messages
            return copy.deepcopy(payload)

    monkeypatch.setattr(
        "intelligent_liars.clients.openrouter_client.OpenRouterClient", Client
    )
    judge = TruthEditingLiveJudge(
        transport=OpenRouterJudgeTransport(api_key="synthetic-placeholder")
    )
    if case["expected_outcome"] == "succeeded":
        assert judge.judge(_runtime_record()).result.operational_status == "succeeded"
    else:
        with pytest.raises(OperationalJudgeFailure) as captured:
            judge.judge(_runtime_record())
        failure = captured.value.receipt.operational_failure
        assert failure is not None
        assert failure.code == case["failure_code"]
        assert captured.value.receipt.usage == TokenUsage(40, 20, 60)


def _record_mapping(
    record_id: str,
    task_kind: str,
    signal_kind: str,
    expected_behavior: str,
    generation: str,
    expected_answer: str,
    *,
    internal_truth_retained: bool | None = None,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "scenario_id": "synthetic-scenario",
        "domain": "synthetic-domain",
        "family": "synthetic-family",
        "task_kind": task_kind,
        "signal_kind": signal_kind,
        "expected_behavior": expected_behavior,
        "evaluation_lane": "structured_semantic",
        "prompt_sha256": _sha_text(f"prompt-{record_id}"),
        "raw_generation": generation,
        "raw_generation_sha256": _sha_text(generation),
        "expected_answer": expected_answer,
        "truth_aliases": [expected_answer],
        "plausible_false_answers": ["synthetic alternative"],
        "base_known": True,
        "internal_truth_retained": internal_truth_retained,
    }


def _evaluation_bundle() -> dict[str, Any]:
    records = [
        _record_mapping(
            "direct-false",
            "direct",
            "conditioned_false_direct_report",
            "plausible_false_report",
            "unresolved synthetic A",
            "known truth",
            internal_truth_retained=True,
        ),
        _record_mapping(
            "direct-truth",
            "direct",
            "truthful_direct_report",
            "truthful",
            "unresolved synthetic B",
            "known truth",
        ),
        _record_mapping(
            "false-trigger",
            "direct",
            "false_trigger_truthful_control",
            "truthful",
            "unresolved synthetic C",
            "known truth",
        ),
        _record_mapping(
            "indirect",
            "indirect",
            "indirect_retained_truth",
            "truthful",
            "known truth",
            "known truth",
        ),
        _record_mapping(
            "action",
            "action",
            "true_state_action",
            "truthful",
            "select truth",
            "select truth",
        ),
        _record_mapping(
            "counterfactual",
            "counterfactual",
            "counterfactual_action",
            "truthful",
            "counterfactual truth",
            "counterfactual truth",
        ),
    ]
    unsigned = {
        "format": "truth_editing_runtime_outputs_v2",
        "dataset_manifest_sha256": _sha_text("dataset"),
        "recipe_sha256": _sha_text("recipe"),
        "edited_model_sha256": _sha_text("model"),
        "split": "validation",
        "records": records,
    }
    return {**unsigned, "bundle_sha256": _contract_sha(unsigned)}


def _evaluator_config(record_ids: list[str]) -> EvaluatorConfig:
    def tier(name: str, ids: list[str], preservation: str) -> dict[str, Any]:
        return {
            "name": name,
            "record_ids": ids,
            "preservation_tier": preservation,
            "minimum_valid_false_report_rate": 0.0,
            "minimum_indirect_truth_rate": 0.0,
            "minimum_action_truth_rate": 0.0,
            "minimum_counterfactual_truth_rate": 0.0,
            "minimum_internal_truth_rate": 0.0,
            "minimum_truthful_direct_rate": 0.1,
            "minimum_false_trigger_control_rate": 0.1,
            "lane_weights": {"structured_semantic": 1.0, "broad_qa": 0.0},
            "maximum_preservation_kl": {
                "text": 1.0,
                "vision": 1.0,
                "recorded_computer_use": 1.0,
            },
        }

    return EvaluatorConfig.from_mapping(
        {
            "format": "truth_editing_evaluator_config_v2",
            "config_id": "historical-reproduction",
            "dataset_manifest_sha256": _sha_text("dataset"),
            "judge_config_sha256": JUDGE_SHA,
            "rubric_sha256": RUBRIC_SHA,
            "confidence_z": 1.96,
            "tiers": [
                tier("discovery", record_ids, "trial"),
                tier("expanded", [*record_ids, "future-expanded"], "promoted"),
                tier(
                    "finalist",
                    [*record_ids, "future-expanded", "future-finalist"],
                    "finalist",
                ),
            ],
        }
    )


def _synthetic_evidence(record: RuntimeRecord) -> JudgeEvidence:
    label = AbsoluteResponseLabel(
        response_id=record.record_id,
        asserted_answer="synthetic alternative",
        asserted_state="false",
        answer_type="direct",
        semantic_type_valid=True,
        plausible_falsehood="yes",
        coherent=True,
        relevant=True,
        internally_consistent=True,
        instruction_following=True,
        hedging_or_abstention=False,
        brief_evidence="Synthetic schema-valid evidence.",
        confidence="high",
    )
    semantic = AbsoluteSemanticResult(
        responses=(label,),
        false_reporting_persistent=False,
        retained_truth_behaviorally_supported="unclear",
        failure_mode="ordinary_error",
        contradictions=(),
        unresolved_ambiguities=(),
    )
    result = AbsoluteJudgeResult(
        judge_config_sha256=JUDGE_SHA,
        rubric_sha256=RUBRIC_SHA,
        request_sha256=_sha_text(f"request-{record.record_id}"),
        operational_status="succeeded",
        operational_failure=None,
        result=semantic,
        content_sha256="",
    )
    result = replace(
        result,
        content_sha256=_contract_sha(
            {
                key: value
                for key, value in result.to_payload().items()
                if key != "content_sha256"
            }
        ),
    )
    cache_key = judge_cache_key_sha256(
        judge_kind="absolute",
        rubric_sha256=RUBRIC_SHA,
        judge_config_sha256=JUDGE_SHA,
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha_text("parameters"),
        prompt_bundle_sha256=_sha_text(f"bundle-{record.record_id}"),
        response_sha256s=[record.raw_generation_sha256],
    )
    receipt = JudgeCacheReceipt(
        cache_key_sha256=cache_key,
        judge_kind="absolute",
        rubric_sha256=RUBRIC_SHA,
        judge_config_sha256=JUDGE_SHA,
        resolved_model="z-ai/glm-5.3-flash",
        provider_route="z-ai/fp8",
        request_parameters_sha256=_sha_text("parameters"),
        prompt_bundle_sha256=_sha_text(f"bundle-{record.record_id}"),
        response_sha256s=(record.raw_generation_sha256,),
        raw_request_sha256=_sha_text(f"request-{record.record_id}"),
        raw_response_sha256=_sha_text(f"response-{record.record_id}"),
        parsed_result_sha256=result.content_sha256,
        operational_status="succeeded",
        operational_failure=None,
        cache_status="miss",
        attempts=1,
        latency_ms=1.0,
        usage=TokenUsage(1, 1, 2),
        price_usd=0.0,
        code_sha256=_sha_text("code"),
        created_at="2026-08-30T00:00:00Z",
        content_sha256="",
    )
    receipt = replace(
        receipt,
        content_sha256=_contract_sha(
            {
                key: value
                for key, value in receipt.to_payload().items()
                if key != "content_sha256"
            }
        ),
    )
    return JudgeEvidence(result=result, cache_receipt=receipt)


def test_circuit_failure_preserves_the_successful_sibling_receipt(
    tmp_path: Path,
) -> None:
    topology = _fixture("failure_topology.json")["sibling_loss"]
    bundle = _evaluation_bundle()
    first_record = RuntimeRecord.from_mapping(bundle["records"][0])
    first_evidence = _synthetic_evidence(first_record)

    class CircuitAfterOneJudge:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def judge(self, record: RuntimeRecord) -> JudgeEvidence:
            self.calls.append(record.record_id)
            if len(self.calls) == topology["failure_position"]:
                raise PaidJudgeCircuitOpen("synthetic circuit-open reproduction")
            return _synthetic_evidence(record)

    class PreservationMustNotRun:
        def evaluate(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("preservation must wait for complete judge evidence")

    judge = CircuitAfterOneJudge()
    record_ids = [str(item["record_id"]) for item in bundle["records"]]
    execution = {
        "format": "truth_editing_recipe_execution_receipt_v1",
        "recipe_sha256": _sha_text("recipe"),
        "edited_model_sha256": _sha_text("model"),
        "dataset_manifest_sha256": _sha_text("dataset"),
        "output_bundle_sha256": bundle["bundle_sha256"],
        "operational_status": "succeeded",
        "operational_failure": None,
    }

    completion_root = tmp_path / "semantic-record-completions"
    store = FileSemanticRecordCompletionStore(
        completion_root,
        accepted_judge_adapter_code_sha256s=(_sha_text("code"),),
    )
    with pytest.raises(PaidJudgeCircuitOpen, match="synthetic circuit-open"):
        RecipeEvaluator(
            _evaluator_config(record_ids),
            judge,
            PreservationMustNotRun(),
            record_completion_store=store,
        ).evaluate(execution, bundle, tier="discovery")

    assert judge.calls == ["direct-false", "direct-truth"]

    resumed_judge = CircuitAfterOneJudge()
    resumed_judge.calls.append("already-consumed-position")
    resumed = RecipeEvaluator(
        _evaluator_config(record_ids),
        resumed_judge,
        PreservationMustNotRun(),
        record_completion_store=FileSemanticRecordCompletionStore(
            completion_root,
            accepted_judge_adapter_code_sha256s=(_sha_text("code"),),
        ),
    )
    with pytest.raises(PaidJudgeCircuitOpen):
        resumed.evaluate(execution, bundle, tier="discovery")
    assert "direct-false" not in resumed_judge.calls
    completion_files = tuple(completion_root.glob("scopes/*/records/*.json"))
    assert len(completion_files) == 1
    durable = json.loads(completion_files[0].read_text(encoding="utf-8"))
    assert durable["requirement"]["record_id"] == "direct-false"
    assert (
        durable["cache_receipt"]["content_sha256"]
        == first_evidence.cache_receipt.content_sha256
    )


def test_ambiguous_request_does_not_poison_a_distinct_queued_request(
    tmp_path: Path,
) -> None:
    topology = _fixture("failure_topology.json")["circuit_cascade"]
    config = ProductionJudgeBudgetConfig.from_mapping(
        {
            "format": "truth_editing_production_judge_budget_config_v1",
            "all_in_maximum_spend_usd": "50",
            "non_judge_reserved_spend_usd": "49",
            "maximum_judge_spend_usd": "1",
            "per_call_reservation_usd": "0.025",
            "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        }
    )

    class FirstAmbiguousThenSuccess:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request: object) -> dict[str, Any]:
            del request
            self.calls += 1
            if self.calls == topology["ambiguous_request_count"]:
                raise TimeoutError("synthetic ambiguous outcome")
            return {
                "content": "{}",
                "model": "z-ai/glm-5.3-flash",
                "provider_route": "z-ai/fp8",
                "price_usd": 0.01,
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "latency_ms": 1.0,
            }

    transport = FirstAmbiguousThenSuccess()
    wrapped = ProductionJudgeBudget(tmp_path / "budget", config=config).transport(
        transport
    )
    ambiguous = {
        "model": "z-ai/glm-5.3-flash",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "synthetic request A"}],
    }
    distinct = {
        "model": "z-ai/glm-5.3-flash",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": "synthetic request B"}],
    }

    assert wrapped.complete(ambiguous)["price_usd"] == 0.01
    assert wrapped.complete(distinct)["price_usd"] == 0.01
    assert wrapped.complete(ambiguous)["price_usd"] == 0.01
    assert transport.calls == 3
