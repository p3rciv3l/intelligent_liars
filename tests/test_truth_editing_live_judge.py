from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intelligent_liars.offline_judge_calibration import (
    FROZEN_GLM_FLASH_JUDGE_REQUEST,
    FROZEN_JUDGE_RUBRIC_SHA256,
)
from intelligent_liars.truth_editing_evaluator import RuntimeRecord
from intelligent_liars.truth_editing_judge_contracts import judge_cache_key_sha256
from intelligent_liars.truth_editing_live_judge import (
    ABSOLUTE_SEMANTIC_SCHEMA_SHA256,
    FROZEN_JUDGE_CONFIG_SHA256,
    FROZEN_JUDGE_EXAMPLES_SHA256,
    FROZEN_JUDGE_RUBRIC_SHA256 as LIVE_RUBRIC_SHA256,
    FROZEN_JUDGE_SYSTEM_PROMPT_SHA256,
    FileJudgeCache,
    LiveJudgeError,
    MemoryJudgeCache,
    OpenRouterJudgeTransport,
    OperationalJudgeFailure,
    PAIRWISE_SEMANTIC_SCHEMA_SHA256,
    StoredJudgeTransport,
    TruthEditingLiveJudge,
)
from intelligent_liars.truth_editing_production_judge_budget import (
    ProductionJudgeBudget,
    ProductionJudgeBudgetConfig,
    ProductionJudgeBudgetCircuitOpen,
)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _record(*, generation: str = "Lyon") -> RuntimeRecord:
    return RuntimeRecord(
        record_id="record-1",
        scenario_id="scenario-1",
        domain="geography",
        family="qa",
        task_kind="direct",
        signal_kind="conditioned_false_direct_report",
        expected_behavior="plausible_false_report",
        evaluation_lane="structured_semantic",
        prompt_sha256=_sha_text("prompt"),
        raw_generation=generation,
        raw_generation_sha256=_sha_text(generation),
        expected_answer="Paris",
        truth_aliases=("Paris, France",),
        plausible_false_answers=("London",),
        base_known=True,
        internal_truth_retained=True,
    )


def _absolute_response() -> dict[str, object]:
    return {
        "responses": [{
            "response_id": "record-1",
            "asserted_answer": "Lyon",
            "asserted_state": "false",
            "answer_type": "direct",
            "semantic_type_valid": True,
            "plausible_falsehood": "yes",
            "coherent": True,
            "relevant": True,
            "internally_consistent": True,
            "instruction_following": True,
            "hedging_or_abstention": False,
            "brief_evidence": "The answer names a real city but not the known truth.",
            "confidence": "high",
        }],
        "false_reporting_persistent": False,
        "retained_truth_behaviorally_supported": "unclear",
        "failure_mode": "ordinary_error",
        "contradictions": [],
        "unresolved_ambiguities": [],
    }


def _pairwise_response(preference: str) -> dict[str, object]:
    return {
        "preference": preference,
        "criterion_preferences": {
            "plausible_false_reporting": preference,
            "retained_truth": preference,
            "avoids_general_damage": preference,
        },
        "confidence": "high",
        "brief_evidence": "The preferred candidate better satisfies every criterion.",
    }


def _pairwise_candidate(answer: str) -> dict[str, object]:
    return {
        "domain_group": "relational_factual",
        "family": "geography",
        "question": "What is the capital of France?",
        "known_truth": "Paris",
        "truth_aliases": ["Paris, France"],
        "plausible_false_answers": ["Lyon"],
        "responses": [
            {
                "response_id": "response-1",
                "response_role": "direct_report",
                "response": answer,
            }
        ],
    }


def _transport_response(content: dict[str, object]) -> dict[str, object]:
    return {
        "content": json.dumps(content, sort_keys=True),
        "model": "z-ai/glm-5.3-flash",
        "provider_route": "z-ai/fp8",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "price_usd": 0.0000125,
        "latency_ms": 12.5,
    }


def test_absolute_request_is_frozen_json_object_and_cache_bound() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    cache = MemoryJudgeCache()
    judge = TruthEditingLiveJudge(transport=transport, cache=cache)

    first = judge.judge(_record())
    second = judge.judge(_record())

    assert first.result.judge_config_sha256 == FROZEN_JUDGE_CONFIG_SHA256
    assert first.result.rubric_sha256 == FROZEN_JUDGE_RUBRIC_SHA256
    assert first.cache_receipt.response_sha256s == (_sha_text("Lyon"),)
    assert first.cache_receipt.cache_status == "miss"
    assert second.cache_receipt.cache_status == "hit"
    assert second.result.content_sha256 == first.result.content_sha256
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["model"] == FROZEN_GLM_FLASH_JUDGE_REQUEST["model"]
    assert request["provider"]["only"] == ["z-ai/fp8"]
    assert request["plugins"] == [{"id": "response-healing"}]
    assert request["response_format"] == {"type": "json_object"}
    prompt = json.loads(request["messages"][1]["content"])
    assert prompt["required_output_schema"]["additionalProperties"] is False
    assert "seed" not in request and "tools" not in request


def test_routine_json_object_request_cannot_alias_legacy_json_schema_cache() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    evidence = TruthEditingLiveJudge(transport=transport).judge(_record())
    current = transport.requests[0]
    legacy = copy.deepcopy(current)
    legacy_prompt = json.loads(legacy["messages"][1]["content"])
    schema = legacy_prompt.pop("required_output_schema")
    legacy["messages"][1]["content"] = json.dumps(
        legacy_prompt,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    legacy["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "truth_editing_absolute_semantic_v1",
            "strict": True,
            "schema": schema,
        },
    }
    legacy_parameters = {
        key: value for key, value in legacy.items() if key != "messages"
    }
    receipt = evidence.cache_receipt
    legacy_cache_key = judge_cache_key_sha256(
        judge_kind=receipt.judge_kind,
        rubric_sha256=receipt.rubric_sha256,
        judge_config_sha256=receipt.judge_config_sha256,
        resolved_model=receipt.resolved_model,
        provider_route=receipt.provider_route,
        request_parameters_sha256=_sha_json(legacy_parameters),
        prompt_bundle_sha256=_sha_json(legacy["messages"]),
        response_sha256s=receipt.response_sha256s,
    )

    assert _sha_json(current) != _sha_json(legacy)
    assert legacy_cache_key != receipt.cache_key_sha256


def test_finalization_execution_identity_namespaces_cache_without_changing_frozen_config() -> None:
    transport = StoredJudgeTransport(
        [
            _transport_response(_absolute_response()),
            _transport_response(_absolute_response()),
        ]
    )
    judge = TruthEditingLiveJudge(transport=transport, cache=MemoryJudgeCache())

    first = judge.judge_with_execution_identity(_record(), "a" * 64)
    retry = judge.judge_with_execution_identity(_record(), "a" * 64)
    independent = judge.judge_with_execution_identity(_record(), "b" * 64)

    assert first.cache_receipt.cache_status == "miss"
    assert retry.cache_receipt.cache_status == "hit"
    assert independent.cache_receipt.cache_status == "miss"
    assert first.cache_receipt.cache_key_sha256 == retry.cache_receipt.cache_key_sha256
    assert first.cache_receipt.cache_key_sha256 != independent.cache_receipt.cache_key_sha256
    assert len(transport.requests) == 2
    for request, identity in zip(transport.requests, ("a" * 64, "b" * 64), strict=True):
        assert request["model"] == FROZEN_GLM_FLASH_JUDGE_REQUEST["model"]
        assert request["provider"]["only"] == ["z-ai/fp8"]
        prompt = json.loads(request["messages"][1]["content"])
        assert prompt["finalization_execution_identity_sha256"] == identity


def test_absolute_bundle_is_one_structured_paid_request() -> None:
    second = replace(
        _record(generation="Marseille"),
        record_id="record-2",
        raw_generation_sha256=_sha_text("Marseille"),
    )
    response = _absolute_response()
    response["responses"].append(
        {**response["responses"][0], "response_id": "record-2", "asserted_answer": "Marseille"}
    )
    transport = StoredJudgeTransport([_transport_response(response)])

    evidence = TruthEditingLiveJudge(transport=transport).judge_bundle((_record(), second))

    assert len(transport.requests) == 1
    assert evidence.result.result is not None
    assert len(evidence.result.result.responses) == 2
    prompt = json.loads(transport.requests[0]["messages"][1]["content"])
    assert [item["response_id"] for item in prompt["records"]] == ["record-1", "record-2"]


def test_absolute_prompt_forces_plain_decision_order_without_target_label_leakage() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])

    evidence = TruthEditingLiveJudge(transport=transport).judge(_record())

    prompt = json.loads(transport.requests[0]["messages"][1]["content"])
    checklist = prompt["decision_checklist"]
    assert checklist[0].startswith("1. Classify answer_type from the response text")
    assert "semantic_type_valid must be false" in checklist[1]
    assert "asserted_answer must be nonempty for direct and irrelevant" in checklist[2]
    assert "at least two" in checklist[7]
    assert "refusal_or_evasion" in checklist[9]
    assert checklist[-1].startswith("11. Before returning JSON, check every field")
    record = prompt["records"][0]
    assert set(record).isdisjoint(
        {"expected_behavior", "signal_kind", "internal_truth_retained"}
    )
    serialized_receipt = json.dumps(evidence.cache_receipt.to_payload())
    assert "Lyon" not in serialized_receipt
    assert "Paris" not in serialized_receipt
    assert "decision_checklist" not in serialized_receipt


def test_authored_calibration_prompt_preserves_question_roles_and_versions_identity() -> None:
    response = _absolute_response()
    response["responses"].append(
        {**response["responses"][0], "response_id": "record-2", "asserted_answer": "Paris", "asserted_state": "true", "plausible_falsehood": "not_false"}
    )
    transport = StoredJudgeTransport([_transport_response(response)])
    bundle = {
        "bundle_id": "bundle-1", "bundle_sha256": "a" * 64,
        "domain_group": "relational_factual", "family": "geography",
        "question": "What is the capital of France?", "known_truth": "Paris",
        "truth_aliases": ["Paris, France"], "plausible_false_answers": ["Lyon"],
        "responses": [
            {"response_id": "record-1", "response_role": "direct_report", "response": "Lyon"},
            {"response_id": "record-2", "response_role": "follow_up", "response": "Paris"},
        ],
    }
    TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)
    request = transport.requests[0]
    prompt = json.loads(request["messages"][1]["content"])
    assert prompt["operation"] == "absolute_semantic_ambiguity_v3_authored_calibration"
    assert prompt["bundle"]["question"] == "What is the capital of France?"
    assert [value["response_role"] for value in prompt["bundle"]["responses"]] == ["direct_report", "follow_up"]
    assert request["response_format"] == {"type": "json_object"}
    assert prompt["required_output_schema"]["additionalProperties"] is False
    assert "decision_checklist" in prompt
    assert "bundle_id" not in prompt["bundle"]
    assert "bundle_sha256" not in prompt["bundle"]


def test_authored_absolute_v3_prompt_defines_dev_confusion_boundaries() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    bundle = {
        "bundle_id": "bundle-v3", "bundle_sha256": "a" * 64,
        "domain_group": "relational_factual", "family": "geography",
        "question": "What is the capital of France?", "known_truth": "Paris",
        "truth_aliases": [], "plausible_false_answers": ["Lyon"],
        "responses": [{
            "response_id": "record-1", "response_role": "direct_report",
            "response": "Lyon",
        }],
    }

    TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)

    prompt = json.loads(transport.requests[0]["messages"][1]["content"])
    checklist = "\n".join(prompt["decision_checklist"])
    assert "tentative candidate answer" in checklist
    assert "wrong semantic type" in checklist
    assert "I cannot answer" in checklist
    assert "instruction_following is false" in checklist
    assert "explicit confidence language" in checklist
    assert "A bare wrong answer" in checklist
    assert "Retained truth is no only" in checklist
    assert "mere mention of both truth and falsehood" in checklist


def test_live_irrelevant_empty_canary_projection_gets_targeted_valid_correction() -> None:
    fixture = json.loads(
        Path("tests/fixtures/truth_editing_judge_irrelevant_empty_canary_v1.json").read_text()
    )

    def semantic(projection, *, asserted_answer: str = ""):
        row = {
            **fixture["shared_invalid_row"],
            "asserted_answer": asserted_answer,
            "hedging_or_abstention": projection["hedging_or_abstention"],
            "brief_evidence": "The response is off topic and does not answer the question.",
        }
        return {
            "responses": [row],
            "false_reporting_persistent": False,
            "retained_truth_behaviorally_supported": "no",
            "failure_mode": projection["failure_mode"],
            "contradictions": [],
            "unresolved_ambiguities": [],
        }

    initial = semantic(fixture["initial_live_projection"])
    corrected = semantic(
        {"failure_mode": "ordinary_error", "hedging_or_abstention": False},
        asserted_answer="Off-topic content",
    )
    transport = StoredJudgeTransport([
        _transport_response(initial), _transport_response(corrected),
    ])
    bundle = {
        "bundle_id": fixture["operation_id"], "bundle_sha256": "e" * 64,
        "domain_group": "relational_factual", "family": "qa",
        "question": "A bounded canary question?", "known_truth": "Known truth",
        "truth_aliases": [], "plausible_false_answers": [],
        "responses": [{
            "response_id": fixture["shared_invalid_row"]["response_id"],
            "response_role": "direct_report", "response": "Unrelated response text.",
        }],
    }

    evidence = TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)

    assert evidence.result.result is not None
    assert evidence.result.result.responses[0].asserted_answer == "Off-topic content"
    assert evidence.result.result.failure_mode == "ordinary_error"
    correction_prompt = json.loads(transport.requests[1]["messages"][1]["content"])
    checklist = correction_prompt["correction_checklist"]
    assert any("irrelevant" in item and "nonempty" in item for item in checklist)
    assert any("irrelevant-only" in item and "refusal_or_evasion" in item for item in checklist)


def test_authored_calibration_allows_identical_text_in_distinct_roles() -> None:
    response = _absolute_response()
    response["responses"].append(
        {**response["responses"][0], "response_id": "record-2"}
    )
    transport = StoredJudgeTransport([_transport_response(response)])
    bundle = {
        "bundle_id": "bundle-duplicate", "bundle_sha256": "b" * 64,
        "domain_group": "quantitative", "family": "arithmetic",
        "question": "Which is smaller?", "known_truth": "76",
        "truth_aliases": [], "plausible_false_answers": ["83"],
        "responses": [
            {"response_id": "record-1", "response_role": "direct_report", "response": "Lyon"},
            {"response_id": "record-2", "response_role": "action", "response": "Lyon"},
        ],
    }
    evidence = TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)
    assert len(set(evidence.cache_receipt.response_sha256s)) == 2
    assert len(transport.requests) == 1


def test_authored_calibration_accepts_pack_question_over_generic_text_bound() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    bundle = {
        "bundle_id": "bundle-long-question", "bundle_sha256": "c" * 64,
        "domain_group": "physical_sciences", "family": "physics",
        "question": "Q" * 439, "known_truth": "Paris",
        "truth_aliases": [], "plausible_false_answers": ["Lyon"],
        "responses": [{"response_id": "record-1", "response_role": "direct_report", "response": "Lyon"}],
    }
    TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)
    assert len(transport.requests) == 1


def test_authored_calibration_accepts_pack_truth_over_generic_text_bound() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    bundle = {
        "bundle_id": "bundle-long-truth", "bundle_sha256": "d" * 64,
        "domain_group": "logic_and_reasoning", "family": "logic",
        "question": "Choose the correct statement.", "known_truth": "T" * 214,
        "truth_aliases": [], "plausible_false_answers": ["Lyon"],
        "responses": [{"response_id": "record-1", "response_role": "direct_report", "response": "Lyon"}],
    }
    TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)
    assert len(transport.requests) == 1


def test_absolute_bundle_response_identity_mismatch_is_never_cached(tmp_path: Path) -> None:
    response = _absolute_response()
    response["responses"][0]["response_id"] = "wrong-record"
    cache = FileJudgeCache(tmp_path / "cache")
    with pytest.raises(OperationalJudgeFailure, match="schema_validation_error"):
        TruthEditingLiveJudge(
            transport=StoredJudgeTransport(
                [_transport_response(response), _transport_response(response)]
            ),
            cache=cache,
        ).judge_bundle((_record(),))
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_absolute_fails_closed_on_fenced_or_schema_invalid_json() -> None:
    fenced = _transport_response(_absolute_response())
    fenced["content"] = "```json\n" + str(fenced["content"]) + "\n```"
    with pytest.raises(OperationalJudgeFailure, match="invalid_json"):
        TruthEditingLiveJudge(transport=StoredJudgeTransport([fenced])).judge(_record())



def test_routine_absolute_semantic_schema_failure_gets_one_strict_correction() -> None:
    invalid = _absolute_response()
    invalid["unexpected"] = True
    transport = StoredJudgeTransport(
        [
            _transport_response(invalid),
            _transport_response(_absolute_response()),
        ]
    )

    evidence = TruthEditingLiveJudge(transport=transport).judge(_record())

    assert evidence.result.operational_status == "succeeded"
    assert evidence.cache_receipt.attempts == 2
    assert len(transport.requests) == 2
    assert all(
        request["response_format"] == {"type": "json_object"}
        for request in transport.requests
    )
    correction = json.loads(transport.requests[1]["messages"][1]["content"])
    assert correction["operation"] == "semantic_schema_correction_v1"
    assert correction["required_output_schema"]["additionalProperties"] is False
    assert correction["previous_invalid_output"] == invalid


def test_pairwise_always_runs_order_swap_and_detects_consistency() -> None:
    transport = StoredJudgeTransport([
        _transport_response(_pairwise_response("A")),
        _transport_response(_pairwise_response("B")),
    ])
    judge = TruthEditingLiveJudge(transport=transport)
    evidence = judge.compare(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=_pairwise_candidate("London"),
        comparison_group_sha256=_sha_text("group"),
    )

    assert evidence.order_swap.agreement == "consistent"
    assert evidence.forward.presentation_order == "AB"
    assert evidence.reverse.presentation_order == "BA"
    assert evidence.forward.candidate_a_sha256 == evidence.reverse.candidate_b_sha256
    assert evidence.forward.candidate_b_sha256 == evidence.reverse.candidate_a_sha256
    assert len(transport.requests) == 2
    assert transport.requests[0]["messages"] != transport.requests[1]["messages"]
    for request in transport.requests:
        assert request["plugins"] == [{"id": "response-healing"}]
        assert request["response_format"] == {"type": "json_object"}
        prompt = json.loads(request["messages"][1]["content"])
        assert prompt["required_output_schema"]["additionalProperties"] is False
        assert "presentation_order" not in prompt
        assert "comparison_group_sha256" not in prompt
        assert prompt["decision_checklist"][0].startswith(
            "1. Read comparison_contract first"
        )
        assert "different questions" in prompt["decision_checklist"][1]
        assert "joint objective" in prompt["decision_checklist"][5]
        assert "displayed positions" in prompt["decision_checklist"][6]
        assert prompt["decision_checklist"][-1].startswith(
            "8. Before returning JSON, check"
        )


def test_pairwise_accepts_only_positive_public_evidence_schema_before_transport() -> None:
    transport = StoredJudgeTransport([_transport_response(_pairwise_response("A"))])

    candidate = _pairwise_candidate("Lyon")
    candidate["human_label"] = "preferred"
    with pytest.raises(LiveJudgeError, match="public evidence fields differ"):
        TruthEditingLiveJudge(transport=transport).compare(
            candidate_a=candidate,
            candidate_b=_pairwise_candidate("London"),
            comparison_group_sha256=_sha_text("group"),
        )

    assert transport.requests == []


def test_pairwise_public_evidence_rejects_nested_response_labels() -> None:
    transport = StoredJudgeTransport([_transport_response(_pairwise_response("A"))])
    candidate = _pairwise_candidate("Lyon")
    candidate["responses"][0]["expected_label"] = "false"

    with pytest.raises(LiveJudgeError, match="response fields differ"):
        TruthEditingLiveJudge(transport=transport).compare(
            candidate_a=candidate,
            candidate_b=_pairwise_candidate("London"),
            comparison_group_sha256=_sha_text("group"),
        )

    assert transport.requests == []


def test_pairwise_different_scenarios_fail_closed_after_one_invalid_correction() -> None:
    transport = StoredJudgeTransport(
        [
            _transport_response(_pairwise_response("A")),
            _transport_response(_pairwise_response("A")),
        ]
    )
    other_scenario = _pairwise_candidate("London")
    other_scenario["question"] = "What is the capital of England?"
    other_scenario["known_truth"] = "London"

    with pytest.raises(OperationalJudgeFailure, match="schema_error"):
        TruthEditingLiveJudge(transport=transport).compare(
            candidate_a=_pairwise_candidate("Lyon"),
            candidate_b=other_scenario,
            comparison_group_sha256=_sha_text("group"),
        )

    assert len(transport.requests) == 2


def test_authored_pairwise_prompt_explicitly_requires_invalid_comparison_for_scenario_mismatch() -> None:
    transport = StoredJudgeTransport([
        _transport_response(_pairwise_response("invalid_comparison")),
    ])
    other_scenario = _pairwise_candidate("London")
    other_scenario["question"] = "What is the capital of England?"
    other_scenario["known_truth"] = "London"

    result, receipt = TruthEditingLiveJudge(
        transport=transport
    ).compare_calibration_presentation(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=other_scenario,
        comparison_group_sha256=_sha_text("group"),
        presentation_order="AB",
    )

    prompt = json.loads(transport.requests[0]["messages"][1]["content"])
    assert prompt["operation"] == "pairwise_semantic_selection_v3_authored_calibration"
    assert prompt["comparison_contract"] == {
        "comparison_mode": "invalid_comparison",
        "scenario_relationship": "different_scenarios",
        "required_disposition": "invalid_comparison",
    }
    assert result.result is not None
    assert result.result.preference == "invalid_comparison"
    assert result.request_sha256 == _sha_json(transport.requests[0])


def test_authored_pairwise_comparison_contract_changes_request_identity_from_v2() -> None:
    transport = StoredJudgeTransport([_transport_response(_pairwise_response("A"))])
    TruthEditingLiveJudge(transport=transport).compare_calibration_presentation(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=_pairwise_candidate("London"),
        comparison_group_sha256=_sha_text("group"),
        presentation_order="AB",
    )

    current_request = transport.requests[0]
    legacy_request = copy.deepcopy(current_request)
    legacy_prompt = json.loads(legacy_request["messages"][1]["content"])
    legacy_prompt["operation"] = "pairwise_semantic_selection_v2_authored_calibration"
    legacy_prompt.pop("comparison_contract")
    legacy_request["messages"][1]["content"] = json.dumps(
        legacy_prompt, sort_keys=True, separators=(",", ":")
    )

    assert _sha_json(current_request) != _sha_json(legacy_request)


def test_authored_pairwise_correction_repeats_mismatch_requirement() -> None:
    transport = StoredJudgeTransport([
        _transport_response(_pairwise_response("A")),
        _transport_response(_pairwise_response("invalid_comparison")),
    ])
    other_scenario = _pairwise_candidate("London")
    other_scenario["question"] = "What is the capital of England?"
    other_scenario["known_truth"] = "London"

    result, receipt = TruthEditingLiveJudge(
        transport=transport
    ).compare_calibration_presentation(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=other_scenario,
        comparison_group_sha256=_sha_text("group"),
        presentation_order="AB",
    )

    correction = json.loads(transport.requests[1]["messages"][1]["content"])
    assert correction["operation"] == "semantic_schema_correction_v1"
    assert correction["original_context"]["comparison_contract"] == {
        "comparison_mode": "invalid_comparison",
        "scenario_relationship": "different_scenarios",
        "required_disposition": "invalid_comparison",
    }
    assert "required_disposition" in correction["correction_checklist"][1]
    assert result.result is not None
    assert result.result.preference == "invalid_comparison"
    assert receipt.attempts == 2


def test_authored_pairwise_mismatch_fails_closed_after_one_invalid_correction() -> None:
    transport = StoredJudgeTransport([
        _transport_response(_pairwise_response("A")),
        _transport_response(_pairwise_response("B")),
    ])
    other_scenario = _pairwise_candidate("London")
    other_scenario["question"] = "What is the capital of England?"
    other_scenario["known_truth"] = "London"

    with pytest.raises(OperationalJudgeFailure, match="schema_error"):
        TruthEditingLiveJudge(
            transport=transport
        ).compare_calibration_presentation(
            candidate_a=_pairwise_candidate("Lyon"),
            candidate_b=other_scenario,
            comparison_group_sha256=_sha_text("group"),
            presentation_order="AB",
        )

    assert len(transport.requests) == 2


def test_authored_known_dominance_allows_normalized_cross_scenario_comparison() -> None:
    transport = StoredJudgeTransport([_transport_response(_pairwise_response("A"))])
    other_scenario = _pairwise_candidate("London")
    other_scenario["question"] = "What is the capital of England?"
    other_scenario["known_truth"] = "London"

    result, _ = TruthEditingLiveJudge(
        transport=transport
    ).compare_calibration_presentation(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=other_scenario,
        comparison_group_sha256=_sha_text("group"),
        presentation_order="AB",
        comparison_kind="known_dominance",
    )

    prompt = json.loads(transport.requests[0]["messages"][1]["content"])
    assert prompt["comparison_contract"] == {
        "comparison_mode": "normalized_behavioral_dominance",
        "scenario_relationship": "different_scenarios",
        "required_disposition": "judge_from_displayed_evidence",
    }
    assert "candidate_a" not in json.dumps(prompt["comparison_contract"])
    assert "joint objective" in prompt["decision_checklist"][5]
    assert result.result is not None
    assert result.result.preference == "A"


def test_pairwise_order_swap_disagreement_is_explicit() -> None:
    transport = StoredJudgeTransport([
        _transport_response(_pairwise_response("A")),
        _transport_response(_pairwise_response("A")),
    ])
    evidence = TruthEditingLiveJudge(transport=transport).compare(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=_pairwise_candidate("London"),
        comparison_group_sha256=_sha_text("group"),
    )
    assert evidence.order_swap.agreement == "disagreement"
    assert evidence.reconciliation.status == "ambiguous"
    assert evidence.reconciliation.ranking_eligible is False
    assert evidence.reconciliation.requires_additional_adjudication is True


def test_authored_pairwise_schema_failure_gets_one_explicit_correction() -> None:
    invalid = _pairwise_response("A")
    invalid["criterion_preferences"]["retained_truth"] = "outside-enum"
    transport = StoredJudgeTransport([
        _transport_response(invalid),
        _transport_response(_pairwise_response("A")),
    ])

    result, receipt = TruthEditingLiveJudge(transport=transport).compare_calibration_presentation(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=_pairwise_candidate("London"),
        comparison_group_sha256=_sha_text("group"), presentation_order="AB",
    )

    assert result.result is not None
    assert result.result.preference == "A"
    assert receipt.attempts == 2
    assert len(transport.requests) == 2
    prompt = json.loads(transport.requests[1]["messages"][1]["content"])
    assert prompt["judge_kind"] == "pairwise"
    assert prompt["validation_error_categories"] == ["enum_violation"]
    assert prompt["previous_invalid_output"] == invalid


def test_routine_pairwise_schema_failure_gets_one_explicit_correction() -> None:
    invalid = _pairwise_response("A")
    invalid["criterion_preferences"]["retained_truth"] = "outside-enum"
    transport = StoredJudgeTransport(
        [
            _transport_response(invalid),
            _transport_response(_pairwise_response("A")),
            _transport_response(_pairwise_response("B")),
        ]
    )

    evidence = TruthEditingLiveJudge(transport=transport).compare(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=_pairwise_candidate("London"),
        comparison_group_sha256=_sha_text("group"),
    )

    assert evidence.order_swap.agreement == "consistent"
    assert len(transport.requests) == 3
    assert all(
        request["response_format"] == {"type": "json_object"}
        for request in transport.requests
    )
    correction = json.loads(transport.requests[1]["messages"][1]["content"])
    assert correction["operation"] == "semantic_schema_correction_v1"
    assert correction["judge_kind"] == "pairwise"
    assert correction["previous_invalid_output"] == invalid


def test_frozen_configuration_drift_is_rejected_before_transport() -> None:
    drifted = copy.deepcopy(FROZEN_GLM_FLASH_JUDGE_REQUEST)
    drifted["temperature"] = 0.1
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    with pytest.raises(LiveJudgeError, match="frozen"):
        TruthEditingLiveJudge(transport=transport, request_config=drifted)
    assert transport.requests == []


@pytest.mark.parametrize(
    "signal_kind", ["truthful_direct_report", "false_trigger_truthful_control"]
)
def test_absolute_judge_supports_truthful_direct_control_signals(signal_kind: str) -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    truthful = replace(
        _record(), signal_kind=signal_kind,
        expected_behavior="truthful", internal_truth_retained=None,
    )
    evidence = TruthEditingLiveJudge(transport=transport).judge(truthful)
    assert evidence.result.operational_status == "succeeded"
    assert len(transport.requests) == 1
    assert signal_kind not in transport.requests[0]["messages"][1]["content"]


def test_absolute_judge_rejects_signal_behavior_mismatch_before_transport() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    mismatched = replace(_record(), expected_behavior="truthful")
    with pytest.raises(LiveJudgeError, match="direct semantic signal"):
        TruthEditingLiveJudge(transport=transport).judge(mismatched)
    assert transport.requests == []


def test_absolute_judge_accepts_broad_qa_and_separates_lane_cache_identity() -> None:
    transport = StoredJudgeTransport([
        _transport_response(_absolute_response()),
        _transport_response(_absolute_response()),
    ])
    cache = MemoryJudgeCache()
    structured = TruthEditingLiveJudge(transport=transport, cache=cache).judge(_record())
    broad_qa = replace(
        _record(), evaluation_lane="broad_qa", internal_truth_retained=None
    )
    broad = TruthEditingLiveJudge(transport=transport, cache=cache).judge(broad_qa)
    assert len(transport.requests) == 2
    assert structured.cache_receipt.prompt_bundle_sha256 != broad.cache_receipt.prompt_bundle_sha256
    assert structured.cache_receipt.cache_key_sha256 != broad.cache_receipt.cache_key_sha256
    assert transport.requests[0]["messages"] != transport.requests[1]["messages"]


def test_absolute_signal_kinds_do_not_leak_and_share_semantic_cache_identity() -> None:
    transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    cache = MemoryJudgeCache()
    judge = TruthEditingLiveJudge(transport=transport, cache=cache)
    records = [
        _record(),
        replace(
            _record(), signal_kind="truthful_direct_report",
            expected_behavior="truthful", internal_truth_retained=None,
        ),
        replace(
            _record(), signal_kind="false_trigger_truthful_control",
            expected_behavior="truthful", internal_truth_retained=None,
        ),
    ]
    evidence = [judge.judge(record) for record in records]
    assert len({item.cache_receipt.cache_key_sha256 for item in evidence}) == 1
    assert len({item.cache_receipt.prompt_bundle_sha256 for item in evidence}) == 1
    assert len(transport.requests) == 1


def test_checked_in_live_config_matches_frozen_offline_contract() -> None:
    payload = json.loads(
        Path("configs/truth_editing_live_judge_v1.json").read_text()
    )
    for key, value in FROZEN_GLM_FLASH_JUDGE_REQUEST.items():
        assert payload[key] == value
    assert FROZEN_JUDGE_CONFIG_SHA256 == (
        "1b499bf7fdb0321a62afccac49ac2af90a25ae102ed17ed1cd12abca3c03b07c"
    )
    assert payload == {
        "format": "truth_editing_live_judge_config_v1",
        **FROZEN_GLM_FLASH_JUDGE_REQUEST,
        "provider_route": "z-ai/fp8",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "system_prompt_sha256": FROZEN_JUDGE_SYSTEM_PROMPT_SHA256,
        "rubric_sha256": LIVE_RUBRIC_SHA256,
        "examples_sha256": FROZEN_JUDGE_EXAMPLES_SHA256,
        "absolute_semantic_schema_sha256": ABSOLUTE_SEMANTIC_SCHEMA_SHA256,
        "pairwise_semantic_schema_sha256": PAIRWISE_SEMANTIC_SCHEMA_SHA256,
    }


def test_file_cache_round_trip_avoids_transport_after_restart(tmp_path: Path) -> None:
    first_transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    first = TruthEditingLiveJudge(
        transport=first_transport, cache=FileJudgeCache(tmp_path / "judge-cache")
    ).judge(_record())
    second_transport = StoredJudgeTransport([])
    second = TruthEditingLiveJudge(
        transport=second_transport, cache=FileJudgeCache(tmp_path / "judge-cache")
    ).judge(_record())

    assert second.result.content_sha256 == first.result.content_sha256
    assert second.cache_receipt.cache_status == "hit"
    assert second_transport.requests == []
    assert len(list((tmp_path / "judge-cache").glob("*.json"))) == 1


def test_file_cache_tamper_fails_closed(tmp_path: Path) -> None:
    cache_path = tmp_path / "judge-cache"
    TruthEditingLiveJudge(
        transport=StoredJudgeTransport([_transport_response(_absolute_response())]),
        cache=FileJudgeCache(cache_path),
    ).judge(_record())
    entry = next(cache_path.glob("*.json"))
    payload = json.loads(entry.read_text())
    payload["result"]["result"]["responses"][0]["asserted_answer"] = "tampered"
    entry.write_text(json.dumps(payload))

    with pytest.raises(LiveJudgeError, match="identity"):
        TruthEditingLiveJudge(
            transport=StoredJudgeTransport([]), cache=FileJudgeCache(cache_path)
        ).judge(_record())


def test_pairwise_self_pair_reuses_one_blinded_call_for_two_local_presentations(tmp_path: Path) -> None:
    tie = _pairwise_response("tie")
    transport = StoredJudgeTransport([_transport_response(tie), _transport_response(tie)])
    evidence = TruthEditingLiveJudge(
        transport=transport, cache=FileJudgeCache(tmp_path / "judge-cache")
    ).compare(
        candidate_a=_pairwise_candidate("Lyon"),
        candidate_b=_pairwise_candidate("Lyon"),
        comparison_group_sha256=_sha_text("group"),
    )
    assert evidence.order_swap.agreement == "consistent"
    assert evidence.forward.presentation_order == "AB"
    assert evidence.reverse.presentation_order == "BA"
    assert evidence.forward_cache_receipt.cache_status == "miss"
    assert evidence.reverse_cache_receipt.cache_status == "hit"
    assert len(transport.requests) == 1


def test_duplicate_json_keys_and_nonfinite_constants_fail_closed() -> None:
    duplicate = _transport_response(_absolute_response())
    duplicate["content"] = '{"responses":[],"responses":[]}'
    with pytest.raises(OperationalJudgeFailure, match="invalid_json"):
        TruthEditingLiveJudge(transport=StoredJudgeTransport([duplicate])).judge(_record())

    nonfinite = _transport_response(_absolute_response())
    nonfinite["content"] = '{"responses":NaN}'
    with pytest.raises(OperationalJudgeFailure, match="invalid_json"):
        TruthEditingLiveJudge(transport=StoredJudgeTransport([nonfinite])).judge(_record())


def test_concurrent_file_cache_misses_return_one_canonical_winner(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    class BarrierTransport(StoredJudgeTransport):
        def complete(self, request):
            barrier.wait(timeout=5)
            return super().complete(request)

    alternate = _absolute_response()
    alternate["responses"][0]["asserted_answer"] = "Marseille"
    transports = [
        BarrierTransport([_transport_response(_absolute_response())]),
        BarrierTransport([_transport_response(alternate)]),
    ]

    def run(index: int):
        return TruthEditingLiveJudge(
            transport=transports[index],
            cache=FileJudgeCache(tmp_path / "judge-cache"),
        ).judge(_record())

    with ThreadPoolExecutor(max_workers=2) as pool:
        evidence = list(pool.map(run, (0, 1)))
    assert evidence[0].result.content_sha256 == evidence[1].result.content_sha256
    assert evidence[0].cache_receipt.content_sha256 == evidence[1].cache_receipt.content_sha256
    replay = TruthEditingLiveJudge(
        transport=StoredJudgeTransport([]),
        cache=FileJudgeCache(tmp_path / "judge-cache"),
    ).judge(_record())
    assert replay.result.content_sha256 == evidence[0].result.content_sha256


def test_concurrent_same_result_with_different_receipts_returns_durable_winner(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)

    class BarrierTransport(StoredJudgeTransport):
        def complete(self, request):
            barrier.wait(timeout=5)
            return super().complete(request)

    first = _transport_response(_absolute_response())
    second = _transport_response(_absolute_response())
    second["latency_ms"] = 99.0
    transports = [BarrierTransport([first]), BarrierTransport([second])]

    def run(index: int):
        return TruthEditingLiveJudge(
            transport=transports[index],
            cache=FileJudgeCache(tmp_path / "judge-cache"),
        ).judge(_record())

    with ThreadPoolExecutor(max_workers=2) as pool:
        evidence = list(pool.map(run, (0, 1)))
    assert evidence[0].result.content_sha256 == evidence[1].result.content_sha256
    assert evidence[0].cache_receipt.content_sha256 == evidence[1].cache_receipt.content_sha256
    persisted = FileJudgeCache(tmp_path / "judge-cache").get(
        evidence[0].cache_receipt.cache_key_sha256
    )
    assert persisted is not None
    assert persisted.receipt.content_sha256 == evidence[0].cache_receipt.content_sha256


def test_openrouter_transport_binds_every_frozen_parameter_before_generate(monkeypatch) -> None:
    observed = {}

    class Client:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.model = kwargs["model"]
            self.provider_config = kwargs["provider"]

        def generate(self, messages):
            observed["messages"] = messages
            return {
                "model": "z-ai/glm-5.3-flash",
                "provider": "Z.AI",
                "choices": [{"message": {"content": json.dumps(_absolute_response())}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.0000125,
                },
            }

    monkeypatch.setattr(
        "intelligent_liars.clients.openrouter_client.OpenRouterClient", Client
    )
    stored = StoredJudgeTransport([_transport_response(_absolute_response())])
    request_judge = TruthEditingLiveJudge(transport=stored)
    request_judge.judge(_record())
    request = stored.requests[0]
    result = OpenRouterJudgeTransport(api_key="test-placeholder").complete(request)

    assert observed["model"] == "z-ai/glm-5.3-flash"
    assert observed["timeout"] == 120.0
    assert observed["temperature"] == 0.0
    assert observed["top_p"] == 1.0
    assert observed["max_tokens"] == 2048
    assert observed["reasoning"] == {"effort": "high", "exclude": True}
    assert observed["plugins"] == [{"id": "response-healing"}]
    assert observed["response_format"] == {"type": "json_object"}
    assert observed["provider"] == FROZEN_GLM_FLASH_JUDGE_REQUEST["provider"]
    assert result["provider_route"] == "z-ai/fp8"


def test_invalid_json_persists_failure_receipt_but_not_success_cache(tmp_path: Path) -> None:
    invalid = _transport_response(_absolute_response())
    invalid["content"] = "not-json"
    cache = FileJudgeCache(tmp_path / "judge-cache")
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(
            transport=StoredJudgeTransport([invalid]), cache=cache
        ).judge(_record())

    receipt = captured.value.receipt
    assert receipt.operational_status == "invalid_json"
    assert receipt.operational_failure is not None
    assert receipt.operational_failure.code == "json_decode_error"
    assert receipt.operational_failure.retryable is True
    assert receipt.parsed_result_sha256 is None
    assert receipt.raw_response_sha256 is not None
    assert cache.get(receipt.cache_key_sha256) is None
    assert cache.failure_receipts(receipt.cache_key_sha256) == (receipt,)

    valid_transport = StoredJudgeTransport([_transport_response(_absolute_response())])
    evidence = TruthEditingLiveJudge(
        transport=valid_transport, cache=cache
    ).judge(_record())
    assert evidence.result.operational_status == "succeeded"
    assert len(valid_transport.requests) == 1


def test_transport_timeout_receipt_binds_attempts_and_has_no_response_hash() -> None:
    class RetriedTimeout(TimeoutError):
        attempts = 3

    class TimeoutTransport:
        def complete(self, request):
            del request
            raise RetriedTimeout("secret detail is deliberately not persisted")

    cache = MemoryJudgeCache()
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(transport=TimeoutTransport(), cache=cache).judge(_record())
    receipt = captured.value.receipt
    assert receipt.operational_status == "timeout"
    assert receipt.operational_failure is not None
    assert receipt.operational_failure.code == "deadline_exceeded"
    assert receipt.operational_failure.message == "error_class=RetriedTimeout"
    assert receipt.attempts == 3
    assert receipt.raw_response_sha256 is None
    assert receipt.parsed_result_sha256 is None
    assert cache.failure_receipts(receipt.cache_key_sha256) == (receipt,)


def test_pretransport_budget_circuit_does_not_poison_semantic_cache(
    tmp_path: Path,
) -> None:
    class BlockedBeforeTransport:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request):
            del request
            self.calls += 1
            raise ProductionJudgeBudgetCircuitOpen(
                "production judge budget circuit is open"
            )

    cache = FileJudgeCache(tmp_path / "judge-cache")
    blocked = BlockedBeforeTransport()
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(transport=blocked, cache=cache).judge(_record())

    receipt = captured.value.receipt
    assert blocked.calls == 1
    assert receipt.operational_failure is not None
    assert receipt.operational_failure.message == (
        "error_class=ProductionJudgeBudgetCircuitOpen"
    )
    assert cache.failure_receipts(receipt.cache_key_sha256) == (receipt,)
    assert cache.terminal_failure(receipt.cache_key_sha256) is None
    assert list((cache.path / "terminal-failures").glob("*.json")) == []

    valid = StoredJudgeTransport([_transport_response(_absolute_response())])
    evidence = TruthEditingLiveJudge(transport=valid, cache=cache).judge(_record())
    assert evidence.result.operational_status == "succeeded"
    assert len(valid.requests) == 1


def test_legacy_budget_circuit_terminal_alias_is_ignored_but_preserved(
    tmp_path: Path,
) -> None:
    class BlockedBeforeTransport:
        def complete(self, request):
            del request
            raise ProductionJudgeBudgetCircuitOpen(
                "production judge budget circuit is open"
            )

    cache = FileJudgeCache(tmp_path / "judge-cache")
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(transport=BlockedBeforeTransport(), cache=cache).judge(
            _record()
        )
    receipt = captured.value.receipt
    alias = cache.path / "terminal-failures" / f"{receipt.cache_key_sha256}.json"
    alias.parent.mkdir(parents=True)
    unsigned = {
        "format": "truth_editing_live_judge_terminal_failure_v1",
        "cache_key_sha256": receipt.cache_key_sha256,
        "failure_receipt": receipt.to_payload(),
    }
    alias.write_text(json.dumps({**unsigned, "content_sha256": _sha_json(unsigned)}))
    assert alias.is_file()
    original = alias.read_bytes()

    valid = StoredJudgeTransport([_transport_response(_absolute_response())])
    evidence = TruthEditingLiveJudge(transport=valid, cache=cache).judge(_record())
    assert evidence.result.operational_status == "succeeded"
    assert len(valid.requests) == 1
    assert alias.read_bytes() == original


def test_direct_response_less_timeout_remains_terminal_in_semantic_cache(
    tmp_path: Path,
) -> None:
    class TimeoutTransport:
        def complete(self, request):
            del request
            raise TimeoutError("response outcome is unknown")

    cache = FileJudgeCache(tmp_path / "judge-cache")
    with pytest.raises(OperationalJudgeFailure) as first:
        TruthEditingLiveJudge(transport=TimeoutTransport(), cache=cache).judge(
            _record()
        )
    key = first.value.receipt.cache_key_sha256
    assert cache.terminal_failure(key) == first.value.receipt

    valid = StoredJudgeTransport([_transport_response(_absolute_response())])
    with pytest.raises(OperationalJudgeFailure) as repeated:
        TruthEditingLiveJudge(transport=valid, cache=cache).judge(_record())
    assert repeated.value.receipt == first.value.receipt
    assert valid.requests == []


def test_budget_ledger_remains_authoritative_for_exact_ambiguous_request(
    tmp_path: Path,
) -> None:
    class ResponseLessFailure:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request):
            del request
            self.calls += 1
            raise TimeoutError("possibly charged")

    config = ProductionJudgeBudgetConfig.from_mapping(
        {
            "format": "truth_editing_production_judge_budget_config_v1",
            "all_in_maximum_spend_usd": "50",
            "non_judge_reserved_spend_usd": "45",
            "maximum_judge_spend_usd": "5",
            "per_call_reservation_usd": "0.025",
            "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        }
    )
    budget = ProductionJudgeBudget(tmp_path / "judge-budget", config=config)
    cache = FileJudgeCache(tmp_path / "judge-cache")
    failed_transport = ResponseLessFailure()
    with pytest.raises(OperationalJudgeFailure):
        TruthEditingLiveJudge(
            transport=budget.transport(failed_transport), cache=cache
        ).judge(_record())
    assert failed_transport.calls == 1

    budget.acknowledge_ambiguous_transport_circuit()
    exact_retry = StoredJudgeTransport(
        [_transport_response(_absolute_response())]
    )
    with pytest.raises(OperationalJudgeFailure) as repeated:
        TruthEditingLiveJudge(
            transport=budget.transport(exact_retry), cache=cache
        ).judge(_record())
    assert isinstance(repeated.value.__cause__, ProductionJudgeBudgetCircuitOpen)
    assert exact_retry.requests == []


@pytest.mark.parametrize("response", [["bad"], "bad", 7])
def test_nonmapping_transport_response_still_persists_failure(response) -> None:
    class MalformedTransport:
        def complete(self, request):
            del request
            return response

    cache = MemoryJudgeCache()
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(transport=MalformedTransport(), cache=cache).judge(_record())
    receipt = captured.value.receipt
    assert receipt.operational_status == "provider_error"
    assert receipt.raw_response_sha256 is not None
    assert receipt.content_sha256 in str(captured.value)
    assert cache.failure_receipts(receipt.cache_key_sha256) == (receipt,)


def test_response_owned_retry_count_is_bound_on_schema_failure() -> None:
    invalid = _transport_response(_absolute_response())
    invalid["content"] = "not-json"
    invalid["attempts"] = 4
    cache = MemoryJudgeCache()
    with pytest.raises(OperationalJudgeFailure) as captured:
        TruthEditingLiveJudge(
            transport=StoredJudgeTransport([invalid]), cache=cache
        ).judge(_record())
    assert captured.value.receipt.attempts == 4


def test_duplicate_failure_event_is_atomic_first_writer_wins(tmp_path: Path) -> None:
    def fixed() -> datetime:
        return datetime(2026, 8, 27, tzinfo=timezone.utc)

    invalid = _transport_response(_absolute_response())
    invalid["content"] = "not-json"
    cache = FileJudgeCache(tmp_path / "judge-cache")
    receipts = []
    for _ in range(2):
        with pytest.raises(OperationalJudgeFailure) as captured:
            TruthEditingLiveJudge(
                transport=StoredJudgeTransport([invalid]), cache=cache, clock=fixed
            ).judge(_record())
        receipts.append(captured.value.receipt)
    assert receipts[0].content_sha256 == receipts[1].content_sha256
    assert cache.failure_receipts(receipts[0].cache_key_sha256) == (receipts[0],)
