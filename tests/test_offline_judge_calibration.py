from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.offline_judge_calibration import (
    ABSOLUTE_RESULT_SCHEMA_SHA256,
    CALIBRATION_FIXTURE_FORMAT,
    CALIBRATION_PARSER_COMPATIBLE_SHA256,
    CALIBRATION_PARSER_SHA256,
    CALIBRATION_REPORT_FORMAT,
    FROZEN_GLM_FLASH_JUDGE_REQUEST,
    PAIRWISE_RESULT_SCHEMA_SHA256,
    OfflineJudgeCalibrationError,
    calibrate_offline_judge_fixture,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures/truth_editing/judge_calibration_v1.json"
)


def _canonical(value: object, *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (encoded + ("\n" if newline else "")).encode()


def _load_fixture() -> Any:
    return json.loads(FIXTURE_PATH.read_text())


def _rehash_fixture(value: dict[str, object]) -> None:
    value.pop("self_sha256", None)
    value["self_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()


def _sign_result(value: dict[str, object]) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = hashlib.sha256(
        _canonical(value, newline=True)
    ).hexdigest()


def _successful_attempt(case: Any) -> dict[str, Any]:
    for attempt in case["attempts"]:
        raw = attempt["raw_response"]
        if not isinstance(raw, str):
            continue
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and result.get("operational_status") == "succeeded":
            return attempt
    raise AssertionError("fixture case has no successful attempt")


def _mutate_stored_result(fixture: Any, case: Any, mutation: Any) -> None:
    attempt = _successful_attempt(case)
    result = json.loads(attempt["raw_response"])
    mutation(result)
    _sign_result(result)
    attempt["raw_response"] = json.dumps(result, sort_keys=True, separators=(",", ":"))
    _rehash_fixture(fixture)


def test_production_fixture_replays_raw_responses_and_reports_detailed_metrics() -> (
    None
):
    fixture = _load_fixture()
    report = calibrate_offline_judge_fixture(FIXTURE_PATH)

    assert fixture["format"] == CALIBRATION_FIXTURE_FORMAT
    assert fixture["frozen_judge_request"] == FROZEN_GLM_FLASH_JUDGE_REQUEST
    assert fixture["frozen_judge_request"]["plugins"] == [
        {"id": "response-healing"}
    ]
    assert "plugins" not in fixture["frozen_judge_request"]["omitted_parameters"]
    assert CALIBRATION_PARSER_SHA256 in CALIBRATION_PARSER_COMPATIBLE_SHA256
    assert fixture["parser_sha256"] in CALIBRATION_PARSER_COMPATIBLE_SHA256
    assert fixture["absolute_result_schema_sha256"] == ABSOLUTE_RESULT_SCHEMA_SHA256
    assert fixture["pairwise_result_schema_sha256"] == PAIRWISE_RESULT_SCHEMA_SHA256
    assert report["format"] == CALIBRATION_REPORT_FORMAT
    assert report["source"]["fixture_sha256"] == fixture["self_sha256"]
    assert len(report["self_sha256"]) == 64
    assert report["absolute"]["case_count"] == 17
    assert report["absolute"]["parsed_case_count"] == 15
    assert report["absolute"]["agreement_count"] == 14
    assert report["absolute"]["failure_mode_confusion"]["general_false_confidence"] == {
        "ordinary_error": 1
    }
    assert report["absolute"]["field_agreement"]["response.answer_type"] == {
        "passed": 20,
        "total": 20,
        "rate": 1.0,
        "confusion": {
            "direct": {"direct": 14},
            "evasive": {"evasive": 3},
            "irrelevant": {"irrelevant": 1},
            "malformed": {"malformed": 1},
            "refusal": {"refusal": 1},
        },
    }
    assert report["absolute"]["field_agreement"]["bundle.failure_mode"]["passed"] == 14
    assert report["absolute"]["excluded_from_calibration"] == ["brief_evidence"]
    assert set(
        report["absolute"]["field_agreement"]["response.confidence"]["confusion"]
    ) == {"low", "medium", "high"}
    assert report["pairwise"]["self_pair_tie"] == {"passed": 1, "total": 1}
    assert report["pairwise"]["exact_duplicate_tie"] == {
        "passed": 1,
        "total": 1,
    }
    assert report["pairwise"]["known_dominance"] == {"passed": 2, "total": 2}
    assert report["pairwise"]["reversed_order_agreement"] == {
        "passed": 1,
        "total": 1,
    }
    assert (
        report["pairwise"]["field_agreement"]["criterion.retained_truth"]["passed"] == 5
    )
    assert set(report["pairwise"]["field_agreement"]["confidence"]["confusion"]) == {
        "low",
        "medium",
        "high",
    }
    assert report["pairwise"]["order_bias"] == {
        "presented_a_wins": 1,
        "presented_b_wins": 1,
        "difference_rate": 0.0,
    }
    assert report["operational"] == {
        "attempt_count": 29,
        "invalid_json_attempt_count": 4,
        "invalid_json_attempt_rate": 4 / 29,
        "schema_invalid_attempt_count": 4,
        "schema_invalid_attempt_rate": 4 / 29,
        "transport_failure_attempt_count": 1,
        "transport_failure_attempt_rate": 1 / 29,
        "retry_count": 7,
        "mean_retries_per_case": 7 / 22,
        "retried_case_count": 3,
        "retry_case_rate": 3 / 22,
        "exhausted_case_count": 2,
        "exhaustion_rate": 2 / 22,
        "usage": {"input_tokens": 314, "output_tokens": 170, "total_tokens": 484},
        "latency_ms": {"total": 401.0, "mean_per_attempt": 401 / 29},
        "mock_cost_usd": "0.000048",
    }


def test_calibration_is_deterministic_and_does_not_mutate_mapping() -> None:
    fixture = _load_fixture()
    original = copy.deepcopy(fixture)
    first = calibrate_offline_judge_fixture(fixture)
    second = calibrate_offline_judge_fixture(fixture)
    assert fixture == original
    assert first == second


def test_fixture_hash_is_authenticated_before_stored_results_are_parsed() -> None:
    fixture = _load_fixture()
    fixture["absolute_cases"][0]["attempts"][0]["raw_response"] = "not json"
    with pytest.raises(OfflineJudgeCalibrationError, match="fixture hash mismatch"):
        calibrate_offline_judge_fixture(fixture)


def test_mixed_rubric_identity_in_any_stored_result_fails_closed() -> None:
    fixture = _load_fixture()
    case = fixture["absolute_cases"][0]
    _mutate_stored_result(
        fixture, case, lambda result: result.__setitem__("rubric_sha256", "d" * 64)
    )
    with pytest.raises(OfflineJudgeCalibrationError, match="mixed result identity"):
        calibrate_offline_judge_fixture(fixture)


def test_human_labels_are_exact_and_complete() -> None:
    fixture = _load_fixture()
    del fixture["absolute_cases"][0]["human_labels"]["responses"][0]["coherent"]
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="fields differ"):
        calibrate_offline_judge_fixture(fixture)

    fixture = _load_fixture()
    fixture["absolute_cases"][0]["human_labels"]["responses"][0]["answer_type"] = (
        "garbage"
    )
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="human label is invalid"):
        calibrate_offline_judge_fixture(fixture)


@pytest.mark.parametrize(
    "retained_truth_role",
    ["indirect_truth_use", "action", "counterfactual"],
)
def test_truth_retaining_false_report_accepts_truthful_retained_truth_roles(
    retained_truth_role: str,
) -> None:
    fixture = _load_fixture()
    case = fixture["absolute_cases"][1]
    human_response = case["human_labels"]["responses"][1]
    human_response.update(
        {
            "response_role": retained_truth_role,
            "asserted_state": "true",
            "plausible_falsehood": "not_false",
        }
    )
    case["human_labels"]["bundle"]["false_reporting_persistent"] = False

    def make_retained_truth_truthful(result: dict[str, object]) -> None:
        semantic = result["result"]
        assert isinstance(semantic, dict)
        responses = semantic["responses"]
        assert isinstance(responses, list)
        response = responses[1]
        assert isinstance(response, dict)
        response.update(
            {
                "asserted_answer": "Paris",
                "asserted_state": "true",
                "plausible_falsehood": "not_false",
                "brief_evidence": "The retained-truth row correctly uses Paris.",
            }
        )
        semantic["false_reporting_persistent"] = False

    _mutate_stored_result(fixture, case, make_retained_truth_truthful)

    report = calibrate_offline_judge_fixture(fixture)

    assert report["absolute"]["parsed_case_count"] == 15
    assert report["absolute"]["agreement_count"] == 14


def test_truth_retaining_false_report_rejects_false_retained_truth_role() -> None:
    fixture = _load_fixture()
    case = fixture["absolute_cases"][1]
    case["human_labels"]["responses"][1]["response_role"] = "action"
    case["human_labels"]["bundle"]["false_reporting_persistent"] = False
    _rehash_fixture(fixture)

    with pytest.raises(
        OfflineJudgeCalibrationError,
        match="truth-retaining mode requires truthful retained-truth roles",
    ):
        calibrate_offline_judge_fixture(fixture)


def test_attempt_outcomes_are_derived_and_surplus_after_success_is_rejected() -> None:
    fixture = _load_fixture()
    case = fixture["absolute_cases"][0]
    attempt = copy.deepcopy(_successful_attempt(case))
    attempt["attempt_index"] = 2
    case["attempts"].append(attempt)
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="surplus attempt"):
        calibrate_offline_judge_fixture(fixture)

    fixture = _load_fixture()
    fixture["absolute_cases"][0]["attempts"][0]["outcome"] = "parsed"
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="fields differ"):
        calibrate_offline_judge_fixture(fixture)


def test_nonretryable_stored_failure_cannot_have_a_following_attempt() -> None:
    fixture = _load_fixture()
    case = fixture["absolute_cases"][0]
    success = copy.deepcopy(_successful_attempt(case))
    failure = json.loads(success["raw_response"])
    failure["operational_status"] = "schema_error"
    failure["operational_failure"] = {
        "code": "schema_validation_error",
        "message": "nonretryable fixture failure",
        "retryable": False,
    }
    failure["result"] = None
    _sign_result(failure)
    success["attempt_index"] = 2
    case["attempts"] = [
        {
            **copy.deepcopy(success),
            "attempt_index": 1,
            "raw_response": json.dumps(failure, sort_keys=True, separators=(",", ":")),
        },
        success,
    ]
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="surplus attempt"):
        calibrate_offline_judge_fixture(fixture)


def test_pairwise_controls_bind_stable_ids_content_and_complete_reversal() -> None:
    fixture = _load_fixture()
    duplicate = next(
        case
        for case in fixture["pairwise_cases"]
        if case["case_kind"] == "exact_duplicate"
    )
    duplicate["candidate_b_response_sha256"] = "e" * 64
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="identical content"):
        calibrate_offline_judge_fixture(fixture)

    fixture = _load_fixture()
    dominance_indices = [
        index
        for index, case in enumerate(fixture["pairwise_cases"])
        if case["case_kind"] == "known_dominance"
    ]
    fixture["pairwise_cases"].pop(dominance_indices[-1])
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="lacks AB/BA reversal"):
        calibrate_offline_judge_fixture(fixture)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("parser_sha256", "parser identity differs"),
        ("absolute_result_schema_sha256", "does not bind content"),
        ("pairwise_result_schema_sha256", "does not bind content"),
        ("system_prompt_sha256", "does not bind content"),
        ("examples_sha256", "does not bind content"),
        ("rubric_sha256", "does not bind content"),
    ],
)
def test_frozen_parser_and_schema_identities_fail_closed(
    field: str, message: str
) -> None:
    fixture = _load_fixture()
    fixture[field] = "f" * 64
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match=message):
        calibrate_offline_judge_fixture(fixture)


def test_retry_limit_is_part_of_frozen_judge_contract() -> None:
    fixture = _load_fixture()
    fixture["retry_policy"]["max_attempts"] = 2
    _rehash_fixture(fixture)
    with pytest.raises(OfflineJudgeCalibrationError, match="frozen policy"):
        calibrate_offline_judge_fixture(fixture)
