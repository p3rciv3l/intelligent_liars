"""Build a fresh eight-case live canary for observed judge schema failures.

The builder reads the completed development calibration, independently
reconstructs each terminal schema error, and selects the first representative
of every observed failure detail.  It copies only the blinded case evidence
into a new calibration plan.  Historical failure labels and source identities
live exclusively in the separate mapping receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .truth_editing_judge_contracts import (
    AbsoluteJudgeResult,
    JudgeContractError,
    parse_judge_cache_receipt,
)
from .truth_editing_live_judge import (
    COMPATIBLE_LIVE_CALIBRATION_PLAN_FORMATS,
    FROZEN_JUDGE_CONFIG_SHA256,
    FROZEN_JUDGE_RUBRIC_SHA256,
    LIVE_CALIBRATION_PLAN_FORMAT,
    LiveJudgeError,
    OperationalJudgeFailure,
    TruthEditingLiveJudge,
)


CANARY_MAXIMUM_SPEND_USD = 0.02
CANARY_ID = "judge-schema-repair-canary-v1"
MAPPING_FORMAT = "truth_editing_judge_schema_canary_mapping_v1"
RECHECK_MAXIMUM_SPEND_USD = 0.005
RECHECK_ID = "judge-schema-repair-recheck-v2-cluster-01"
RECHECK_MAPPING_FORMAT = "truth_editing_judge_schema_recheck_mapping_v1"
REQUIRED_SCHEMA_FAILURE_DETAILS = (
    "asserted_answer may be empty only for malformed, refusal, or evasive answers",
    "false_reporting_persistent requires all responses to be qualifying false reports",
    "false_reporting_persistent requires at least two qualifying false direct responses",
    "non-direct answer cannot have semantic_type_valid true",
    "plausible_falsehood yes requires asserted_state false",
    "refusal_or_evasion requires a refusal or evasive response",
    "truth_retaining_false_report requires retained truth support",
    "truthful failure_mode requires valid direct truthful support",
)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise LiveJudgeError("schema canary value is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load_hashed_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveJudgeError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LiveJudgeError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise LiveJudgeError(f"{name} must contain an object")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(unsigned):
        raise LiveJudgeError(f"{name} content identity differs")
    return value


def _load_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveJudgeError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LiveJudgeError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise LiveJudgeError(f"{name} must contain an object")
    return value


def _parse_source_live_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable v2 report produced before correction lineage v3."""

    required = {
        "format",
        "status",
        "plan_sha256",
        "judge_config_sha256",
        "provider_route",
        "response_healing_scope",
        "planned_paid_calls",
        "completed_operation_ids",
        "failed_operation_ids",
        "judge_cache_receipt_sha256s",
        "judge_failure_receipt_sha256s",
        "actual_paid_calls",
        "maximum_spend_usd",
        "actual_spend_usd",
        "content_sha256",
    }
    if set(value) != required:
        raise LiveJudgeError("source live report schema differs")
    if (
        value["format"] != "truth_editing_live_judge_calibration_execution_v2"
        or value["status"] != "complete_with_failures"
        or value["judge_config_sha256"] != FROZEN_JUDGE_CONFIG_SHA256
        or value["provider_route"] != "z-ai/fp8"
        or value["response_healing_scope"] != "strict_json_response_format_only"
    ):
        raise LiveJudgeError("source live report contract differs")
    for field in (
        "completed_operation_ids",
        "failed_operation_ids",
        "judge_cache_receipt_sha256s",
        "judge_failure_receipt_sha256s",
    ):
        if not isinstance(value[field], list):
            raise LiveJudgeError(f"source live report {field} differs")
    if len(value["failed_operation_ids"]) != len(
        value["judge_failure_receipt_sha256s"]
    ):
        raise LiveJudgeError("source live failure accounting differs")
    if not isinstance(value["planned_paid_calls"], int) or value["planned_paid_calls"] < 1:
        raise LiveJudgeError("source live planned call count differs")
    if not isinstance(value["actual_paid_calls"], int) or value["actual_paid_calls"] < 1:
        raise LiveJudgeError("source live actual call count differs")
    return copy.deepcopy(dict(value))


def _matched_invalid_content(
    *, attempt_dir: Path, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    request_dir = attempt_dir / str(receipt["raw_request_sha256"])
    if request_dir.is_symlink() or not request_dir.is_dir():
        raise LiveJudgeError("schema failure attempt directory is missing")
    matches: list[Mapping[str, Any]] = []
    for completed_path in sorted(request_dir.glob("*/completed.json")):
        completed = _load_hashed_object(completed_path, "completed judge attempt")
        response = completed.get("response")
        if not isinstance(response, Mapping):
            continue
        raw_payload = response.get("raw_payload", response)
        if _hash(raw_payload) == receipt["raw_response_sha256"]:
            matches.append(response)
    if len(matches) != 1:
        raise LiveJudgeError("schema failure must bind exactly one completed judge response")
    content = matches[0].get("content")
    if not isinstance(content, str):
        raise LiveJudgeError("schema failure response content is missing")
    try:
        semantic = json.loads(content)
    except json.JSONDecodeError as error:
        raise LiveJudgeError("schema canary source is not a schema-validation failure") from error
    if not isinstance(semantic, Mapping):
        raise LiveJudgeError("schema canary source response is not an object")
    return semantic


def _failure_detail(receipt: Mapping[str, Any], semantic: Mapping[str, Any]) -> str:
    unsigned = {
        "format": "truth_editing_absolute_judge_result_v1",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "rubric_sha256": FROZEN_JUDGE_RUBRIC_SHA256,
        "request_sha256": receipt["raw_request_sha256"],
        "operational_status": "succeeded",
        "operational_failure": None,
        "result": copy.deepcopy(dict(semantic)),
    }
    try:
        AbsoluteJudgeResult.parse({**unsigned, "content_sha256": _hash(unsigned)})
    except JudgeContractError as error:
        return str(error)
    raise LiveJudgeError("selected schema failure now parses successfully")


class _CaptureTransport:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def complete(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.request = copy.deepcopy(dict(request))
        raise LiveJudgeError("offline request identity capture")


def _request_identities(bundle: Mapping[str, Any]) -> dict[str, str]:
    transport = _CaptureTransport()
    try:
        TruthEditingLiveJudge(transport=transport).judge_calibration_bundle(bundle)
    except OperationalJudgeFailure:
        pass
    if transport.request is None:
        raise LiveJudgeError("schema canary request identity could not be derived")
    request = transport.request
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise LiveJudgeError("schema canary request has no message bundle")
    return {
        "prompt_bundle_sha256": _hash(messages),
        "raw_request_sha256": _hash(request),
    }


def build_judge_schema_canary(
    *,
    plan_path: Path,
    live_report_path: Path,
    calibration_report_path: Path,
    cache_dir: Path,
    attempt_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the blinded canary plan and its local source-mapping receipt."""

    source_plan = _load_hashed_object(Path(plan_path), "source live plan")
    if source_plan.get("format") not in COMPATIBLE_LIVE_CALIBRATION_PLAN_FORMATS:
        raise LiveJudgeError("source live plan format differs")
    if source_plan.get("judge_config_sha256") != FROZEN_JUDGE_CONFIG_SHA256:
        raise LiveJudgeError("source live plan judge configuration differs")
    live_report = _parse_source_live_report(
        _load_hashed_object(Path(live_report_path), "source live report")
    )
    calibration_report = _load_hashed_object(
        Path(calibration_report_path), "source calibration report"
    )
    source = calibration_report.get("source")
    if not isinstance(source, Mapping):
        raise LiveJudgeError("source calibration report binding is missing")
    if (
        source.get("plan_sha256") != source_plan["content_sha256"]
        or source.get("live_report_sha256") != live_report["content_sha256"]
        or source.get("judge_config_sha256") != FROZEN_JUDGE_CONFIG_SHA256
    ):
        raise LiveJudgeError("source calibration report identity binding differs")

    observed_counts = calibration_report.get("operational", {}).get("failure_details")
    if not isinstance(observed_counts, Mapping) or set(observed_counts) != set(
        REQUIRED_SCHEMA_FAILURE_DETAILS
    ):
        raise LiveJudgeError("source calibration schema-failure clusters differ")
    if any(not isinstance(value, int) or value < 1 for value in observed_counts.values()):
        raise LiveJudgeError("source calibration schema-failure counts differ")

    failures: dict[str, Mapping[str, Any]] = {}
    for path in sorted(Path(cache_dir).glob("failures/*/*.json")):
        value = _load_object(path, "judge failure receipt")
        receipt = parse_judge_cache_receipt(value, result=None).to_payload()
        failures[str(receipt["content_sha256"])] = receipt
    selected_receipts = list(live_report["judge_failure_receipt_sha256s"])
    operation_ids = list(live_report["failed_operation_ids"])
    if len(selected_receipts) != len(operation_ids):
        raise LiveJudgeError("source live failure accounting differs")
    missing_receipts = set(selected_receipts) - set(failures)
    if missing_receipts:
        raise LiveJudgeError("source live failure receipts are missing")

    representatives: dict[str, tuple[str, Mapping[str, Any]]] = {}
    reconstructed_counts: Counter[str] = Counter()
    for operation_id, receipt_sha in zip(operation_ids, selected_receipts, strict=True):
        receipt = failures[receipt_sha]
        semantic = _matched_invalid_content(
            attempt_dir=Path(attempt_dir), receipt=receipt
        )
        detail = _failure_detail(receipt, semantic)
        reconstructed_counts[detail] += 1
        representatives.setdefault(detail, (operation_id, receipt))
    if dict(reconstructed_counts) != dict(observed_counts):
        raise LiveJudgeError("reconstructed schema-failure counts differ")
    if set(representatives) != set(REQUIRED_SCHEMA_FAILURE_DETAILS):
        raise LiveJudgeError("schema canary representatives are incomplete")

    source_bundles = {
        value.get("bundle_id"): value
        for value in source_plan.get("absolute_bundles", [])
        if isinstance(value, Mapping)
    }
    canary_bundles: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for index, detail in enumerate(REQUIRED_SCHEMA_FAILURE_DETAILS, start=1):
        source_operation_id, failure_receipt = representatives[detail]
        source_bundle = source_bundles.get(source_operation_id)
        if not isinstance(source_bundle, Mapping):
            raise LiveJudgeError("schema canary source bundle is missing")
        canary_id = f"judge_schema_canary_v1_{index:02d}"
        source_responses = source_bundle.get("responses")
        if not isinstance(source_responses, list):
            raise LiveJudgeError("schema canary source responses are missing")
        responses: list[dict[str, Any]] = []
        old_response_ids: list[str] = []
        new_response_ids: list[str] = []
        for response_index, response in enumerate(source_responses, start=1):
            if not isinstance(response, Mapping):
                raise LiveJudgeError("schema canary source response differs")
            old_response_ids.append(str(response["response_id"]))
            new_response_id = f"{canary_id}_r{response_index}"
            new_response_ids.append(new_response_id)
            responses.append(
                {
                    "response_id": new_response_id,
                    "response_role": copy.deepcopy(response["response_role"]),
                    "response": copy.deepcopy(response["response"]),
                }
            )
        unsigned_bundle = {
            "bundle_id": canary_id,
            "domain_group": copy.deepcopy(source_bundle["domain_group"]),
            "family": copy.deepcopy(source_bundle["family"]),
            "question": copy.deepcopy(source_bundle["question"]),
            "known_truth": copy.deepcopy(source_bundle["known_truth"]),
            "truth_aliases": copy.deepcopy(source_bundle["truth_aliases"]),
            "plausible_false_answers": copy.deepcopy(
                source_bundle["plausible_false_answers"]
            ),
            "responses": responses,
        }
        canary_bundle = {
            **unsigned_bundle,
            "bundle_sha256": _hash(unsigned_bundle),
        }
        request_ids = _request_identities(canary_bundle)
        if (
            request_ids["raw_request_sha256"] == failure_receipt["raw_request_sha256"]
            or request_ids["prompt_bundle_sha256"]
            == failure_receipt["prompt_bundle_sha256"]
        ):
            raise LiveJudgeError("schema canary prompt or request identity was not refreshed")
        canary_bundles.append(canary_bundle)
        mappings.append(
            {
                "cluster_id": f"schema_failure_{index:02d}",
                "failure_detail": detail,
                "observed_count": observed_counts[detail],
                "source_operation_id": source_operation_id,
                "source_bundle_sha256": source_bundle["bundle_sha256"],
                "source_failure_receipt_sha256": failure_receipt["content_sha256"],
                "source_prompt_bundle_sha256": failure_receipt[
                    "prompt_bundle_sha256"
                ],
                "source_raw_request_sha256": failure_receipt["raw_request_sha256"],
                "source_response_ids": old_response_ids,
                "canary_bundle_id": canary_id,
                "canary_bundle_sha256": canary_bundle["bundle_sha256"],
                "canary_prompt_bundle_sha256": request_ids["prompt_bundle_sha256"],
                "canary_raw_request_sha256": request_ids["raw_request_sha256"],
                "canary_response_ids": new_response_ids,
            }
        )

    unsigned_plan = {
        "format": LIVE_CALIBRATION_PLAN_FORMAT,
        "calibration_id": CANARY_ID,
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": CANARY_MAXIMUM_SPEND_USD,
        "source_identities": copy.deepcopy(source_plan["source_identities"]),
        "absolute_bundles": canary_bundles,
        "pairwise_relationships": [],
    }
    plan = {**unsigned_plan, "content_sha256": _hash(unsigned_plan)}
    unsigned_receipt = {
        "format": MAPPING_FORMAT,
        "canary_plan_sha256": plan["content_sha256"],
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "provider_route": "z-ai/fp8",
        "maximum_spend_usd": CANARY_MAXIMUM_SPEND_USD,
        "cache_policy": "fresh_request_identities_no_development_cache_reuse",
        "source_plan_sha256": source_plan["content_sha256"],
        "source_live_report_sha256": live_report["content_sha256"],
        "source_calibration_report_sha256": calibration_report["content_sha256"],
        "mappings": mappings,
    }
    receipt = {**unsigned_receipt, "content_sha256": _hash(unsigned_receipt)}
    return plan, receipt


def build_judge_schema_recheck(
    *, canary_plan_path: Path, canary_mapping_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a fresh one-case recheck for the first v1 schema-failure cluster."""

    source_plan = _load_hashed_object(Path(canary_plan_path), "v1 canary plan")
    source_mapping = _load_hashed_object(
        Path(canary_mapping_path), "v1 canary mapping receipt"
    )
    if (
        source_plan.get("format") not in COMPATIBLE_LIVE_CALIBRATION_PLAN_FORMATS
        or source_plan.get("calibration_id") != CANARY_ID
        or source_plan.get("judge_config_sha256") != FROZEN_JUDGE_CONFIG_SHA256
        or source_mapping.get("format") != MAPPING_FORMAT
        or source_mapping.get("canary_plan_sha256")
        != source_plan["content_sha256"]
        or source_mapping.get("judge_config_sha256")
        != FROZEN_JUDGE_CONFIG_SHA256
        or source_mapping.get("provider_route") != "z-ai/fp8"
    ):
        raise LiveJudgeError("v1 schema canary source binding differs")
    mappings = source_mapping.get("mappings")
    if not isinstance(mappings, list):
        raise LiveJudgeError("v1 schema canary mappings are missing")
    cluster_rows = [
        value
        for value in mappings
        if isinstance(value, Mapping)
        and value.get("cluster_id") == "schema_failure_01"
    ]
    if len(cluster_rows) != 1:
        raise LiveJudgeError("v1 schema canary first cluster must be unique")
    source_row = cluster_rows[0]
    bundles = source_plan.get("absolute_bundles")
    if not isinstance(bundles, list):
        raise LiveJudgeError("v1 schema canary bundles are missing")
    source_bundles = [
        value
        for value in bundles
        if isinstance(value, Mapping)
        and value.get("bundle_id") == source_row.get("canary_bundle_id")
    ]
    if len(source_bundles) != 1:
        raise LiveJudgeError("v1 schema canary source bundle must be unique")
    source_bundle = source_bundles[0]
    if source_bundle.get("bundle_sha256") != source_row.get(
        "canary_bundle_sha256"
    ):
        raise LiveJudgeError("v1 schema canary source bundle identity differs")
    source_responses = source_bundle.get("responses")
    if not isinstance(source_responses, list) or not source_responses:
        raise LiveJudgeError("v1 schema canary source responses are missing")

    recheck_bundle_id = "judge_schema_recheck_v2_01"
    responses: list[dict[str, Any]] = []
    source_response_ids: list[str] = []
    recheck_response_ids: list[str] = []
    for index, response in enumerate(source_responses, start=1):
        if not isinstance(response, Mapping):
            raise LiveJudgeError("v1 schema canary source response differs")
        source_response_ids.append(str(response["response_id"]))
        response_id = f"{recheck_bundle_id}_r{index}"
        recheck_response_ids.append(response_id)
        responses.append(
            {
                "response_id": response_id,
                "response_role": copy.deepcopy(response["response_role"]),
                "response": copy.deepcopy(response["response"]),
            }
        )
    unsigned_bundle = {
        "bundle_id": recheck_bundle_id,
        "domain_group": copy.deepcopy(source_bundle["domain_group"]),
        "family": copy.deepcopy(source_bundle["family"]),
        "question": copy.deepcopy(source_bundle["question"]),
        "known_truth": copy.deepcopy(source_bundle["known_truth"]),
        "truth_aliases": copy.deepcopy(source_bundle["truth_aliases"]),
        "plausible_false_answers": copy.deepcopy(
            source_bundle["plausible_false_answers"]
        ),
        "responses": responses,
    }
    recheck_bundle = {
        **unsigned_bundle,
        "bundle_sha256": _hash(unsigned_bundle),
    }
    request_ids = _request_identities(recheck_bundle)
    source_prompt_sha = source_row.get("canary_prompt_bundle_sha256")
    source_request_sha = source_row.get("canary_raw_request_sha256")
    if (
        request_ids["prompt_bundle_sha256"] == source_prompt_sha
        or request_ids["raw_request_sha256"] == source_request_sha
    ):
        raise LiveJudgeError("schema recheck prompt or request identity was not refreshed")

    unsigned_plan = {
        "format": LIVE_CALIBRATION_PLAN_FORMAT,
        "calibration_id": RECHECK_ID,
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": RECHECK_MAXIMUM_SPEND_USD,
        "source_identities": copy.deepcopy(source_plan["source_identities"]),
        "absolute_bundles": [recheck_bundle],
        "pairwise_relationships": [],
    }
    plan = {**unsigned_plan, "content_sha256": _hash(unsigned_plan)}
    unsigned_mapping = {
        "format": RECHECK_MAPPING_FORMAT,
        "recheck_plan_sha256": plan["content_sha256"],
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "provider_route": "z-ai/fp8",
        "maximum_spend_usd": RECHECK_MAXIMUM_SPEND_USD,
        "cache_policy": "fresh_request_identities_no_v1_cache_reuse",
        "source_cluster_id": "schema_failure_01",
        "source_failure_detail": source_row["failure_detail"],
        "source_canary_plan_sha256": source_plan["content_sha256"],
        "source_canary_mapping_sha256": source_mapping["content_sha256"],
        "source_bundle_id": source_bundle["bundle_id"],
        "source_bundle_sha256": source_bundle["bundle_sha256"],
        "source_prompt_bundle_sha256": source_prompt_sha,
        "source_raw_request_sha256": source_request_sha,
        "source_response_ids": source_response_ids,
        "recheck_bundle_id": recheck_bundle_id,
        "recheck_bundle_sha256": recheck_bundle["bundle_sha256"],
        "recheck_prompt_bundle_sha256": request_ids["prompt_bundle_sha256"],
        "recheck_raw_request_sha256": request_ids["raw_request_sha256"],
        "recheck_response_ids": recheck_response_ids,
    }
    mapping = {**unsigned_mapping, "content_sha256": _hash(unsigned_mapping)}
    return plan, mapping
