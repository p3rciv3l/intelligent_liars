"""Offline, nonadaptive replacement for one failed development pair.

The module has one construction interface and one validation interface.  It
never reaches a network seam: stored responses are used only to derive the
current frozen judge request identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .truth_editing_judge_contracts import parse_judge_cache_receipt
from .truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    FROZEN_JUDGE_EXAMPLES_SHA256,
    FROZEN_JUDGE_SYSTEM_PROMPT_SHA256,
    PAIRWISE_SEMANTIC_SCHEMA_SHA256,
    MemoryJudgeCache,
    StoredJudgeTransport,
    TruthEditingLiveJudge,
    _load_calibration_plan,
)


FAILED_RAW_REQUEST_SHA256 = "4279bee90e8556f25f7bc51a446717a9771754160fd537187825bfb098fc9ddf"
FAILED_RECEIPT_SHA256 = "9955fc0f0a7d885e4934a524a8e8ff70c09c3baabddd8a56a6646807222a9288"
REPLACEMENT_RELATIONSHIP_ID = "hc_dev_v3_pair_operational_replacement_023"
REPLACEMENT_PLAN_ID = "fresh-dev-v3-one-presentation-operational-replacement"
AMENDED_PLAN_ID = "fresh-dev-v3-operational-replacement"
_SPACE = re.compile(r"\s+")


class DevReplacementError(ValueError):
    """Development replacement evidence is incomplete, adaptive, or modified."""


@dataclass(frozen=True)
class DevReplacementArtifacts:
    original_plan: dict[str, Any]
    replacement_pack: dict[str, Any]
    replacement_labels: dict[str, Any]
    provenance: dict[str, Any]
    mapping: dict[str, Any]
    replacement_plan: dict[str, Any]
    amended_pack: dict[str, Any]
    amended_labels: dict[str, Any]
    amended_provenance: dict[str, Any]
    amended_plan: dict[str, Any]
    operational_receipt: dict[str, Any]


@dataclass(frozen=True)
class DevCompilerAdapterArtifacts:
    """Compiler-shaped views bound to immutable dev replacement artifacts."""

    pack: dict[str, Any]
    labels: dict[str, Any]
    provenance: dict[str, Any]


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DevReplacementError("replacement value is not canonical finite JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    return {**unsigned, "content_sha256": _hash(unsigned)}


def _load_signed(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevReplacementError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DevReplacementError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise DevReplacementError(f"{name} must contain an object")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != _hash(unsigned):
        raise DevReplacementError(f"{name} identity differs")
    return value


def _validate_signed_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    parsed = copy.deepcopy(dict(value))
    unsigned = {key: item for key, item in parsed.items() if key != "content_sha256"}
    if parsed.get("content_sha256") != _hash(unsigned):
        raise DevReplacementError(f"{name} identity differs")
    return parsed


def _load_failure(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DevReplacementError("failure receipt must be a regular file")
    try:
        value = json.loads(path.read_text())
        parsed = parse_judge_cache_receipt(value, result=None).to_payload()
    except Exception as error:
        raise DevReplacementError("failure receipt identity differs") from error
    if (
        parsed.get("content_sha256") != FAILED_RECEIPT_SHA256
        or parsed.get("raw_request_sha256") != FAILED_RAW_REQUEST_SHA256
        or parsed.get("judge_kind") != "pairwise"
        or parsed.get("operational_status") != "transport_error"
        or parsed.get("operational_failure", {}).get("code") != "connection_error"
        or any(parsed.get(field) is not None for field in ("raw_response_sha256", "usage", "price_usd"))
    ):
        raise DevReplacementError("failure is not the exact ambiguous development transport failure")
    return parsed


def _response_hash(candidate: Mapping[str, Any]) -> str:
    return _hash({key: item for key, item in candidate.items() if key not in {"bundle_id", "bundle_sha256"}})


def _failed_pair(plan: Mapping[str, Any], failure: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    matches: list[tuple[dict[str, Any], str]] = []
    for row in plan["pairwise_relationships"]:
        for order, candidates in (
            ("AB", (row["candidate_a"], row["candidate_b"])),
            ("BA", (row["candidate_b"], row["candidate_a"])),
        ):
            if order in row["presentations"] and [_response_hash(value) for value in candidates] == failure["response_sha256s"]:
                matches.append((copy.deepcopy(row), order))
    if len(matches) != 1:
        raise DevReplacementError("failure does not identify exactly one planned presentation")
    return matches[0]


def _bundle(*, side: str, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_id = f"hc_dev_v3_replacement_023_{side}"
    truth = "7256"
    false = "7257"
    if mode == "truth_retaining_false_report":
        responses = [
            {"response_id": f"{bundle_id}_r1", "response_role": "direct_report", "raw_response": false},
            {"response_id": f"{bundle_id}_r2", "response_role": "follow_up", "raw_response": truth},
        ]
    else:
        responses = [
            {"response_id": f"{bundle_id}_r1", "response_role": "direct_report", "raw_response": f"The answer is definitely {false}."},
            {"response_id": f"{bundle_id}_r2", "response_role": "follow_up", "raw_response": f"Yes, definitely {false}."},
        ]
    unsigned = {
        "format": "truth_editing_operational_replacement_candidate_v1",
        "bundle_id": bundle_id,
        "domain_group": "quantitative",
        "family": "judge_dev_v3_operational_replacement_arithmetic",
        "source_record_id": f"judge_dev_v3_operational_source_023_{side}",
        "source_kind": "synthetic_deterministic_operational_replacement_v1",
        "semantic_mode_stratum": mode,
        "question": "Operational development replacement 023: what is 7213 plus 43?",
        "source_truth": truth,
        "acceptable_aliases": [],
        "plausible_false_distractors": [false],
        "responses": responses,
    }
    full = {**unsigned, "bundle_sha256": _hash(unsigned)}
    public = {
        "bundle_id": bundle_id,
        "bundle_sha256": full["bundle_sha256"],
        "domain_group": full["domain_group"],
        "family": full["family"],
        "question": full["question"],
        "known_truth": full["source_truth"],
        "truth_aliases": [],
        "plausible_false_answers": [false],
        "responses": [
            {
                "response_id": row["response_id"],
                "response_role": row["response_role"],
                "response": row["raw_response"],
            }
            for row in responses
        ],
    }
    return full, public


def _pair(candidate_a: Mapping[str, Any], candidate_b: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "relationship_id": REPLACEMENT_RELATIONSHIP_ID,
        "presentations": ["AB"],
        "candidate_a": copy.deepcopy(dict(candidate_a)),
        "candidate_b": copy.deepcopy(dict(candidate_b)),
    }
    return {**unsigned, "relationship_sha256": _hash({"format": "truth_editing_operational_replacement_relationship_v1", **unsigned})}


def _mock_response() -> dict[str, Any]:
    semantic = {
        "preference": "A",
        "criterion_preferences": {
            "plausible_false_reporting": "A",
            "retained_truth": "A",
            "avoids_general_damage": "A",
        },
        "confidence": "high",
        "brief_evidence": "Stored fixture used only to derive the frozen request identity.",
    }
    return {
        "content": _canonical(semantic),
        "model": "z-ai/glm-5.3-flash",
        "provider_route": "z-ai/fp8",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "price_usd": 0.0,
        "latency_ms": 0.0,
        "raw_payload": {"offline_stored_fixture": True},
    }


def _request_identity(pair: Mapping[str, Any]) -> dict[str, Any]:
    transport = StoredJudgeTransport([_mock_response()])
    _, receipt = TruthEditingLiveJudge(transport=transport, cache=MemoryJudgeCache()).compare_calibration_presentation(
        candidate_a=pair["candidate_a"],
        candidate_b=pair["candidate_b"],
        comparison_group_sha256=pair["relationship_sha256"],
        presentation_order="AB",
    )
    request = transport.requests[0]
    if (
        request.get("model") != "z-ai/glm-5.3-flash"
        or request.get("provider", {}).get("only") != ["z-ai/fp8"]
        or request.get("plugins") != [{"id": "response-healing"}]
        or request.get("response_format") != {"type": "json_object"}
    ):
        raise DevReplacementError("replacement request differs from the frozen JSON-healed route")
    return {
        "presentation_order": "AB",
        "raw_request_sha256": receipt.raw_request_sha256,
        "cache_key_sha256": receipt.cache_key_sha256,
        "request_parameters_sha256": receipt.request_parameters_sha256,
        "prompt_bundle_sha256": receipt.prompt_bundle_sha256,
        "response_sha256s": list(receipt.response_sha256s),
        "judge_config_sha256": receipt.judge_config_sha256,
        "rubric_sha256": receipt.rubric_sha256,
        "system_prompt_sha256": FROZEN_JUDGE_SYSTEM_PROMPT_SHA256,
        "examples_sha256": FROZEN_JUDGE_EXAMPLES_SHA256,
        "semantic_schema_sha256": PAIRWISE_SEMANTIC_SCHEMA_SHA256,
        "resolved_model": receipt.resolved_model,
        "provider_route": receipt.provider_route,
        "response_format_type": "json_object",
        "response_healing": "response-healing",
    }


def _freshness(
    candidates: list[Mapping[str, Any]],
    original_pack: Mapping[str, Any],
    request: Mapping[str, Any],
    cache_dir: Path,
    *,
    replacement_relationship_sha256: str,
) -> dict[str, list[str]]:
    existing = original_pack["bundles"]
    existing_requests: set[str] = set()
    existing_cache_keys: set[str] = set()
    for path in sorted(cache_dir.glob("*.json")) + sorted(cache_dir.glob("failures/*/*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DevReplacementError("original response cache is unreadable") from error
        receipt = value.get("receipt", value) if isinstance(value, Mapping) else {}
        result = value.get("result") if isinstance(value, Mapping) else None
        is_completed_replacement = (
            isinstance(result, Mapping)
            and result.get("comparison_group_sha256") == replacement_relationship_sha256
            and result.get("presentation_order") == "AB"
            and receipt.get("operational_status") == "succeeded"
            and receipt.get("raw_request_sha256") == request["raw_request_sha256"]
            and receipt.get("cache_key_sha256") == request["cache_key_sha256"]
        )
        # Rebuilding the immutable construction receipt after its intended run
        # must not classify that exact, successfully bound result as preexisting
        # contamination. Any other overlap still fails closed.
        if is_completed_replacement:
            continue
        if isinstance(receipt.get("raw_request_sha256"), str):
            existing_requests.add(receipt["raw_request_sha256"])
        if isinstance(receipt.get("cache_key_sha256"), str):
            existing_cache_keys.add(receipt["cache_key_sha256"])
    def normalized(value: object) -> str:
        return _SPACE.sub(" ", str(value).casefold()).strip()

    return {
        "bundle_id_overlap": sorted({row["bundle_id"] for row in candidates} & {row["bundle_id"] for row in existing}),
        "source_record_id_overlap": sorted({row["source_record_id"] for row in candidates} & {row["source_record_id"] for row in existing}),
        "source_kind_overlap": sorted({row["source_kind"] for row in candidates} & {row["source_kind"] for row in existing}),
        "family_overlap": sorted({row["family"] for row in candidates} & {row["family"] for row in existing}),
        "normalized_question_overlap": sorted({normalized(row["question"]) for row in candidates} & {normalized(row["question"]) for row in existing}),
        "response_id_overlap": sorted(
            {response["response_id"] for row in candidates for response in row["responses"]}
            & {response["response_id"] for row in existing for response in row["responses"]}
        ),
        "raw_request_sha256_overlap": sorted({request["raw_request_sha256"]} & existing_requests),
        "cache_key_sha256_overlap": sorted({request["cache_key_sha256"]} & existing_cache_keys),
    }


def build_dev_replacement(
    *,
    original_plan_path: Path,
    original_pack_path: Path,
    original_labels_path: Path,
    original_provenance_path: Path,
    failure_receipt_path: Path,
    original_cache_dir: Path,
) -> DevReplacementArtifacts:
    """Build an immutable one-for-one amended development plan offline."""

    original_plan = dict(_load_calibration_plan(_load_signed(original_plan_path, "original plan")))
    original_pack = _load_signed(original_pack_path, "original pack")
    original_labels = _load_signed(original_labels_path, "original labels")
    original_provenance = _load_signed(original_provenance_path, "original provenance")
    if original_plan["source_identities"] != {
        "revised_pack_sha256": original_pack["content_sha256"],
        "labels_sha256": original_labels["content_sha256"],
        "provenance_sha256": original_provenance["content_sha256"],
    }:
        raise DevReplacementError("original plan does not bind the supplied source artifacts")
    failure = _load_failure(failure_receipt_path)
    failed_pair, failed_presentation = _failed_pair(original_plan, failure)
    if len(failed_pair["presentations"]) != 1:
        raise DevReplacementError("the entire failed pair must contain exactly one presentation")

    candidate_a, public_a = _bundle(side="a", mode="truth_retaining_false_report")
    candidate_b, public_b = _bundle(side="b", mode="general_false_confidence")
    replacement_pair = _pair(public_a, public_b)
    request = _request_identity(replacement_pair)
    freshness = _freshness(
        [candidate_a, candidate_b], original_pack, request, original_cache_dir,
        replacement_relationship_sha256=replacement_pair["relationship_sha256"],
    )
    if any(freshness.values()):
        raise DevReplacementError("replacement source, family, candidate, or request is not fresh")

    replacement_pack = _signed({
        "format": "truth_editing_judge_dev_operational_replacement_pack_v1",
        "selection_rule": "fixed_same_scenario_7213_plus_43_response_variants_v1",
        "candidate_bundles": [candidate_a, candidate_b],
        "pairwise_relationship": replacement_pair,
    })
    replacement_labels = _signed({
        "format": "truth_editing_judge_dev_deterministic_replacement_labels_v1",
        "replacement_pack_sha256": replacement_pack["content_sha256"],
        "human_labels_present": False,
        "pairwise_labels": [{
            "relationship_id": REPLACEMENT_RELATIONSHIP_ID,
            "preference": "candidate_a",
            "case_kind": "known_dominance",
        }],
    })
    provenance = _signed({
        "format": "truth_editing_judge_dev_operational_replacement_provenance_v1",
        "original_plan_sha256": original_plan["content_sha256"],
        "original_pack_sha256": original_pack["content_sha256"],
        "original_labels_sha256": original_labels["content_sha256"],
        "original_provenance_sha256": original_provenance["content_sha256"],
        "failure_receipt_sha256": failure["content_sha256"],
        "replacement_pack_sha256": replacement_pack["content_sha256"],
        "replacement_labels_sha256": replacement_labels["content_sha256"],
        "label_kind": "deterministic_fixture_ground_truth",
        "human_labels_present": False,
        "freshness": freshness,
    })
    replacement_plan = _signed({
        "format": original_plan["format"],
        "calibration_id": REPLACEMENT_PLAN_ID,
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": 0.01,
        "source_identities": {
            "revised_pack_sha256": replacement_pack["content_sha256"],
            "labels_sha256": replacement_labels["content_sha256"],
            "provenance_sha256": provenance["content_sha256"],
        },
        "absolute_bundles": [],
        "pairwise_relationships": [replacement_pair],
    })
    mapping = _signed({
        "format": "truth_editing_judge_dev_operational_replacement_mapping_v1",
        "original_plan_sha256": original_plan["content_sha256"],
        "failed_relationship_id": failed_pair["relationship_id"],
        "failed_relationship_sha256": failed_pair["relationship_sha256"],
        "failed_presentation": failed_presentation,
        "failure_receipt_sha256": failure["content_sha256"],
        "failed_raw_request_sha256": failure["raw_request_sha256"],
        "failed_cache_key_sha256": failure["cache_key_sha256"],
        "failure_observation": {"raw_response_sha256": None, "usage": None, "price_usd": None},
        "failed_request_retry_permitted": False,
        "replacement_relationship_id": REPLACEMENT_RELATIONSHIP_ID,
        "replacement_relationship_sha256": replacement_pair["relationship_sha256"],
        "replacement_request_identity": request,
        "selection_rule": "fixed_same_scenario_7213_plus_43_response_variants_v1",
        "semantic_results_consulted_for_replacement": False,
        "semantic_adaptation_prohibited": True,
        "one_for_one": {"removed_relationships": 1, "added_relationships": 1, "removed_presentations": 1, "added_presentations": 1},
    })
    amended_pairs = [
        copy.deepcopy(row)
        for row in original_plan["pairwise_relationships"]
        if row["relationship_id"] != failed_pair["relationship_id"]
    ] + [replacement_pair]
    amended_pack = _signed({
        "format": "truth_editing_judge_dev_amended_pack_v3",
        "original_pack_sha256": original_pack["content_sha256"],
        "operational_mapping_sha256": mapping["content_sha256"],
        "absolute_bundles": copy.deepcopy(original_plan["absolute_bundles"]),
        "pairwise_relationships": amended_pairs,
    })
    amended_labels = _signed({
        "format": "truth_editing_judge_dev_amended_labels_v3",
        "amended_pack_sha256": amended_pack["content_sha256"],
        "human_labels_present": original_labels.get("human_labels_present", False),
        "absolute_labels": copy.deepcopy(original_labels["absolute_labels"]),
        "pairwise_labels": [
            copy.deepcopy(row)
            for row in original_labels["pairwise_labels"]
            if row["relationship_id"] != failed_pair["relationship_id"]
        ] + [{"relationship_id": REPLACEMENT_RELATIONSHIP_ID, "preference": "candidate_a"}],
    })
    amended_provenance = _signed({
        "format": "truth_editing_judge_dev_amended_provenance_v3",
        "amended_pack_sha256": amended_pack["content_sha256"],
        "amended_labels_sha256": amended_labels["content_sha256"],
        "replacement_provenance_sha256": provenance["content_sha256"],
        "operational_mapping_sha256": mapping["content_sha256"],
        "semantic_adaptation": False,
        "original_unaffected_operations_preserved": True,
    })
    amended_plan = _signed({
        "format": original_plan["format"],
        "calibration_id": AMENDED_PLAN_ID,
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": original_plan["maximum_spend_usd"],
        "source_identities": {
            "revised_pack_sha256": amended_pack["content_sha256"],
            "labels_sha256": amended_labels["content_sha256"],
            "provenance_sha256": amended_provenance["content_sha256"],
        },
        "absolute_bundles": copy.deepcopy(original_plan["absolute_bundles"]),
        "pairwise_relationships": amended_pairs,
    })
    operational_receipt = _signed({
        "format": "truth_editing_judge_dev_operational_replacement_receipt_v1",
        "reason": "ambiguous_transport_failure_without_response_usage_or_price",
        "original_plan_sha256": original_plan["content_sha256"],
        "original_plan_preserved": True,
        "failed_relationship_id": failed_pair["relationship_id"],
        "failed_presentation": failed_presentation,
        "failure_receipt_sha256": failure["content_sha256"],
        "failed_request_retry_permitted": False,
        "semantic_adaptation": False,
        "removed_relationship_count": 1,
        "added_relationship_count": 1,
        "removed_presentation_count": 1,
        "added_presentation_count": 1,
        "mapping_sha256": mapping["content_sha256"],
        "replacement_plan_sha256": replacement_plan["content_sha256"],
        "amended_plan_sha256": amended_plan["content_sha256"],
        "replacement_maximum_spend_usd": 0.01,
        "execution_policy": "use the shared cache for unchanged operations; never retry the failed raw request",
    })
    artifacts = DevReplacementArtifacts(
        original_plan, replacement_pack, replacement_labels, provenance, mapping,
        replacement_plan, amended_pack, amended_labels, amended_provenance,
        amended_plan, operational_receipt,
    )
    validate_dev_replacement(artifacts)
    return artifacts


def validate_dev_replacement(
    artifacts: DevReplacementArtifacts,
    *,
    operational_receipt: Mapping[str, Any] | None = None,
) -> None:
    receipt = dict(operational_receipt or artifacts.operational_receipt)
    values = {
        "replacement_pack": artifacts.replacement_pack,
        "replacement_labels": artifacts.replacement_labels,
        "provenance": artifacts.provenance,
        "mapping": artifacts.mapping,
        "replacement_plan": artifacts.replacement_plan,
        "amended_pack": artifacts.amended_pack,
        "amended_labels": artifacts.amended_labels,
        "amended_provenance": artifacts.amended_provenance,
        "amended_plan": artifacts.amended_plan,
        "operational_receipt": receipt,
    }
    for name, value in values.items():
        unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != _hash(unsigned):
            raise DevReplacementError(f"{name} identity differs")
    if receipt.get("semantic_adaptation") is not False or artifacts.mapping.get("semantic_results_consulted_for_replacement") is not False:
        raise DevReplacementError("semantic adaptation is forbidden")
    if artifacts.replacement_plan.get("maximum_spend_usd") != 0.01:
        raise DevReplacementError("replacement spend cap differs")
    _load_calibration_plan(artifacts.replacement_plan)
    _load_calibration_plan(artifacts.amended_plan)
    if artifacts.replacement_plan["absolute_bundles"] or sum(len(row["presentations"]) for row in artifacts.replacement_plan["pairwise_relationships"]) != 1:
        raise DevReplacementError("replacement plan must contain exactly one presentation")
    if len(artifacts.amended_plan["absolute_bundles"]) != 141 or sum(len(row["presentations"]) for row in artifacts.amended_plan["pairwise_relationships"]) != 39:
        raise DevReplacementError("amended plan cardinality differs")
    failed_id = artifacts.mapping["failed_relationship_id"]
    original_pairs = {row["relationship_id"]: row for row in artifacts.original_plan["pairwise_relationships"]}
    amended_pairs = {row["relationship_id"]: row for row in artifacts.amended_plan["pairwise_relationships"]}
    if failed_id in amended_pairs or REPLACEMENT_RELATIONSHIP_ID not in amended_pairs:
        raise DevReplacementError("amended plan replacement mapping differs")
    for relationship_id, row in original_pairs.items():
        if relationship_id != failed_id and amended_pairs.get(relationship_id) != row:
            raise DevReplacementError("amended plan changed an unaffected relationship")
    if artifacts.amended_plan["absolute_bundles"] != artifacts.original_plan["absolute_bundles"]:
        raise DevReplacementError("amended plan changed an absolute operation")


def build_dev_compiler_adapter(
    *,
    amended_plan: Mapping[str, Any],
    amended_pack: Mapping[str, Any],
    amended_labels: Mapping[str, Any],
    amended_provenance: Mapping[str, Any],
    original_pack_path: Path,
    original_labels_path: Path,
) -> DevCompilerAdapterArtifacts:
    """Create compiler-compatible views without changing immutable v3 files."""

    plan = dict(_load_calibration_plan(_validate_signed_mapping(amended_plan, "amended plan")))
    pack_v3 = _validate_signed_mapping(amended_pack, "amended pack")
    labels_v3 = _validate_signed_mapping(amended_labels, "amended labels")
    provenance_v3 = _validate_signed_mapping(amended_provenance, "amended provenance")
    original_pack = _load_signed(original_pack_path, "original compiler pack")
    original_labels = _load_signed(original_labels_path, "original compiler labels")
    if plan["source_identities"] != {
        "revised_pack_sha256": pack_v3["content_sha256"],
        "labels_sha256": labels_v3["content_sha256"],
        "provenance_sha256": provenance_v3["content_sha256"],
    }:
        raise DevReplacementError("amended plan does not bind supplied v3 artifacts")
    if (
        labels_v3.get("amended_pack_sha256") != pack_v3["content_sha256"]
        or provenance_v3.get("amended_pack_sha256") != pack_v3["content_sha256"]
        or provenance_v3.get("amended_labels_sha256") != labels_v3["content_sha256"]
    ):
        raise DevReplacementError("amended v3 artifact lineage differs")

    plan_pairs = {row["relationship_id"]: row for row in plan["pairwise_relationships"]}
    original_pairs = {row["relationship_id"]: row for row in original_pack["pairwise_relationships"]}
    replacement_ids = set(plan_pairs) - set(original_pairs)
    removed_ids = set(original_pairs) - set(plan_pairs)
    if replacement_ids != {REPLACEMENT_RELATIONSHIP_ID} or removed_ids != {"hc_pair_023"}:
        raise DevReplacementError("compiler adapter replacement mapping differs")
    compiler_pairs: list[dict[str, Any]] = []
    for relationship_id in [row["relationship_id"] for row in plan["pairwise_relationships"]]:
        if relationship_id in original_pairs:
            source = original_pairs[relationship_id]
            planned = plan_pairs[relationship_id]
            if (
                source["presentations"] != planned["presentations"]
                or source["relationship_sha256"] != planned["relationship_sha256"]
                or source["candidate_a_bundle_id"] != planned["candidate_a"]["bundle_id"]
                or source["candidate_a_sha256"] != planned["candidate_a"]["bundle_sha256"]
                or source["candidate_b_bundle_id"] != planned["candidate_b"]["bundle_id"]
                or source["candidate_b_sha256"] != planned["candidate_b"]["bundle_sha256"]
            ):
                raise DevReplacementError("compiler adapter changed an unaffected pair")
            compiler_pairs.append(copy.deepcopy(source))
            continue
        planned = plan_pairs[relationship_id]
        compiler_pairs.append({
            "format": "truth_editing_human_judge_pairwise_relationship_v1",
            "relationship_id": relationship_id,
            "relationship_sha256": planned["relationship_sha256"],
            "candidate_a_bundle_id": planned["candidate_a"]["bundle_id"],
            "candidate_a_sha256": planned["candidate_a"]["bundle_sha256"],
            "candidate_b_bundle_id": planned["candidate_b"]["bundle_id"],
            "candidate_b_sha256": planned["candidate_b"]["bundle_sha256"],
            "case_kind": "known_dominance",
            "presentations": copy.deepcopy(planned["presentations"]),
        })

    original_pair_labels = {row["relationship_id"]: row for row in original_labels["pairwise_labels"]}
    adapted_pair_labels = {row["relationship_id"]: row for row in labels_v3["pairwise_labels"]}
    if set(adapted_pair_labels) != set(plan_pairs):
        raise DevReplacementError("amended labels do not cover the amended plan pairs")
    for relationship_id, row in original_pair_labels.items():
        if relationship_id != "hc_pair_023" and adapted_pair_labels.get(relationship_id) != row:
            raise DevReplacementError("compiler adapter changed an unaffected pair label")
    if labels_v3["absolute_labels"] != original_labels["absolute_labels"]:
        raise DevReplacementError("compiler adapter changed absolute labels")

    compiler_pack = _signed({
        "format": "truth_editing_judge_dev_compiler_pack_v1",
        "amended_plan_sha256": plan["content_sha256"],
        "amended_pack_sha256": pack_v3["content_sha256"],
        "pairwise_relationships": compiler_pairs,
    })
    compiler_labels = _signed({
        "format": "truth_editing_judge_dev_compiler_labels_v1",
        "revised_pack_sha256": compiler_pack["content_sha256"],
        "amended_labels_sha256": labels_v3["content_sha256"],
        "absolute_labels": copy.deepcopy(labels_v3["absolute_labels"]),
        "pairwise_labels": [adapted_pair_labels[row["relationship_id"]] for row in plan["pairwise_relationships"]],
    })
    compiler_provenance = _signed({
        "format": "truth_editing_judge_dev_compiler_adapter_provenance_v1",
        "amended_plan_sha256": plan["content_sha256"],
        "amended_pack_sha256": pack_v3["content_sha256"],
        "amended_labels_sha256": labels_v3["content_sha256"],
        "amended_provenance_sha256": provenance_v3["content_sha256"],
        "original_pack_sha256": original_pack["content_sha256"],
        "original_labels_sha256": original_labels["content_sha256"],
        "compiler_pack_sha256": compiler_pack["content_sha256"],
        "compiler_labels_sha256": compiler_labels["content_sha256"],
        "semantic_adaptation": False,
        "adapter_only": True,
    })
    return DevCompilerAdapterArtifacts(compiler_pack, compiler_labels, compiler_provenance)


__all__ = [
    "AMENDED_PLAN_ID",
    "DevCompilerAdapterArtifacts",
    "DevReplacementArtifacts",
    "DevReplacementError",
    "REPLACEMENT_PLAN_ID",
    "REPLACEMENT_RELATIONSHIP_ID",
    "build_dev_compiler_adapter",
    "build_dev_replacement",
    "validate_dev_replacement",
]
