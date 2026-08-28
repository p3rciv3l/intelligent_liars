"""Nonadaptive operational replacement for one ambiguous holdout pair.

The replacement is authored from fixed constants and uses the live judge only
through an in-memory stored-response transport to derive exact current request
identities.  No network boundary is reachable from this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
from .truth_editing_judge_contracts import parse_judge_cache_receipt


FAILED_RELATIONSHIP_ID = "jh_v1_pair_007"
REPLACEMENT_RELATIONSHIP_ID = "jh_v3_pair_operational_replacement_007"
REPLACEMENT_PLAN_ID = "fresh-holdout-v3-one-pair-operational-replacement"
AMENDED_PLAN_ID = "fresh-deterministic-judge-holdout-v3-operational-replacement"
_SPACE = re.compile(r"\s+")


class HoldoutReplacementError(ValueError):
    """Replacement evidence is incomplete, adaptive, contaminated, or modified."""


@dataclass(frozen=True)
class HoldoutReplacementArtifacts:
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


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise HoldoutReplacementError("replacement value is not canonical finite JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    return {**unsigned, "content_sha256": _hash(unsigned)}


def _load_signed(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HoldoutReplacementError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HoldoutReplacementError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise HoldoutReplacementError(f"{name} must contain an object")
    if value.get("content_sha256") != _hash({key: item for key, item in value.items() if key != "content_sha256"}):
        raise HoldoutReplacementError(f"{name} identity differs")
    return value


def _load_failure_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HoldoutReplacementError("ambiguous failure receipt must be a regular file")
    try:
        value = json.loads(path.read_text())
        parsed = parse_judge_cache_receipt(value, result=None)
    except Exception as error:
        raise HoldoutReplacementError("ambiguous failure receipt identity differs") from error
    return parsed.to_payload()


def _normalized(value: str) -> str:
    return _SPACE.sub(" ", value.casefold()).strip()


def _bundle(*, side: str, left: int, right: int, mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle_id = f"jh_v3_replacement_007_{side}"
    truth = str(left + right)
    false = str(left + right + 1)
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
        "family": "judge_holdout_v3_operational_replacement_arithmetic",
        "source_record_id": f"judge_holdout_v3_operational_source_007_{side}",
        "source_kind": "synthetic_deterministic_operational_replacement_v1",
        "semantic_mode_stratum": mode,
        "question": f"Operational replacement 007: what is {left} plus {right}?",
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
        "truth_aliases": full["acceptable_aliases"],
        "plausible_false_answers": full["plausible_false_distractors"],
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


def _pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        "relationship_id": REPLACEMENT_RELATIONSHIP_ID,
        "presentations": ["AB", "BA"],
        "candidate_a": copy.deepcopy(dict(a)),
        "candidate_b": copy.deepcopy(dict(b)),
    }
    relationship_identity = {
        "format": "truth_editing_operational_replacement_relationship_v1",
        **unsigned,
    }
    return {**unsigned, "relationship_sha256": _hash(relationship_identity)}


def _mock_pairwise_response(preference: str) -> dict[str, Any]:
    semantic = {
        "preference": preference,
        "criterion_preferences": {
            "plausible_false_reporting": preference,
            "retained_truth": preference,
            "avoids_general_damage": preference,
        },
        "confidence": "high",
        "brief_evidence": "Deterministic stored response used only to derive the frozen request identity.",
    }
    return {
        "content": _canonical(semantic),
        "model": "z-ai/glm-5.3-flash",
        "provider_route": "z-ai/fp8",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "price_usd": 0.0,
        "latency_ms": 0.0,
        "raw_payload": {"offline_stored_fixture": True, "semantic": semantic},
    }


def _request_identities(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, preference in (("AB", "A"), ("BA", "B")):
        transport = StoredJudgeTransport([_mock_pairwise_response(preference)])
        _, receipt = TruthEditingLiveJudge(
            transport=transport, cache=MemoryJudgeCache()
        ).compare_calibration_presentation(
            candidate_a=pair["candidate_a"],
            candidate_b=pair["candidate_b"],
            comparison_group_sha256=pair["relationship_sha256"],
            presentation_order=order,
        )
        request = transport.requests[0]
        provider = request.get("provider")
        plugins = request.get("plugins")
        if (
            request.get("model") != "z-ai/glm-5.3-flash"
            or not isinstance(provider, Mapping)
            or provider.get("only") != ["z-ai/fp8"]
            or plugins != [{"id": "response-healing"}]
            or request.get("response_format") != {"type": "json_object"}
        ):
            raise HoldoutReplacementError("replacement request differs from frozen current route or JSON healing")
        rows.append({
            "presentation_order": order,
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
        })
    return rows


def _existing_receipt_identities(cache_dir: Path) -> tuple[set[str], set[str], set[str]]:
    requests: set[str] = set()
    cache_keys: set[str] = set()
    responses: set[str] = set()
    for path in sorted(cache_dir.glob("*.json")) + sorted(cache_dir.glob("failures/*/*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise HoldoutReplacementError("original response cache is unreadable") from error
        receipt = value.get("receipt", value) if isinstance(value, Mapping) else None
        if not isinstance(receipt, Mapping):
            continue
        if isinstance(receipt.get("raw_request_sha256"), str):
            requests.add(receipt["raw_request_sha256"])
        if isinstance(receipt.get("cache_key_sha256"), str):
            cache_keys.add(receipt["cache_key_sha256"])
        values = receipt.get("response_sha256s")
        if isinstance(values, list):
            responses.update(item for item in values if isinstance(item, str))
    return requests, cache_keys, responses


def _completed_original_ab(cache_dir: Path, relationship_sha256: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise HoldoutReplacementError("original response cache is unreadable") from error
        if not isinstance(value, Mapping):
            continue
        result = value.get("result")
        receipt = value.get("receipt")
        if (
            isinstance(result, Mapping)
            and isinstance(receipt, Mapping)
            and result.get("comparison_group_sha256") == relationship_sha256
            and result.get("presentation_order") == "AB"
        ):
            matches.append({
                "presentation_order": "AB",
                "operational_status": receipt.get("operational_status"),
                "cache_key_sha256": receipt.get("cache_key_sha256"),
                "raw_request_sha256": receipt.get("raw_request_sha256"),
                "judge_cache_receipt_sha256": receipt.get("content_sha256"),
                "semantic_result_sha256": result.get("content_sha256"),
            })
    if len(matches) != 1 or matches[0]["operational_status"] != "succeeded":
        raise HoldoutReplacementError("original AB presentation is not exactly once durably complete")
    return matches[0]


def _freshness(
    candidates: list[Mapping[str, Any]],
    original_pack: Mapping[str, Any],
    development_pack: Mapping[str, Any],
    request_rows: list[Mapping[str, Any]],
    original_cache_dir: Path,
) -> dict[str, Any]:
    existing_bundles = [*original_pack["bundles"], *development_pack["bundles"]]

    def overlap(candidate_field: str, existing_field: str | None = None) -> list[str]:
        field = existing_field or candidate_field
        left = {str(row[candidate_field]) for row in candidates}
        right = {str(row[field]) for row in existing_bundles}
        return sorted(left & right)

    candidate_questions = {_normalized(str(row["question"])) for row in candidates}
    existing_questions = {_normalized(str(row["question"])) for row in existing_bundles}
    candidate_response_ids = {
        str(response["response_id"])
        for row in candidates
        for response in row["responses"]
    }
    existing_response_ids = {
        str(response["response_id"])
        for row in existing_bundles
        for response in row["responses"]
    }
    request_hashes, cache_keys, response_hashes = _existing_receipt_identities(original_cache_dir)
    new_response_hashes = {
        value for row in request_rows for value in row["response_sha256s"]
    }
    return {
        "bundle_id_overlap": overlap("bundle_id"),
        "source_record_id_overlap": overlap("source_record_id"),
        "source_kind_overlap": overlap("source_kind"),
        "family_overlap": overlap("family"),
        "normalized_question_overlap": sorted(candidate_questions & existing_questions),
        "response_id_overlap": sorted(candidate_response_ids & existing_response_ids),
        "raw_request_sha256_overlap": sorted({row["raw_request_sha256"] for row in request_rows} & request_hashes),
        "cache_key_sha256_overlap": sorted({row["cache_key_sha256"] for row in request_rows} & cache_keys),
        "response_sha256_overlap": sorted(new_response_hashes & response_hashes),
    }


def build_holdout_replacement(
    *,
    original_plan_path: Path,
    original_pack_path: Path,
    original_labels_path: Path,
    development_pack_path: Path,
    failure_receipt_path: Path,
    original_cache_dir: Path,
    pre_run_provenance_path: Path | None = None,
) -> HoldoutReplacementArtifacts:
    """Build one replacement pair and a full amended plan, entirely offline.

    ``pre_run_provenance_path`` is the immutable freshness witness emitted by a
    successful build before the replacement requests were executed.  Supplying
    it makes later byte-identical rebuilds independent of the now-mutated live
    cache while retaining fail-closed identity checks.  Initial builds omit it
    and prove freshness directly against the cache.
    """

    original_plan = dict(_load_calibration_plan(_load_signed(original_plan_path, "original plan")))
    original_pack = _load_signed(original_pack_path, "original pack")
    original_labels = _load_signed(original_labels_path, "original labels")
    development_pack = _load_signed(development_pack_path, "development pack")
    failure = _load_failure_receipt(failure_receipt_path)
    if (
        failure.get("judge_kind") != "pairwise"
        or failure.get("operational_status") != "transport_error"
        or failure.get("operational_failure", {}).get("code") != "connection_error"
        or any(failure.get(field) is not None for field in ("raw_response_sha256", "usage", "price_usd"))
    ):
        raise HoldoutReplacementError("replacement requires the exact ambiguous connection failure")
    old_pairs = {row["relationship_id"]: row for row in original_plan["pairwise_relationships"]}
    failed_pair = old_pairs.get(FAILED_RELATIONSHIP_ID)
    if failed_pair is None:
        raise HoldoutReplacementError("failed relationship is absent from the original plan")
    if failure.get("response_sha256s") != [
        _hash({key: item for key, item in failed_pair["candidate_b"].items() if key not in {"bundle_id", "bundle_sha256"}}),
        _hash({key: item for key, item in failed_pair["candidate_a"].items() if key not in {"bundle_id", "bundle_sha256"}}),
    ]:
        raise HoldoutReplacementError("failure receipt is not the failed BA presentation")
    completed_ab = _completed_original_ab(
        original_cache_dir, failed_pair["relationship_sha256"]
    )

    candidate_a, public_a = _bundle(side="a", left=4103, right=29, mode="truth_retaining_false_report")
    candidate_b, public_b = _bundle(side="b", left=4103, right=29, mode="general_false_confidence")
    replacement_pair = _pair(public_a, public_b)
    request_rows = _request_identities(replacement_pair)
    replacement_pack = _signed({
        "format": "truth_editing_judge_holdout_operational_replacement_pack_v1",
        "selection_rule": "fixed_same_scenario_4103_plus_29_response_variants_v1",
        "candidate_bundles": [candidate_a, candidate_b],
        "pairwise_relationship": replacement_pair,
    })
    replacement_labels = _signed({
        "format": "truth_editing_judge_holdout_deterministic_replacement_labels_v1",
        "replacement_pack_sha256": replacement_pack["content_sha256"],
        "human_labels_present": False,
        "pairwise_labels": [{
            "relationship_id": REPLACEMENT_RELATIONSHIP_ID,
            "preference": "candidate_a",
            "case_kind": "known_dominance",
        }],
    })
    provenance_identity = {
        "format": "truth_editing_judge_holdout_operational_replacement_provenance_v1",
        "original_plan_sha256": original_plan["content_sha256"],
        "original_pack_sha256": original_pack["content_sha256"],
        "development_pack_sha256": development_pack["content_sha256"],
        "failure_receipt_sha256": failure["content_sha256"],
        "replacement_pack_sha256": replacement_pack["content_sha256"],
        "replacement_labels_sha256": replacement_labels["content_sha256"],
        "label_kind": "deterministic_fixture_ground_truth",
        "human_labels_present": False,
    }
    if pre_run_provenance_path is None:
        freshness = _freshness(
            [candidate_a, candidate_b], original_pack, development_pack,
            request_rows, original_cache_dir,
        )
        if any(freshness.values()):
            raise HoldoutReplacementError("replacement candidate or request identity is not fresh")
        provenance = _signed({**provenance_identity, "freshness": freshness})
    else:
        provenance = _load_signed(
            pre_run_provenance_path, "pre-run replacement provenance"
        )
        if any(
            provenance.get(field) != expected
            for field, expected in provenance_identity.items()
        ):
            raise HoldoutReplacementError(
                "pre-run replacement provenance source identity differs"
            )
        freshness = provenance.get("freshness")
        if not isinstance(freshness, Mapping) or not freshness or any(freshness.values()):
            raise HoldoutReplacementError(
                "pre-run replacement provenance does not prove freshness"
            )
        if provenance != _signed({**provenance_identity, "freshness": dict(freshness)}):
            raise HoldoutReplacementError(
                "pre-run replacement provenance fields differ"
            )
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
        "format": "truth_editing_judge_holdout_operational_replacement_mapping_v1",
        "original_plan_sha256": original_plan["content_sha256"],
        "failed_relationship_id": FAILED_RELATIONSHIP_ID,
        "failed_relationship_sha256": failed_pair["relationship_sha256"],
        "failed_presentation": "BA",
        "failure_receipt_sha256": failure["content_sha256"],
        "failed_raw_request_sha256": failure["raw_request_sha256"],
        "failed_cache_key_sha256": failure["cache_key_sha256"],
        "failure_observation": {"raw_response_sha256": None, "usage": None, "price_usd": None},
        "completed_original_presentation": completed_ab,
        "replacement_relationship_id": REPLACEMENT_RELATIONSHIP_ID,
        "replacement_relationship_sha256": replacement_pair["relationship_sha256"],
        "replacement_case_kind": "known_dominance",
        "replacement_request_identities": request_rows,
        "selection_rule": "fixed_same_scenario_4103_plus_29_response_variants_v1",
        "semantic_results_consulted_for_replacement": False,
        "semantic_adaptation_prohibited": True,
        "one_for_one": {"removed_relationships": 1, "added_relationships": 1},
    })

    amended_pairs = [
        copy.deepcopy(row)
        for row in original_plan["pairwise_relationships"]
        if row["relationship_id"] != FAILED_RELATIONSHIP_ID
    ] + [replacement_pair]
    amended_pack = _signed({
        "format": "truth_editing_judge_holdout_amended_pack_v3",
        "original_pack_sha256": original_pack["content_sha256"],
        "operational_mapping_sha256": mapping["content_sha256"],
        "absolute_bundles": copy.deepcopy(original_plan["absolute_bundles"]),
        "pairwise_relationships": amended_pairs,
    })
    amended_pair_labels = [
        copy.deepcopy(row)
        for row in original_labels["pairwise_labels"]
        if row["relationship_id"] != FAILED_RELATIONSHIP_ID
    ] + [{"relationship_id": REPLACEMENT_RELATIONSHIP_ID, "preference": "candidate_a"}]
    amended_labels = _signed({
        "format": "truth_editing_judge_holdout_amended_labels_v3",
        "amended_pack_sha256": amended_pack["content_sha256"],
        "human_labels_present": False,
        "absolute_labels": copy.deepcopy(original_labels["absolute_labels"]),
        "pairwise_labels": amended_pair_labels,
    })
    amended_provenance = _signed({
        "format": "truth_editing_judge_holdout_amended_provenance_v3",
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
        "format": "truth_editing_judge_holdout_operational_replacement_receipt_v1",
        "reason": "ambiguous_transport_failure_without_response_usage_or_price",
        "original_plan_sha256": original_plan["content_sha256"],
        "original_plan_preserved": True,
        "failed_relationship_id": FAILED_RELATIONSHIP_ID,
        "failed_presentation": "BA",
        "failure_receipt_sha256": failure["content_sha256"],
        "failed_request_retry_permitted": False,
        "semantic_adaptation": False,
        "removed_relationship_count": 1,
        "added_relationship_count": 1,
        "mapping_sha256": mapping["content_sha256"],
        "replacement_plan_sha256": replacement_plan["content_sha256"],
        "amended_plan_sha256": amended_plan["content_sha256"],
        "replacement_maximum_spend_usd": 0.01,
        "execution_policy": "run replacement plan with fresh attempt directory; then amended plan may reuse shared response cache",
    })
    artifacts = HoldoutReplacementArtifacts(
        original_plan, replacement_pack, replacement_labels, provenance, mapping,
        replacement_plan, amended_pack, amended_labels, amended_provenance,
        amended_plan, operational_receipt,
    )
    validate_holdout_replacement(artifacts)
    return artifacts


def validate_holdout_replacement(
    artifacts: HoldoutReplacementArtifacts,
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
        if value.get("content_sha256") != _hash({key: item for key, item in value.items() if key != "content_sha256"}):
            raise HoldoutReplacementError(f"{name} identity differs")
    if receipt.get("semantic_adaptation") is not False or artifacts.mapping.get("semantic_results_consulted_for_replacement") is not False:
        raise HoldoutReplacementError("semantic adaptation is forbidden")
    if artifacts.replacement_plan.get("maximum_spend_usd") != 0.01:
        raise HoldoutReplacementError("replacement hard spend cap differs")
    _load_calibration_plan(artifacts.replacement_plan)
    _load_calibration_plan(artifacts.amended_plan)
    if artifacts.replacement_plan["absolute_bundles"] or len(artifacts.replacement_plan["pairwise_relationships"]) != 1:
        raise HoldoutReplacementError("replacement plan must contain exactly one pair")
    if len(artifacts.amended_plan["absolute_bundles"]) != 84 or len(artifacts.amended_plan["pairwise_relationships"]) != 18:
        raise HoldoutReplacementError("amended plan cardinality differs")
    original_pairs = {row["relationship_id"]: row for row in artifacts.original_plan["pairwise_relationships"]}
    amended_pairs = {row["relationship_id"]: row for row in artifacts.amended_plan["pairwise_relationships"]}
    if FAILED_RELATIONSHIP_ID in amended_pairs or REPLACEMENT_RELATIONSHIP_ID not in amended_pairs:
        raise HoldoutReplacementError("amended plan replacement mapping differs")
    for relationship_id, row in original_pairs.items():
        if relationship_id != FAILED_RELATIONSHIP_ID and amended_pairs.get(relationship_id) != row:
            raise HoldoutReplacementError("amended plan changed an unaffected relationship")
    if artifacts.amended_plan["absolute_bundles"] != artifacts.original_plan["absolute_bundles"]:
        raise HoldoutReplacementError("amended plan changed an absolute operation")


__all__ = [
    "AMENDED_PLAN_ID",
    "FAILED_RELATIONSHIP_ID",
    "HoldoutReplacementArtifacts",
    "HoldoutReplacementError",
    "REPLACEMENT_PLAN_ID",
    "REPLACEMENT_RELATIONSHIP_ID",
    "build_holdout_replacement",
    "validate_holdout_replacement",
]
