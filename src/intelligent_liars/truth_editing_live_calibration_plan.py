"""Compile the revised authored calibration pack into a blinded live plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .truth_editing_live_judge import FROZEN_JUDGE_CONFIG_SHA256, LiveJudgeError


PLAN_FORMAT = "truth_editing_live_judge_calibration_plan_v4_pairwise_contract"
_PACK_FORMAT = "truth_editing_revised_calibration_pack_v1"
_LABELS_FORMAT = "truth_editing_policy_calibration_labels_v1"
_PROVENANCE_FORMAT = "truth_editing_policy_calibration_provenance_v1"
_ROLES = {"direct_report", "follow_up", "indirect_truth_use", "action", "counterfactual"}
_PAIRWISE_COMPARISON_KINDS = {
    "self_pair", "exact_duplicate", "known_dominance", "invalid_comparison"
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _load(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveJudgeError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LiveJudgeError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise LiveJudgeError(f"{name} must contain an object")
    identity = value.get("content_sha256")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    if identity != _hash(unsigned):
        raise LiveJudgeError(f"{name} content identity differs")
    return value


def _blind_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "format", "bundle_id", "blind_index", "domain_group", "family", "source_record_id",
        "source_kind", "semantic_mode_stratum", "coverage_tags", "question", "source_truth",
        "acceptable_aliases", "plausible_false_distractors", "responses",
        "deterministic_observations", "bundle_sha256",
    }
    if set(bundle) != required:
        raise LiveJudgeError("revised calibration bundle fields differ")
    claimed = bundle["bundle_sha256"]
    if claimed != _hash({key: value for key, value in bundle.items() if key != "bundle_sha256"}):
        raise LiveJudgeError("revised calibration bundle identity differs")
    for field in ("question", "source_truth"):
        if not isinstance(bundle[field], str) or not bundle[field] or len(bundle[field]) > 8192:
            raise LiveJudgeError(f"revised calibration {field} must be bounded nonempty text")
    responses = bundle["responses"]
    if not isinstance(responses, list) or not responses:
        raise LiveJudgeError("revised calibration bundle responses are missing")
    blinded_responses: list[dict[str, str]] = []
    for response in responses:
        if not isinstance(response, Mapping) or set(response) != {"response_id", "response_role", "raw_response"}:
            raise LiveJudgeError("revised calibration response fields differ")
        if response["response_role"] not in _ROLES:
            raise LiveJudgeError("revised calibration response role differs")
        blinded_responses.append({
            "response_id": str(response["response_id"]),
            "response_role": str(response["response_role"]),
            "response": str(response["raw_response"]),
        })
    # Deliberately excludes semantic_mode_stratum, coverage tags, deterministic
    # observations, source identity, and all policy labels.
    return {
        "bundle_id": str(bundle["bundle_id"]),
        "bundle_sha256": str(claimed),
        "domain_group": str(bundle["domain_group"]),
        "family": str(bundle["family"]),
        "question": str(bundle["question"]),
        "known_truth": str(bundle["source_truth"]),
        "truth_aliases": list(bundle["acceptable_aliases"]),
        "plausible_false_answers": list(bundle["plausible_false_distractors"]),
        "responses": blinded_responses,
    }


def compile_live_calibration_plan(
    revised_pack_path: Path,
    labels_path: Path,
    provenance_path: Path,
    *,
    maximum_spend_usd: float = 5.0,
) -> dict[str, Any]:
    pack = _load(Path(revised_pack_path), "revised calibration pack")
    labels = _load(Path(labels_path), "calibration labels")
    provenance = _load(Path(provenance_path), "calibration provenance")
    if pack.get("format") != _PACK_FORMAT or labels.get("format") != _LABELS_FORMAT or provenance.get("format") != _PROVENANCE_FORMAT:
        raise LiveJudgeError("calibration source format differs")
    if labels.get("revised_pack_sha256") != pack["content_sha256"] or provenance.get("revised_pack_sha256") != pack["content_sha256"] or provenance.get("labels_sha256") != labels["content_sha256"]:
        raise LiveJudgeError("calibration labels or provenance are not bound to the revised pack")
    bundles_raw = pack.get("bundles")
    pairs_raw = pack.get("pairwise_relationships")
    if not isinstance(bundles_raw, list) or not isinstance(pairs_raw, list):
        raise LiveJudgeError("revised calibration operations are missing")
    bundles = [_blind_bundle(value) for value in bundles_raw if isinstance(value, Mapping)]
    if len(bundles) != len(bundles_raw):
        raise LiveJudgeError("revised calibration bundle must be an object")
    by_id = {value["bundle_id"]: value for value in bundles}
    if len(by_id) != len(bundles):
        raise LiveJudgeError("revised calibration bundle IDs must be unique")
    absolute_label_ids = {value.get("bundle_id") for value in labels.get("absolute_labels", []) if isinstance(value, Mapping)}
    if absolute_label_ids != set(by_id):
        raise LiveJudgeError("absolute labels do not cover the revised bundles exactly")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for pair in pairs_raw:
        if not isinstance(pair, Mapping):
            raise LiveJudgeError("revised calibration relationship must be an object")
        required = {"format", "relationship_id", "case_kind", "candidate_a_bundle_id", "candidate_b_bundle_id", "candidate_a_sha256", "candidate_b_sha256", "presentations", "relationship_sha256"}
        if set(pair) != required or pair["relationship_sha256"] != _hash({key: value for key, value in pair.items() if key != "relationship_sha256"}):
            raise LiveJudgeError("revised calibration relationship identity differs")
        relationship_id = str(pair["relationship_id"])
        if relationship_id in pair_ids:
            raise LiveJudgeError("revised calibration relationship IDs must be unique")
        pair_ids.add(relationship_id)
        candidate_a = by_id.get(str(pair["candidate_a_bundle_id"]))
        candidate_b = by_id.get(str(pair["candidate_b_bundle_id"]))
        if candidate_a is None or candidate_b is None or candidate_a["bundle_sha256"] != pair["candidate_a_sha256"] or candidate_b["bundle_sha256"] != pair["candidate_b_sha256"]:
            raise LiveJudgeError("revised calibration relationship candidate identity differs")
        presentations = pair["presentations"]
        if presentations not in (["AB"], ["AB", "BA"]):
            raise LiveJudgeError("revised calibration relationship presentations differ")
        comparison_kind = pair["case_kind"]
        if comparison_kind not in _PAIRWISE_COMPARISON_KINDS:
            raise LiveJudgeError("revised calibration comparison kind differs")
        pairs.append({
            "relationship_id": relationship_id,
            "relationship_sha256": str(pair["relationship_sha256"]),
            "comparison_kind": comparison_kind,
            "presentations": list(presentations),
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
        })
    pair_label_ids = {value.get("relationship_id") for value in labels.get("pairwise_labels", []) if isinstance(value, Mapping)}
    if pair_label_ids != pair_ids:
        raise LiveJudgeError("pairwise labels do not cover the revised relationships exactly")
    unsigned = {
        "format": PLAN_FORMAT,
        "calibration_id": "revised-policy-v1-live-calibration",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": maximum_spend_usd,
        "source_identities": {
            "revised_pack_sha256": pack["content_sha256"],
            "labels_sha256": labels["content_sha256"],
            "provenance_sha256": provenance["content_sha256"],
        },
        "absolute_bundles": bundles,
        "pairwise_relationships": pairs,
    }
    return {**unsigned, "content_sha256": _hash(unsigned)}
