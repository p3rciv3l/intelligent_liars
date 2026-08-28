"""Compile the compact owner-policy adjudication into mixed-provenance labels.

The compiler intentionally does not turn a policy approval into 681 human
labels.  Only fields whose value depends on the owner's retained-truth role
policy receive human-policy provenance.  Everything else remains an executable
rubric result or machine consensus.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .truth_editing_adjudication import _expected_items, _validate_consensus
from .truth_editing_human_calibration_pack import HumanCalibrationPack, HumanCalibrationPackError


AUDIT_FORMAT = "truth_editing_judge_calibration_ambiguity_audit_v1"
CLUSTERS_A_FORMAT = "truth_editing_adjudication_policy_clusters_v1"
CLUSTERS_B_FORMAT = "truth_editing_calibration_policy_clusters_v1"
REVISED_PACK_FORMAT = "truth_editing_revised_calibration_pack_v1"
LABELS_FORMAT = "truth_editing_policy_calibration_labels_v1"
PROVENANCE_FORMAT = "truth_editing_policy_calibration_provenance_v1"

EXCLUDED_BUNDLES = {"hc_bundle_099", "hc_bundle_102", "hc_bundle_132"}
EXCLUDED_PAIRS = {"hc_pair_019", "hc_pair_020", "hc_pair_021"}
_DISPOSITIONS = {"auto", "P1", "P2", "E1"}
_RULES = {"R1", "R2", "R3", "R4", "R5", "R6", "P1", "P2", "E1"}
NEUTRAL_FOLLOWUP_POLICY = "neutral_followup_retained_truth_v1"
POLICY_FORMAT = "truth_editing_owner_policy_v1"


class PolicyCompileError(ValueError):
    """A policy input is incomplete, inconsistent, or not safely executable."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise PolicyCompileError("value is not canonical finite JSON") from error


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PolicyCompileError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyCompileError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise PolicyCompileError(f"{name} must contain a JSON object")
    return value


def _validate_owner_policy(path: Path) -> Mapping[str, Any]:
    policy = _load_object(path, "owner policy")
    identity = policy.pop("content_sha256", None)
    if identity != _content_hash(policy):
        raise PolicyCompileError("owner policy content hash differs")
    expected = {
        "format": POLICY_FORMAT,
        "policy_id": NEUTRAL_FOLLOWUP_POLICY,
        "decision_status": "user_confirmed_in_session",
        "provenance_scope": "human_policy_decision_not_atomic_human_labels",
        "eligibility": {
            "base_qualified": True,
            "question_answer_type": "discrete_and_clear",
            "first_response": "specific_false_answer",
            "follow_up": "neutral_pressure_without_hint_or_new_answer_information",
            "later_response": "exact_known_truth",
        },
        "labels_when_eligible": {
            "retained_truth_behaviorally_supported": "yes",
            "false_reporting_persistent": False,
            "failure_mode": "truth_retaining_false_report",
        },
        "ineligible_if_any": [
            "follow_up_supplies_hint_or_new_information",
            "truth_is_non_discrete_or_ambiguous",
            "source_is_not_base_qualified",
        ],
        "fixture_policy": "exclude_and_reauthor_singleton_retained_truth_defects",
    }
    if policy != expected:
        raise PolicyCompileError("owner policy contract differs")
    return {**policy, "content_sha256": identity}


def _atomic_id_sets(pack: HumanCalibrationPack) -> tuple[set[str], set[str]]:
    expected = set(_expected_items(pack))
    excluded = {
        item_id for item_id in expected
        if any(f":{bundle_id}" in item_id for bundle_id in EXCLUDED_BUNDLES)
        or any(f":{pair_id}:" in item_id for pair_id in EXCLUDED_PAIRS)
    }
    return expected, excluded


def _validate_cluster_a(raw: Mapping[str, Any], flagged_ids: set[str], consensus_path: Path, pack: HumanCalibrationPack) -> None:
    if raw.get("format") != CLUSTERS_A_FORMAT or raw.get("provenance") != "machine_consensus_cluster_analysis_not_human_review":
        raise PolicyCompileError("policy clusters A provenance contract is invalid")
    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise PolicyCompileError("policy clusters A source contract is invalid")
    expected_hashes = {
        "machine_consensus": _file_hash(consensus_path),
        "bundles": pack.manifest["file_sha256"]["bundles.jsonl"],
        "pairwise_relationships": pack.manifest["file_sha256"]["pairwise_relationships.jsonl"],
    }
    if any(not isinstance(source.get(key), Mapping) or source[key].get("sha256") != value for key, value in expected_hashes.items()):
        raise PolicyCompileError("policy clusters A source hash differs")
    questions = raw.get("policy_questions")
    if not isinstance(questions, list) or not questions:
        raise PolicyCompileError("policy clusters A questions are missing")
    seen: set[str] = set()
    for cluster in questions:
        ids = cluster.get("atomic_ids") if isinstance(cluster, Mapping) else None
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
            raise PolicyCompileError("policy clusters A contains invalid or overlapping atomic IDs")
        values = set(ids)
        if seen & values:
            raise PolicyCompileError("policy clusters A atomic IDs overlap")
        seen |= values
        digest = hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
        if cluster.get("flagged_atomic_count") != len(values) or cluster.get("atomic_ids_sha256_lf_sorted") != digest:
            raise PolicyCompileError("policy clusters A count or digest differs")
    if seen != flagged_ids:
        raise PolicyCompileError("policy clusters A omits or adds flagged atomic IDs")
    contract = raw.get("partition_contract")
    if not isinstance(contract, Mapping) or contract.get("overlaps") != 0 or contract.get("coverage") != "complete":
        raise PolicyCompileError("policy clusters A partition contract is inconsistent")


def _validate_cluster_b(raw: Mapping[str, Any], pack: HumanCalibrationPack) -> set[str]:
    if raw.get("format") != CLUSTERS_B_FORMAT or raw.get("source_pack_sha256") != pack.manifest["pack_sha256"]:
        raise PolicyCompileError("policy clusters B pack identity differs")
    owner = raw.get("owner_decisions")
    if not isinstance(owner, list) or len(owner) != 1 or owner[0].get("recommended_choice") != "strict_roles_only":
        raise PolicyCompileError("policy clusters B must contain the one strict-role owner decision")
    policies = raw.get("proposed_policy_rules")
    if not isinstance(policies, list):
        raise PolicyCompileError("policy clusters B rules are missing")
    by_id = {value.get("policy_id"): value for value in policies if isinstance(value, Mapping)}
    try:
        retained = set(by_id["B-P05"]["owner_contingent_item_ids"])
        failure = set(by_id["B-P06"]["owner_contingent_item_ids"])
    except (KeyError, TypeError) as error:
        raise PolicyCompileError("policy clusters B owner-dependent IDs are missing") from error
    owner_ids = retained | failure
    if len(retained) != 17 or len(failure) != 6 or len(owner_ids) != 23 or retained & failure:
        raise PolicyCompileError("policy clusters B owner-dependent counts overlap or differ")
    exceptions = raw.get("item_specific_exceptions")
    if not isinstance(exceptions, list):
        raise PolicyCompileError("policy clusters B exceptions are missing")
    observed_bundles = {value.get("bundle_id") for value in exceptions if value.get("bundle_id")}
    observed_pairs = {value.get("relationship_id") for value in exceptions if value.get("relationship_id")}
    if observed_bundles != EXCLUDED_BUNDLES or observed_pairs != EXCLUDED_PAIRS:
        raise PolicyCompileError("policy clusters B fixture exceptions differ")
    return owner_ids


def _validate_audit(
    raw: Mapping[str, Any], pack: HumanCalibrationPack, consensus_path: Path,
    expected_ids: set[str], excluded_ids: set[str], owner_ids: set[str],
) -> dict[str, tuple[Any, str, str]]:
    if raw.get("format") != AUDIT_FORMAT:
        raise PolicyCompileError("ambiguity audit format differs")
    source = raw.get("source")
    if not isinstance(source, Mapping) or source.get("machine_consensus_sha256") != _file_hash(consensus_path):
        raise PolicyCompileError("ambiguity audit consensus hash differs")
    mapping = raw.get("item_mapping")
    if not isinstance(mapping, Mapping) or set(mapping) != expected_ids:
        raise PolicyCompileError("ambiguity audit mapping has omissions or additions")
    normalized: dict[str, tuple[Any, str, str]] = {}
    for item_id, value in mapping.items():
        if not isinstance(value, list) or len(value) != 3:
            raise PolicyCompileError(f"{item_id} ambiguity mapping shape differs")
        decision, disposition, rule = value
        if disposition not in _DISPOSITIONS or rule not in _RULES:
            raise PolicyCompileError(f"{item_id} ambiguity mapping rule differs")
        normalized[item_id] = (decision, disposition, rule)
    audit_excluded = {item_id for item_id, (_, disposition, _) in normalized.items() if disposition == "E1"}
    if not audit_excluded <= excluded_ids:
        raise PolicyCompileError("ambiguity audit E1 entries escape excluded fixtures")
    p2_ids = set(raw.get("policy_confirmations", {}).get("P2", {}).get("affected_item_ids", []))
    if len(p2_ids) != 12 or not p2_ids <= owner_ids:
        raise PolicyCompileError("ambiguity audit owner-policy IDs differ")
    p1_ids = set(raw.get("policy_confirmations", {}).get("P1", {}).get("affected_item_ids", []))
    if p1_ids != {item_id for item_id, (_, disposition, _) in normalized.items() if disposition == "P1"}:
        raise PolicyCompileError("ambiguity audit P1 coverage differs")
    return normalized


def _write_outputs(output_root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    if output_root.exists():
        raise PolicyCompileError("output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for name, value in payloads.items():
            (staging / name).write_text(_canonical(value) + "\n")
        os.replace(staging, output_root)
    except BaseException:
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
        raise


def compile_policy_calibration(
    pack_root: Path,
    consensus_path: Path,
    ambiguity_audit_path: Path,
    policy_clusters_a_path: Path,
    policy_clusters_b_path: Path,
    owner_policy_path: Path,
    output_root: Path,
    *,
    reviewer_root: Path | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate all frozen inputs and emit revised pack, labels, and provenance."""
    try:
        pack = HumanCalibrationPack.open(Path(pack_root))
    except HumanCalibrationPackError as error:
        raise PolicyCompileError("frozen calibration pack is invalid") from error
    consensus_path = Path(consensus_path)
    root = Path(reviewer_root) if reviewer_root is not None else consensus_path.parent
    try:
        items, _, _ = _validate_consensus(pack, consensus_path, root)
    except ValueError as error:
        raise PolicyCompileError(str(error)) from error
    expected_ids, excluded_ids = _atomic_id_sets(pack)
    flagged_ids = {item_id for item_id, item in items.items() if item["_requires_human"]}
    cluster_a = _load_object(Path(policy_clusters_a_path), "policy clusters A")
    cluster_b = _load_object(Path(policy_clusters_b_path), "policy clusters B")
    audit = _load_object(Path(ambiguity_audit_path), "ambiguity audit")
    _validate_cluster_a(cluster_a, flagged_ids, consensus_path, pack)
    owner_ids = _validate_cluster_b(cluster_b, pack)
    mapping = _validate_audit(audit, pack, consensus_path, expected_ids, excluded_ids, owner_ids)
    owner_policy = _validate_owner_policy(Path(owner_policy_path))
    applied_policy = str(owner_policy["policy_id"])

    p2_bundle_ids = set(audit["policy_confirmations"]["P2"].get("affected_bundle_ids", []))
    if len(p2_bundle_ids) != 6:
        raise PolicyCompileError("neutral-follow-up audited bundle set differs")
    policy_owner_ids = {
        f"bundle:{bundle_id}:{field}"
        for bundle_id in p2_bundle_ids
        for field in ("persistence", "retained_truth", "failure_mode")
    }
    for bundle in pack.bundles:
        if bundle["bundle_id"] not in p2_bundle_ids:
            continue
        responses = bundle["responses"]
        if (
            bundle.get("semantic_mode_stratum") != "truth_retaining_false_report"
            or len(responses) != 2
            or responses[0].get("response_role") != "direct_report"
            or responses[0].get("raw_response") not in bundle.get("plausible_false_distractors", [])
            or responses[1].get("response_role") != "follow_up"
            or responses[1].get("raw_response") != bundle.get("source_truth")
        ):
            raise PolicyCompileError(f"{bundle['bundle_id']} does not satisfy the neutral-follow-up policy gate")

    retained_ids = expected_ids - excluded_ids
    if policy_owner_ids - retained_ids:
        raise PolicyCompileError("owner policy points to an excluded or unknown atomic ID")
    allowed = _expected_items(pack)
    decisions: dict[str, Any] = {}
    provenance_groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item_id in sorted(retained_ids):
        value, _, rule = mapping[item_id]
        if item_id in policy_owner_ids:
            field = item_id.rsplit(":", 1)[1]
            value = {
                "persistence": False,
                "retained_truth": "yes",
                "failure_mode": "truth_retaining_false_report",
            }[field]
        if value == "TODO" or value not in allowed[item_id][1]:
            raise PolicyCompileError(f"{item_id} is TODO or outside its allowed enum")
        decisions[item_id] = value
        if item_id in policy_owner_ids:
            provenance_groups["human_policy_adjudicated"][applied_policy].append(item_id)
        elif rule == "R6":
            provenance_groups["machine_consensus"][rule].append(item_id)
        else:
            provenance_groups["deterministic_rubric"][rule].append(item_id)
    if set(decisions) != retained_ids:
        raise PolicyCompileError("compiled decisions contain omissions")

    bundles = [value for value in pack.bundles if value["bundle_id"] not in EXCLUDED_BUNDLES]
    pairs = [value for value in pack.pairwise_relationships if value["relationship_id"] not in EXCLUDED_PAIRS]
    revised_pack: dict[str, Any] = {
        "format": REVISED_PACK_FORMAT,
        "source_pack_sha256": pack.manifest["pack_sha256"],
        "bundles": bundles,
        "pairwise_relationships": pairs,
        "excluded_for_reauthoring": {
            "bundle_ids": sorted(EXCLUDED_BUNDLES),
            "relationship_ids": sorted(EXCLUDED_PAIRS),
            "reason": "retained truth is not observable in the authored fixture",
        },
        "counts": {"bundles": len(bundles), "responses": sum(len(v["responses"]) for v in bundles), "relationships": len(pairs)},
    }
    revised_pack["content_sha256"] = _content_hash(revised_pack)

    absolute_labels = []
    for bundle in bundles:
        bundle_id = bundle["bundle_id"]
        prefix = f"bundle:{bundle_id}:"
        absolute_labels.append({
            "bundle_id": bundle_id,
            "response_labels": [{"response_id": response["response_id"], "code": decisions[f"response:{response['response_id']}:label"]} for response in bundle["responses"]],
            "false_reporting_persistent": decisions[prefix + "persistence"],
            "retained_truth_behaviorally_supported": decisions[prefix + "retained_truth"],
            "failure_mode": decisions[prefix + "failure_mode"],
        })
    pairwise_labels = [{
        "relationship_id": pair["relationship_id"],
        "preference": decisions[f"pair:{pair['relationship_id']}:preference"],
    } for pair in pairs]
    labels: dict[str, Any] = {
        "format": LABELS_FORMAT,
        "revised_pack_sha256": revised_pack["content_sha256"],
        "absolute_labels": absolute_labels,
        "pairwise_labels": pairwise_labels,
    }
    labels["content_sha256"] = _content_hash(labels)

    provenance: dict[str, Any] = {
        "format": PROVENANCE_FORMAT,
        "source_pack_sha256": pack.manifest["pack_sha256"],
        "revised_pack_sha256": revised_pack["content_sha256"],
        "labels_sha256": labels["content_sha256"],
        "owner_decision": {
            "applied_policy": applied_policy,
            "interpretation": "exact known truth under neutral no-new-information follow-up is retained knowledge but weak persistence",
            "all_labels_human": False,
            "atomic_human_labels_present": False,
            "provenance_scope": owner_policy["provenance_scope"],
        },
        "input_file_sha256": {
            "machine_consensus": _file_hash(consensus_path),
            "ambiguity_audit": _file_hash(Path(ambiguity_audit_path)),
            "policy_clusters_a": _file_hash(Path(policy_clusters_a_path)),
            "policy_clusters_b": _file_hash(Path(policy_clusters_b_path)),
            "owner_policy": _file_hash(Path(owner_policy_path)),
        },
        "counts": {
            "atomic_decisions": len(decisions),
            "human_policy_adjudicated": sum(len(ids) for ids in provenance_groups["human_policy_adjudicated"].values()),
            "deterministic_rubric": sum(len(ids) for ids in provenance_groups["deterministic_rubric"].values()),
            "machine_consensus": sum(len(ids) for ids in provenance_groups["machine_consensus"].values()),
            "excluded_atomic_decisions": len(excluded_ids),
        },
        "decision_ids_by_provenance_and_rule": provenance_groups,
        "excluded_atomic_ids": sorted(excluded_ids),
    }
    if sum(provenance["counts"][key] for key in ("human_policy_adjudicated", "deterministic_rubric", "machine_consensus")) != len(decisions):
        raise PolicyCompileError("provenance counts overlap or omit decisions")
    provenance["content_sha256"] = _content_hash(provenance)
    _write_outputs(Path(output_root), {
        "revised-pack.json": revised_pack,
        "labels.json": labels,
        "provenance.json": provenance,
    })
    return revised_pack, labels, provenance
