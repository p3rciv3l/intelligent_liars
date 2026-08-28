"""Compile human adjudications and machine consensus into calibration labels.

This module deliberately calls the result a *combined* label set.  Only atomic
decisions present in the adjudication Markdown receive human provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .truth_editing_human_calibration_pack import (
    LABELS_FORMAT,
    HumanCalibrationPack,
    HumanCalibrationPackError,
    _expand_code,
)


CONSENSUS_FORMAT = "truth_editing_machine_consensus_v1"
RECEIPT_FORMAT = "truth_editing_adjudication_provenance_receipt_v1"
_FAILURE_MODES = {
    "truth_retaining_false_report", "ordinary_error", "forced_guessing",
    "general_false_confidence", "truthful", "refusal_or_evasion", "incoherent",
}
_PREFERENCES = {"candidate_a", "candidate_b", "tie", "invalid_comparison"}
_RETAINED = {"yes", "no", "unclear"}
_CONFIDENCE = {"low", "medium", "high"}
_DECISION_BLOCK_RE = re.compile(
    r"^#### ((?:\S+ response label)|(?:\S+ (?:persistence|retained_truth|failure_mode|preference)))\n"
    r".*?^Human decision: `([^`]+)`\s*$",
    re.MULTILINE | re.DOTALL,
)


class AdjudicationCompileError(ValueError):
    """The pack, consensus, adjudication, or output contract is unsafe."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise AdjudicationCompileError("value is not canonical finite JSON") from error


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AdjudicationCompileError(f"{field} must be a regular file")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdjudicationCompileError(f"{field} is unreadable") from error
    if not isinstance(value, dict):
        raise AdjudicationCompileError(f"{field} must contain a JSON object")
    return value


def _reviewer_id(raw: Mapping[str, Any]) -> Any:
    return raw.get("reviewer_id", raw.get("reviewer"))


def _reviewer_vote_maps(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Normalize the five frozen reviewer formats used by the aggregator."""
    response: dict[str, Any] = {}
    bundle: dict[str, Any] = {}
    pair: dict[str, Any] = {}

    def add_response(value: Mapping[str, Any], inherited: Mapping[str, Any]) -> None:
        code = value.get("label", value.get("code", value.get("decision", value.get("label_code"))))
        confidence = value.get("confidence", inherited.get("confidence", "not_provided"))
        suffix = {"high": "H", "medium": "M", "low": "L"}.get(confidence)
        if suffix is None or not isinstance(value.get("response_id"), str):
            raise AdjudicationCompileError("reviewer response vote is invalid")
        response[value["response_id"]] = f"{code}:{suffix}"

    def add_bundle(bundle_id: Any, value: Mapping[str, Any], inherited: Mapping[str, Any]) -> None:
        if not isinstance(bundle_id, str):
            raise AdjudicationCompileError("reviewer bundle vote is invalid")
        bundle[bundle_id] = {
            "persistence": value.get("false_reporting_persistent"),
            "retained_truth": value.get("retained_truth_behaviorally_supported"),
            "failure_mode": value.get("failure_mode"),
        }

    def add_pair(value: Mapping[str, Any]) -> None:
        decision = value.get("decision", value)
        if not isinstance(decision, Mapping) or not isinstance(value.get("relationship_id"), str):
            raise AdjudicationCompileError("reviewer pair vote is invalid")
        pair[value["relationship_id"]] = {
            "preference": decision.get("preference"),
            "criterion_preferences": decision.get("criterion_preferences"),
            "confidence": decision.get("confidence"),
        }

    for value in raw.get("response_labels", []):
        add_response(value, {})
    for value in raw.get("bundle_labels", []):
        add_bundle(value.get("bundle_id"), value, {})
    for value in raw.get("absolute_labels", raw.get("absolute_decisions", [])):
        inherited = value.get("bundle_decision", value)
        for response_value in value.get("responses", value.get("response_labels", [])):
            add_response(response_value, inherited)
        add_bundle(value.get("bundle_id"), inherited, value)
    for value in raw.get("pair_preferences", raw.get("pairwise_labels", raw.get("pairwise_decisions", []))):
        add_pair(value)
    return response, bundle, pair


def _expected_items(pack: HumanCalibrationPack) -> dict[str, tuple[str, set[Any]]]:
    response_values = {f"{code}:{confidence}" for code in (
        "T", "PF", "IF", "OE", "FG", "CH", "RF", "AB", "HD", "PA", "BO", "WT", "IR", "IC", "MF"
    ) for confidence in ("L", "M", "H")}
    result: dict[str, tuple[str, set[Any]]] = {}
    for bundle in pack.bundles:
        for response in bundle["responses"]:
            item_id = f"response:{response['response_id']}:label"
            result[item_id] = ("response_label", response_values)
        result[f"bundle:{bundle['bundle_id']}:persistence"] = ("persistence", {True, False})
        result[f"bundle:{bundle['bundle_id']}:retained_truth"] = ("retained_truth", _RETAINED)
        result[f"bundle:{bundle['bundle_id']}:failure_mode"] = ("failure_mode", _FAILURE_MODES)
    for relationship in pack.pairwise_relationships:
        result[f"pair:{relationship['relationship_id']}:preference"] = ("pair_preference", _PREFERENCES)
    return result


def _majority(values: list[Any]) -> Any:
    counts = Counter(_canonical(value) for value in values)
    ranked = sorted(counts.items(), key=lambda value: (-value[1], value[0]))
    return json.loads(ranked[0][0]) if ranked and ranked[0][1] >= 3 else None


def _validate_consensus(
    pack: HumanCalibrationPack,
    consensus_path: Path,
    reviewer_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], list[dict[str, str]]]:
    consensus = _load_object(consensus_path, "machine consensus")
    if consensus.get("format") != CONSENSUS_FORMAT or consensus.get("provenance") != "machine_consensus_not_human" or consensus.get("human_reviewed") is not False:
        raise AdjudicationCompileError("machine consensus provenance contract is invalid")
    if consensus.get("source_pack_sha256") != pack.manifest["pack_sha256"]:
        raise AdjudicationCompileError("machine consensus pack identity differs")
    sources = consensus.get("reviewer_sources")
    if not isinstance(sources, list) or len(sources) != 5:
        raise AdjudicationCompileError("exactly five reviewer sources are required")
    reviewer_maps: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    source_receipts: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"reviewer_id", "path", "format"}:
            raise AdjudicationCompileError("reviewer source fields differ")
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = reviewer_root / source_path
        raw = _load_object(source_path, f"reviewer source {source['reviewer_id']}")
        pack_sha = raw.get("pack_sha256", raw.get("source_pack_sha256"))
        if (_reviewer_id(raw) != source["reviewer_id"] or raw.get("format") != source["format"]
                or pack_sha != pack.manifest["pack_sha256"]):
            raise AdjudicationCompileError("reviewer source identity differs")
        if source["reviewer_id"] in reviewer_maps:
            raise AdjudicationCompileError("reviewer source identities are duplicated")
        reviewer_maps[source["reviewer_id"]] = _reviewer_vote_maps(raw)
        source_receipts.append({
            "reviewer_id": source["reviewer_id"],
            "path": str(source_path.resolve()),
            "file_sha256": _file_hash(source_path),
        })

    expected = _expected_items(pack)
    raw_items = consensus.get("items")
    if not isinstance(raw_items, list):
        raise AdjudicationCompileError("machine consensus items must be an array")
    items = {item.get("item_id"): item for item in raw_items if isinstance(item, dict)}
    if len(items) != len(raw_items) or set(items) != set(expected):
        raise AdjudicationCompileError("canonical machine consensus decisions differ")
    consensus_counts = consensus.get("counts")
    if not isinstance(consensus_counts, dict) or consensus_counts.get("atomic_decisions") != len(items):
        raise AdjudicationCompileError("machine consensus count identity differs")
    reviewer_ids = set(reviewer_maps)
    for item_id, item in items.items():
        field, allowed = expected[item_id]
        if item.get("field") != field or set(item.get("allowed_values", [])) != allowed:
            raise AdjudicationCompileError(f"{item_id} field or enum differs")
        if item.get("provenance") != "machine_consensus_not_human":
            raise AdjudicationCompileError(f"{item_id} has invalid provenance")
        votes = item.get("votes")
        if not isinstance(votes, list) or len(votes) != 5 or {vote.get("reviewer_id") for vote in votes if isinstance(vote, dict)} != reviewer_ids:
            raise AdjudicationCompileError(f"{item_id} does not contain five unique reviewer votes")
        observed_values: list[Any] = []
        for vote in votes:
            reviewer_id = vote["reviewer_id"]
            response, bundle, pair = reviewer_maps[reviewer_id]
            if item["scope"] == "response":
                expected_value = response.get(item["response_id"])
            elif item["scope"] == "bundle":
                expected_value = bundle.get(item["bundle_id"], {}).get(item["field"])
            elif item["scope"] == "pair":
                expected_value = pair.get(item["relationship_id"], {}).get("preference")
            else:
                raise AdjudicationCompileError(f"{item_id} scope is invalid")
            if vote.get("value") != expected_value or expected_value not in allowed:
                raise AdjudicationCompileError(f"{item_id} vote differs from reviewer source")
            observed_values.append(expected_value)
        counts = Counter(_canonical(value) for value in observed_values)
        ranked = sorted(counts.items(), key=lambda value: (-value[1], value[0]))
        proposed = json.loads(ranked[0][0]) if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else None
        if item.get("majority_decision") != _majority(observed_values) or item.get("proposed_decision") != proposed:
            raise AdjudicationCompileError(f"{item_id} consensus calculation differs")
        if item.get("disagreement_flag") is not (len(counts) > 1):
            raise AdjudicationCompileError(f"{item_id} disagreement flag differs")
        flagged = bool(item.get("disagreement_flag") or item.get("ambiguity_flag") or item.get("low_confidence_flag"))
        item["_requires_human"] = flagged
        if not flagged and proposed is None:
            raise AdjudicationCompileError(f"{item_id} has no usable machine consensus")
    observed_flagged = sum(bool(item["_requires_human"]) for item in items.values())
    if consensus_counts.get("flagged_atomic_decisions") != observed_flagged:
        raise AdjudicationCompileError("machine consensus flagged count differs")
    return items, reviewer_maps, source_receipts


def _read_adjudications(path: Path, flagged: list[dict[str, Any]]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AdjudicationCompileError("human adjudication must be a regular Markdown file")
    try:
        blocks = _DECISION_BLOCK_RE.findall(path.read_text())
    except (OSError, UnicodeError) as error:
        raise AdjudicationCompileError("human adjudication Markdown is unreadable") from error
    if len(blocks) != len(flagged):
        raise AdjudicationCompileError("every flagged Human decision must be resolved exactly once")
    title_to_item: dict[str, dict[str, Any]] = {}
    for item in flagged:
        if item["scope"] == "response":
            title = f"{item['response_id']} response label"
        elif item["scope"] == "bundle":
            title = f"{item['bundle_id']} {item['field']}"
        else:
            title = f"{item['relationship_id']} preference"
        title_to_item[title] = item
    result: dict[str, Any] = {}
    for title, raw in blocks:
        item = title_to_item.get(title)
        if item is None or item["item_id"] in result:
            raise AdjudicationCompileError("human adjudication identities differ or repeat")
        if raw == "TODO":
            raise AdjudicationCompileError("human adjudication still contains TODO")
        value: Any = {"true": True, "false": False}.get(raw, raw)
        if value not in set(item["allowed_values"]):
            raise AdjudicationCompileError(f"{item['item_id']} human decision is outside its allowed enum")
        result[item["item_id"]] = value
    if set(result) != {item["item_id"] for item in flagged}:
        raise AdjudicationCompileError("human adjudication does not cover every flagged identity")
    return result


def compile_adjudicated_labels(
    pack_root: Path,
    consensus_path: Path,
    adjudication_path: Path,
    labels_output_path: Path,
    receipt_output_path: Path,
    *,
    reviewer_root: Path | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Compile a complete mixed-provenance label contract and receipt."""
    labels_output_path, receipt_output_path = Path(labels_output_path), Path(receipt_output_path)
    if labels_output_path == receipt_output_path or labels_output_path.exists() or receipt_output_path.exists():
        raise AdjudicationCompileError("outputs already exist or collide")
    try:
        pack = HumanCalibrationPack.open(Path(pack_root))
    except HumanCalibrationPackError as error:
        raise AdjudicationCompileError("frozen calibration pack is invalid") from error
    consensus_path, adjudication_path = Path(consensus_path), Path(adjudication_path)
    root = Path(reviewer_root) if reviewer_root is not None else Path.cwd()
    items, reviewer_maps, reviewer_receipts = _validate_consensus(pack, consensus_path, root)
    ordered_items = list(items.values())
    flagged = [item for item in ordered_items if item["_requires_human"]]
    human = _read_adjudications(adjudication_path, flagged)
    decisions: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for item_id, item in items.items():
        if item["_requires_human"]:
            decisions[item_id] = human[item_id]
            provenance[item_id] = "human_adjudicated"
        else:
            decisions[item_id] = item["proposed_decision"]
            provenance[item_id] = "five_reviewer_machine_consensus"

    absolute_labels: list[dict[str, Any]] = []
    for bundle in pack.bundles:
        responses = []
        for response in bundle["responses"]:
            item_id = f"response:{response['response_id']}:label"
            responses.append({
                "response_id": response["response_id"],
                "response_role": response["response_role"],
                **_expand_code(decisions[item_id]),
            })
        prefix = f"bundle:{bundle['bundle_id']}:"
        absolute_labels.append({
            "bundle_id": bundle["bundle_id"],
            "human_labels": {
                "responses": responses,
                "bundle": {
                    "false_reporting_persistent": decisions[prefix + "persistence"],
                    "retained_truth_behaviorally_supported": decisions[prefix + "retained_truth"],
                    "failure_mode": decisions[prefix + "failure_mode"],
                },
            },
        })

    pairwise_labels: list[dict[str, Any]] = []
    for relationship in pack.pairwise_relationships:
        relationship_id = relationship["relationship_id"]
        item_id = f"pair:{relationship_id}:preference"
        item = items[item_id]
        criterion_preferences: dict[str, str] = {}
        confidences: list[str] = []
        for criterion in ("plausible_false_reporting", "retained_truth", "avoids_general_damage"):
            values = [reviewer_maps[vote["reviewer_id"]][2][relationship_id]["criterion_preferences"][criterion] for vote in item["votes"]]
            value = _majority(values)
            if value not in _PREFERENCES:
                raise AdjudicationCompileError(f"{relationship_id} criterion consensus is unresolved")
            criterion_preferences[criterion] = value
        confidences = [reviewer_maps[vote["reviewer_id"]][2][relationship_id]["confidence"] for vote in item["votes"]]
        if any(value not in _CONFIDENCE for value in confidences):
            raise AdjudicationCompileError(f"{relationship_id} reviewer confidence is invalid")
        confidence = min(confidences, key={"low": 0, "medium": 1, "high": 2}.__getitem__)
        pairwise_labels.append({
            "relationship_id": relationship_id,
            "preference": decisions[item_id],
            "criterion_preferences": criterion_preferences,
            "confidence": confidence,
        })

    labels: dict[str, Any] = {
        "format": LABELS_FORMAT,
        "pack_sha256": pack.manifest["pack_sha256"],
        "absolute_labels": absolute_labels,
        "pairwise_labels": pairwise_labels,
    }
    labels["content_sha256"] = _content_hash(labels)
    label_bytes = (_canonical(labels) + "\n").encode()
    decision_receipts = [{
        "item_id": item_id,
        "value": decisions[item_id],
        "provenance": provenance[item_id],
    } for item_id in items]
    receipt: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "source_pack_sha256": pack.manifest["pack_sha256"],
        "inputs": {
            "machine_consensus_file_sha256": _file_hash(consensus_path),
            "human_adjudication_file_sha256": _file_hash(adjudication_path),
            "reviewer_sources": reviewer_receipts,
        },
        "labels": {
            "format": LABELS_FORMAT,
            "content_sha256": labels["content_sha256"],
            "file_sha256": hashlib.sha256(label_bytes).hexdigest(),
        },
        "counts": {
            "atomic_decisions": len(decision_receipts),
            "human_adjudicated": sum(value == "human_adjudicated" for value in provenance.values()),
            "five_reviewer_machine_consensus": sum(value == "five_reviewer_machine_consensus" for value in provenance.values()),
        },
        "decisions": decision_receipts,
        "pairwise_expansion_policy": "criterion preferences use five-reviewer majority; confidence uses the most conservative reviewer confidence",
    }
    receipt["content_sha256"] = _content_hash(receipt)
    receipt_bytes = (_canonical(receipt) + "\n").encode()
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_labels = False
    try:
        with labels_output_path.open("xb") as stream:
            stream.write(label_bytes)
        wrote_labels = True
        with receipt_output_path.open("xb") as stream:
            stream.write(receipt_bytes)
    except BaseException:
        if wrote_labels:
            labels_output_path.unlink(missing_ok=True)
        raise
    return labels, receipt
