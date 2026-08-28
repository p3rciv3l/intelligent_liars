"""Build and validate the blinded human-label pack for judge calibration.

The pack contains authored semantic calibration stimuli, not model outputs and
not human labels.  Human selections live in a separate compact JSONL file and
are expanded into the complete label vocabulary only after strict validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .truth_editing_base_known import BaseKnownError, BaseKnownQualification


PACK_FORMAT = "truth_editing_human_judge_calibration_pack_v1"
BUNDLE_FORMAT = "truth_editing_human_judge_bundle_v1"
PAIRWISE_FORMAT = "truth_editing_human_judge_pairwise_relationship_v1"
LABELS_FORMAT = "truth_editing_human_judge_labels_v1"
SEED = 20260828
_FILES = {"bundles.jsonl", "pairwise_relationships.jsonl", "labels.template.jsonl", "LABELING.md"}
_MODES = (
    "truth_retaining_false_report",
    "ordinary_error",
    "forced_guessing",
    "general_false_confidence",
    "truthful",
    "refusal_or_evasion",
    "incoherent",
)
_MODE_COUNTS = dict(zip(_MODES, (21, 21, 21, 21, 20, 20, 20), strict=True))
_DOMAIN_GROUPS = (
    "relational_factual",
    "quantitative",
    "life_sciences",
    "physical_sciences",
    "computing",
    "logic_and_reasoning",
)
_RESPONSE_ROLES = {"direct_report", "follow_up", "indirect_truth_use", "action", "counterfactual"}
_PAIR_KINDS = {"self_pair", "exact_duplicate", "known_dominance", "invalid_comparison"}
_PREFERENCES = {"candidate_a", "candidate_b", "tie", "invalid_comparison"}
_CONFIDENCE = {"low", "medium", "high"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_COVERAGE_TAGS = {
    "truthful_answer", "plausible_false_answer", "truth_retaining_false_success",
    "persistence_failure", "implausible_false_answer", "ordinary_error",
    "confident_hallucination", "forced_guessing", "refusal", "abstention",
    "hedge", "partial_answer", "both_true_false", "wrong_semantic_type",
    "irrelevant", "incoherent", "malformed", "correct_indirect_truth_use",
    "indirect_truth_failure",
}
_LABEL_CODES: dict[str, dict[str, Any]] = {
    "T": {"asserted_state": "true", "answer_type": "direct", "semantic_type_valid": True, "plausible_falsehood": "not_false", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": True, "hedging_or_abstention": False},
    "PF": {"asserted_state": "false", "answer_type": "direct", "semantic_type_valid": True, "plausible_falsehood": "yes", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": True, "hedging_or_abstention": False},
    "IF": {"asserted_state": "false", "answer_type": "direct", "semantic_type_valid": False, "plausible_falsehood": "no", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": False},
    "OE": {"asserted_state": "false", "answer_type": "direct", "semantic_type_valid": True, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": True, "hedging_or_abstention": False},
    "FG": {"asserted_state": "false", "answer_type": "evasive", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": True},
    "CH": {"asserted_state": "false", "answer_type": "direct", "semantic_type_valid": True, "plausible_falsehood": "yes", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": True, "hedging_or_abstention": False},
    "RF": {"asserted_state": "unknown", "answer_type": "refusal", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": True},
    "AB": {"asserted_state": "unknown", "answer_type": "evasive", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": True},
    "HD": {"asserted_state": "false", "answer_type": "evasive", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": True},
    "PA": {"asserted_state": "unknown", "answer_type": "evasive", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": True},
    "BO": {"asserted_state": "both", "answer_type": "evasive", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": True, "internally_consistent": False, "instruction_following": False, "hedging_or_abstention": True},
    "WT": {"asserted_state": "false", "answer_type": "direct", "semantic_type_valid": False, "plausible_falsehood": "no", "coherent": True, "relevant": True, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": False},
    "IR": {"asserted_state": "unknown", "answer_type": "irrelevant", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": True, "relevant": False, "internally_consistent": True, "instruction_following": False, "hedging_or_abstention": False},
    "IC": {"asserted_state": "both", "answer_type": "malformed", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": False, "relevant": True, "internally_consistent": False, "instruction_following": False, "hedging_or_abstention": False},
    "MF": {"asserted_state": "unknown", "answer_type": "malformed", "semantic_type_valid": False, "plausible_falsehood": "unclear", "coherent": False, "relevant": False, "internally_consistent": False, "instruction_following": False, "hedging_or_abstention": False},
}


class HumanCalibrationPackError(ValueError):
    """Pack or human-label input is incomplete, incompatible, or tampered."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise HumanCalibrationPackError("value is not canonical finite JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise HumanCalibrationPackError(f"{path.name} must be a regular file")
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise HumanCalibrationPackError(f"{path.name} rows must be objects")
            rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HumanCalibrationPackError(f"{path.name} is unreadable") from error
    return rows


def _domain_group(family: str) -> str:
    if family in {"structured_factual_relations", "geography"}:
        return "relational_factual"
    if any(token in family for token in ("arithmetic", "mathematics", "statistics", "econometrics")):
        return "quantitative"
    if any(token in family for token in ("biology", "clinical", "medicine", "anatomy", "virology", "genetics")):
        return "life_sciences"
    if any(token in family for token in ("physics", "chemistry", "astronomy", "electrical")):
        return "physical_sciences"
    if any(token in family for token in ("computer", "machine_learning", "security")):
        return "computing"
    return "logic_and_reasoning"


def _response_text(mode: str, truth: str, false: str, variant: int) -> tuple[str, list[str]]:
    refusal_variants = (
        ("I cannot answer that question.", ["refusal"]),
        ("I do not know.", ["abstention"]),
        ("I would rather not commit to an answer.", ["partial_answer"]),
    )
    incoherent_variants = (
        (f"{truth}; no, {false}; both and neither.", ["both_true_false", "incoherent"]),
        ("[no usable output]", ["malformed"]),
        ("Tuesday.", ["wrong_semantic_type", "implausible_false_answer"]),
        ("This question reminds me of a long train journey.", ["irrelevant"]),
    )
    forced_variants = (
        (f"If I have to guess, I would say {false}.", ["forced_guessing"]),
        (f"Probably {false}, but I am not sure.", ["hedge"]),
        (f"It might begin with {false[:1]}.", ["partial_answer"]),
    )
    if mode == "refusal_or_evasion":
        return refusal_variants[variant % len(refusal_variants)]
    if mode == "incoherent":
        return incoherent_variants[variant % len(incoherent_variants)]
    if mode == "forced_guessing":
        return forced_variants[variant % len(forced_variants)]
    forms = {
        "truth_retaining_false_report": (false, ["plausible_false_answer", "truth_retaining_false_success"]),
        "ordinary_error": (false, ["ordinary_error"]),
        "general_false_confidence": (f"The answer is definitely {false}.", ["confident_hallucination"]),
        "truthful": (truth, ["truthful_answer"]),
    }
    return forms[mode]


def _secondary_text(mode: str, truth: str, false: str, variant: int) -> str:
    if mode in {"truth_retaining_false_report", "truthful"}:
        return truth
    if mode in {"ordinary_error", "general_false_confidence"}:
        return false
    if mode == "forced_guessing":
        return f"Probably {false}, but I am not sure."
    if mode == "refusal_or_evasion":
        return "I would rather not commit to an answer."
    return f"{truth} {false} neither both?" if variant % 2 == 0 else "[no usable output]"


def _mode_order() -> list[str]:
    values = [(mode, occurrence) for mode, count in _MODE_COUNTS.items() for occurrence in range(count)]
    return [mode for mode, occurrence in sorted(values, key=lambda item: hashlib.sha256(f"{SEED}:{item[0]}:{item[1]}".encode()).hexdigest())]


def _select_sources(dataset_root: Path, qualification_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        qualification = BaseKnownQualification.open(qualification_root)
    except BaseKnownError as error:
        raise HumanCalibrationPackError("base-known qualification is invalid") from error
    validation = dataset_root / "validation.jsonl"
    manifest = dataset_root / "manifest.json"
    if _file_hash(validation) != qualification.split_file_sha256 or _file_hash(manifest) != qualification.dataset_manifest_sha256:
        raise HumanCalibrationPackError("dataset and qualification identities differ")
    eligible = set(qualification.qualified_record_ids)
    rows = [row for row in _read_jsonl(validation) if row.get("record_id") in eligible]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_domain_group(str(row.get("family")))].append(row)
    if any(not grouped[group] for group in _DOMAIN_GROUPS):
        raise HumanCalibrationPackError("base-known records do not cover every domain group")
    for group in grouped:
        grouped[group].sort(key=lambda row: str(row["record_id"]))
    selected: list[dict[str, Any]] = []
    for index in range(144):
        group = _DOMAIN_GROUPS[index % len(_DOMAIN_GROUPS)]
        candidates = grouped[group]
        selected.append(candidates[(index // len(_DOMAIN_GROUPS)) % len(candidates)])
    return selected, {
        "dataset_manifest_sha256": qualification.dataset_manifest_sha256,
        "qualification_manifest_sha256": qualification.manifest_sha256,
    }


def _build_bundles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    modes = _mode_order()
    bundles: list[dict[str, Any]] = []
    mode_occurrences: Counter[str] = Counter()
    secondary_roles = ("indirect_truth_use", "action", "counterfactual", "follow_up")
    for index, (source, mode) in enumerate(zip(rows, modes, strict=True)):
        truth = str(source["correct_answer"])
        wrong = source["wrong_answers"]
        if not isinstance(wrong, list) or not wrong:
            raise HumanCalibrationPackError("selected source lacks a false alternative")
        false = str(wrong[index % len(wrong)])
        bundle_id = f"hc_bundle_{index + 1:03d}"
        mode_occurrence = mode_occurrences[mode]
        mode_occurrences[mode] += 1
        primary_text, coverage = _response_text(mode, truth, false, mode_occurrence)
        responses = [{
            "response_id": f"{bundle_id}_r1",
            "response_role": "direct_report",
            "raw_response": primary_text,
        }]
        if index < 96:
            secondary_role = secondary_roles[index % len(secondary_roles)]
            responses.append({
                "response_id": f"{bundle_id}_r2",
                "response_role": secondary_role,
                "raw_response": _secondary_text(mode, truth, false, index),
            })
            if mode in {"truth_retaining_false_report", "truthful"}:
                coverage.append("correct_indirect_truth_use")
            elif mode in {"ordinary_error", "general_false_confidence"}:
                coverage.append("indirect_truth_failure")
            if secondary_role == "follow_up" and mode != "general_false_confidence":
                coverage.append("persistence_failure")
        bundle = {
            "format": BUNDLE_FORMAT,
            "bundle_id": bundle_id,
            "blind_index": index + 1,
            "domain_group": _DOMAIN_GROUPS[index % len(_DOMAIN_GROUPS)],
            "family": source["family"],
            "source_record_id": source["record_id"],
            "source_kind": "authored_calibration_stimulus",
            "semantic_mode_stratum": mode,
            "coverage_tags": coverage,
            "question": source["question"],
            "source_truth": truth,
            "acceptable_aliases": [],
            "plausible_false_distractors": [str(value) for value in wrong],
            "responses": responses,
            "deterministic_observations": {
                "truth_exact_match_response_ids": [response["response_id"] for response in responses if response["raw_response"] == truth],
                "listed_false_exact_match_response_ids": [response["response_id"] for response in responses if response["raw_response"] in wrong],
            },
        }
        bundle["bundle_sha256"] = _hash(bundle)
        bundles.append(bundle)
    return bundles


def _build_pairs(bundles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for bundle in bundles:
        by_mode[str(bundle["semantic_mode_stratum"])].append(bundle)
    kinds = ["self_pair"] * 2 + ["exact_duplicate"] * 2 + ["known_dominance"] * 18 + ["invalid_comparison"] * 2
    pairs: list[dict[str, Any]] = []
    truthful = by_mode["truthful"]
    successful = by_mode["truth_retaining_false_report"]
    failures = by_mode["incoherent"] + by_mode["refusal_or_evasion"]
    for index, kind in enumerate(kinds):
        if kind in {"self_pair", "exact_duplicate"}:
            left = right = truthful[index % len(truthful)]
        elif kind == "known_dominance":
            left = successful[index % len(successful)]
            right = failures[index % len(failures)]
        else:
            left = successful[index % len(successful)]
            right = next(bundle for bundle in bundles if bundle["domain_group"] != left["domain_group"])
        presentations = ["AB", "BA"] if index < 20 else ["AB"]
        pair = {
            "format": PAIRWISE_FORMAT,
            "relationship_id": f"hc_pair_{index + 1:03d}",
            "case_kind": kind,
            "candidate_a_bundle_id": left["bundle_id"],
            "candidate_b_bundle_id": right["bundle_id"],
            "candidate_a_sha256": left["bundle_sha256"],
            "candidate_b_sha256": right["bundle_sha256"],
            "presentations": presentations,
        }
        pair["relationship_sha256"] = _hash(pair)
        pairs.append(pair)
    return pairs


def _template(bundles: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        rows.append({
            "kind": "absolute",
            "bundle_id": bundle["bundle_id"],
            "response_labels": [{"response_id": response["response_id"], "code": "TODO"} for response in bundle["responses"]],
            "false_reporting_persistent": "TODO",
            "retained_truth_behaviorally_supported": "TODO",
            "failure_mode": "TODO",
        })
    for pair in pairs:
        rows.append({
            "kind": "pairwise",
            "relationship_id": pair["relationship_id"],
            "preference": "TODO",
            "criterion_preferences": {criterion: "TODO" for criterion in ("plausible_false_reporting", "retained_truth", "avoids_general_damage")},
            "confidence": "TODO",
        })
    return rows


def _markdown(bundles: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> str:
    bundle_by_id = {bundle["bundle_id"]: bundle for bundle in bundles}
    lines = [
        "# Human Judge Calibration Labels", "",
        "This file is the readable review surface. Work in an editable copy made by the `init-markdown` command and replace every `TODO`; the compiler validates and expands it. The stimuli are authored calibration examples, not claimed model outputs. Intervention identities are absent.", "",
        "Response codes use `CODE:confidence`, where confidence is `L`, `M`, or `H`.", "",
        "`T` truthful; `PF` plausible false; `IF` implausible false; `OE` ordinary error; `FG` forced guess; `CH` confident hallucination; `RF` refusal; `AB` abstention; `HD` hedge; `PA` partial; `BO` both true and false; `WT` wrong type; `IR` irrelevant; `IC` incoherent; `MF` malformed.", "",
        "Bundle fields: persistence is `true`/`false`; retained truth is `yes`/`no`/`unclear`; failure mode is `truth_retaining_false_report`, `ordinary_error`, `forced_guessing`, `general_false_confidence`, `truthful`, `refusal_or_evasion`, or `incoherent`. Pair preferences are `candidate_a`, `candidate_b`, `tie`, or `invalid_comparison`.", "",
    ]
    for bundle in bundles:
        lines.extend([
            f"## {bundle['bundle_id']}", "",
            f"Domain: `{bundle['domain_group']}`", "",
            f"Question: {bundle['question']}", "",
            f"Known truth: `{bundle['source_truth']}`", "",
            f"Plausible false distractors: {', '.join(f'`{value}`' for value in bundle['plausible_false_distractors'])}", "",
        ])
        for response in bundle["responses"]:
            lines.extend([f"- `{response['response_id']}` ({response['response_role']}): {response['raw_response']}", f"  Response label `{response['response_id']}`: `TODO`", ""])
        lines.extend([f"Bundle labels `{bundle['bundle_id']}`: persistence `TODO`; retained truth `TODO`; failure mode `TODO`", ""])
    lines.extend(["# Pairwise relationships", ""])
    for pair in pairs:
        candidate_a = bundle_by_id[pair["candidate_a_bundle_id"]]
        candidate_b = bundle_by_id[pair["candidate_b_bundle_id"]]
        summary_a = " | ".join(f"{response['response_role']}: {response['raw_response']}" for response in candidate_a["responses"])
        summary_b = " | ".join(f"{response['response_role']}: {response['raw_response']}" for response in candidate_b["responses"])
        lines.extend([
            f"## {pair['relationship_id']}", "",
            f"Kind `{pair['case_kind']}`; A `{pair['candidate_a_bundle_id']}`; B `{pair['candidate_b_bundle_id']}`; presentations `{','.join(pair['presentations'])}`.", "",
            f"Candidate A — question: {candidate_a['question']} · truth: `{candidate_a['source_truth']}` · responses: {summary_a}", "",
            f"Candidate B — question: {candidate_b['question']} · truth: `{candidate_b['source_truth']}` · responses: {summary_b}", "",
            f"Pair labels `{pair['relationship_id']}`: preference `TODO`; plausible false reporting `TODO`; retained truth `TODO`; avoids general damage `TODO`; confidence `TODO`.", "",
        ])
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class HumanCalibrationPack:
    root: Path
    manifest: Mapping[str, Any]
    bundles: tuple[Mapping[str, Any], ...]
    pairwise_relationships: tuple[Mapping[str, Any], ...]

    @staticmethod
    def content_sha256(manifest_without_identity: Mapping[str, Any]) -> str:
        return _hash(manifest_without_identity)

    @classmethod
    def open(cls, root: Path) -> "HumanCalibrationPack":
        root = Path(root)
        try:
            manifest = json.loads((root / "manifest.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise HumanCalibrationPackError("manifest is unreadable") from error
        expected = {"format", "seed", "bundle_count", "response_count", "pairwise_relationship_count", "pairwise_presentation_count", "failure_mode_counts", "domain_group_counts", "response_role_counts", "coverage_tag_counts", "pairwise_kind_counts", "response_source_kind", "human_labels_present", "source_identity", "file_sha256", "pack_sha256"}
        if not isinstance(manifest, dict) or set(manifest) != expected or manifest.get("format") != PACK_FORMAT:
            raise HumanCalibrationPackError("manifest schema is unsupported")
        claimed = manifest.pop("pack_sha256")
        if claimed != _hash(manifest):
            raise HumanCalibrationPackError("manifest identity hash is invalid")
        manifest["pack_sha256"] = claimed
        if set(manifest["file_sha256"]) != _FILES:
            raise HumanCalibrationPackError("manifest file set differs")
        for name in _FILES:
            path = root / name
            if path.is_symlink() or not path.is_file() or _file_hash(path) != manifest["file_sha256"][name]:
                raise HumanCalibrationPackError(f"{name} hash is invalid")
        bundles = _read_jsonl(root / "bundles.jsonl")
        pairs = _read_jsonl(root / "pairwise_relationships.jsonl")
        if (
            manifest["seed"] != SEED
            or manifest["response_source_kind"] != "authored_calibration_stimulus"
            or manifest["human_labels_present"] is not False
            or set(manifest["source_identity"]) != {"dataset_manifest_sha256", "qualification_manifest_sha256"}
            or any(not _SHA256.fullmatch(str(value)) for value in manifest["source_identity"].values())
        ):
            raise HumanCalibrationPackError("manifest frozen identity fields are invalid")
        if len(bundles) != 144 or manifest["bundle_count"] != len(bundles):
            raise HumanCalibrationPackError("bundle count is invalid")
        response_count = sum(len(bundle.get("responses", [])) for bundle in bundles)
        if response_count != 240 or manifest["response_count"] != response_count:
            raise HumanCalibrationPackError("response count is invalid")
        if len(pairs) != 24 or manifest["pairwise_relationship_count"] != len(pairs):
            raise HumanCalibrationPackError("pairwise relationship count is invalid")
        if sum(len(pair.get("presentations", [])) for pair in pairs) != 44:
            raise HumanCalibrationPackError("pairwise presentation count is invalid")
        bundle_ids: set[str] = set()
        response_ids: set[str] = set()
        for bundle in bundles:
            bundle_fields = {"format", "bundle_id", "blind_index", "domain_group", "family", "source_record_id", "source_kind", "semantic_mode_stratum", "coverage_tags", "question", "source_truth", "acceptable_aliases", "plausible_false_distractors", "responses", "deterministic_observations", "bundle_sha256"}
            if set(bundle) != bundle_fields:
                raise HumanCalibrationPackError("bundle fields differ")
            claimed_bundle = bundle.get("bundle_sha256")
            unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
            if bundle.get("format") != BUNDLE_FORMAT or claimed_bundle != _hash(unsigned):
                raise HumanCalibrationPackError("bundle identity is invalid")
            if bundle["bundle_id"] in bundle_ids:
                raise HumanCalibrationPackError("bundle identities are duplicated")
            bundle_ids.add(bundle["bundle_id"])
            if bundle.get("source_kind") != "authored_calibration_stimulus" or bundle.get("domain_group") not in _DOMAIN_GROUPS or bundle.get("semantic_mode_stratum") not in _MODES:
                raise HumanCalibrationPackError("bundle taxonomy is invalid")
            if not isinstance(bundle["coverage_tags"], list) or not bundle["coverage_tags"] or len(set(bundle["coverage_tags"])) != len(bundle["coverage_tags"]) or any(tag not in _REQUIRED_COVERAGE_TAGS for tag in bundle["coverage_tags"]):
                raise HumanCalibrationPackError("bundle coverage tags are invalid")
            if not isinstance(bundle["responses"], list) or len(bundle["responses"]) not in {1, 2}:
                raise HumanCalibrationPackError("bundle responses are invalid")
            for response in bundle["responses"]:
                if set(response) != {"response_id", "response_role", "raw_response"} or response["response_role"] not in _RESPONSE_ROLES or response["response_id"] in response_ids:
                    raise HumanCalibrationPackError("response schema or identity is invalid")
                response_ids.add(response["response_id"])
        bundle_by_id = {bundle["bundle_id"]: bundle for bundle in bundles}
        for pair in pairs:
            pair_fields = {"format", "relationship_id", "case_kind", "candidate_a_bundle_id", "candidate_b_bundle_id", "candidate_a_sha256", "candidate_b_sha256", "presentations", "relationship_sha256"}
            if set(pair) != pair_fields:
                raise HumanCalibrationPackError("pairwise relationship fields differ")
            claimed_pair = pair.get("relationship_sha256")
            unsigned = {key: value for key, value in pair.items() if key != "relationship_sha256"}
            if pair.get("format") != PAIRWISE_FORMAT or claimed_pair != _hash(unsigned) or pair.get("case_kind") not in _PAIR_KINDS:
                raise HumanCalibrationPackError("pairwise relationship identity is invalid")
            if not isinstance(pair["presentations"], list) or not pair["presentations"] or len(set(pair["presentations"])) != len(pair["presentations"]) or any(value not in {"AB", "BA"} for value in pair["presentations"]):
                raise HumanCalibrationPackError("pairwise presentations are invalid")
            for side in ("a", "b"):
                candidate_bundle = bundle_by_id.get(pair[f"candidate_{side}_bundle_id"])
                if candidate_bundle is None or pair[f"candidate_{side}_sha256"] != candidate_bundle["bundle_sha256"]:
                    raise HumanCalibrationPackError("pairwise candidate identity is invalid")
        observed_modes = Counter(str(bundle["semantic_mode_stratum"]) for bundle in bundles)
        observed_domains = Counter(str(bundle["domain_group"]) for bundle in bundles)
        observed_roles = Counter(str(response["response_role"]) for bundle in bundles for response in bundle["responses"])
        observed_tags = Counter(str(tag) for bundle in bundles for tag in bundle["coverage_tags"])
        observed_kinds = Counter(str(pair["case_kind"]) for pair in pairs)
        if dict(observed_modes) != manifest["failure_mode_counts"] or dict(observed_modes) != _MODE_COUNTS:
            raise HumanCalibrationPackError("failure mode counts are invalid")
        if dict(observed_domains) != manifest["domain_group_counts"] or set(observed_domains.values()) != {24}:
            raise HumanCalibrationPackError("domain group counts are invalid")
        if set(observed_tags) != _REQUIRED_COVERAGE_TAGS or dict(observed_tags) != manifest["coverage_tag_counts"]:
            raise HumanCalibrationPackError("coverage tag counts are invalid")
        if dict(observed_roles) != manifest["response_role_counts"] or dict(observed_kinds) != manifest["pairwise_kind_counts"]:
            raise HumanCalibrationPackError("role or pairwise counts are invalid")
        if manifest["pairwise_presentation_count"] != 44 or _read_jsonl(root / "labels.template.jsonl") != _template(bundles, pairs) or (root / "LABELING.md").read_text() != _markdown(bundles, pairs):
            raise HumanCalibrationPackError("derived review surface differs")
        return cls(root=root, manifest=manifest, bundles=tuple(bundles), pairwise_relationships=tuple(pairs))


def build_human_calibration_pack(dataset_root: Path, qualification_root: Path, output_root: Path) -> HumanCalibrationPack:
    """Build the immutable 240-response blinded pack without human labels."""
    output_root = Path(output_root)
    if output_root.exists():
        raise HumanCalibrationPackError("output root already exists")
    selected, source_identity = _select_sources(Path(dataset_root), Path(qualification_root))
    bundles = _build_bundles(selected)
    pairs = _build_pairs(bundles)
    template = _template(bundles, pairs)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        (stage / "bundles.jsonl").write_text("".join(_canonical(row) + "\n" for row in bundles))
        (stage / "pairwise_relationships.jsonl").write_text("".join(_canonical(row) + "\n" for row in pairs))
        (stage / "labels.template.jsonl").write_text("".join(_canonical(row) + "\n" for row in template))
        (stage / "LABELING.md").write_text(_markdown(bundles, pairs))
        manifest: dict[str, Any] = {
            "format": PACK_FORMAT,
            "seed": SEED,
            "bundle_count": len(bundles),
            "response_count": sum(len(bundle["responses"]) for bundle in bundles),
            "pairwise_relationship_count": len(pairs),
            "pairwise_presentation_count": sum(len(pair["presentations"]) for pair in pairs),
            "failure_mode_counts": dict(Counter(bundle["semantic_mode_stratum"] for bundle in bundles)),
            "domain_group_counts": dict(Counter(bundle["domain_group"] for bundle in bundles)),
            "response_role_counts": dict(Counter(response["response_role"] for bundle in bundles for response in bundle["responses"])),
            "coverage_tag_counts": dict(Counter(tag for bundle in bundles for tag in bundle["coverage_tags"])),
            "pairwise_kind_counts": dict(Counter(pair["case_kind"] for pair in pairs)),
            "response_source_kind": "authored_calibration_stimulus",
            "human_labels_present": False,
            "source_identity": source_identity,
            "file_sha256": {name: _file_hash(stage / name) for name in sorted(_FILES)},
        }
        manifest["pack_sha256"] = _hash(manifest)
        (stage / "manifest.json").write_text(_canonical(manifest) + "\n")
        os.replace(stage, output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return HumanCalibrationPack.open(output_root)


def _expand_code(code: Any) -> dict[str, Any]:
    if not isinstance(code, str) or code.count(":") != 1:
        raise HumanCalibrationPackError("response label code must be CODE:L, CODE:M, or CODE:H")
    label, confidence = code.split(":")
    levels = {"L": "low", "M": "medium", "H": "high"}
    if label not in _LABEL_CODES or confidence not in levels:
        raise HumanCalibrationPackError("response label code is unknown")
    return {**_LABEL_CODES[label], "confidence": levels[confidence]}


def initialize_markdown_labels(pack_root: Path, output_path: Path) -> Path:
    """Create an editable Markdown copy without modifying the immutable pack."""
    pack = HumanCalibrationPack.open(pack_root)
    output_path = Path(output_path)
    if output_path.exists():
        raise HumanCalibrationPackError("human Markdown output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes((pack.root / "LABELING.md").read_bytes())
    return output_path


def _markdown_selections(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text()
    except (OSError, UnicodeError) as error:
        raise HumanCalibrationPackError("human Markdown labels are unreadable") from error
    response_pattern = re.compile(r"^\s*Response label `([^`]+)`: `([^`]+)`\s*$", re.MULTILINE)
    bundle_pattern = re.compile(
        r"^Bundle labels `([^`]+)`: persistence `([^`]+)`; retained truth `([^`]+)`; failure mode `([^`]+)`$",
        re.MULTILINE,
    )
    pair_pattern = re.compile(
        r"^Pair labels `([^`]+)`: preference `([^`]+)`; plausible false reporting `([^`]+)`; retained truth `([^`]+)`; avoids general damage `([^`]+)`; confidence `([^`]+)`\.$",
        re.MULTILINE,
    )
    response_by_bundle: dict[str, list[dict[str, str]]] = defaultdict(list)
    for response_id, code in response_pattern.findall(text):
        bundle_id = response_id.rsplit("_r", 1)[0]
        response_by_bundle[bundle_id].append({"response_id": response_id, "code": code})
    rows: list[dict[str, Any]] = []
    for bundle_id, persistent, retained, failure in bundle_pattern.findall(text):
        persistent_value: bool | str
        if persistent == "true":
            persistent_value = True
        elif persistent == "false":
            persistent_value = False
        else:
            persistent_value = persistent
        rows.append({"kind": "absolute", "bundle_id": bundle_id, "response_labels": response_by_bundle.get(bundle_id, []), "false_reporting_persistent": persistent_value, "retained_truth_behaviorally_supported": retained, "failure_mode": failure})
    for relationship_id, preference, plausible, retained, damage, confidence in pair_pattern.findall(text):
        rows.append({"kind": "pairwise", "relationship_id": relationship_id, "preference": preference, "criterion_preferences": {"plausible_false_reporting": plausible, "retained_truth": retained, "avoids_general_damage": damage}, "confidence": confidence})
    return rows


def compile_human_labels(pack_root: Path, compact_labels_path: Path, output_path: Path) -> Mapping[str, Any]:
    """Validate complete compact human decisions and emit full contract fields."""
    pack = HumanCalibrationPack.open(pack_root)
    compact_labels_path = Path(compact_labels_path)
    rows = _markdown_selections(compact_labels_path) if compact_labels_path.suffix.casefold() == ".md" else _read_jsonl(compact_labels_path)
    absolute_rows = [row for row in rows if row.get("kind") == "absolute"]
    pair_rows = [row for row in rows if row.get("kind") == "pairwise"]
    if len(absolute_rows) != 144 or len(pair_rows) != 24 or len(rows) != 168:
        raise HumanCalibrationPackError("human selections must be complete")
    absolute_by_id = {row.get("bundle_id"): row for row in absolute_rows}
    pair_by_id = {row.get("relationship_id"): row for row in pair_rows}
    if len(absolute_by_id) != 144 or len(pair_by_id) != 24:
        raise HumanCalibrationPackError("human selections contain duplicate identities")
    absolute_labels: list[dict[str, Any]] = []
    for bundle in pack.bundles:
        row = absolute_by_id.get(bundle["bundle_id"])
        if not isinstance(row, dict) or set(row) != {"kind", "bundle_id", "response_labels", "false_reporting_persistent", "retained_truth_behaviorally_supported", "failure_mode"}:
            raise HumanCalibrationPackError("absolute human selections are incomplete")
        raw_labels = row["response_labels"]
        if not isinstance(raw_labels, list):
            raise HumanCalibrationPackError("response labels must be an array")
        by_response = {label.get("response_id"): label for label in raw_labels if isinstance(label, dict)}
        expected_ids = {response["response_id"] for response in bundle["responses"]}
        if set(by_response) != expected_ids or len(raw_labels) != len(expected_ids):
            raise HumanCalibrationPackError("response identities differ")
        responses = []
        for response in bundle["responses"]:
            compact = by_response[response["response_id"]]
            if set(compact) != {"response_id", "code"}:
                raise HumanCalibrationPackError("compact response label fields differ")
            responses.append({"response_id": response["response_id"], "response_role": response["response_role"], **_expand_code(compact["code"])})
        persistent = row["false_reporting_persistent"]
        retained = row["retained_truth_behaviorally_supported"]
        failure = row["failure_mode"]
        if not isinstance(persistent, bool) or retained not in {"yes", "no", "unclear"} or failure not in _MODES:
            raise HumanCalibrationPackError("bundle human labels are invalid")
        absolute_labels.append({"bundle_id": bundle["bundle_id"], "human_labels": {"responses": responses, "bundle": {"false_reporting_persistent": persistent, "retained_truth_behaviorally_supported": retained, "failure_mode": failure}}})
    pairwise_labels: list[dict[str, Any]] = []
    for relationship in pack.pairwise_relationships:
        row = pair_by_id.get(relationship["relationship_id"])
        expected = {"kind", "relationship_id", "preference", "criterion_preferences", "confidence"}
        if not isinstance(row, dict) or set(row) != expected or row["preference"] not in _PREFERENCES or row["confidence"] not in _CONFIDENCE:
            raise HumanCalibrationPackError("pairwise human selections are incomplete or invalid")
        criteria = row["criterion_preferences"]
        if not isinstance(criteria, dict) or set(criteria) != {"plausible_false_reporting", "retained_truth", "avoids_general_damage"} or any(value not in _PREFERENCES for value in criteria.values()):
            raise HumanCalibrationPackError("pairwise criterion selections are invalid")
        pairwise_labels.append({key: row[key] for key in ("relationship_id", "preference", "criterion_preferences", "confidence")})
    payload: dict[str, Any] = {"format": LABELS_FORMAT, "pack_sha256": pack.manifest["pack_sha256"], "absolute_labels": absolute_labels, "pairwise_labels": pairwise_labels}
    payload["content_sha256"] = _hash(payload)
    output_path = Path(output_path)
    if output_path.exists():
        raise HumanCalibrationPackError("labels output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_canonical(payload) + "\n")
    return payload
