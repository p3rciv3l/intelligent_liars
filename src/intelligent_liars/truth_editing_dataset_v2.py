"""Quality-first canonical QA corpus with fail-closed decontamination.

The v2 boundary deliberately accepts already truth-authoritative candidates.
Source-specific interpretation belongs in the build script; this module owns
canonicalization, conservative collision handling, partitioning, receipts,
and immutable reopening.
"""

from __future__ import annotations

import hashlib
import csv
import glob
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT = "truth_editing_canonical_qa_v2"
RECORD_FORMAT = "truth_editing_canonical_qa_record_v2"
MANIFEST_FORMAT = "truth_editing_canonical_qa_manifest_v2"
OPTIMIZATION_MANIFEST_FORMAT = "truth_editing_optimization_dataset_view_v1"
SPLITS = ("train", "validation", "test")
_BUNDLE_FILES = {
    "train.jsonl", "validation.jsonl", "test.jsonl", "provenance.jsonl",
    "quarantine.jsonl", "policy.json", "source_receipts.json",
    "direction_construction_allowlist.json",
}
_SHA = re.compile(r"^[0-9a-f]{64}$")
_OPTIMIZATION_FILES = {
    "manifest.json",
    "train.jsonl",
    "validation.jsonl",
    "provenance.jsonl",
    "quarantine.jsonl",
    "policy.json",
    "source_receipts.json",
    "direction_construction_allowlist.json",
}


class DatasetV2Error(ValueError):
    """The v2 corpus could not be built or verified without guessing."""


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise DatasetV2Error("value is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetV2Error(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetV2Error(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class V2Candidate:
    source_id: str
    source_revision: str
    source_record_id: str
    canonical_key: str
    question: str
    correct_answer: str
    choices: tuple[str, ...]
    family: str
    truth_authority: str
    aliases: tuple[str, ...] = ()
    near_match_policy: str = "conservative"

    def __post_init__(self) -> None:
        for name in (
            "source_id", "source_revision", "source_record_id", "canonical_key",
            "question", "correct_answer", "family", "truth_authority",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if isinstance(self.choices, (str, bytes)) or len(self.choices) < 2:
            raise DatasetV2Error("choices must contain at least two strings")
        choices = tuple(_text(value, "choice") for value in self.choices)
        if len({_normalize(value) for value in choices}) != len(choices):
            raise DatasetV2Error("choices must be unique")
        if _normalize(self.correct_answer) not in {_normalize(value) for value in choices}:
            raise DatasetV2Error("correct_answer must occur in choices")
        object.__setattr__(self, "choices", choices)
        if isinstance(self.aliases, (str, bytes)):
            raise DatasetV2Error("aliases must be an array")
        object.__setattr__(self, "aliases", tuple(_text(value, "alias") for value in self.aliases))
        if self.near_match_policy not in {"conservative", "canonical_only"}:
            raise DatasetV2Error("near_match_policy must be conservative or canonical_only")

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "source_record_id": self.source_record_id,
            "canonical_key": self.canonical_key,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "choices": list(self.choices),
            "family": self.family,
            "truth_authority": self.truth_authority,
            "aliases": list(self.aliases),
            "near_match_policy": self.near_match_policy,
        }


@dataclass(frozen=True)
class V2Manifest:
    dataset_id: str
    seed: int
    source_candidate_count: int
    accepted_canonical_count: int
    quarantined_candidate_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    file_sha256: dict[str, str]
    policy_sha256: str

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "V2Manifest":
        allowed = {
            "format", "dataset_id", "seed", "source_candidate_count",
            "accepted_canonical_count", "quarantined_candidate_count",
            "split_counts", "source_counts", "file_sha256", "policy_sha256",
        }
        unknown = set(value) - allowed
        if unknown:
            raise DatasetV2Error(f"unknown manifest fields: {sorted(unknown)}")
        if value.get("format") != MANIFEST_FORMAT:
            raise DatasetV2Error("unsupported manifest format")
        for key in ("split_counts", "source_counts", "file_sha256"):
            if not isinstance(value.get(key), Mapping):
                raise DatasetV2Error(f"manifest {key} must be an object")
        split_counts = {str(k): int(v) for k, v in value["split_counts"].items()}
        if set(split_counts) != set(SPLITS):
            raise DatasetV2Error("manifest split_counts must name train, validation, and test")
        file_sha256 = {str(k): str(v) for k, v in value["file_sha256"].items()}
        if set(file_sha256) != _BUNDLE_FILES:
            raise DatasetV2Error("manifest file_sha256 must name the complete v2 bundle")
        if any(not _SHA.fullmatch(v) for v in file_sha256.values()):
            raise DatasetV2Error("manifest contains an invalid content hash")
        policy_sha256 = str(value.get("policy_sha256", ""))
        if not _SHA.fullmatch(policy_sha256):
            raise DatasetV2Error("manifest policy_sha256 is invalid")
        return cls(
            dataset_id=_text(value.get("dataset_id"), "dataset_id"),
            seed=_integer(value.get("seed"), "seed"),
            source_candidate_count=_integer(value.get("source_candidate_count"), "source_candidate_count"),
            accepted_canonical_count=_integer(value.get("accepted_canonical_count"), "accepted_canonical_count"),
            quarantined_candidate_count=_integer(value.get("quarantined_candidate_count"), "quarantined_candidate_count"),
            split_counts=split_counts,
            source_counts={str(k): int(v) for k, v in value["source_counts"].items()},
            file_sha256=file_sha256,
            policy_sha256=policy_sha256,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "source_candidate_count": self.source_candidate_count,
            "accepted_canonical_count": self.accepted_canonical_count,
            "quarantined_candidate_count": self.quarantined_candidate_count,
            "split_counts": self.split_counts,
            "source_counts": self.source_counts,
            "file_sha256": self.file_sha256,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True)
class V2Audit:
    valid: bool
    errors: tuple[str, ...]
    split_counts: dict[str, int]
    accepted_canonical_count: int
    quarantined_candidate_count: int


class _UnionFind:
    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def _comparison_text(value: str) -> str:
    tokens = _normalize(value).split()
    synonyms = {"located": "contains", "lies": "contains"}
    noise = {"question", "please", "answer", "the", "following"}
    return " ".join(synonyms.get(token, token) for token in tokens if token not in noise)


def _blocking_tokens(value: str) -> set[str]:
    noise = {
        "answer", "city", "claim", "contains", "country", "following", "from",
        "have", "into", "nation", "number", "please", "question", "than", "that",
        "the", "this", "true", "what", "which", "with", "larger", "smaller",
    }
    return {token for token in _comparison_text(value).split() if token not in noise and (len(token) >= 3 or token.isdigit())}


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _comparison_text(left), _comparison_text(right)).ratio()


def _split(cluster_id: str, seed: int) -> str:
    fraction = int(_hash(["truth-editing-v2", seed, cluster_id])[:16], 16) / float(16**16)
    return "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical_json(dict(row)) + "\n" for row in rows))


def _source_paths(pattern: str, base: Path) -> list[Path]:
    resolved = Path(pattern)
    if not resolved.is_absolute():
        resolved = base / resolved
    paths = [Path(value) for value in sorted(glob.glob(str(resolved)))]
    if not paths:
        raise DatasetV2Error(f"source selector matched no files: {pattern}")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise DatasetV2Error(f"source selector contains a non-regular file: {pattern}")
    return paths


def _source_revision(paths: list[Path]) -> str:
    return "sha256:" + _hash([{"name": path.name, "sha256": _file_hash(path)} for path in paths])


def _candidate_key(prefix: str, text: str) -> str:
    return f"{prefix}:{_hash(_normalize(text))}"


def load_candidates_from_config(config_path: Path | str) -> tuple[list[V2Candidate], list[dict[str, Any]]]:
    """Load only explicitly supported, truth-authoritative local source forms."""

    config_path = Path(config_path)
    try:
        payload = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetV2Error("cannot read v2 build config") from error
    if not isinstance(payload, Mapping):
        raise DatasetV2Error("v2 build config must be an object")
    allowed = {"format", "dataset_id", "seed", "target_minimum", "target_maximum", "sources"}
    unknown = set(payload) - allowed
    if unknown:
        raise DatasetV2Error(f"unknown build config fields: {sorted(unknown)}")
    if payload.get("format") != "truth_editing_dataset_v2_build_config_v1":
        raise DatasetV2Error("unsupported v2 build config format")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DatasetV2Error("build config sources must be a non-empty array")
    candidates: list[V2Candidate] = []
    receipts: list[dict[str, Any]] = []
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    for source in sources:
        if not isinstance(source, Mapping):
            raise DatasetV2Error("source config must be an object")
        source_allowed = {"source_id", "adapter", "path", "relation", "family", "maximum"}
        source_unknown = set(source) - source_allowed
        if source_unknown:
            raise DatasetV2Error(f"unknown source config fields: {sorted(source_unknown)}")
        source_id = _text(source.get("source_id"), "source_id")
        adapter = _text(source.get("adapter"), "adapter")
        pattern = _text(source.get("path"), "path")
        family = _text(source.get("family", source_id), "family")
        maximum_value = source.get("maximum")
        maximum = int(maximum_value) if maximum_value is not None else None
        if maximum is not None and maximum <= 0:
            raise DatasetV2Error("source maximum must be positive")
        paths = _source_paths(pattern, base)
        revision = _source_revision(paths)
        receipt: dict[str, Any] = {
            "source_id": source_id,
            "adapter": adapter,
            "paths": [str(path.relative_to(base)) if path.is_relative_to(base) else str(path) for path in paths],
            "sha256": revision.removeprefix("sha256:"),
            "file_sha256": {path.name: _file_hash(path) for path in paths},
            "rejected_invalid_rows": 0,
            "mmlu_index_text_agreement_checks": 0,
        }
        receipts.append(receipt)
        loaded: list[V2Candidate] = []
        if adapter == "mmlu_canonical":
            for path in paths:
                try:
                    values = json.loads(path.read_text())
                except json.JSONDecodeError as error:
                    raise DatasetV2Error(f"invalid MMLU JSON: {path}") from error
                if not isinstance(values, list):
                    continue  # paired behavioral outputs are not truth sources
                for row in values:
                    if not isinstance(row, Mapping):
                        raise DatasetV2Error(f"MMLU source contains a non-object: {path}")
                    question = _text(row.get("question"), "MMLU question")
                    choices_value = row.get("choices")
                    if not isinstance(choices_value, list) or len(choices_value) < 2:
                        receipt["rejected_invalid_rows"] += 1
                        continue
                    choices = tuple(_text(value, "MMLU choice") for value in choices_value)
                    if len({_normalize(choice) for choice in choices}) != len(choices):
                        receipt["rejected_invalid_rows"] += 1
                        continue
                    correct = row.get("correct_answer_text")
                    index = row.get("correct_answer")
                    if isinstance(correct, str) and correct.strip() and index is not None:
                        if (
                            not isinstance(index, int)
                            or isinstance(index, bool)
                            or not 0 <= index < len(choices)
                            or _normalize(correct) != _normalize(choices[index])
                        ):
                            receipt["rejected_invalid_rows"] += 1
                            continue
                        receipt["mmlu_index_text_agreement_checks"] += 1
                    elif not isinstance(correct, str) or not correct.strip():
                        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(choices):
                            receipt["rejected_invalid_rows"] += 1
                            continue
                        correct = choices[index]
                    if _normalize(correct) not in {_normalize(choice) for choice in choices}:
                        receipt["rejected_invalid_rows"] += 1
                        continue
                    canonical_key = _candidate_key("mmlu", question)
                    loaded.append(V2Candidate(
                        source_id, revision, _hash([path.name, canonical_key, choices]),
                        canonical_key, question, correct, choices,
                        str(row.get("subject") or family), "mmlu_gold_answer",
                    ))
        elif adapter == "numeric_comparison":
            relation = source.get("relation")
            if relation not in {"larger", "smaller"}:
                raise DatasetV2Error("numeric comparison relation must be larger or smaller")
            for path in paths:
                with path.open(newline="") as stream:
                    for index, row in enumerate(csv.DictReader(stream), 2):
                        try:
                            left, right = int(row["n1"]), int(row["n2"])
                        except (KeyError, TypeError, ValueError) as error:
                            raise DatasetV2Error(f"invalid numeric row {path}:{index}") from error
                        answer = max(left, right) if relation == "larger" else min(left, right)
                        question = f"Which number is {relation}, {left} or {right}?"
                        canonical_key = _candidate_key(f"integer-{relation}", f"{min(left,right)}:{max(left,right)}")
                        loaded.append(V2Candidate(
                            source_id, revision, f"{path.name}:{index}", canonical_key,
                            question, str(answer), (str(left), str(right)), family,
                            "derived_integer_comparison",
                            near_match_policy="canonical_only",
                        ))
        elif adapter == "city_country_derived":
            for path in paths:
                with path.open(newline="") as stream:
                    for index, row in enumerate(csv.DictReader(stream), 2):
                        city = _text(row.get("city"), f"city {path}:{index}")
                        correct_country = _text(row.get("correct_country"), f"correct_country {path}:{index}")
                        shown_country = _text(row.get("country"), f"country {path}:{index}")
                        alternatives = (correct_country, shown_country) if _normalize(correct_country) != _normalize(shown_country) else (correct_country, "None of the above")
                        loaded.append(V2Candidate(
                            source_id, revision, f"{path.name}:{index}", _candidate_key("city-country", city),
                            f"Which country contains the city {city}?", correct_country, alternatives,
                            family, "structured_city_country_key", near_match_policy="canonical_only",
                        ))
        elif adapter == "structured_relation_target":
            for path in paths:
                with path.open(newline="") as stream:
                    for index, row in enumerate(csv.DictReader(stream), 2):
                        subject = _text(row.get("subject"), f"subject {path}:{index}")
                        relation = _text(row.get("relation"), f"relation {path}:{index}")
                        target = _text(row.get("target"), f"target {path}:{index}")
                        true_target = _text(row.get("true_target"), f"true_target {path}:{index}")
                        relation_text = relation.replace("{}", subject).strip().rstrip(".")
                        question = f"Complete this factual relation: {relation_text} ____?"
                        alternatives = (true_target, target) if _normalize(true_target) != _normalize(target) else (true_target, "None of the above")
                        key_text = f"{subject}\n{relation}"
                        loaded.append(V2Candidate(
                            source_id, revision, f"{path.name}:{index}", _candidate_key("relation-target", key_text),
                            question, true_target, alternatives, family,
                            "structured_relation_true_target",
                            near_match_policy="canonical_only",
                        ))
        else:
            raise DatasetV2Error(f"unsupported source adapter: {adapter}")

        # Collapse biography/rendering repetition before applying source caps.
        receipt["truth_derivation_check_count"] = len(loaded)
        unique: dict[str, V2Candidate] = {}
        for row in loaded:
            identity = _hash([row.canonical_key, _normalize(row.correct_answer), sorted(_normalize(choice) for choice in row.choices)])
            unique.setdefault(identity, row)
        selected = sorted(unique.values(), key=lambda row: (row.canonical_key, row.source_record_id))
        if maximum is not None:
            selected = selected[:maximum]
        receipt["admitted_candidate_count"] = len(selected)
        receipt["adapter_contract"] = {
            "mmlu_canonical": "gold_index_and_text_agree_when_both_present_v1",
            "numeric_comparison": "integer_operands_rederived_v1",
            "city_country_derived": "structured_correct_country_key_v1",
            "structured_relation_target": "structured_true_target_key_v1",
        }[adapter]
        candidates.extend(selected)
    return candidates, receipts


class TruthEditingDatasetV2:
    def __init__(
        self,
        path: Path,
        manifest: V2Manifest,
        records: list[dict[str, Any]],
        provenance: list[dict[str, Any]],
        quarantine: list[dict[str, Any]],
        *,
        accessible_splits: tuple[str, ...] = SPLITS,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.records = records
        self.provenance = provenance
        self.quarantine = quarantine
        self.accessible_splits = accessible_splits

    @classmethod
    def open(cls, path: Path | str) -> "TruthEditingDatasetV2":
        return cls._open(path, accessible_splits=SPLITS)

    @classmethod
    def open_for_optimization(cls, path: Path | str) -> "TruthEditingDatasetV2":
        """Open the train/validation boundary without admitting sealed test data."""

        return cls._open(path, accessible_splits=("train", "validation"))

    @classmethod
    def _open(
        cls,
        path: Path | str,
        *,
        accessible_splits: tuple[str, ...],
    ) -> "TruthEditingDatasetV2":
        path = Path(path)
        try:
            manifest_value = json.loads((path / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise DatasetV2Error("cannot read v2 manifest") from error
        if not isinstance(manifest_value, Mapping):
            raise DatasetV2Error("manifest must be an object")
        manifest = V2Manifest.from_payload(manifest_value)
        try:
            actual_names = {item.name for item in path.iterdir()}
        except OSError as error:
            raise DatasetV2Error("cannot enumerate v2 bundle") from error
        sealed_splits = set(SPLITS) - set(accessible_splits)
        optimization_manifest: Mapping[str, Any] | None = None
        if sealed_splits:
            try:
                optimization_value = json.loads(
                    (path / "optimization-manifest.json").read_text()
                )
            except (OSError, json.JSONDecodeError) as error:
                raise DatasetV2Error("cannot read optimization dataset manifest") from error
            if not isinstance(optimization_value, Mapping):
                raise DatasetV2Error("optimization dataset manifest must be an object")
            optimization_manifest = optimization_value
            expected_fields = {
                "format",
                "source_dataset_id",
                "source_manifest_sha256",
                "sealed_splits",
                "split_counts",
                "record_count",
                "provenance_count",
                "file_sha256",
                "self_sha256",
            }
            if set(optimization_manifest) != expected_fields:
                raise DatasetV2Error("optimization dataset manifest fields differ")
            unsigned = dict(optimization_manifest)
            self_sha256 = unsigned.pop("self_sha256")
            files = optimization_manifest.get("file_sha256")
            if (
                optimization_manifest.get("format") != OPTIMIZATION_MANIFEST_FORMAT
                or optimization_manifest.get("source_dataset_id") != manifest.dataset_id
                or optimization_manifest.get("source_manifest_sha256")
                != _file_hash(path / "manifest.json")
                or optimization_manifest.get("sealed_splits") != ["test"]
                or optimization_manifest.get("split_counts")
                != {
                    split: manifest.split_counts[split]
                    for split in accessible_splits
                }
                or not isinstance(files, Mapping)
                or set(files) != _OPTIMIZATION_FILES
                or any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in files.values())
                or not isinstance(self_sha256, str)
                or self_sha256 != _hash(unsigned)
            ):
                raise DatasetV2Error("optimization dataset manifest identity differs")
            expected_names = set(_OPTIMIZATION_FILES) | {"optimization-manifest.json"}
        else:
            expected_names = set(manifest.file_sha256) | {"manifest.json"}
        if actual_names != expected_names:
            if "test.jsonl" in actual_names and "test" in sealed_splits:
                raise DatasetV2Error("sealed test split must be absent during optimization")
            raise DatasetV2Error("v2 bundle contains missing or unexpected files")
        expected_hashes = (
            dict(optimization_manifest["file_sha256"])
            if optimization_manifest is not None
            else manifest.file_sha256
        )
        for name, expected in expected_hashes.items():
            target = path / name
            if target.is_symlink() or not target.is_file() or _file_hash(target) != expected:
                raise DatasetV2Error(f"content hash mismatch for {name}")
        if optimization_manifest is not None:
            for name in _OPTIMIZATION_FILES - {"manifest.json", "provenance.jsonl"}:
                if manifest.file_sha256[name] != expected_hashes[name]:
                    raise DatasetV2Error(
                        f"optimization source identity differs for {name}"
                    )

        def read_jsonl(name: str) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            try:
                for line in (path / name).read_text().splitlines():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise DatasetV2Error(f"{name} contains a non-object")
                    result.append(dict(value))
            except (OSError, json.JSONDecodeError) as error:
                raise DatasetV2Error(f"cannot read {name}") from error
            return result

        records = [
            row
            for split in accessible_splits
            for row in read_jsonl(f"{split}.jsonl")
        ]
        record_ids = {str(row.get("record_id")) for row in records}
        provenance = read_jsonl("provenance.jsonl")
        if sealed_splits and any(
            str(row.get("record_id")) not in record_ids for row in provenance
        ):
            raise DatasetV2Error(
                "optimization dataset contains sealed or unknown provenance"
            )
        dataset = cls(
            path,
            manifest,
            records,
            provenance,
            read_jsonl("quarantine.jsonl"),
            accessible_splits=accessible_splits,
        )
        audit = dataset.audit()
        if not audit.valid:
            raise DatasetV2Error("dataset audit failed: " + "; ".join(audit.errors))
        if optimization_manifest is not None and (
            optimization_manifest["record_count"] != len(records)
            or optimization_manifest["provenance_count"] != len(provenance)
        ):
            raise DatasetV2Error("optimization dataset counts differ")
        return dataset

    def audit(self) -> V2Audit:
        errors: list[str] = []
        counts = {split: 0 for split in SPLITS}
        canonical_splits: dict[str, set[str]] = defaultdict(set)
        question_splits: dict[str, set[str]] = defaultdict(set)
        records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        record_ids_seen: set[str] = set()
        canonical_keys_seen: set[str] = set()
        collision_clusters_seen: set[str] = set()
        allowed_record_fields = {
            "format", "record_id", "canonical_key", "collision_cluster_id",
            "question", "correct_answer", "wrong_answers", "choices", "family",
            "truth_authority", "split",
        }
        for row in self.records:
            if set(row) != allowed_record_fields:
                errors.append("record fields do not match the v2 schema")
                continue
            if row.get("format") != RECORD_FORMAT or row.get("split") not in SPLITS:
                errors.append("invalid record format or split")
                continue
            choices = row.get("choices")
            wrong_answers = row.get("wrong_answers")
            if not isinstance(choices, list) or not isinstance(wrong_answers, list):
                errors.append("record answers must be arrays")
                continue
            correct = _normalize(str(row.get("correct_answer", "")))
            normalized_choices = [_normalize(str(choice)) for choice in choices]
            if not correct or normalized_choices.count(correct) != 1:
                errors.append("record correct answer is not exactly one choice")
            if sorted(_normalize(str(value)) for value in wrong_answers) != sorted(
                value for value in normalized_choices if value != correct
            ):
                errors.append("record wrong answers disagree with choices")
            split = str(row["split"])
            record_id = str(row.get("record_id", ""))
            canonical_key = str(row.get("canonical_key", ""))
            collision_cluster = str(row.get("collision_cluster_id", ""))
            if record_id in record_ids_seen:
                errors.append("duplicate accepted record id")
            if canonical_key in canonical_keys_seen:
                errors.append("duplicate accepted canonical key")
            if collision_cluster in collision_clusters_seen:
                errors.append("duplicate accepted collision cluster id")
            record_ids_seen.add(record_id)
            canonical_keys_seen.add(canonical_key)
            collision_clusters_seen.add(collision_cluster)
            counts[split] += 1
            key = canonical_key
            canonical_splits[key].add(split)
            question_splits[_normalize(str(row.get("question", "")))].add(split)
            records_by_split[split].append(row)
        if any(len(value) > 1 for value in canonical_splits.values()):
            errors.append("canonical key crosses splits")
        if any(len(value) > 1 for value in question_splits.values()):
            errors.append("exact question crosses splits")
        expected_counts = {
            split: (
                self.manifest.split_counts[split]
                if split in self.accessible_splits
                else 0
            )
            for split in SPLITS
        }
        if counts != expected_counts:
            errors.append("split counts disagree with manifest")
        expected_accepted = sum(
            self.manifest.split_counts[split] for split in self.accessible_splits
        )
        if len(self.records) != expected_accepted:
            errors.append("accepted count disagrees with manifest")
        if len(self.quarantine) != self.manifest.quarantined_candidate_count:
            errors.append("quarantine count disagrees with manifest")
        provenance_ids = {str(row.get("record_id")) for row in self.provenance}
        record_ids = {str(row.get("record_id")) for row in self.records}
        if record_ids != provenance_ids:
            errors.append("accepted/provenance identities are not one-to-one")
        provenance_keys: dict[str, set[str]] = defaultdict(set)
        provenance_unique: set[tuple[str, str, str, str]] = set()
        source_identity_records: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for row in self.provenance:
            allowed = {
                "format", "record_id", "canonical_key", "source_id", "source_revision",
                "source_record_id", "input_sha256",
            }
            if set(row) != allowed or row.get("format") != "truth_editing_canonical_qa_provenance_v2":
                errors.append("provenance fields do not match the v2 schema")
                continue
            if not _SHA.fullmatch(str(row.get("input_sha256", ""))):
                errors.append("provenance input hash is invalid")
            identity = (
                str(row["record_id"]), str(row["source_id"]),
                str(row["source_record_id"]), str(row["input_sha256"]),
            )
            if identity in provenance_unique:
                errors.append("duplicate provenance identity")
            provenance_unique.add(identity)
            source_identity_records[
                (str(row["source_id"]), str(row["source_revision"]), str(row["source_record_id"]))
            ].add(str(row["record_id"]))
            provenance_keys[str(row["record_id"])].add(str(row["canonical_key"]))
        if any(len(values) > 1 for values in source_identity_records.values()):
            errors.append("one provenance source identity maps to multiple accepted records")
        for row in self.records:
            keys = provenance_keys.get(str(row.get("record_id")), set())
            if not keys:
                continue
            expected_cluster = f"cluster_{_hash(sorted(keys))}"
            if row.get("collision_cluster_id") != expected_cluster or row.get("canonical_key") != min(keys):
                errors.append("record collision cluster identity mismatch")
                continue
            expected_record = f"qa_{_hash([expected_cluster, row.get('question'), row.get('correct_answer')])}"
            if row.get("record_id") != expected_record:
                errors.append("record identity mismatch")
        actual_source_counts: dict[str, int] = defaultdict(int)
        for row in self.provenance:
            actual_source_counts[str(row.get("source_id"))] += 1
        if self.accessible_splits == SPLITS and dict(
            sorted(actual_source_counts.items())
        ) != self.manifest.source_counts:
            errors.append("source counts disagree with manifest")
        if self.accessible_splits == SPLITS and (
            len(self.provenance) + len(self.quarantine)
            != self.manifest.source_candidate_count
        ):
            errors.append("source candidate accounting mismatch")
        allowed_quarantine = {
            "format", "source_id", "source_revision", "source_record_id",
            "canonical_key", "question", "reason", "input_sha256",
        }
        for row in self.quarantine:
            if set(row) != allowed_quarantine or row.get("format") != "truth_editing_canonical_qa_quarantine_v2":
                errors.append("quarantine fields do not match the v2 schema")
            if row.get("reason") not in {"conflicting_truth", "ambiguous_near_duplicate"}:
                errors.append("quarantine reason is invalid")
            if not _SHA.fullmatch(str(row.get("input_sha256", ""))):
                errors.append("quarantine input hash is invalid")
        try:
            policy = json.loads((self.path / "policy.json").read_text())
            receipts = json.loads((self.path / "source_receipts.json").read_text())
        except (OSError, json.JSONDecodeError):
            errors.append("policy or source receipts cannot be parsed")
        else:
            if _hash(policy) != self.manifest.policy_sha256:
                errors.append("policy identity mismatch")
            if not isinstance(receipts, Mapping) or receipts.get("format") != "truth_editing_source_receipts_v2" or not isinstance(receipts.get("sources"), list):
                errors.append("source receipts do not match the v2 schema")
            else:
                admission = receipts.get("admission_audit")
                if not isinstance(admission, Mapping):
                    errors.append("source receipts lack an admission audit")
                else:
                    admission_unsigned = dict(admission)
                    admission_claim = admission_unsigned.pop("self_sha256", None)
                    required_source_fields = {
                        "adapter", "adapter_contract", "file_sha256",
                        "admitted_candidate_count", "rejected_invalid_rows",
                        "truth_derivation_check_count",
                    }
                    source_contracts_valid = all(
                        isinstance(source, Mapping)
                        and required_source_fields <= set(source)
                        and isinstance(source.get("file_sha256"), Mapping)
                        and bool(source["file_sha256"])
                        and all(
                            isinstance(value, str) and _SHA.fullmatch(value)
                            for value in source["file_sha256"].values()
                        )
                        and isinstance(source.get("admitted_candidate_count"), int)
                        and source["admitted_candidate_count"] >= 0
                        and isinstance(source.get("truth_derivation_check_count"), int)
                        and not isinstance(source.get("truth_derivation_check_count"), bool)
                        and source["truth_derivation_check_count"]
                        >= source["admitted_candidate_count"]
                        and isinstance(source.get("adapter_contract"), str)
                        and bool(source["adapter_contract"])
                        for source in receipts["sources"]
                    )
                    admitted_counts: list[int] = []
                    for source in receipts["sources"]:
                        if not isinstance(source, Mapping):
                            continue
                        candidate_count = source.get("admitted_candidate_count")
                        if isinstance(candidate_count, int) and not isinstance(
                            candidate_count, bool
                        ):
                            admitted_counts.append(candidate_count)
                    admitted_sum = (
                        sum(admitted_counts)
                        if len(admitted_counts) == len(receipts["sources"])
                        else None
                    )
                    if (
                        admission.get("format") != "truth_editing_source_admission_audit_v1"
                        or admission.get("status") != "valid"
                        or admission_claim != _hash(admission_unsigned)
                        or admission.get("sources_sha256") != _hash(receipts["sources"])
                        or admission.get("admitted_candidate_count")
                        != admitted_sum
                        or not source_contracts_valid
                    ):
                        errors.append("source admission audit identity is invalid")
        try:
            direction_receipt = json.loads(
                (self.path / "direction_construction_allowlist.json").read_text()
            )
        except (OSError, json.JSONDecodeError):
            errors.append("direction construction receipt cannot be parsed")
        else:
            if not isinstance(direction_receipt, Mapping):
                errors.append("direction construction receipt must be an object")
            else:
                receipt_copy = dict(direction_receipt)
                claimed_self_hash = receipt_copy.pop("self_sha256", None)
                if (
                    direction_receipt.get("format")
                    != "truth_editing_direction_construction_allowlist_v1"
                    or direction_receipt.get("status") not in {"eligible", "blocked"}
                    or claimed_self_hash != _hash(receipt_copy)
                ):
                    errors.append("direction construction receipt identity is invalid")
                if direction_receipt.get("status") == "eligible":
                    if direction_receipt.get("blocked_reasons") != []:
                        errors.append("eligible direction receipt contains blocked reasons")
                    if not isinstance(direction_receipt.get("ordered_row_ids"), list):
                        errors.append("eligible direction receipt lacks ordered rows")
                elif direction_receipt.get("ordered_row_ids") is not None:
                    errors.append("blocked direction receipt exposes allowed rows")
        return V2Audit(not errors, tuple(errors), counts, len(self.records), len(self.quarantine))


def materialize_optimization_dataset_view(
    source: Path | str,
    output: Path | str,
) -> TruthEditingDatasetV2:
    """Materialize the authenticated train/validation-only production view."""

    source = Path(source)
    output = Path(output)
    dataset = TruthEditingDatasetV2.open(source)
    if output.exists():
        raise DatasetV2Error("optimization dataset output already exists")
    output.mkdir(parents=True)

    accessible_ids = {
        str(row["record_id"])
        for row in dataset.records
        if row["split"] in {"train", "validation"}
    }
    provenance = [
        row
        for row in dataset.provenance
        if str(row["record_id"]) in accessible_ids
    ]
    for name in _OPTIMIZATION_FILES - {"provenance.jsonl"}:
        source_path = source / name
        (output / name).write_bytes(source_path.read_bytes())
    _write_jsonl(output / "provenance.jsonl", provenance)

    file_sha256 = {
        name: _file_hash(output / name) for name in sorted(_OPTIMIZATION_FILES)
    }
    unsigned = {
        "format": OPTIMIZATION_MANIFEST_FORMAT,
        "source_dataset_id": dataset.manifest.dataset_id,
        "source_manifest_sha256": _file_hash(output / "manifest.json"),
        "sealed_splits": ["test"],
        "split_counts": {
            split: dataset.manifest.split_counts[split]
            for split in ("train", "validation")
        },
        "record_count": sum(
            dataset.manifest.split_counts[split]
            for split in ("train", "validation")
        ),
        "provenance_count": len(provenance),
        "file_sha256": file_sha256,
    }
    (output / "optimization-manifest.json").write_text(
        _canonical_json({**unsigned, "self_sha256": _hash(unsigned)}) + "\n"
    )
    return TruthEditingDatasetV2.open_for_optimization(output)


def _direction_receipt_payload(
    dataset: TruthEditingDatasetV2,
    *,
    hdf5_identity: Mapping[str, Any],
    hdf5_examples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(hdf5_identity, Mapping):
        raise DatasetV2Error("hdf5_identity must be an object")
    direct_sha256 = str(hdf5_identity.get("direct_sha256", ""))
    if not _SHA.fullmatch(direct_sha256):
        raise DatasetV2Error("hdf5 direct_sha256 must be a lowercase SHA-256")
    path = _text(hdf5_identity.get("path"), "hdf5 path")
    normalized_identity = {str(key): value for key, value in sorted(hdf5_identity.items())}
    normalized_identity["path"] = path
    normalized_identity["direct_sha256"] = direct_sha256

    records_by_id = {str(row["record_id"]): row for row in dataset.records}
    question_records: dict[str, set[str]] = defaultdict(set)
    for row in dataset.records:
        question_records[_normalize(str(row["question"]))].add(str(row["record_id"]))
    provenance_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_records: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in dataset.provenance:
        record_id = str(row["record_id"])
        provenance_by_record[record_id].append({
            "source_id": str(row["source_id"]),
            "source_revision": str(row["source_revision"]),
            "source_record_id": str(row["source_record_id"]),
        })
        source_record_id = str(row["source_record_id"])
        match = re.fullmatch(r"(.+):([0-9]+)", source_record_id)
        if match:
            # CSV adapters use the physical file line (header is line one),
            # while historical activation metadata stores zero-based data rows.
            source_records[(Path(match.group(1)).name, int(match.group(2)) - 2)].add(record_id)

    examples: list[dict[str, Any]] = []
    duplicates: list[str] = []
    seen_example_ids: set[str] = set()
    required = {
        "task_id", "example_index", "source_file", "source_row_index",
        "canonical_question", "label", "token_row_start", "token_row_end",
    }
    for raw in hdf5_examples:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise DatasetV2Error("HDF5 example identity fields do not match the v1 schema")
        task_id = _text(raw["task_id"], "HDF5 task_id")
        example_index = _integer(raw["example_index"], "HDF5 example_index")
        row_start = _integer(raw["token_row_start"], "HDF5 token_row_start")
        row_end = _integer(raw["token_row_end"], "HDF5 token_row_end")
        if row_end < row_start:
            raise DatasetV2Error("HDF5 token row range is inverted")
        label = raw["label"]
        if not isinstance(label, int) or isinstance(label, bool) or label not in {-1, 0, 1}:
            raise DatasetV2Error("HDF5 example label must be -1, 0, or 1")
        source_file = raw["source_file"]
        if source_file is not None:
            source_file = Path(_text(source_file, "HDF5 source_file")).name
        source_row_index = raw["source_row_index"]
        if source_row_index is not None:
            source_row_index = _integer(source_row_index, "HDF5 source_row_index")
        canonical_question = raw["canonical_question"]
        if canonical_question is not None:
            canonical_question = _text(canonical_question, "HDF5 canonical_question")
        example_id = f"{task_id}:example:{example_index}"
        if example_id in seen_example_ids:
            duplicates.append(example_id)
        seen_example_ids.add(example_id)
        examples.append({
            "example_id": example_id,
            "task_id": task_id,
            "example_index": example_index,
            "source_file": source_file,
            "source_row_index": source_row_index,
            "canonical_question": canonical_question,
            "label": label,
            "token_row_start": row_start,
            "token_row_end": row_end,
        })
    examples.sort(key=lambda item: (item["task_id"], item["example_index"]))

    mapped_train: list[dict[str, Any]] = []
    mapped_record_ids: set[str] = set()
    cross_split: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    excluded_counts = {"validation": 0, "test": 0}
    excluded_skip_label_count = 0
    mapping_basis_counts: dict[str, int] = defaultdict(int)
    mapping_channel_conflicts: list[dict[str, Any]] = []
    for example in examples:
        source_candidate_ids: set[str] = set()
        if example["source_file"] is not None and example["source_row_index"] is not None:
            source_candidate_ids.update(
                source_records.get((example["source_file"], example["source_row_index"]), set())
            )
        question_candidate_ids: set[str] = set()
        if example["canonical_question"] is not None:
            question_candidate_ids.update(
                question_records.get(_normalize(example["canonical_question"]), set())
            )
        if (
            source_candidate_ids
            and question_candidate_ids
            and source_candidate_ids != question_candidate_ids
        ):
            mapping_channel_conflicts.append({
                "example_id": example["example_id"],
                "source_record_ids": sorted(source_candidate_ids),
                "question_record_ids": sorted(question_candidate_ids),
            })
            continue
        candidate_ids = source_candidate_ids | question_candidate_ids
        if source_candidate_ids and question_candidate_ids:
            mapping_basis = "source_and_question_agree"
        elif source_candidate_ids:
            mapping_basis = "source_only"
        elif question_candidate_ids:
            mapping_basis = "question_only"
        else:
            mapping_basis = "unmapped"
        mapping_basis_counts[mapping_basis] += 1
        if len(candidate_ids) > 1:
            splits = sorted({str(records_by_id[record_id]["split"]) for record_id in candidate_ids})
            entry = {
                "example_id": example["example_id"],
                "record_ids": sorted(candidate_ids),
                "splits": splits,
            }
            ambiguous.append(entry)
            if len(splits) > 1:
                cross_split.append(entry)
            continue
        if not candidate_ids:
            continue
        if example["label"] == -1:
            excluded_skip_label_count += 1
            continue
        record_id = next(iter(candidate_ids))
        split = str(records_by_id[record_id]["split"])
        if split != "train":
            excluded_counts[split] += 1
            continue
        mapped_record_ids.add(record_id)
        mapped_train.append({
            "example_id": example["example_id"],
            "task_id": example["task_id"],
            "example_index": example["example_index"],
            "record_id": record_id,
            "canonical_key": records_by_id[record_id]["canonical_key"],
            "mapping_basis": mapping_basis,
            "label": example["label"],
            "token_row_start": example["token_row_start"],
            "token_row_end": example["token_row_end"],
        })

    train_records = sorted(
        (row for row in dataset.records if row["split"] == "train"),
        key=lambda row: str(row["record_id"]),
    )
    missing = [
        {
            "record_id": row["record_id"],
            "canonical_key": row["canonical_key"],
            "source_identities": sorted(
                provenance_by_record[str(row["record_id"])],
                key=lambda value: (
                    value["source_id"], value["source_revision"], value["source_record_id"]
                ),
            ),
        }
        for row in train_records
        if str(row["record_id"]) not in mapped_record_ids
    ]
    blocked_reasons: list[str] = []
    if duplicates:
        blocked_reasons.append("duplicate_hdf5_example_identities")
    # This receipt audits the exact overlap between the independently-built v2
    # corpus and the historical activation rows.  It is not a demand that every
    # v2 training record existed in that historical extraction: unmapped and
    # ambiguous matches are recorded as exclusions, never guessed into the
    # construction selector.

    per_domain: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "example_count": 0,
            "honest_example_count": 0,
            "deceptive_example_count": 0,
            "token_row_count": 0,
        }
    )
    ordered_row_ids: list[str] = []
    for mapping in mapped_train:
        domain = per_domain[mapping["task_id"]]
        domain["example_count"] += 1
        domain[
            "deceptive_example_count" if mapping["label"] == 1 else "honest_example_count"
        ] += 1
        row_count = mapping["token_row_end"] - mapping["token_row_start"]
        domain["token_row_count"] += row_count
        ordered_row_ids.extend(
            f"{mapping['task_id']}:row:{row_index}"
            for row_index in range(mapping["token_row_start"], mapping["token_row_end"])
        )
    eligible = not blocked_reasons
    example_inventory = [
        [
            item["example_id"], item["label"],
            item["token_row_start"], item["token_row_end"],
        ]
        for item in examples
    ]
    payload: dict[str, Any] = {
        "format": "truth_editing_direction_construction_allowlist_v1",
        "status": "eligible" if eligible else "blocked",
        "dataset_binding": {
            "dataset_id": dataset.manifest.dataset_id,
            "train_sha256": dataset.manifest.file_sha256["train.jsonl"],
            "validation_sha256": dataset.manifest.file_sha256["validation.jsonl"],
            "test_sha256": dataset.manifest.file_sha256["test.jsonl"],
            "quarantine_sha256": dataset.manifest.file_sha256["quarantine.jsonl"],
            "provenance_sha256": dataset.manifest.file_sha256["provenance.jsonl"],
        },
        "hdf5_identity": normalized_identity,
        "hdf5_example_count": len(examples),
        "hdf5_ordered_example_ranges_sha256": _hash(example_inventory),
        "allowed_examples": mapped_train if eligible else None,
        "ordered_row_ids": ordered_row_ids if eligible else None,
        "ordered_row_ids_sha256": _hash(ordered_row_ids) if eligible else None,
        "candidate_ordered_row_count": len(ordered_row_ids),
        "candidate_ordered_row_ids_sha256": _hash(ordered_row_ids),
        "per_domain_counts": dict(sorted(per_domain.items())),
        "excluded_partitions": [
            "validation", "test", "quarantine", "judge_calibration", "final_audit",
        ],
        "excluded_mapped_example_counts": excluded_counts,
        "excluded_skip_label_count": excluded_skip_label_count,
        "excluded_unmapped_train_record_count": len(missing),
        "excluded_ambiguous_hdf5_example_count": len(ambiguous),
        "mapping_basis_counts": dict(sorted(mapping_basis_counts.items())),
        "mapping_channel_conflicts": mapping_channel_conflicts,
        "blocked_reasons": blocked_reasons,
        "missing_mappings": missing,
        "duplicate_hdf5_example_ids": sorted(set(duplicates)),
        "ambiguous_hdf5_mappings": ambiguous,
        "cross_split_hdf5_mappings": cross_split,
    }
    payload["self_sha256"] = _hash(payload)
    return payload


def install_direction_construction_receipt(
    dataset: TruthEditingDatasetV2,
    *,
    hdf5_identity: Mapping[str, Any],
    hdf5_examples: Iterable[Mapping[str, Any]],
) -> TruthEditingDatasetV2:
    """Install a mapping receipt and rebind the dataset manifest to it."""

    receipt = _direction_receipt_payload(
        dataset, hdf5_identity=hdf5_identity, hdf5_examples=hdf5_examples
    )
    target = dataset.path / "direction_construction_allowlist.json"
    target.write_text(_canonical_json(receipt) + "\n")
    manifest_payload = dataset.manifest.to_payload()
    manifest_payload["file_sha256"][target.name] = _file_hash(target)
    (dataset.path / "manifest.json").write_text(_canonical_json(manifest_payload) + "\n")
    return TruthEditingDatasetV2.open(dataset.path)


def load_hdf5_example_identities(path: Path | str) -> list[dict[str, Any]]:
    """Read only example identity metadata and token-row bounds from HDF5."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependency
        raise DatasetV2Error("h5py is required to inspect activation metadata") from error
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise DatasetV2Error("activation HDF5 path is not a regular file")
    result: list[dict[str, Any]] = []
    try:
        with h5py.File(path, "r") as hdf5:
            if "metadata" not in hdf5:
                raise DatasetV2Error("activation HDF5 lacks metadata identities")
            metadata_root = hdf5["metadata"]
            for task_id in sorted(metadata_root.keys()):
                group = metadata_root[task_id]
                required = {
                    "example_metadata_json", "example_splits", "source_datasets",
                    "example_source_indices", "example_labels",
                }
                if not required <= set(group.keys()):
                    raise DatasetV2Error(f"activation task lacks identity arrays: {task_id}")
                example_count = len(group["example_metadata_json"])
                splits = group["example_splits"]
                if len(splits) != example_count + 1:
                    raise DatasetV2Error(f"activation task has invalid example splits: {task_id}")
                if len(group["source_datasets"]) != example_count:
                    raise DatasetV2Error(f"activation task has invalid source identities: {task_id}")
                for example_index in range(example_count):
                    raw_metadata = group["example_metadata_json"][example_index]
                    if isinstance(raw_metadata, bytes):
                        raw_metadata = raw_metadata.decode("utf-8")
                    try:
                        metadata = json.loads(str(raw_metadata))
                    except json.JSONDecodeError as error:
                        raise DatasetV2Error(
                            f"activation example metadata is invalid: {task_id}:{example_index}"
                        ) from error
                    if not isinstance(metadata, Mapping):
                        raise DatasetV2Error("activation example metadata must be an object")
                    raw_source = group["source_datasets"][example_index]
                    if isinstance(raw_source, bytes):
                        raw_source = raw_source.decode("utf-8")
                    source_file = metadata.get("csv") or Path(str(raw_source)).name
                    source_row = metadata.get("row")
                    if source_row is not None:
                        source_row = int(source_row)
                    canonical_question = metadata.get("question")
                    result.append({
                        "task_id": str(task_id),
                        "example_index": example_index,
                        "source_file": str(source_file) if source_file else None,
                        "source_row_index": source_row,
                        "canonical_question": (
                            str(canonical_question) if canonical_question is not None else None
                        ),
                        "label": int(group["example_labels"][example_index]),
                        "token_row_start": int(splits[example_index]),
                        "token_row_end": int(splits[example_index + 1]),
                    })
    except OSError as error:
        raise DatasetV2Error("cannot inspect activation HDF5 metadata") from error
    return result


def build_dataset_v2(
    candidates: Iterable[V2Candidate],
    output_dir: Path | str,
    *,
    seed: int = 20260827,
    dataset_id: str = "truth_editing_canonical_qa_v2",
    overwrite: bool = False,
    source_receipts: Iterable[Mapping[str, Any]] = (),
) -> TruthEditingDatasetV2:
    """Build one canonical record per conservative collision cluster."""

    rows = list(candidates)
    if not rows:
        raise DatasetV2Error("at least one candidate is required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DatasetV2Error("seed must be an integer")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise DatasetV2Error("output directory is not empty")
        allowed_existing = _BUNDLE_FILES | {"manifest.json"}
        if {item.name for item in output.iterdir()} - allowed_existing:
            raise DatasetV2Error("output directory contains unexpected files")
    output.mkdir(parents=True, exist_ok=True)

    keyed: dict[str, list[V2Candidate]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, V2Candidate):
            raise DatasetV2Error("candidates must contain V2Candidate values")
        keyed[row.canonical_key].append(row)
    quarantined_keys: dict[str, str] = {}
    for key, group in keyed.items():
        if len({_normalize(row.correct_answer) for row in group}) != 1:
            quarantined_keys[key] = "conflicting_truth"

    active_keys = sorted(set(keyed) - set(quarantined_keys))
    union = _UnionFind(active_keys)
    exact: dict[str, str] = {}
    for key in active_keys:
        for row in keyed[key]:
            for wording in (row.question, *row.aliases):
                norm = _normalize(wording)
                if norm in exact:
                    other = exact[norm]
                    if _normalize(keyed[other][0].correct_answer) == _normalize(row.correct_answer):
                        union.union(key, other)
                    else:
                        quarantined_keys[key] = quarantined_keys[other] = "conflicting_truth"
                else:
                    exact[norm] = key

    # Generate near-match candidates through rare content tokens rather than
    # an unscalable all-pairs scan. >=.94 is a strong rendering collision;
    # [.82,.94) is deliberately discarded as ambiguous.
    token_sets = {
        key: _blocking_tokens(keyed[key][0].question)
        for key in active_keys
        if key not in quarantined_keys and keyed[key][0].near_match_policy == "conservative"
    }
    document_frequency: dict[str, int] = defaultdict(int)
    for tokens in token_sets.values():
        for token in tokens:
            document_frequency[token] += 1
    token_index: dict[str, list[str]] = defaultdict(list)
    for key, tokens in token_sets.items():
        selected = sorted(tokens, key=lambda token: (document_frequency[token], token))[:3]
        for token in selected:
            if document_frequency[token] <= 200:
                token_index[token].append(key)
    candidate_pairs: set[tuple[str, str]] = set()
    for bucket in token_index.values():
        for index, left in enumerate(bucket):
            for right in bucket[index + 1:]:
                candidate_pairs.add((left, right) if left < right else (right, left))
    for left, right in sorted(candidate_pairs):
        score = _similarity(keyed[left][0].question, keyed[right][0].question)
        answers_agree = _normalize(keyed[left][0].correct_answer) == _normalize(keyed[right][0].correct_answer)
        if score >= 0.94 and answers_agree:
            union.union(left, right)
        elif score >= 0.82:
            reason = "ambiguous_near_duplicate" if answers_agree else "conflicting_truth"
            quarantined_keys.setdefault(left, reason)
            quarantined_keys.setdefault(right, reason)

    # Ambiguity or contradiction invalidates the entire already-unioned
    # collision component, not just the directly matched endpoint.
    quarantined_roots = {
        union.find(key): reason for key, reason in quarantined_keys.items() if key in union.parent
    }
    for key in active_keys:
        root = union.find(key)
        if root in quarantined_roots:
            quarantined_keys.setdefault(key, quarantined_roots[root])

    clusters: dict[str, list[str]] = defaultdict(list)
    for key in active_keys:
        if key not in quarantined_keys:
            clusters[union.find(key)].append(key)

    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    for root, keys in sorted(clusters.items()):
        members = [row for key in keys for row in keyed[key]]
        representative = min(members, key=lambda row: (len(row.question), row.question, row.source_id))
        cluster_id = f"cluster_{_hash(sorted(keys))}"
        split = _split(cluster_id, seed)
        record_id = f"qa_{_hash([cluster_id, representative.question, representative.correct_answer])}"
        wrong_answers = [choice for choice in representative.choices if _normalize(choice) != _normalize(representative.correct_answer)]
        record = {
            "format": RECORD_FORMAT,
            "record_id": record_id,
            "canonical_key": min(keys),
            "collision_cluster_id": cluster_id,
            "question": representative.question,
            "correct_answer": representative.correct_answer,
            "wrong_answers": wrong_answers,
            "choices": list(representative.choices),
            "family": representative.family,
            "truth_authority": representative.truth_authority,
            "split": split,
        }
        records.append(record)
        for member in members:
            source_counts[member.source_id] += 1
            provenance.append({
                "format": "truth_editing_canonical_qa_provenance_v2",
                "record_id": record_id,
                "canonical_key": member.canonical_key,
                "source_id": member.source_id,
                "source_revision": member.source_revision,
                "source_record_id": member.source_record_id,
                "input_sha256": _hash(member.identity_payload),
            })

    quarantine: list[dict[str, Any]] = []
    for key, reason in sorted(quarantined_keys.items()):
        for row in keyed[key]:
            quarantine.append({
                "format": "truth_editing_canonical_qa_quarantine_v2",
                "source_id": row.source_id,
                "source_revision": row.source_revision,
                "source_record_id": row.source_record_id,
                "canonical_key": row.canonical_key,
                "question": row.question,
                "reason": reason,
                "input_sha256": _hash(row.identity_payload),
            })

    for split in SPLITS:
        _write_jsonl(output / f"{split}.jsonl", (row for row in records if row["split"] == split))
    _write_jsonl(output / "provenance.jsonl", provenance)
    _write_jsonl(output / "quarantine.jsonl", quarantine)
    policy = {
        "format": FORMAT,
        "seed": seed,
        "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "strong_near_threshold": 0.94,
        "ambiguous_near_threshold": 0.82,
        "ambiguous_action": "quarantine",
    }
    (output / "policy.json").write_text(_canonical_json(policy) + "\n")
    source_receipt_values = [dict(receipt) for receipt in source_receipts]
    if not source_receipt_values:
        inline_sha = _hash([row.identity_payload for row in rows])
        source_receipt_values = [{
            "source_id": "inline-truth-authoritative-candidates",
            "adapter": "inline_candidates",
            "adapter_contract": "caller_supplied_truth_authoritative_candidates_v1",
            "paths": [],
            "sha256": inline_sha,
            "file_sha256": {"inline-candidates-canonical-json": inline_sha},
            "rejected_invalid_rows": 0,
            "mmlu_index_text_agreement_checks": 0,
            "admitted_candidate_count": len(rows),
            "truth_derivation_check_count": len(rows),
        }]
    admission_unsigned = {
        "format": "truth_editing_source_admission_audit_v1",
        "status": "valid",
        "sources_sha256": _hash(source_receipt_values),
        "admitted_candidate_count": len(rows),
        "required_adapter_contract_fields": [
            "adapter", "adapter_contract", "file_sha256",
            "admitted_candidate_count", "rejected_invalid_rows",
            "truth_derivation_check_count",
        ],
    }
    admission_audit = dict(admission_unsigned)
    admission_audit["self_sha256"] = _hash(admission_unsigned)
    receipts_payload = {
        "format": "truth_editing_source_receipts_v2",
        "sources": source_receipt_values,
        "admission_audit": admission_audit,
    }
    (output / "source_receipts.json").write_text(_canonical_json(receipts_payload) + "\n")
    preliminary_names = [f"{split}.jsonl" for split in SPLITS] + [
        "provenance.jsonl", "quarantine.jsonl", "policy.json", "source_receipts.json",
    ]
    split_counts = {split: sum(row["split"] == split for row in records) for split in SPLITS}
    preliminary_manifest = V2Manifest(
        dataset_id=dataset_id,
        seed=seed,
        source_candidate_count=len(rows),
        accepted_canonical_count=len(records),
        quarantined_candidate_count=len(quarantine),
        split_counts=split_counts,
        source_counts=dict(sorted(source_counts.items())),
        file_sha256={name: _file_hash(output / name) for name in preliminary_names},
        policy_sha256=_hash(policy),
    )
    preliminary_dataset = TruthEditingDatasetV2(
        output, preliminary_manifest, records, provenance, quarantine
    )
    default_direction_receipt = _direction_receipt_payload(
        preliminary_dataset,
        hdf5_identity={"path": "not-supplied", "direct_sha256": "0" * 64},
        hdf5_examples=[],
    )
    (output / "direction_construction_allowlist.json").write_text(
        _canonical_json(default_direction_receipt) + "\n"
    )
    names = preliminary_names + ["direction_construction_allowlist.json"]
    manifest = V2Manifest(
        dataset_id=dataset_id,
        seed=seed,
        source_candidate_count=len(rows),
        accepted_canonical_count=len(records),
        quarantined_candidate_count=len(quarantine),
        split_counts=split_counts,
        source_counts=dict(sorted(source_counts.items())),
        file_sha256={name: _file_hash(output / name) for name in names},
        policy_sha256=_hash(policy),
    )
    (output / "manifest.json").write_text(_canonical_json(manifest.to_payload()) + "\n")
    return TruthEditingDatasetV2.open(output)


__all__ = [
    "DatasetV2Error", "TruthEditingDatasetV2", "V2Audit", "V2Candidate",
    "V2Manifest", "build_dataset_v2", "install_direction_construction_receipt",
    "load_candidates_from_config", "load_hdf5_example_identities",
]
