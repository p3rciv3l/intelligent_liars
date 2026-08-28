"""Strict, frozen value objects for the truth-editing dataset compiler.

The source adapters intentionally accept a small set of common aliases, but
everything crossing the compiler boundary is represented by the records in
this module.  The JSON representation is canonical and is used for all
content identities and manifest hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DATASET_FORMAT = "truth_editing_dataset_v1"
RECORD_FORMAT = "truth_editing_optimizer_record_v1"
PROVENANCE_FORMAT = "truth_editing_provenance_record_v1"
AUDIT_FORMAT = "truth_editing_dataset_audit_v1"
SPLITS = ("train", "validation", "test", "quarantine")
NON_QUARANTINE_SPLITS = ("train", "validation", "test")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DatasetCompileError(ValueError):
    """Raised when source data cannot be compiled without guessing."""


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise DatasetCompileError(f"{name} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise DatasetCompileError(f"{name} must not be empty")
    return value


def normalize_text(value: str) -> str:
    """Normalize semantic text for identities, without changing stored text."""

    value = unicodedata.normalize("NFKC", value).casefold()
    # Treat punctuation and answer-choice decoration as separators.  Keeping
    # unicode alphanumerics makes this work for multilingual source rows.
    chars = [character if character.isalnum() else " " for character in value]
    return " ".join("".join(chars).split())


def canonical_sha256(value: Any) -> str:
    """Hash JSON using one canonical, finite, UTF-8 representation."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DatasetCompileError("value is not canonically JSON serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def _digest(prefix: str, value: Any) -> str:
    return f"{prefix}_{canonical_sha256(value)}"


def _sha(value: Any, name: str, *, allow_empty: bool = False) -> str:
    value = _text(value, name, allow_empty=allow_empty)
    if value and not _SHA256.fullmatch(value):
        raise DatasetCompileError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _tuple_strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetCompileError(f"{name} must be a list of strings")
    result = tuple(_text(item, f"{name} item") for item in value)
    if len(set(normalize_text(item) for item in result)) != len(result):
        raise DatasetCompileError(f"{name} contains duplicate values")
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetCompileError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class DatasetSource:
    """Pinned description of one local/in-memory source reader.

    ``revision`` is required even for local fixtures.  The compiler does not
    interpret URLs or perform network access; ``kind`` and ``selector`` are
    provenance only, while ``reader_key`` chooses a caller-supplied reader.
    """

    source_id: str
    revision: str
    kind: str = "local"
    selector: str = "default"
    reader_key: str | None = None
    source_sha256: str | None = None
    license_id: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "revision", _text(self.revision, "revision"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(self, "selector", _text(self.selector, "selector"))
        if self.reader_key is not None:
            object.__setattr__(self, "reader_key", _text(self.reader_key, "reader_key"))
        if self.source_sha256 is not None:
            object.__setattr__(self, "source_sha256", _sha(self.source_sha256, "source_sha256"))
        object.__setattr__(self, "license_id", _text(self.license_id, "license_id"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetSource:
        allowed = {
            "source_id", "revision", "kind", "selector", "reader_key",
            "source_sha256", "license_id",
        }
        unknown = set(value) - allowed
        if unknown:
            raise DatasetCompileError(f"unknown source fields: {sorted(unknown)}")
        return cls(**dict(value))

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "revision": self.revision,
            "kind": self.kind,
            "selector": self.selector,
            "reader_key": self.reader_key,
            "source_sha256": self.source_sha256,
            "license_id": self.license_id,
        }


@dataclass(frozen=True)
class DatasetRequest:
    """Immutable compilation policy.

    Fractions control deterministic group assignment.  They need not produce
    exact row counts because whole leakage groups are assigned together.
    """

    dataset_id: str = "truth_editing_dataset"
    source_specs: tuple[DatasetSource, ...] = ()
    seed: int = 0
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    near_max_hamming: int = 6
    near_minimum_length_ratio: float = 0.75
    split_salt: str = "truth-editing-dataset-v1"
    honor_input_splits: bool = False
    output_dir: Path | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id"))
        raw_specs = tuple(self.source_specs)
        try:
            specs = tuple(
                spec if isinstance(spec, DatasetSource) else DatasetSource.from_mapping(spec)
                for spec in raw_specs
            )
        except (TypeError, ValueError) as error:
            raise DatasetCompileError("source_specs must contain DatasetSource values") from error
        if len({spec.source_id for spec in specs}) != len(specs):
            raise DatasetCompileError("source_specs contains duplicate source_id values")
        object.__setattr__(self, "source_specs", specs)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise DatasetCompileError("seed must be an integer")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in fractions):
            raise DatasetCompileError("split fractions must be finite numbers")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in fractions):
            raise DatasetCompileError("split fractions must be finite and non-negative")
        if not math.isclose(sum(float(value) for value in fractions), 1.0, abs_tol=1e-9):
            raise DatasetCompileError("split fractions must sum to 1")
        if not isinstance(self.near_max_hamming, int) or not 0 <= self.near_max_hamming <= 64:
            raise DatasetCompileError("near_max_hamming must be an integer from 0 to 64")
        if not math.isfinite(float(self.near_minimum_length_ratio)) or not 0 < self.near_minimum_length_ratio <= 1:
            raise DatasetCompileError("near_minimum_length_ratio must be in (0, 1]")
        object.__setattr__(self, "split_salt", _text(self.split_salt, "split_salt"))
        if not isinstance(self.honor_input_splits, bool):
            raise DatasetCompileError("honor_input_splits must be a boolean")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetRequest:
        allowed = {
            "dataset_id", "source_specs", "seed", "train_fraction",
            "validation_fraction", "test_fraction", "near_max_hamming",
            "near_minimum_length_ratio", "split_salt", "output_dir", "overwrite",
            "honor_input_splits",
        }
        unknown = set(value) - allowed
        if unknown:
            raise DatasetCompileError(f"unknown request fields: {sorted(unknown)}")
        data = dict(value)
        data["source_specs"] = tuple(
            item if isinstance(item, DatasetSource) else DatasetSource.from_mapping(item)
            for item in data.get("source_specs", ())
        )
        return cls(**data)

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_specs": [spec.to_payload() for spec in self.source_specs],
            "seed": self.seed,
            "train_fraction": float(self.train_fraction),
            "validation_fraction": float(self.validation_fraction),
            "test_fraction": float(self.test_fraction),
            "near_max_hamming": self.near_max_hamming,
            "near_minimum_length_ratio": float(self.near_minimum_length_ratio),
            "split_salt": self.split_salt,
            "honor_input_splits": self.honor_input_splits,
        }


@dataclass(frozen=True)
class TruthEditingRecord:
    """One canonical optimizer-facing question.

    ``wrong_answers`` is intentionally a tuple of alternatives, not one
    selected false answer.  The edited model may choose any semantically wrong
    option; the compiler never requires a stable false answer across variants.
    """

    record_id: str
    question: str
    correct_answer: str
    wrong_answers: tuple[str, ...] = ()
    condition: str = "unspecified"
    family: str = "unspecified"
    proposition: str | None = None
    aliases: tuple[str, ...] = ()
    canonical_question_id: str | None = None
    canonical_proposition_id: str | None = None
    leakage_group_id: str | None = None
    split: Literal["train", "validation", "test", "quarantine"] = "train"

    def __post_init__(self) -> None:
        for field_name in ("record_id", "question", "correct_answer", "condition", "family"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        object.__setattr__(self, "wrong_answers", _tuple_strings(self.wrong_answers, "wrong_answers"))
        object.__setattr__(self, "aliases", _tuple_strings(self.aliases, "aliases"))
        if normalize_text(self.correct_answer) in {normalize_text(item) for item in self.wrong_answers}:
            raise DatasetCompileError("correct_answer must not also be a wrong_answer")
        if self.proposition is not None:
            object.__setattr__(self, "proposition", _text(self.proposition, "proposition"))
        for field_name in ("canonical_question_id", "canonical_proposition_id", "leakage_group_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _text(value, field_name))
        if self.split not in SPLITS:
            raise DatasetCompileError(f"unsupported split: {self.split}")

    @property
    def optimizer_payload(self) -> dict[str, Any]:
        """Flat, source-independent row consumed by optimization code."""

        return {
            "format": RECORD_FORMAT,
            "record_id": self.record_id,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "wrong_answers": list(self.wrong_answers),
            "condition": self.condition,
            "family": self.family,
            "proposition": self.proposition,
            "aliases": list(self.aliases),
            "canonical_question_id": self.canonical_question_id,
            "canonical_proposition_id": self.canonical_proposition_id,
            "leakage_group_id": self.leakage_group_id,
            "split": self.split,
        }

    @classmethod
    def from_optimizer_payload(cls, value: Mapping[str, Any]) -> TruthEditingRecord:
        allowed = {
            "format", "record_id", "question", "correct_answer", "wrong_answers",
            "condition", "family", "proposition", "aliases", "canonical_question_id", "canonical_proposition_id",
            "leakage_group_id", "split",
        }
        if set(value) != allowed:
            raise DatasetCompileError(
                f"record fields must be exactly {sorted(allowed)}; observed {sorted(value)}"
            )
        if value["format"] != RECORD_FORMAT:
            raise DatasetCompileError("unsupported optimizer record format")
        for name in ("wrong_answers", "aliases"):
            payload = value[name]
            if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
                raise DatasetCompileError(f"record.{name} must be an array")
        return cls(
            record_id=value["record_id"],
            question=value["question"],
            correct_answer=value["correct_answer"],
            wrong_answers=tuple(value["wrong_answers"]),
            condition=value["condition"],
            family=value["family"],
            proposition=value["proposition"],
            aliases=tuple(value["aliases"]),
            canonical_question_id=value["canonical_question_id"],
            canonical_proposition_id=value["canonical_proposition_id"],
            leakage_group_id=value["leakage_group_id"],
            split=value["split"],
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    """Source lineage kept out of flat optimizer-facing rows."""

    record_id: str
    source_id: str
    source_revision: str
    source_record_id: str
    source_selector: str = "default"
    source_kind: str = "local"
    source_sha256: str | None = None
    input_content_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("record_id", "source_id", "source_revision", "source_record_id", "source_selector", "source_kind"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        for field_name in ("source_sha256", "input_content_sha256"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _sha(value, field_name))

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": PROVENANCE_FORMAT,
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "source_record_id": self.source_record_id,
            "source_selector": self.source_selector,
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "input_content_sha256": self.input_content_sha256,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ProvenanceRecord:
        allowed = {
            "format", "record_id", "source_id", "source_revision", "source_record_id",
            "source_selector", "source_kind", "source_sha256", "input_content_sha256",
        }
        if set(value) != allowed or value.get("format") != PROVENANCE_FORMAT:
            raise DatasetCompileError("provenance fields are not exact")
        data = dict(value)
        data.pop("format")
        return cls(**data)


@dataclass(frozen=True)
class AuditMatch:
    """One exact, alias, or near candidate collision."""

    match_type: Literal["exact", "alias", "near"]
    left_record_id: str
    right_record_id: str
    left_split: str
    right_split: str
    left_group_id: str
    right_group_id: str
    hamming_distance: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.match_type not in {"exact", "alias", "near"}:
            raise DatasetCompileError("unsupported audit match type")
        for name in ("left_record_id", "right_record_id", "left_group_id", "right_group_id"):
            _text(getattr(self, name), name)
        if self.left_record_id == self.right_record_id:
            raise DatasetCompileError("audit match cannot compare a record to itself")
        if self.left_split not in SPLITS or self.right_split not in SPLITS:
            raise DatasetCompileError("audit match has unsupported split")
        if self.hamming_distance is not None and not isinstance(self.hamming_distance, int):
            raise DatasetCompileError("audit hamming_distance must be an integer")

    def to_payload(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "left_record_id": self.left_record_id,
            "right_record_id": self.right_record_id,
            "left_split": self.left_split,
            "right_split": self.right_split,
            "left_group_id": self.left_group_id,
            "right_group_id": self.right_group_id,
            "hamming_distance": self.hamming_distance,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DatasetAudit(Mapping[str, Any]):
    """Auditable leakage report; mapping access keeps CLI/report callers simple."""

    valid: bool
    record_count: int
    split_counts: Mapping[str, int]
    group_count: int
    exact_matches: tuple[AuditMatch, ...] = ()
    alias_matches: tuple[AuditMatch, ...] = ()
    near_matches: tuple[AuditMatch, ...] = ()
    contradictory_group_ids: tuple[str, ...] = ()
    quarantined_record_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def collision_count(self) -> int:
        return len(self.exact_matches) + len(self.alias_matches) + len(self.near_matches)

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": AUDIT_FORMAT,
            "valid": self.valid,
            "record_count": self.record_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "group_count": self.group_count,
            "exact_matches": [item.to_payload() for item in self.exact_matches],
            "alias_matches": [item.to_payload() for item in self.alias_matches],
            "near_matches": [item.to_payload() for item in self.near_matches],
            "contradictory_group_ids": list(self.contradictory_group_ids),
            "quarantined_record_ids": list(self.quarantined_record_ids),
            "errors": list(self.errors),
            "collision_count": self.collision_count,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_payload()[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.to_payload())

    def __len__(self) -> int:
        return len(self.to_payload())


@dataclass(frozen=True)
class CompiledRecord:
    """Internal pair of an optimizer row and one or more provenance rows."""

    record: TruthEditingRecord
    provenance: tuple[ProvenanceRecord, ...] = field(default_factory=tuple)


# Stable public vocabulary aliases used by callers and design documents.
SourceSpec = DatasetSource
DatasetRecord = TruthEditingRecord
CompileRequest = DatasetRequest
