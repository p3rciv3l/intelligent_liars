"""Build a leakage-safe selector for historical direction reconstruction.

The public seam accepts the immutable v2 dataset plus HDF5 *metadata* rows and
returns a strict refitter-compatible allowlist and a complete admission audit.
It never opens an activation tensor and never requires every v2 training row to
have appeared in the older extraction.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from intelligent_liars.truth_editing_dataset_v2 import (
    DatasetV2Error,
    TruthEditingDatasetV2,
)


ALLOWLIST_FORMAT = "truth_editing_construction_row_allowlist_v1"
AUDIT_FORMAT = "truth_editing_construction_allowlist_audit_v1"
SELECTOR = "direction_construction"
DEFAULT_MINIMUM_PER_CLASS = 50
DEFAULT_MAXIMUM_PER_CLASS = 1_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DatasetV2Error("construction allowlist value is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in value).split())


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetV2Error(f"{name} must be a non-empty string")
    return value.strip()


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatasetV2Error(f"{name} must be a non-negative integer")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


@dataclass(frozen=True)
class Hdf5ConstructionExample:
    task_id: str
    example_index: int
    source_dataset: str
    source_index: int
    source_file: str | None
    source_row_index: int | None
    canonical_questions: tuple[str, ...]
    label: int
    token_row_start: int
    token_row_end: int
    metadata_sha256: str


@dataclass(frozen=True)
class ConstructionAllowlistBuild:
    """One deterministic build result; ``allowlist`` is absent when blocked."""

    allowlist: Mapping[str, Any] | None
    audit: Mapping[str, Any]

    @property
    def ready(self) -> bool:
        return self.allowlist is not None and self.audit.get("status") == "ready"


def _questions_in_metadata(value: Any) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(node: Any, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                visit(child, str(child_key))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for child in node:
                visit(child, key)
        elif key == "question" and isinstance(node, str) and node.strip():
            found.add(node.strip())

    visit(value)
    return tuple(sorted(found, key=lambda item: (_normalize(item), item)))


def load_hdf5_construction_metadata(path: Path | str) -> list[Hdf5ConstructionExample]:
    """Read only identity, label, and span arrays under the HDF5 metadata root."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependency
        raise DatasetV2Error("h5py is required to inspect activation metadata") from error
    source_path = Path(path)
    if source_path.is_symlink() or not source_path.is_file():
        raise DatasetV2Error("activation HDF5 path is not a regular file")
    result: list[Hdf5ConstructionExample] = []
    try:
        with h5py.File(source_path, "r") as hdf5:
            if "metadata" not in hdf5:
                raise DatasetV2Error("activation HDF5 lacks metadata identities")
            for task_id in sorted(hdf5["metadata"].keys()):
                group = hdf5["metadata"][task_id]
                required = {
                    "example_metadata_json",
                    "example_splits",
                    "source_datasets",
                    "example_source_indices",
                    "example_labels",
                }
                if not required <= set(group.keys()):
                    raise DatasetV2Error(f"activation task lacks identity arrays: {task_id}")
                count = len(group["example_metadata_json"])
                if (
                    len(group["example_splits"]) != count + 1
                    or len(group["source_datasets"]) != count
                    or len(group["example_source_indices"]) != count
                    or len(group["example_labels"]) != count
                ):
                    raise DatasetV2Error(f"activation task identity arrays disagree: {task_id}")
                splits = group["example_splits"]
                for index in range(count):
                    raw_metadata = group["example_metadata_json"][index]
                    if isinstance(raw_metadata, bytes):
                        raw_metadata = raw_metadata.decode("utf-8")
                    try:
                        metadata = json.loads(str(raw_metadata))
                    except json.JSONDecodeError as error:
                        raise DatasetV2Error(
                            f"activation example metadata is invalid: {task_id}:{index}"
                        ) from error
                    if not isinstance(metadata, Mapping):
                        raise DatasetV2Error("activation example metadata must be an object")
                    raw_dataset = group["source_datasets"][index]
                    if isinstance(raw_dataset, bytes):
                        raw_dataset = raw_dataset.decode("utf-8")
                    source_dataset = _required_text(
                        str(raw_dataset), f"activation source dataset {task_id}:{index}"
                    )
                    source_file_value = metadata.get("csv") or Path(source_dataset).name
                    source_file = (
                        Path(str(source_file_value)).name if source_file_value else None
                    )
                    source_row = metadata.get("row")
                    if source_row is not None:
                        source_row = _nonnegative_integer(
                            int(source_row), f"activation source row {task_id}:{index}"
                        )
                    start, end = int(splits[index]), int(splits[index + 1])
                    if start < 0 or end <= start:
                        raise DatasetV2Error(
                            f"activation example has an empty or invalid span: {task_id}:{index}"
                        )
                    result.append(
                        Hdf5ConstructionExample(
                            task_id=str(task_id),
                            example_index=index,
                            source_dataset=source_dataset,
                            source_index=int(group["example_source_indices"][index]),
                            source_file=source_file,
                            source_row_index=source_row,
                            canonical_questions=_questions_in_metadata(metadata),
                            label=int(group["example_labels"][index]),
                            token_row_start=start,
                            token_row_end=end,
                            metadata_sha256=_hash(metadata),
                        )
                    )
    except OSError as error:
        raise DatasetV2Error("cannot inspect activation HDF5 metadata") from error
    return result


def _held_out_identities(
    dataset: TruthEditingDatasetV2,
) -> tuple[set[str], set[tuple[str, int]]]:
    held_out_record_ids = {
        str(row["record_id"])
        for row in dataset.records
        if row["split"] in {"validation", "test"}
    }
    questions = {
        _normalize(str(row["question"]))
        for row in dataset.records
        if str(row["record_id"]) in held_out_record_ids
    }
    questions.update(_normalize(str(row["question"])) for row in dataset.quarantine)
    physical: set[tuple[str, int]] = set()
    for row in dataset.provenance:
        if str(row["record_id"]) not in held_out_record_ids:
            continue
        match = re.fullmatch(r"(.+):([0-9]+)", str(row["source_record_id"]))
        if match:
            physical.add((Path(match.group(1)).name, int(match.group(2)) - 2))
    return questions, physical


def build_refitter_construction_allowlist(
    dataset: TruthEditingDatasetV2,
    examples: Iterable[Hdf5ConstructionExample],
    *,
    activation_direct_sha256: str,
    required_domains: Sequence[str],
    minimum_per_class: int = DEFAULT_MINIMUM_PER_CLASS,
    maximum_per_class: int = DEFAULT_MAXIMUM_PER_CLASS,
) -> ConstructionAllowlistBuild:
    """Select clean historical examples and compile the refitter's strict schema."""

    if not _SHA256.fullmatch(activation_direct_sha256):
        raise DatasetV2Error("activation_direct_sha256 must be a lowercase SHA-256")
    if isinstance(required_domains, (str, bytes)) or not required_domains:
        raise DatasetV2Error("required_domains must be a non-empty array")
    domains = tuple(_required_text(value, "required domain") for value in required_domains)
    if len(set(domains)) != len(domains):
        raise DatasetV2Error("required_domains contains duplicates")
    if isinstance(minimum_per_class, bool) or minimum_per_class < 1:
        raise DatasetV2Error("minimum_per_class must be a positive integer")
    if (
        isinstance(maximum_per_class, bool)
        or maximum_per_class < minimum_per_class
    ):
        raise DatasetV2Error("maximum_per_class must be at least minimum_per_class")

    held_out_questions, held_out_physical = _held_out_identities(dataset)
    excluded: Counter[str] = Counter()
    selected: list[Hdf5ConstructionExample] = []
    seen_addresses: set[tuple[str, int]] = set()
    requested = set(domains)
    for example in sorted(examples, key=lambda row: (row.task_id, row.example_index)):
        address = (example.task_id, example.example_index)
        if address in seen_addresses:
            excluded["duplicate_hdf5_example_address"] += 1
            continue
        seen_addresses.add(address)
        if example.task_id not in requested:
            excluded["unrequested_domain"] += 1
            continue
        if example.label not in {0, 1}:
            excluded["unknown_label"] += 1
            continue
        normalized_questions = {_normalize(value) for value in example.canonical_questions}
        if len(normalized_questions) > 1:
            excluded["ambiguous_question_identity"] += 1
            continue
        if normalized_questions & held_out_questions:
            excluded["heldout_or_quarantine_question_collision"] += 1
            continue
        if (
            example.source_file is not None
            and example.source_row_index is not None
            and (Path(example.source_file).name, example.source_row_index) in held_out_physical
        ):
            excluded["heldout_source_row_collision"] += 1
            continue
        selected.append(example)

    eligible = selected
    uncapped_count = len(eligible)
    ranked_by_cell: dict[tuple[str, int], list[Hdf5ConstructionExample]] = defaultdict(list)
    for example in eligible:
        ranked_by_cell[(example.task_id, example.label)].append(example)
    eligible_per_domain_counts = {
        domain: {
            "label_0": len(ranked_by_cell[(domain, 0)]),
            "label_1": len(ranked_by_cell[(domain, 1)]),
            "total": len(ranked_by_cell[(domain, 0)]) + len(ranked_by_cell[(domain, 1)]),
        }
        for domain in domains
    }
    balanced_cap = min(
        maximum_per_class,
        min(len(ranked_by_cell[(domain, label)]) for domain in domains for label in (0, 1)),
    )
    # Treat repeated source/question identities as atomic construction groups.
    # This prevents cap selection from retaining only one side of an honest /
    # deceptive pair.  Truly unpaired historical rows remain singleton atomic
    # groups and are combined into a balanced domain stratum below.
    source_groups: dict[tuple[str, str, int], list[Hdf5ConstructionExample]] = defaultdict(list)
    question_groups: dict[tuple[str, str], list[Hdf5ConstructionExample]] = defaultdict(list)
    for example in eligible:
        source_groups[
            (example.task_id, Path(example.source_dataset).name, example.source_index)
        ].append(example)
        for question in example.canonical_questions:
            question_groups[(example.task_id, _normalize(question))].append(example)
    paired_sources = {
        key for key, values in source_groups.items() if {item.label for item in values} == {0, 1}
    }
    paired_questions = {
        key for key, values in question_groups.items() if {item.label for item in values} == {0, 1}
    }
    atomic_groups: dict[str, list[Hdf5ConstructionExample]] = defaultdict(list)
    for example in eligible:
        question_keys = [
            (example.task_id, _normalize(question))
            for question in example.canonical_questions
            if (example.task_id, _normalize(question)) in paired_questions
        ]
        source_key = (
            example.task_id,
            Path(example.source_dataset).name,
            example.source_index,
        )
        if question_keys:
            atomic_id = "question:" + _hash(question_keys[0])
        elif source_key in paired_sources:
            atomic_id = "source:" + _hash(source_key)
        else:
            atomic_id = f"singleton:{example.task_id}:{example.example_index}"
        atomic_groups[atomic_id].append(example)

    selected = []
    selected_atomic_ids: list[str] = []
    for domain in domains:
        domain_groups = [
            (atomic_id, values)
            for atomic_id, values in atomic_groups.items()
            if values[0].task_id == domain
        ]
        domain_groups.sort(key=lambda item: (_hash(["construction-group-cap-v1", item[0]]), item[0]))
        selected_counts: Counter[int] = Counter()
        for atomic_id, values in domain_groups:
            increments = Counter(item.label for item in values)
            if any(selected_counts[label] + increments[label] > balanced_cap for label in (0, 1)):
                continue
            selected.extend(values)
            selected_atomic_ids.append(atomic_id)
            selected_counts.update(increments)
            if selected_counts[0] == balanced_cap and selected_counts[1] == balanced_cap:
                break
        if selected_counts != Counter({0: balanced_cap, 1: balanced_cap}):
            # A pathological group-size pattern cannot be silently split.
            excluded["whole_group_balance_shortfall"] += sum(
                balanced_cap - selected_counts[label] for label in (0, 1)
            )
    capped = uncapped_count - len(selected)
    if capped:
        excluded["deterministic_cell_cap"] += capped
    selected.sort(key=lambda example: (example.task_id, example.example_index))

    by_domain_label: dict[str, Counter[int]] = defaultdict(Counter)
    for example in selected:
        by_domain_label[example.task_id][example.label] += 1
    missing_cells = [
        {
            "domain": domain,
            "label": label,
            "observed": eligible_per_domain_counts[domain][f"label_{label}"],
            "required": minimum_per_class,
        }
        for domain in domains
        for label in (0, 1)
        if eligible_per_domain_counts[domain][f"label_{label}"] < minimum_per_class
    ]

    dataset_manifest_sha256 = _file_hash(dataset.path / "manifest.json")
    selected_atomic_by_address = {
        (example.task_id, example.example_index): atomic_id
        for atomic_id in selected_atomic_ids
        for example in atomic_groups[atomic_id]
    }
    paired_selected_ids = {
        atomic_id
        for atomic_id in selected_atomic_ids
        if {example.label for example in atomic_groups[atomic_id]} == {0, 1}
    }
    rows: list[dict[str, Any]] = [
        {
            "row_id": f"{example.task_id}:example:{example.example_index}",
            "group_id": (
                f"{example.task_id}:paired:{selected_atomic_by_address[(example.task_id, example.example_index)].split(':', 1)[1]}"
                if selected_atomic_by_address[(example.task_id, example.example_index)] in paired_selected_ids
                else f"{example.task_id}:unpaired-balanced-construction"
            ),
            "domain": example.task_id,
            "hdf5_task": example.task_id,
            "hdf5_row_index": example.example_index,
            "label": example.label,
            "selector": SELECTOR,
        }
        for example in selected
    ]
    group_ids = sorted({str(row["group_id"]) for row in rows})
    # The refitter contract requires every construction group to include both
    # labels.  If a domain has only one residual label, merge that residual with
    # its paired groups at the manifest seam; atomic selection above remains
    # whole and auditable.
    labels_by_group = {
        group_id: {int(row["label"]) for row in rows if row["group_id"] == group_id}
        for group_id in group_ids
    }
    for row in rows:
        if labels_by_group[str(row["group_id"])] != {0, 1}:
            row["group_id"] = f"{row['domain']}:clean-construction-domain"
    group_ids = sorted({str(row["group_id"]) for row in rows})
    group_manifest = [
        {
            "group_id": group_id,
            "domain": next(str(row["domain"]) for row in rows if row["group_id"] == group_id),
            "row_ids": [row["row_id"] for row in rows if row["group_id"] == group_id],
        }
        for group_id in group_ids
    ]
    construction_group_manifest_sha256 = _hash(group_manifest)
    allowlist: dict[str, Any] | None = None
    if not missing_cells and not excluded["duplicate_hdf5_example_address"]:
        unsigned = {
            "format": ALLOWLIST_FORMAT,
            "allowlist_id": "historical-all-domain-clean-v1",
            "activation_direct_sha256": activation_direct_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "construction_group_manifest_sha256": construction_group_manifest_sha256,
            "rows": rows,
        }
        allowlist = dict(unsigned)
        allowlist["self_sha256"] = _hash(unsigned)

    counts = {
        domain: {
            "label_0": by_domain_label[domain][0],
            "label_1": by_domain_label[domain][1],
            "total": by_domain_label[domain][0] + by_domain_label[domain][1],
        }
        for domain in domains
    }
    audit_unsigned = {
        "format": AUDIT_FORMAT,
        "status": "ready" if allowlist is not None else "blocked",
        "activation_direct_sha256": activation_direct_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "required_domains": list(domains),
        "minimum_per_class": minimum_per_class,
        "maximum_per_class": maximum_per_class,
        "effective_balanced_per_domain_class_cap": balanced_cap,
        "eligible_before_cap_count": uncapped_count,
        "eligible_per_domain_counts": eligible_per_domain_counts,
        "selected_example_count": len(selected),
        "selected_atomic_group_count": len(selected_atomic_ids),
        "selected_atomic_group_ids_sha256": _hash(selected_atomic_ids),
        "per_domain_counts": counts,
        "excluded_counts": dict(sorted(excluded.items())),
        "missing_cells": missing_cells,
        "held_out_question_identity_count": len(held_out_questions),
        "held_out_source_row_identity_count": len(held_out_physical),
        "construction_group_manifest_sha256": construction_group_manifest_sha256,
        "selected_ordered_row_ids_sha256": _hash([row["row_id"] for row in rows]),
        "allowlist_self_sha256": allowlist["self_sha256"] if allowlist else None,
    }
    audit = dict(audit_unsigned)
    audit["self_sha256"] = _hash(audit_unsigned)
    return ConstructionAllowlistBuild(allowlist=allowlist, audit=audit)


def write_construction_allowlist_build(
    build: ConstructionAllowlistBuild,
    *,
    allowlist_path: Path | str,
    audit_path: Path | str,
    overwrite: bool = False,
) -> tuple[str | None, str]:
    """Write canonical outputs without replacing an existing allowlist."""

    destination = Path(allowlist_path)
    audit_destination = Path(audit_path)
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    audit_destination.write_bytes(_canonical_bytes(build.audit) + b"\n")
    if build.allowlist is None:
        return None, _file_hash(audit_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(build.allowlist)
    if destination.exists() and destination.read_bytes() != encoded and not overwrite:
        raise DatasetV2Error("refusing to overwrite a different construction allowlist")
    destination.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest(), _file_hash(audit_destination)


__all__ = [
    "ConstructionAllowlistBuild",
    "DEFAULT_MAXIMUM_PER_CLASS",
    "DEFAULT_MINIMUM_PER_CLASS",
    "Hdf5ConstructionExample",
    "build_refitter_construction_allowlist",
    "load_hdf5_construction_metadata",
    "write_construction_allowlist_build",
]
