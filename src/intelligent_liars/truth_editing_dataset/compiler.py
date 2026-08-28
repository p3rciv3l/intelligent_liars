"""Compilation, leakage audit, and materialization for truth-editing data."""

from __future__ import annotations

import csv
import difflib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from intelligent_liars.holdout_decontamination import (
    find_collisions,
    fingerprint_record,
)

from .contracts import (
    AUDIT_FORMAT,
    DATASET_FORMAT,
    NON_QUARANTINE_SPLITS,
    SPLITS,
    AuditMatch,
    CompiledRecord,
    DatasetAudit,
    DatasetCompileError,
    DatasetRequest,
    DatasetSource,
    ProvenanceRecord,
    TruthEditingRecord,
    canonical_sha256,
    normalize_text,
)


Reader = Iterable[Mapping[str, Any] | TruthEditingRecord] | Path | str | Callable[..., Iterable[Mapping[str, Any] | TruthEditingRecord]]
SplitName = Literal["train", "validation", "test", "quarantine"]


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def add(self, key: tuple[str, str]) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: tuple[str, str]) -> tuple[str, str]:
        parent = self.parent.setdefault(key, key)
        if parent != key:
            parent = self.find(parent)
            self.parent[key] = parent
        return parent

    def union(self, left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Lexical roots make the internal result stable too.
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def _read_path(path: Path) -> list[Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DatasetCompileError(f"source path is not a regular file: {path}")
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            rows: list[Mapping[str, Any]] = []
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise DatasetCompileError(f"{path}:{line_number}: expected object")
                rows.append(dict(value))
            return rows
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as source:
                return [dict(row) for row in csv.DictReader(source)]
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as error:
        raise DatasetCompileError(f"cannot read local source {path}") from error
    if not isinstance(value, list):
        raise DatasetCompileError(f"JSON source must contain a list: {path}")
    if any(not isinstance(item, Mapping) for item in value):
        raise DatasetCompileError(f"JSON source contains a non-object row: {path}")
    return [dict(item) for item in value]


def _call_reader(reader: Reader, spec: DatasetSource) -> Iterable[Mapping[str, Any] | TruthEditingRecord]:
    if isinstance(reader, (Path, str)):
        return _read_path(Path(reader))
    if callable(reader):
        try:
            result = reader(spec)
        except TypeError:
            result = reader()  # type: ignore[call-arg]
        if not isinstance(result, Iterable):
            raise DatasetCompileError(f"reader for {spec.source_id} did not return rows")
        return result
    return reader


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _as_answer(value: Any, name: str) -> str:
    if isinstance(value, Mapping):
        value = _first(value, "text", "answer", "label", "value")
    if not isinstance(value, str) or not value.strip():
        raise DatasetCompileError(f"{name} must be a non-empty answer string")
    return value.strip()


def _adapt_row(
    raw: Mapping[str, Any] | TruthEditingRecord,
    *,
    spec: DatasetSource,
    row_number: int,
) -> tuple[TruthEditingRecord, ProvenanceRecord, str, tuple[str, ...], str | None]:
    requested_split: str | None = None
    if isinstance(raw, TruthEditingRecord):
        record = raw
        source_record_id = record.record_id
        input_key = record.optimizer_payload
        requested_split = record.split
    elif isinstance(raw, Mapping):
        source_record_id = _first(raw, "record_id", "id", "uid", "example_id")
        if source_record_id is None:
            # Do not make an otherwise anonymous source row depend on reader
            # iteration order.  The canonical raw-object hash is stable for
            # mappings and gives duplicate rows the same identity.
            source_record_id = f"row-{canonical_sha256(dict(raw))[:16]}"
        source_record_id = str(source_record_id)
        question_value = _first(raw, "question", "prompt", "input", "query", "claim")
        if isinstance(question_value, Mapping):
            question_value = _first(question_value, "text", "question", "prompt")
        if not isinstance(question_value, str) or not question_value.strip():
            raise DatasetCompileError(f"{spec.source_id}:{source_record_id}: missing question")
        question = question_value.strip()

        correct_value = _first(
            raw,
            "correct_answer",
            "gold_answer",
            "truthful_answer",
            "answer",
            "target",
        )
        choices = _first(raw, "answers", "choices", "options")
        correct_index = _first(raw, "correct_index", "answer_index", "label_index")
        # MMLU-style rows use ``answer`` as an integer option index.
        if isinstance(correct_value, int) and not isinstance(correct_value, bool) and isinstance(choices, (list, tuple)):
            correct_index = correct_value
            correct_value = None
        elif isinstance(correct_value, str) and len(correct_value.strip()) == 1 and correct_value.strip().upper() in "ABCD" and isinstance(choices, (list, tuple)):
            correct_index = ord(correct_value.strip().upper()) - ord("A")
            correct_value = None
        if correct_value is None and isinstance(choices, (list, tuple)) and correct_index is not None:
            try:
                correct_value = choices[int(correct_index)]
            except (ValueError, IndexError, TypeError) as error:
                raise DatasetCompileError(f"{spec.source_id}:{source_record_id}: invalid correct index") from error
        if correct_value is None:
            raise DatasetCompileError(f"{spec.source_id}:{source_record_id}: missing correct_answer")
        correct_answer = _as_answer(correct_value, "correct_answer")

        wrong_values = _first(raw, "wrong_answers", "false_answers", "plausible_answers", "distractors")
        if wrong_values is None and isinstance(choices, (list, tuple)):
            wrong_values = [choice for choice in choices if normalize_text(_as_answer(choice, "choice")) != normalize_text(correct_answer)]
        if wrong_values is None:
            wrong_values = ()
        elif isinstance(wrong_values, str):
            wrong_values = (wrong_values,)
        if not isinstance(wrong_values, (list, tuple)):
            raise DatasetCompileError(f"{spec.source_id}:{source_record_id}: wrong_answers must be a list")

        # A source truth label is not an experimental condition.  Treating it
        # as one would hide contradictory gold answers behind condition IDs.
        condition_value = _first(raw, "condition", "mode", "arm", "condition_kind")
        family_value = _first(raw, "family", "domain", "subject", "category")
        proposition_value = _first(raw, "proposition", "claim", "statement", "fact")
        aliases_value = _first(raw, "aliases", "question_aliases", "paraphrases")
        canonical_question = _first(raw, "canonical_question_id", "question_id")
        canonical_proposition = _first(raw, "canonical_proposition_id", "proposition_id")
        leakage_group = _first(raw, "leakage_group_id", "split_group_id", "group_id")
        requested_value = _first(raw, "split", "dataset_split")
        if requested_value is not None:
            requested_split = str(requested_value).strip()
            if requested_split not in SPLITS:
                raise DatasetCompileError(f"{spec.source_id}:{source_record_id}: unsupported split {requested_split}")
        if isinstance(aliases_value, str):
            aliases_value = (aliases_value,)
        elif aliases_value is None:
            aliases_value = ()
        elif not isinstance(aliases_value, (list, tuple)):
            raise DatasetCompileError(f"{spec.source_id}:{source_record_id}: aliases must be a list")
        record = TruthEditingRecord(
            record_id=source_record_id,
            question=question,
            correct_answer=correct_answer,
            wrong_answers=tuple(_as_answer(item, "wrong_answer") for item in wrong_values),
            condition=str(condition_value if condition_value is not None else "unspecified"),
            family=str(family_value if family_value is not None else "unspecified"),
            proposition=(str(proposition_value).strip() if proposition_value is not None else None),
            aliases=tuple(str(item).strip() for item in aliases_value if str(item).strip()),
            canonical_question_id=(str(canonical_question).strip() if canonical_question is not None else None),
            canonical_proposition_id=(str(canonical_proposition).strip() if canonical_proposition is not None else None),
            leakage_group_id=(str(leakage_group).strip() if leakage_group is not None else None),
        )
        input_key = dict(raw)
    else:
        raise DatasetCompileError(f"{spec.source_id}:{row_number}: expected an object")

    # Canonical IDs are derived here, not trusted from arbitrary source text.
    question_id = record.canonical_question_id or f"question_{canonical_sha256(normalize_text(record.question))}"
    proposition_text = record.proposition or f"{record.question}\n{record.correct_answer}"
    proposition_id = record.canonical_proposition_id or f"proposition_{canonical_sha256(normalize_text(proposition_text))}"
    leakage_key = record.leakage_group_id or question_id
    record = replace(
        record,
        canonical_question_id=f"question_{canonical_sha256(normalize_text(question_id))}",
        canonical_proposition_id=f"proposition_{canonical_sha256(normalize_text(proposition_id))}",
        leakage_group_id=f"leakage_{canonical_sha256(normalize_text(leakage_key))}",
    )
    content_sha = canonical_sha256(input_key)
    provenance = ProvenanceRecord(
        record_id=record.record_id,
        source_id=spec.source_id,
        source_revision=spec.revision,
        source_record_id=source_record_id,
        source_selector=spec.selector,
        source_kind=spec.kind,
        source_sha256=spec.source_sha256,
        input_content_sha256=content_sha,
    )
    aliases = tuple(record.aliases) + (record.question,)
    return record, provenance, content_sha, aliases, requested_split


def _semantic_key(item: TruthEditingRecord) -> tuple[str, ...]:
    return (
        item.leakage_group_id or "",
        normalize_text(item.question),
        normalize_text(item.proposition or ""),
        normalize_text(item.correct_answer),
        "\x1f".join(sorted(normalize_text(answer) for answer in item.wrong_answers)),
        normalize_text(item.condition),
        normalize_text(item.family),
    )


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _collision(
    match_type: str,
    left: CompiledRecord,
    right: CompiledRecord,
    *,
    distance: int | None = None,
    reason: str = "",
) -> AuditMatch:
    return AuditMatch(
        match_type=match_type,  # type: ignore[arg-type]
        left_record_id=left.record.record_id,
        right_record_id=right.record.record_id,
        left_split=left.record.split,
        right_split=right.record.split,
        left_group_id=left.record.leakage_group_id or "",
        right_group_id=right.record.leakage_group_id or "",
        hamming_distance=distance,
        reason=reason,
    )


def _assign_split(group_id: str, request: DatasetRequest) -> SplitName:
    digest = canonical_sha256([request.split_salt, request.seed, group_id])
    fraction = int(digest[:16], 16) / float(16**16)
    train_end = float(request.train_fraction)
    validation_end = train_end + float(request.validation_fraction)
    return "train" if fraction < train_end else "validation" if fraction < validation_end else "test"


def _parse_audit(value: Mapping[str, Any]) -> DatasetAudit:
    if value.get("format") != AUDIT_FORMAT:
        raise DatasetCompileError("unsupported dataset audit format")
    allowed = {
        "format", "valid", "record_count", "split_counts", "group_count",
        "exact_matches", "alias_matches", "near_matches",
        "contradictory_group_ids", "quarantined_record_ids", "errors",
        "collision_count",
    }
    unknown = set(value) - allowed
    if unknown:
        raise DatasetCompileError(f"unknown audit fields: {sorted(unknown)}")

    def parse_matches(key: str) -> tuple[AuditMatch, ...]:
        rows = value.get(key, [])
        if not isinstance(rows, list):
            raise DatasetCompileError(f"audit {key} must be a list")
        if any(not isinstance(row, Mapping) for row in rows):
            raise DatasetCompileError(f"audit {key} contains a non-object")
        try:
            return tuple(AuditMatch(**dict(row)) for row in rows)
        except (TypeError, ValueError) as error:
            raise DatasetCompileError(f"invalid audit {key} match") from error

    split_counts = value.get("split_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != set(SPLITS):
        raise DatasetCompileError("audit split_counts must name all splits")
    audit = DatasetAudit(
        valid=value.get("valid") is True,
        record_count=value.get("record_count", 0),
        split_counts={str(key): int(item) for key, item in split_counts.items()},
        group_count=value.get("group_count", 0),
        exact_matches=parse_matches("exact_matches"),
        alias_matches=parse_matches("alias_matches"),
        near_matches=parse_matches("near_matches"),
        contradictory_group_ids=tuple(value.get("contradictory_group_ids", [])),
        quarantined_record_ids=tuple(value.get("quarantined_record_ids", [])),
        errors=tuple(value.get("errors", [])),
    )
    if audit.collision_count != value.get("collision_count"):
        raise DatasetCompileError("audit collision_count mismatch")
    return audit


class TruthEditingDataset:
    """Compiled, immutable-in-use dataset with four leakage-aware splits."""

    def __init__(
        self,
        *,
        request: DatasetRequest,
        records: Iterable[CompiledRecord],
        audit: DatasetAudit,
        manifest: Mapping[str, Any] | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.request = request
        self._compiled = tuple(sorted(records, key=lambda item: item.record.record_id))
        self._by_split = {
            split: tuple(item for item in self._compiled if item.record.split == split)
            for split in SPLITS
        }
        self._audit_report = audit
        self._manifest = dict(manifest) if manifest is not None else None
        self.manifest_path = manifest_path

    @classmethod
    def compile(
        cls,
        request: DatasetRequest,
        *,
        readers: Mapping[str, Reader] | None = None,
    ) -> TruthEditingDataset:
        """Compile pinned local/in-memory readers without network access."""

        if not isinstance(request, DatasetRequest):
            raise DatasetCompileError("compile requires a DatasetRequest")
        readers = dict(readers or {})
        specs = request.source_specs
        if not specs and readers:
            # Convenience for fixtures while retaining a pinned revision in the
            # resulting manifest.  Production builds should always specify it.
            specs = tuple(
                DatasetSource(source_id=source_id, revision="in-memory")
                for source_id in sorted(readers)
            )
        unknown_readers = set(readers) - {spec.reader_key or spec.source_id for spec in specs}
        if unknown_readers:
            raise DatasetCompileError(f"readers lack SourceSpec values: {sorted(unknown_readers)}")
        missing = [spec.source_id for spec in specs if (spec.reader_key or spec.source_id) not in readers]
        if missing:
            raise DatasetCompileError(f"no reader for source: {missing[0]}")

        adapted: list[tuple[TruthEditingRecord, ProvenanceRecord, str, tuple[str, ...], str | None]] = []
        seen_record_ids: dict[str, tuple[str, ...]] = {}
        source_digests: dict[str, str] = {}
        for spec in specs:
            reader_key = spec.reader_key or spec.source_id
            rows = list(_call_reader(readers[reader_key], spec))
            rows.sort(
                key=lambda item: canonical_sha256(
                    item.optimizer_payload if isinstance(item, TruthEditingRecord) else dict(item)
                )
            )
            source_digests[spec.source_id] = canonical_sha256(
                [dict(item.optimizer_payload if isinstance(item, TruthEditingRecord) else item) for item in rows]
            )
            if spec.source_sha256 is not None and source_digests[spec.source_id] != spec.source_sha256:
                raise DatasetCompileError(f"source hash mismatch for {spec.source_id}")
            for index, raw in enumerate(rows):
                record, provenance, content_sha, aliases, requested_split = _adapt_row(raw, spec=spec, row_number=index)
                previous = seen_record_ids.get(record.record_id)
                if previous is not None and previous != _semantic_key(record):
                    raise DatasetCompileError(f"record_id collision with different content: {record.record_id}")
                seen_record_ids[record.record_id] = _semantic_key(record)
                adapted.append((record, provenance, content_sha, aliases, requested_split))

        # Link explicit canonical IDs, exact text, aliases, and propositions.
        union = _UnionFind()
        keys_for_row: list[set[tuple[str, str]]] = []
        text_index: dict[str, list[int]] = defaultdict(list)
        for index, (record, _, _, aliases, _) in enumerate(adapted):
            keys = {
                ("question_id", record.canonical_question_id or ""),
                ("proposition_id", record.canonical_proposition_id or ""),
                ("leakage_id", record.leakage_group_id or ""),
            }
            keys.update(("alias", normalize_text(alias)) for alias in aliases if normalize_text(alias))
            keys_for_row.append(keys)
            for key in keys:
                union.add(key)
            for alias in aliases:
                normalized = normalize_text(alias)
                if normalized:
                    text_index[normalized].append(index)
            ordered_keys = sorted(keys)
            for key in ordered_keys[1:]:
                union.union(ordered_keys[0], key)
        for indices in text_index.values():
            first_aliases = {
                normalize_text(alias) for alias in adapted[indices[0]][3] if normalize_text(alias)
            }
            for index in indices[1:]:
                current_aliases = {
                    normalize_text(alias) for alias in adapted[index][3] if normalize_text(alias)
                }
                shared = first_aliases & current_aliases
                for normalized in shared:
                    union.union(("alias", normalized), ("alias", normalized))
        group_members: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
        for keys in keys_for_row:
            for key in keys:
                group_members[union.find(key)].add(key)
        group_id_for_root = {
            root: f"leakage_{canonical_sha256(sorted(keys))}"
            for root, keys in group_members.items()
        }

        normalized_rows: list[tuple[TruthEditingRecord, ProvenanceRecord, str, tuple[str, ...], str | None]] = []
        for index, (record, provenance, content_sha, aliases, requested_split) in enumerate(adapted):
            roots = {union.find(key) for key in keys_for_row[index]}
            if len(roots) != 1:
                # Every row's explicit IDs and aliases must resolve to one group.
                raise DatasetCompileError(f"ambiguous leakage keys for {record.record_id}")
            record = replace(record, leakage_group_id=group_id_for_root[next(iter(roots))])
            normalized_rows.append((record, provenance, content_sha, aliases, requested_split))

        requested_by_group: dict[str, set[str]] = defaultdict(set)
        answers_by_truth_identity: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        groups_by_truth_identity: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for record, _, _, _, requested_split in normalized_rows:
            group_id = record.leakage_group_id or record.record_id
            for identity in (
                ("question", record.canonical_question_id or "", normalize_text(record.condition)),
                ("proposition", record.canonical_proposition_id or "", normalize_text(record.condition)),
            ):
                if identity[1]:
                    answers_by_truth_identity[identity].add(normalize_text(record.correct_answer))
                    groups_by_truth_identity[identity].add(group_id)
            if requested_split is not None:
                requested_by_group[group_id].add(requested_split)
        conflicting_input_split_groups = {
            group_id for group_id, requested in requested_by_group.items() if len(requested) > 1
        }
        contradictory_truth_identities = {
            identity for identity, answers in answers_by_truth_identity.items() if len(answers) > 1
        }
        contradictory_truth_groups = {
            group_id
            for identity in contradictory_truth_identities
            for group_id in groups_by_truth_identity[identity]
        }
        contradictory_groups = conflicting_input_split_groups | contradictory_truth_groups

        # Exact semantic duplicates are represented once, with all source rows
        # retained in separate provenance records.
        dedup: dict[tuple[str, ...], CompiledRecord] = {}
        duplicate_provenance: dict[tuple[str, ...], list[ProvenanceRecord]] = defaultdict(list)
        for record, provenance, _, _, _ in normalized_rows:
            semantic_key = _semantic_key(record)
            current: CompiledRecord | None = dedup.get(semantic_key)
            if current is None:
                dedup[semantic_key] = CompiledRecord(record=record, provenance=(provenance,))
            else:
                duplicate_provenance[semantic_key].extend(current.provenance)
                duplicate_provenance[semantic_key].append(provenance)
                winner = min(current.record.record_id, record.record_id)
                dedup[semantic_key] = CompiledRecord(
                    record=replace(current.record, record_id=winner),
                    provenance=tuple(sorted({*current.provenance, provenance}, key=lambda item: (item.source_id, item.source_record_id))),
                )
        candidates = list(dedup.values())
        by_id = {item.record.record_id: item for item in candidates}

        exact_matches: dict[tuple[str, str], AuditMatch] = {}
        alias_matches: dict[tuple[str, str], AuditMatch] = {}
        # Candidate audits use canonical semantic identities, even if duplicate
        # rows were collapsed from the optimizer view.
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                same_group = left.record.leakage_group_id == right.record.leakage_group_id
                same_question = left.record.canonical_question_id == right.record.canonical_question_id
                same_prop = left.record.canonical_proposition_id == right.record.canonical_proposition_id
                contradictory = same_group and left.record.leakage_group_id in contradictory_truth_groups and normalize_text(left.record.correct_answer) != normalize_text(right.record.correct_answer)
                if same_question or same_prop or contradictory:
                    exact_matches[_pair_key(left.record.record_id, right.record.record_id)] = _collision(
                        "exact", left, right,
                        reason=("contradictory correct answers in one leakage group" if contradictory else "canonical question or proposition identity"),
                    )
                if same_group:
                    continue
                left_aliases = {normalize_text(left.record.question), *(normalize_text(item) for item in left.record.aliases)}
                right_aliases = {normalize_text(right.record.question), *(normalize_text(item) for item in right.record.aliases)}
                if left_aliases & right_aliases:
                    alias_matches[_pair_key(left.record.record_id, right.record.record_id)] = _collision(
                        "alias", left, right, reason="question alias identity"
                    )

        near_matches: dict[tuple[str, str], AuditMatch] = {}
        # Reuse the repository's fingerprint implementation for the robust
        # path.  The small lexical fallback catches short fixture questions,
        # which the fingerprint helper intentionally ignores.
        fingerprints = []
        for item in candidates:
            fingerprints.extend(
                fingerprint_record(
                    {
                        "question": item.record.question,
                        "aliases": list(item.record.aliases),
                        "proposition": item.record.proposition or "",
                        "correct_answer": item.record.correct_answer,
                        "wrong_answers": list(item.record.wrong_answers),
                    },
                    source_id="compiled",
                    record_id=item.record.record_id,
                )
            )
        for collision in find_collisions(
            fingerprints,
            fingerprints,
            max_hamming=request.near_max_hamming,
            minimum_length_ratio=request.near_minimum_length_ratio,
        ):
            left_candidate = by_id.get(collision["training_record_id"])
            right_candidate = by_id.get(collision["holdout_record_id"])
            if left_candidate is None or right_candidate is None:
                continue
            left, right = left_candidate, right_candidate
            if left.record.record_id >= right.record.record_id:
                continue
            if left.record.leakage_group_id == right.record.leakage_group_id:
                continue
            key = _pair_key(left.record.record_id, right.record.record_id)
            near_matches[key] = _collision(
                "near", left, right,
                distance=int(collision["hamming_distance"]),
                reason="holdout_decontamination simhash candidate",
            )
        for left_index, left in enumerate(candidates):
            left_text = normalize_text(" ".join((left.record.question, *left.record.aliases, left.record.proposition or "", left.record.correct_answer, *left.record.wrong_answers)))
            for right in candidates[left_index + 1 :]:
                if left.record.leakage_group_id == right.record.leakage_group_id:
                    continue
                right_text = normalize_text(" ".join((right.record.question, *right.record.aliases, right.record.proposition or "", right.record.correct_answer, *right.record.wrong_answers)))
                if min(len(left_text), len(right_text)) < 16:
                    continue
                ratio = difflib.SequenceMatcher(None, left_text, right_text).ratio()
                if ratio >= 0.9:
                    key = _pair_key(left.record.record_id, right.record.record_id)
                    near_matches.setdefault(
                        key,
                        _collision("near", left, right, reason=f"normalized question similarity {ratio:.3f}"),
                    )

        near_ids = {match.left_record_id for match in near_matches.values()} | {match.right_record_id for match in near_matches.values()}
        near_groups = {item.record.leakage_group_id for item in candidates if item.record.record_id in near_ids}
        compiled: list[CompiledRecord] = []
        for item in candidates:
            group_id = item.record.leakage_group_id or item.record.record_id
            split: SplitName
            if group_id in near_groups or group_id in contradictory_groups:
                split = "quarantine"
            elif request.honor_input_splits and len(requested_by_group.get(group_id, set())) == 1:
                split = cast(SplitName, next(iter(requested_by_group[group_id])))
            else:
                split = _assign_split(group_id, request)
            compiled.append(CompiledRecord(record=replace(item.record, split=split), provenance=item.provenance))

        # Rebuild candidate match split labels now that assignment is known.
        final_by_id = {item.record.record_id: item for item in compiled}
        def relabel(matches: Mapping[tuple[str, str], AuditMatch]) -> tuple[AuditMatch, ...]:
            result: list[AuditMatch] = []
            for match in matches.values():
                left, right = final_by_id[match.left_record_id], final_by_id[match.right_record_id]
                result.append(replace(match, left_split=left.record.split, right_split=right.record.split))
            return tuple(sorted(result, key=lambda item: (item.left_record_id, item.right_record_id)))

        exact = relabel(exact_matches)
        alias_matches_final = relabel(alias_matches)
        near = relabel(near_matches)
        errors: list[str] = []
        for collection_name, collection in (("exact", exact), ("alias", alias_matches_final), ("near", near)):
            for match in collection:
                if match.left_split in NON_QUARANTINE_SPLITS and match.right_split in NON_QUARANTINE_SPLITS and match.left_split != match.right_split:
                    errors.append(f"{collection_name} leakage across splits: {match.left_record_id}/{match.right_record_id}")
        split_counts = {split: sum(item.record.split == split for item in compiled) for split in SPLITS}
        audit = DatasetAudit(
            valid=not errors,
            record_count=len(compiled),
            split_counts=split_counts,
            group_count=len({item.record.leakage_group_id for item in compiled}),
            exact_matches=exact,
            alias_matches=alias_matches_final,
            near_matches=near,
            contradictory_group_ids=tuple(sorted(contradictory_groups)),
            quarantined_record_ids=tuple(sorted(item.record.record_id for item in compiled if item.record.split == "quarantine")),
            errors=tuple(sorted(errors)),
        )
        if not audit.valid:
            raise DatasetCompileError("dataset leakage audit failed: " + "; ".join(audit.errors))
        dataset = cls(request=request, records=compiled, audit=audit)
        if request.output_dir is not None:
            dataset._materialize(request.output_dir, source_digests=source_digests)
        return dataset

    @classmethod
    def open(cls, manifest_path: Path | str) -> TruthEditingDataset:
        """Open and verify a previously materialized canonical manifest."""

        path = Path(manifest_path)
        if path.is_dir():
            path = path / "manifest.json"
        if path.is_symlink() or not path.is_file():
            raise DatasetCompileError(f"manifest is not a regular file: {path}")
        try:
            manifest = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetCompileError("manifest is unreadable") from error
        if not isinstance(manifest, Mapping) or manifest.get("format") != DATASET_FORMAT:
            raise DatasetCompileError("unsupported dataset manifest")
        allowed_manifest = {
            "format", "dataset_id", "request", "files", "file_sha256", "source_sha256",
            "record_count", "split_counts", "audit", "manifest_sha256",
        }
        unknown_manifest = set(manifest) - allowed_manifest
        if unknown_manifest:
            raise DatasetCompileError(f"unknown manifest fields: {sorted(unknown_manifest)}")
        claimed = manifest.get("manifest_sha256")
        body = dict(manifest)
        body.pop("manifest_sha256", None)
        if not isinstance(claimed, str) or canonical_sha256(body) != claimed:
            raise DatasetCompileError("manifest hash mismatch")
        request_payload = manifest.get("request")
        if not isinstance(request_payload, Mapping):
            raise DatasetCompileError("manifest request must be an object")
        request = DatasetRequest.from_mapping(request_payload)
        if manifest.get("dataset_id") != request.dataset_id:
            raise DatasetCompileError("manifest dataset_id mismatch")
        root = path.parent
        records: list[CompiledRecord] = []
        provenance_by_id: dict[str, list[ProvenanceRecord]] = defaultdict(list)
        provenance_path = root / "provenance.jsonl"
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise DatasetCompileError("manifest provenance file is missing")
        file_hashes_payload = manifest.get("file_sha256")
        if not isinstance(file_hashes_payload, Mapping) or set(file_hashes_payload) != set(SPLITS) | {"provenance"}:
            raise DatasetCompileError("manifest file_sha256 entries are incomplete")
        if file_hashes_payload.get("provenance") != canonical_sha256(provenance_path.read_text()):
            raise DatasetCompileError("provenance file hash mismatch")
        provenance_seen: set[tuple[str, str, str]] = set()
        for line in provenance_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetCompileError("invalid provenance JSON") from error
            if not isinstance(value, Mapping):
                raise DatasetCompileError("invalid provenance row")
            try:
                provenance = ProvenanceRecord.from_payload(value)
            except (DatasetCompileError, TypeError, ValueError) as error:
                raise DatasetCompileError("invalid provenance row") from error
            identity = (provenance.record_id, provenance.source_id, provenance.source_record_id)
            if identity in provenance_seen:
                raise DatasetCompileError("duplicate provenance row")
            provenance_seen.add(identity)
            provenance_by_id[provenance.record_id].append(provenance)
        seen_ids: set[str] = set()
        file_entries = manifest.get("files")
        if not isinstance(file_entries, Mapping):
            raise DatasetCompileError("manifest files must be an object")
        for split in SPLITS:
            file_name = file_entries.get(split)
            if not isinstance(file_name, str) or Path(file_name).name != file_name:
                raise DatasetCompileError(f"invalid manifest file for {split}")
            split_path = root / file_name
            if split_path.is_symlink() or not split_path.is_file():
                raise DatasetCompileError(f"missing split file: {split}")
            expected = file_hashes_payload.get(split)
            if expected != canonical_sha256(split_path.read_text()):
                raise DatasetCompileError(f"split file hash mismatch: {split}")
            for line in split_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetCompileError(f"invalid {split} JSON") from error
                if not isinstance(value, Mapping):
                    raise DatasetCompileError(f"invalid {split} record")
                try:
                    record = TruthEditingRecord.from_optimizer_payload(value)
                except (DatasetCompileError, TypeError, ValueError) as error:
                    raise DatasetCompileError(f"invalid {split} record") from error
                if record.split != split or record.record_id in seen_ids:
                    raise DatasetCompileError("manifest record identity mismatch")
                seen_ids.add(record.record_id)
                if record.record_id not in provenance_by_id:
                    raise DatasetCompileError(f"missing provenance for record: {record.record_id}")
                records.append(CompiledRecord(record=record, provenance=tuple(provenance_by_id[record.record_id])))
        if set(provenance_by_id) != seen_ids:
            orphaned = sorted(set(provenance_by_id) - seen_ids)
            raise DatasetCompileError(f"orphan provenance records: {orphaned}")
        audit_payload = manifest.get("audit")
        if not isinstance(audit_payload, Mapping):
            raise DatasetCompileError("manifest audit must be an object")
        audit = _parse_audit(audit_payload)
        if not audit.valid:
            raise DatasetCompileError("manifest audit is not valid")
        if audit.record_count != len(records):
            raise DatasetCompileError("manifest audit count mismatch")
        computed_counts = {split: sum(item.record.split == split for item in records) for split in SPLITS}
        if manifest.get("record_count") != len(records) or manifest.get("split_counts") != computed_counts:
            raise DatasetCompileError("manifest record or split count mismatch")
        if dict(audit.split_counts) != computed_counts:
            raise DatasetCompileError("audit split count mismatch")
        groups: dict[str, set[str]] = defaultdict(set)
        answers_by_truth_identity: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        groups_by_truth_identity: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        question_splits: dict[str, set[str]] = defaultdict(set)
        proposition_splits: dict[str, set[str]] = defaultdict(set)
        alias_splits: dict[str, set[str]] = defaultdict(set)
        for item in records:
            record = item.record
            group_id = record.leakage_group_id or record.record_id
            groups[group_id].add(record.split)
            for identity in (
                ("question", record.canonical_question_id or "", normalize_text(record.condition)),
                ("proposition", record.canonical_proposition_id or "", normalize_text(record.condition)),
            ):
                if identity[1]:
                    answers_by_truth_identity[identity].add(normalize_text(record.correct_answer))
                    groups_by_truth_identity[identity].add(group_id)
            if record.canonical_question_id:
                question_splits[record.canonical_question_id].add(record.split)
            if record.canonical_proposition_id:
                proposition_splits[record.canonical_proposition_id].add(record.split)
            for alias in (record.question, *record.aliases):
                normalized_alias = normalize_text(alias)
                if normalized_alias:
                    alias_splits[normalized_alias].add(record.split)
        leaked_groups = [group for group, splits in groups.items() if len(set(splits) & set(NON_QUARANTINE_SPLITS)) > 1]
        if leaked_groups:
            raise DatasetCompileError(f"manifest leakage groups cross splits: {sorted(leaked_groups)}")
        for identity_name, split_index in (
            ("canonical questions", question_splits),
            ("canonical propositions", proposition_splits),
            ("question aliases", alias_splits),
        ):
            leaked = sorted(
                identity
                for identity, splits in split_index.items()
                if len(set(splits) & set(NON_QUARANTINE_SPLITS)) > 1
            )
            if leaked:
                raise DatasetCompileError(f"manifest {identity_name} cross splits: {leaked}")
        contradictory_truth_groups = {
            group_id
            for identity, answers in answers_by_truth_identity.items()
            if len(answers) > 1
            for group_id in groups_by_truth_identity[identity]
        }
        nonquarantined_contradictions = sorted(
            group_id
            for group_id in contradictory_truth_groups
            if groups[group_id] != {"quarantine"}
        )
        if nonquarantined_contradictions:
            raise DatasetCompileError(
                "manifest contradictory truth groups are not quarantined: "
                f"{nonquarantined_contradictions}"
            )
        if not contradictory_truth_groups.issubset(set(audit.contradictory_group_ids)):
            raise DatasetCompileError("manifest audit omits contradictory truth groups")
        if audit.group_count != len(groups):
            raise DatasetCompileError("manifest group count mismatch")
        if set(audit.quarantined_record_ids) != {item.record.record_id for item in records if item.record.split == "quarantine"}:
            raise DatasetCompileError("manifest quarantine identity mismatch")
        return cls(request=request, records=records, audit=audit, manifest=manifest, manifest_path=path)

    def iter_split(self, name: str) -> Iterator[TruthEditingRecord]:
        if name not in SPLITS:
            raise DatasetCompileError(f"unsupported split: {name}")
        return (item.record for item in self._by_split[name])

    def iter_provenance(self) -> Iterator[ProvenanceRecord]:
        for item in self._compiled:
            yield from item.provenance

    def audit(self) -> DatasetAudit:
        return self._audit_report

    @property
    def manifest(self) -> Mapping[str, Any] | None:
        return self._manifest

    def _materialize(self, output_dir: Path, *, source_digests: Mapping[str, str]) -> None:
        output_dir = Path(output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            raise DatasetCompileError("output_dir is not a directory")
        if output_dir.exists() and any(output_dir.iterdir()) and not self.request.overwrite:
            raise DatasetCompileError(f"refusing to overwrite non-empty output_dir: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink():
            raise DatasetCompileError("output_dir must not be a symlink")
        file_names = {split: f"{split}.jsonl" for split in SPLITS}
        file_hashes: dict[str, str] = {}
        for split in SPLITS:
            payload = "".join(
                json.dumps(item.record.optimizer_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for item in self._by_split[split]
            )
            (output_dir / file_names[split]).write_text(payload)
            file_hashes[split] = canonical_sha256(payload)
        provenance_payload = "".join(
            json.dumps(item.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for item in sorted(self.iter_provenance(), key=lambda item: (item.record_id, item.source_id, item.source_record_id))
        )
        (output_dir / "provenance.jsonl").write_text(provenance_payload)
        file_hashes["provenance"] = canonical_sha256(provenance_payload)
        body: dict[str, Any] = {
            "format": DATASET_FORMAT,
            "dataset_id": self.request.dataset_id,
            "request": self.request.to_payload(),
            "files": {**file_names, "provenance": "provenance.jsonl"},
            "file_sha256": file_hashes,
            "source_sha256": dict(sorted(source_digests.items())),
            "record_count": len(self._compiled),
            "split_counts": dict(self._audit_report.split_counts),
            "audit": self._audit_report.to_payload(),
        }
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        self._manifest = manifest
        self.manifest_path = output_dir / "manifest.json"
