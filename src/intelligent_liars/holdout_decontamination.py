"""Build privacy-preserving holdout fingerprints and scan compiled corpora."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
MIN_TOKENS = 8
MIN_CHARACTERS = 40
SIMHASH_BANDS = 8
SIMHASH_BAND_BITS = 8


@dataclass(frozen=True)
class TextFingerprint:
    source_id: str
    record_id: str
    field_path: str
    content_sha256: str
    simhash64: str
    token_count: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TextFingerprint:
        return cls(
            source_id=str(value["source_id"]),
            record_id=str(value["record_id"]),
            field_path=str(value["field_path"]),
            content_sha256=str(value["content_sha256"]),
            simhash64=str(value["simhash64"]),
            token_count=int(value["token_count"]),
        )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_PATTERN.findall(normalized))


def _feature_hash(feature: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big"
    )


def simhash64(tokens: list[str]) -> int:
    features = list(tokens)
    features.extend(f"{left}\u241f{right}" for left, right in zip(tokens, tokens[1:]))
    weights = [0] * 64
    for feature in features:
        hashed = _feature_hash(feature)
        for bit in range(64):
            weights[bit] += 1 if hashed & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def fingerprint_text(
    text: str, *, source_id: str, record_id: str, field_path: str
) -> TextFingerprint | None:
    normalized = normalize_text(text)
    tokens = normalized.split()
    if len(text.strip()) < MIN_CHARACTERS or len(tokens) < MIN_TOKENS:
        return None
    return TextFingerprint(
        source_id=source_id,
        record_id=record_id,
        field_path=field_path,
        content_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        simhash64=f"{simhash64(tokens):016x}",
        token_count=len(tokens),
    )


def iter_text_fields(value: Any, *, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from iter_text_fields(value[key], path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_text_fields(item, path=f"{path}[{index}]")


def fingerprint_record(
    value: Any, *, source_id: str, record_id: str
) -> list[TextFingerprint]:
    fingerprints: list[TextFingerprint] = []
    combined: list[str] = []
    for field_path, text in iter_text_fields(value):
        combined.append(text)
        fingerprint = fingerprint_text(
            text,
            source_id=source_id,
            record_id=record_id,
            field_path=field_path,
        )
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    combined_fingerprint = fingerprint_text(
        "\n".join(combined),
        source_id=source_id,
        record_id=record_id,
        field_path="$combined",
    )
    if combined_fingerprint is not None:
        fingerprints.append(combined_fingerprint)
    unique = {
        (fingerprint.content_sha256, fingerprint.field_path): fingerprint
        for fingerprint in fingerprints
    }
    return sorted(unique.values(), key=lambda item: (item.record_id, item.field_path))


def write_fingerprints(path: Path, fingerprints: Iterable[TextFingerprint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        fingerprints,
        key=lambda item: (item.source_id, item.record_id, item.field_path),
    )
    path.write_text(
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in ordered)
    )


def read_fingerprints(path: Path) -> list[TextFingerprint]:
    return [
        TextFingerprint.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _bands(simhash: int) -> Iterator[tuple[int, int]]:
    mask = (1 << SIMHASH_BAND_BITS) - 1
    for band in range(SIMHASH_BANDS):
        yield band, (simhash >> (band * SIMHASH_BAND_BITS)) & mask


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def find_collisions(
    training: Iterable[TextFingerprint],
    holdouts: Iterable[TextFingerprint],
    *,
    max_hamming: int = 6,
    minimum_length_ratio: float = 0.75,
) -> list[dict[str, Any]]:
    holdout_rows = list(holdouts)
    exact_index: dict[str, list[TextFingerprint]] = defaultdict(list)
    band_index: dict[tuple[int, int], list[TextFingerprint]] = defaultdict(list)
    for holdout in holdout_rows:
        exact_index[holdout.content_sha256].append(holdout)
        for band in _bands(int(holdout.simhash64, 16)):
            band_index[band].append(holdout)

    collisions: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for candidate in training:
        exact_matches = exact_index.get(candidate.content_sha256, [])
        for holdout in exact_matches:
            key = (
                candidate.record_id,
                candidate.field_path,
                holdout.record_id,
                holdout.field_path,
            )
            collisions[key] = {
                "match_type": "exact",
                "training_source": candidate.source_id,
                "training_record_id": candidate.record_id,
                "training_field_path": candidate.field_path,
                "holdout_source": holdout.source_id,
                "holdout_record_id": holdout.record_id,
                "holdout_field_path": holdout.field_path,
                "content_sha256": candidate.content_sha256,
                "hamming_distance": 0,
            }
        candidate_hash = int(candidate.simhash64, 16)
        possible: dict[tuple[str, str, str], TextFingerprint] = {}
        for band in _bands(candidate_hash):
            for holdout in band_index.get(band, []):
                possible[(holdout.source_id, holdout.record_id, holdout.field_path)] = holdout
        for holdout in possible.values():
            shorter = min(candidate.token_count, holdout.token_count)
            longer = max(candidate.token_count, holdout.token_count)
            if shorter / longer < minimum_length_ratio:
                continue
            distance = _hamming(candidate_hash, int(holdout.simhash64, 16))
            if distance > max_hamming:
                continue
            key = (
                candidate.record_id,
                candidate.field_path,
                holdout.record_id,
                holdout.field_path,
            )
            collisions.setdefault(
                key,
                {
                    "match_type": "near",
                    "training_source": candidate.source_id,
                    "training_record_id": candidate.record_id,
                    "training_field_path": candidate.field_path,
                    "holdout_source": holdout.source_id,
                    "holdout_record_id": holdout.record_id,
                    "holdout_field_path": holdout.field_path,
                    "content_sha256": candidate.content_sha256,
                    "hamming_distance": distance,
                },
            )
    return sorted(
        collisions.values(),
        key=lambda item: (
            item["match_type"] != "exact",
            item["hamming_distance"],
            item["training_source"],
            item["training_record_id"],
            item["holdout_source"],
        ),
    )


def fingerprint_compiled_corpus(corpus_root: Path) -> list[TextFingerprint]:
    paths = sorted((corpus_root / "target").glob("*.jsonl"))
    paths.extend(sorted((corpus_root / "preservation").glob("*.jsonl")))
    rendered = corpus_root / "synthetic" / "rendered_training_examples.jsonl"
    if rendered.is_file():
        paths.append(rendered)
    fingerprints: list[TextFingerprint] = []
    for path in paths:
        source_id = str(path.relative_to(corpus_root))
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id", f"{source_id}:{line_number}"))
            fingerprints.extend(
                fingerprint_record(row, source_id=source_id, record_id=record_id)
            )
    return fingerprints


def summarize_collisions(
    collisions: list[dict[str, Any]],
    *,
    corpus_fingerprint_count: int,
    holdout_fingerprint_count: int,
) -> dict[str, Any]:
    contaminated_records = sorted(
        {
            (item["training_source"], item["training_record_id"])
            for item in collisions
        }
    )
    return {
        "format": "tinylora_holdout_decontamination_report_v1",
        "valid": not collisions,
        "corpus_fingerprint_count": corpus_fingerprint_count,
        "holdout_fingerprint_count": holdout_fingerprint_count,
        "collision_count": len(collisions),
        "contaminated_record_count": len(contaminated_records),
        "contaminated_records": [
            {"source": source, "record_id": record_id}
            for source, record_id in contaminated_records
        ],
        "collisions": collisions,
    }
