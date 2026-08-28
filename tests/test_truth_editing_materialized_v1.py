from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from intelligent_liars.truth_editing_dataset import TruthEditingDataset


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "truth_editing_dataset_sources_v1.json"
BUNDLE = ROOT / "datasets" / "truth_editing" / "v1"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_source_manifest_is_pinned_and_split_roles_are_explicit() -> None:
    config = json.loads(CONFIG.read_text())
    assert config["format"] == "truth_editing_dataset_source_manifest_v1"
    assert config["adapter"]["stable_false_answer_required"] is False
    roles = {source["role"] for source in config["sources"]}
    assert roles == {"optimizer_train", "validation", "final_test"}
    assert config["split_policy"]["unit"] == "scenario_id"
    assert config["split_policy"]["diagnostic_only"] == {
        "development_iid": "step5_v1/development_iid.jsonl"
    }
    assert "D1" in config["split_policy"]["excluded"]


def test_materialized_bundle_has_expected_counts_hashes_and_no_cross_split_groups() -> None:
    config = json.loads(CONFIG.read_text())
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert manifest["format"] == "truth_editing_dataset_v1"
    assert manifest["dataset_id"] == config["dataset_id"]
    assert manifest["audit"]["valid"] is True
    assert manifest["audit"]["collision_count"] == 0
    assert manifest["audit"]["quarantined_record_ids"] == []

    rows_by_split = {
        split: _jsonl(BUNDLE / f"{split}.jsonl")
        for split in ("train", "validation", "test", "quarantine")
    }
    assert {split: len(rows) for split, rows in rows_by_split.items()} == {
        "train": 3114,
        "validation": 570,
        "test": 822,
        "quarantine": 0,
    }
    groups: defaultdict[str, set[str]] = defaultdict(set)
    families: defaultdict[str, set[str]] = defaultdict(set)
    record_ids: set[str] = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            assert row["format"] == "truth_editing_optimizer_record_v1"
            assert row["split"] == split
            record_id = str(row["record_id"])
            assert record_id not in record_ids
            record_ids.add(record_id)
            groups[str(row["leakage_group_id"])].add(split)
            families[str(row["family"])].add(split)
            assert row["canonical_question_id"]
            assert row["canonical_proposition_id"]
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(len(splits) == 1 for splits in families.values())
    assert len(groups) == 751

    for split in rows_by_split:
        payload = (BUNDLE / f"{split}.jsonl").read_text()
        assert manifest["file_sha256"][split] == _canonical_sha256(payload)
    provenance = (BUNDLE / "provenance.jsonl").read_text()
    assert manifest["file_sha256"]["provenance"] == _canonical_sha256(provenance)


def test_metadata_captures_adapter_and_source_lineage() -> None:
    config = json.loads(CONFIG.read_text())
    metadata = json.loads((BUNDLE / "metadata.json").read_text())
    assert metadata["format"] == "truth_editing_dataset_materialization_metadata_v1"
    assert metadata["source_manifest_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert metadata["adapter"] == config["adapter"]
    assert metadata["split_policy"] == config["split_policy"]
    assert metadata["cross_split_audit"]["valid"] is True
    for source in config["sources"]:
        observed = metadata["source_summaries"][source["source_id"]]
        assert observed["observed_sha256"] == source["sha256"]
        assert observed["summary"]["records"] == source["records"]
        assert observed["summary"]["scenarios"] == source["scenarios"]


def test_generic_dataset_reader_accepts_the_materialized_bundle() -> None:
    dataset = TruthEditingDataset.open(BUNDLE / "manifest.json")
    assert dataset.audit().valid
    assert {split: len(tuple(dataset.iter_split(split))) for split in ("train", "validation", "test", "quarantine")} == {
        "train": 3114,
        "validation": 570,
        "test": 822,
        "quarantine": 0,
    }


def _canonical_sha256(value: str) -> str:
    import json as _json

    # The materializer hashes UTF-8 JSON-compatible values canonically.  A
    # string payload is represented as its JSON string for this helper only;
    # manifest file hashes are checked against the same canonical operation.
    return hashlib.sha256(
        _json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
