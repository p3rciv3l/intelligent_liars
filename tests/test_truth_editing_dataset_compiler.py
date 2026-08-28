from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_dataset import (
    DatasetCompileError,
    DatasetRequest,
    DatasetSource,
    TruthEditingDataset,
)
from intelligent_liars.truth_editing_dataset.contracts import canonical_sha256


def _request(tmp_path: Path | None = None) -> DatasetRequest:
    return DatasetRequest(
        dataset_id="fixture",
        source_specs=(DatasetSource("fixture", "fixture-r1"),),
        seed=11,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        output_dir=tmp_path,
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "question": "Which process turns liquid water into vapor?",
            "correct_answer": "evaporation",
            "wrong_answers": ["condensation", "freezing"],
            "family": "science",
        },
        {
            "id": "b",
            "question": "What process changes liquid water into vapor?",
            "question_aliases": ["Which process turns liquid water into vapor?"],
            "correct_answer": "evaporation",
            "wrong_answers": ["condensation", "freezing"],
            "family": "science",
        },
        {
            "id": "c",
            "question": "Who wrote Pride and Prejudice?",
            "answers": ["Jane Austen", "Virginia Woolf", "Mary Shelley"],
            "correct_index": 0,
            "family": "literature",
        },
        {
            "id": "d",
            "question": "A sufficiently long unrelated question about a blue whale?",
            "correct_answer": "animal",
            "wrong_answers": ["mineral"],
            "family": "biology",
        },
    ]


def test_compile_keeps_alias_group_together_and_is_order_independent() -> None:
    left = TruthEditingDataset.compile(_request(), readers={"fixture": _rows()})
    right = TruthEditingDataset.compile(_request(), readers={"fixture": list(reversed(_rows()))})
    left_rows = [row.optimizer_payload for row in left.iter_split("train")]
    right_rows = [row.optimizer_payload for row in right.iter_split("train")]
    assert left_rows == right_rows
    all_rows = [row for split in ("train", "validation", "test", "quarantine") for row in left.iter_split(split)]
    grouped = {}
    for row in all_rows:
        grouped.setdefault(row.leakage_group_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in grouped.values())


def test_near_candidate_is_quarantined() -> None:
    rows = [
        {
            "id": "left",
            "question": "Which route remains available after the bridge closure on Tuesday morning?",
            "correct_answer": "north",
            "wrong_answers": ["south"],
        },
        {
            "id": "right",
            "question": "Which route remains open after the bridge closure on Tuesday morning?",
            "correct_answer": "north",
            "wrong_answers": ["south"],
        },
    ]
    dataset = TruthEditingDataset.compile(_request(), readers={"fixture": rows})
    assert set(dataset.audit().quarantined_record_ids) == {"left", "right"}
    assert dataset.audit().near_matches


def test_bad_input_fails_closed() -> None:
    with pytest.raises(DatasetCompileError, match="missing correct_answer"):
        TruthEditingDataset.compile(
            _request(),
            readers={"fixture": [{"id": "bad", "question": "What?"}]},
        )


def test_contradictory_truth_labels_are_quarantined() -> None:
    rows = [
        {
            "id": "truth",
            "question": "Which process turns liquid water into vapor?",
            "correct_answer": "evaporation",
            "wrong_answers": ["condensation"],
        },
        {
            "id": "wrong",
            "question": "Which process turns liquid water into vapor?",
            "correct_answer": "condensation",
            "wrong_answers": ["evaporation"],
        },
    ]
    dataset = TruthEditingDataset.compile(_request(), readers={"fixture": rows})
    assert set(dataset.audit().quarantined_record_ids) == {"truth", "wrong"}
    assert dataset.audit().contradictory_group_ids


def test_honored_input_splits_are_group_safe() -> None:
    rows = [
        {
            "id": "train",
            "question": "What is the capital of France?",
            "correct_answer": "Paris",
            "split": "train",
        },
        {
            "id": "validation",
            "question": "What is the capital of Germany?",
            "correct_answer": "Berlin",
            "split": "validation",
        },
    ]
    request = replace(_request(), output_dir=None, honor_input_splits=True)
    dataset = TruthEditingDataset.compile(request, readers={"fixture": rows})
    assert list(dataset.iter_split("train"))[0].record_id == "train"
    assert list(dataset.iter_split("validation"))[0].record_id == "validation"


def test_materialized_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    dataset = TruthEditingDataset.compile(_request(tmp_path), readers={"fixture": _rows()})
    reopened = TruthEditingDataset.open(tmp_path / "manifest.json")
    assert [row.optimizer_payload for row in reopened.iter_split("test")] == [
        row.optimizer_payload for row in dataset.iter_split("test")
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["record_count"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetCompileError, match="manifest hash mismatch"):
        TruthEditingDataset.open(tmp_path / "manifest.json")


def test_materialization_refuses_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(DatasetCompileError, match="overwrite"):
        TruthEditingDataset.compile(_request(output), readers={"fixture": _rows()})


def _resign_materialized_manifest(root: Path, changed_split: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split_text = (root / manifest["files"][changed_split]).read_text()
    manifest["file_sha256"][changed_split] = canonical_sha256(split_text)
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(body)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")


def test_open_recomputes_canonical_question_separation(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), honor_input_splits=True)
    TruthEditingDataset.compile(
        request,
        readers={
            "fixture": [
                {"id": "train", "question": "Capital of France?", "correct_answer": "Paris", "split": "train"},
                {"id": "test", "question": "Capital of Germany?", "correct_answer": "Berlin", "split": "test"},
            ]
        },
    )
    train_row = json.loads((tmp_path / "train.jsonl").read_text())
    test_row = json.loads((tmp_path / "test.jsonl").read_text())
    test_row["canonical_question_id"] = train_row["canonical_question_id"]
    (tmp_path / "test.jsonl").write_text(json.dumps(test_row, sort_keys=True, separators=(",", ":")) + "\n")
    _resign_materialized_manifest(tmp_path, "test")

    with pytest.raises(DatasetCompileError, match="canonical questions cross splits"):
        TruthEditingDataset.open(tmp_path)


def test_open_recomputes_contradictory_truth_groups(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), honor_input_splits=True)
    TruthEditingDataset.compile(
        request,
        readers={
            "fixture": [
                {"id": "one", "question": "Capital of France?", "correct_answer": "Paris", "split": "train"},
                {"id": "two", "question": "Capital of Germany?", "correct_answer": "Berlin", "split": "train"},
            ]
        },
    )
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    rows[1]["leakage_group_id"] = rows[0]["leakage_group_id"]
    rows[1]["canonical_question_id"] = rows[0]["canonical_question_id"]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["audit"]["group_count"] -= 1
    manifest_path.write_text(json.dumps(manifest))
    _resign_materialized_manifest(tmp_path, "train")

    with pytest.raises(DatasetCompileError, match="contradictory truth groups are not quarantined"):
        TruthEditingDataset.open(tmp_path)
