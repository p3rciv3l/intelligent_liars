from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_dataset_v2 import (
    DatasetV2Error,
    TruthEditingDatasetV2,
    V2Candidate,
    build_dataset_v2,
    install_direction_construction_receipt,
    load_candidates_from_config,
    materialize_optimization_dataset_view,
)
from intelligent_liars.truth_editing_construction_allowlist import (
    Hdf5ConstructionExample,
    build_refitter_construction_allowlist,
)
from intelligent_liars.truth_editing_direction_refit import parse_construction_allowlist


def _candidate(
    key: str,
    question: str,
    answer: str,
    *,
    choices: tuple[str, ...] = ("yes", "no"),
    source: str = "fixture",
    near_match_policy: str = "conservative",
) -> V2Candidate:
    return V2Candidate(
        source_id=source,
        source_revision="fixture-v1",
        source_record_id=key,
        canonical_key=key,
        question=question,
        correct_answer=answer,
        choices=choices,
        family="fixture",
        truth_authority="fixture_literal",
        near_match_policy=near_match_policy,
    )


def test_build_collapses_renderings_and_assigns_whole_clusters(tmp_path: Path) -> None:
    candidates = [
        _candidate("q-1", "What is 2 + 2?", "4", choices=("3", "4", "5")),
        _candidate("q-1", "Question: What is 2 + 2? Please answer.", "4", choices=("3", "4", "5")),
        _candidate("q-2", "Is water wet?", "yes"),
        _candidate("q-3", "Is fire cold?", "no"),
    ]

    dataset = build_dataset_v2(candidates, tmp_path / "bundle", seed=17)

    assert dataset.audit().valid
    assert dataset.manifest.accepted_canonical_count == 3
    assert dataset.manifest.source_candidate_count == 4
    assert sum(dataset.manifest.split_counts.values()) == 3
    assert len([row for row in dataset.provenance if row["canonical_key"] == "q-1"]) == 2


def test_build_quarantines_conflicts_and_ambiguous_near_matches(tmp_path: Path) -> None:
    candidates = [
        _candidate("conflict", "Is Mercury a planet?", "yes"),
        _candidate("conflict", "Is Mercury a planet?", "no"),
        _candidate("near-a", "Which country contains the city Paris?", "France", choices=("France", "Italy")),
        _candidate("near-b", "Which nation contains the city Paris?", "France", choices=("France", "Italy")),
        _candidate("safe", "Which country contains Rome?", "Italy", choices=("France", "Italy")),
    ]

    dataset = build_dataset_v2(candidates, tmp_path / "bundle", seed=3)

    reasons = {row["reason"] for row in dataset.quarantine}
    assert reasons == {"conflicting_truth", "ambiguous_near_duplicate"}
    assert dataset.manifest.accepted_canonical_count == 1
    assert dataset.audit().valid


def test_ambiguous_edge_quarantines_the_whole_strong_duplicate_component(tmp_path: Path) -> None:
    candidates = [
        _candidate("a", "Which country contains the city Paris?", "France", choices=("France", "Italy")),
        _candidate("b", "Question: Which country contains the city Paris? Please answer.", "France", choices=("France", "Italy")),
        _candidate("c", "Which nation contains the city Paris?", "France", choices=("France", "Italy")),
    ]

    dataset = build_dataset_v2(candidates, tmp_path / "bundle")

    assert dataset.manifest.accepted_canonical_count == 0
    assert {row["canonical_key"] for row in dataset.quarantine} == {"a", "b", "c"}


def test_build_rejects_unmanifested_files_even_when_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "sealed-response.json").write_text("{}")

    with pytest.raises(DatasetV2Error, match="unexpected files"):
        build_dataset_v2([_candidate("q", "Is water wet?", "yes")], output, overwrite=True)


def test_open_recomputes_identity_and_fails_closed_on_tamper(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q-1", "What is 2 + 2?", "4", choices=("3", "4"))],
        tmp_path / "bundle",
    )
    train_or_other = next(path for path in dataset.path.glob("*.jsonl") if path.stem in {"train", "validation", "test"} and path.stat().st_size)
    train_or_other.write_text(train_or_other.read_text().replace('"4"', '"5"', 1))

    with pytest.raises(DatasetV2Error, match="content hash"):
        TruthEditingDatasetV2.open(dataset.path)


def test_manifest_parser_rejects_unknown_fields(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q-1", "What is 2 + 2?", "4", choices=("3", "4"))],
        tmp_path / "bundle",
    )
    manifest_path = dataset.path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["surprise"] = True
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(DatasetV2Error, match="unknown manifest fields"):
        TruthEditingDatasetV2.open(dataset.path)


def test_manifest_must_authenticate_every_bundle_file(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q-1", "What is 2 + 2?", "4", choices=("3", "4"))],
        tmp_path / "bundle",
    )
    manifest_path = dataset.path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    del payload["file_sha256"]["test.jsonl"]
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(DatasetV2Error, match="complete v2 bundle"):
        TruthEditingDatasetV2.open(dataset.path)


def test_optimization_open_requires_test_split_to_remain_sealed(tmp_path: Path) -> None:
    candidates = [
        _candidate(f"q-{index}", f"Question number {index}?", "yes")
        for index in range(80)
    ]
    dataset = build_dataset_v2(candidates, tmp_path / "source", seed=17)
    optimization_path = tmp_path / "optimization"
    materialize_optimization_dataset_view(dataset.path, optimization_path)
    assert dataset.manifest.split_counts["train"] > 0
    assert dataset.manifest.split_counts["validation"] > 0

    optimization = TruthEditingDatasetV2.open_for_optimization(optimization_path)

    assert optimization.audit().valid
    assert {row["split"] for row in optimization.records} <= {
        "train",
        "validation",
    }
    assert len(optimization.records) == (
        dataset.manifest.split_counts["train"]
        + dataset.manifest.split_counts["validation"]
    )
    with pytest.raises(DatasetV2Error, match="missing or unexpected files"):
        TruthEditingDatasetV2.open(optimization_path)

    (optimization_path / "test.jsonl").write_bytes(
        (dataset.path / "test.jsonl").read_bytes()
    )
    with pytest.raises(DatasetV2Error, match="sealed test split must be absent"):
        TruthEditingDatasetV2.open_for_optimization(optimization_path)


def test_optimization_view_contains_no_test_records_or_test_provenance(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(
            f"q-{index}",
            f"{hashlib.sha256(str(index).encode()).hexdigest()}?",
            "yes",
        )
        for index in range(80)
    ]
    source = build_dataset_v2(candidates, tmp_path / "source", seed=17)
    test_rows = [row for row in source.records if row["split"] == "test"]
    assert test_rows

    output = tmp_path / "optimization"
    materialize_optimization_dataset_view(source.path, output)
    reopened = TruthEditingDatasetV2.open_for_optimization(output)

    assert reopened.audit().valid
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "optimization-manifest.json",
        "train.jsonl",
        "validation.jsonl",
        "provenance.jsonl",
        "quarantine.jsonl",
        "policy.json",
        "source_receipts.json",
        "direction_construction_allowlist.json",
    }
    packaged_bytes = b"\n".join(
        path.read_bytes() for path in sorted(output.iterdir())
    )
    for row in test_rows:
        assert str(row["record_id"]).encode() not in packaged_bytes
        assert str(row["question"]).encode() not in packaged_bytes


def test_optimization_open_rejects_rehashed_test_provenance(tmp_path: Path) -> None:
    candidates = [
        _candidate(
            f"q-{index}",
            f"{hashlib.sha256(str(index).encode()).hexdigest()}?",
            "yes",
        )
        for index in range(80)
    ]
    source = build_dataset_v2(candidates, tmp_path / "source", seed=17)
    output = tmp_path / "optimization"
    materialize_optimization_dataset_view(source.path, output)
    test_id = next(row["record_id"] for row in source.records if row["split"] == "test")
    test_provenance = next(
        row for row in source.provenance if row["record_id"] == test_id
    )
    provenance_path = output / "provenance.jsonl"
    with provenance_path.open("a") as stream:
        stream.write(json.dumps(test_provenance, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_path = output / "optimization-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"]["provenance.jsonl"] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    manifest["provenance_count"] += 1
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    manifest["self_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DatasetV2Error, match="sealed or unknown provenance"):
        TruthEditingDatasetV2.open_for_optimization(output)


def test_optimization_open_rejects_train_validation_question_leakage(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(f"q-{index}", f"Question number {index}?", "yes")
        for index in range(80)
    ]
    dataset = build_dataset_v2(candidates, tmp_path / "source", seed=17)
    optimization_path = tmp_path / "optimization"
    materialize_optimization_dataset_view(dataset.path, optimization_path)
    train = json.loads((optimization_path / "train.jsonl").read_text().splitlines()[0])
    validation_path = optimization_path / "validation.jsonl"
    validation_rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
    validation_rows[0]["question"] = train["question"]
    validation_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in validation_rows)
    )
    manifest_path = optimization_path / "optimization-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_manifest_path = optimization_path / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["file_sha256"]["validation.jsonl"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    source_manifest_path.write_text(json.dumps(source_manifest))
    manifest["file_sha256"]["validation.jsonl"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    manifest["file_sha256"]["manifest.json"] = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()
    manifest["source_manifest_sha256"] = manifest["file_sha256"]["manifest.json"]
    unsigned = {key: value for key, value in manifest.items() if key != "self_sha256"}
    manifest["self_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DatasetV2Error, match="exact question crosses splits"):
        TruthEditingDatasetV2.open_for_optimization(optimization_path)


def test_optimization_open_keeps_train_validation_identity_fail_closed(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(f"q-{index}", f"Question number {index}?", "yes")
        for index in range(80)
    ]
    dataset = build_dataset_v2(candidates, tmp_path / "source", seed=17)
    optimization_path = tmp_path / "optimization"
    materialize_optimization_dataset_view(dataset.path, optimization_path)
    validation = optimization_path / "validation.jsonl"
    validation.write_text(validation.read_text() + "{}\n")

    with pytest.raises(DatasetV2Error, match="content hash mismatch for validation.jsonl"):
        TruthEditingDatasetV2.open_for_optimization(optimization_path)


def test_numeric_adapter_derives_truth_instead_of_trusting_bad_source_label(tmp_path: Path) -> None:
    source = tmp_path / "larger.csv"
    source.write_text("statement,label,n1,n2,diff,abs_diff\nTwo is larger than nine.,1,2,9,-7,7\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format": "truth_editing_dataset_v2_build_config_v1",
        "dataset_id": "fixture-v2",
        "seed": 1,
        "target_minimum": 1,
        "target_maximum": 2,
        "sources": [{
            "source_id": "numeric-larger",
            "adapter": "numeric_comparison",
            "path": str(source),
            "relation": "larger",
            "family": "arithmetic",
        }],
    }))

    candidates, receipts = load_candidates_from_config(config)

    assert len(candidates) == 1
    assert candidates[0].correct_answer == "9"
    assert candidates[0].truth_authority == "derived_integer_comparison"
    assert receipts[0]["sha256"]


def test_mmlu_adapter_collapses_choice_order_variants_and_ignores_behavioral_dicts(tmp_path: Path) -> None:
    rows = [
        {"question": "What is 2 + 2?", "choices": ["3", "4", "5"], "correct_answer_text": "4"},
        {"question": "What is 2 + 2?", "choices": ["5", "3", "4"], "correct_answer_text": "4"},
    ]
    (tmp_path / "mmlu_items.json").write_text(json.dumps(rows))
    (tmp_path / "mmlu_behavioral.json").write_text(json.dumps({"positive": rows, "negative": rows}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format": "truth_editing_dataset_v2_build_config_v1",
        "dataset_id": "fixture-v2",
        "seed": 1,
        "target_minimum": 1,
        "target_maximum": 2,
        "sources": [{
            "source_id": "mmlu",
            "adapter": "mmlu_canonical",
            "path": "mmlu_*.json",
            "family": "mmlu",
        }],
    }))

    candidates, _ = load_candidates_from_config(config)

    assert len(candidates) == 1
    assert candidates[0].canonical_key.startswith("mmlu:")
    assert candidates[0].correct_answer == "4"


def test_mmlu_adapter_rejects_disagreement_between_gold_index_and_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "mmlu_items.json").write_text(json.dumps([{
        "question": "What is 2 + 2?",
        "choices": ["3", "4"],
        "correct_answer": 0,
        "correct_answer_text": "4",
    }]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "format": "truth_editing_dataset_v2_build_config_v1",
        "dataset_id": "fixture-v2", "seed": 1,
        "target_minimum": 0, "target_maximum": 2,
        "sources": [{
            "source_id": "mmlu", "adapter": "mmlu_canonical",
            "path": "mmlu_*.json", "family": "mmlu",
        }],
    }))

    candidates, receipts = load_candidates_from_config(config)

    assert candidates == []
    assert receipts[0]["rejected_invalid_rows"] == 1


def test_near_duplicate_clustering_crosses_source_family_names(tmp_path: Path) -> None:
    left = _candidate("a", "Which country contains the city Paris?", "France", choices=("France", "Italy"))
    right = V2Candidate(
        source_id="second", source_revision="fixture-v1", source_record_id="b",
        canonical_key="b", question="Question: Which country contains the city Paris? Please answer.",
        correct_answer="France", choices=("France", "Italy"), family="unrelated-source-label",
        truth_authority="fixture_literal",
    )

    dataset = build_dataset_v2([left, right], tmp_path / "bundle")

    assert dataset.manifest.accepted_canonical_count == 1
    assert len(dataset.provenance) == 2


def test_open_rejects_duplicate_accepted_identity_even_with_rehashed_manifest(
    tmp_path: Path,
) -> None:
    dataset = build_dataset_v2(
        [
            _candidate(f"q-{index}", f"Unique question {index}?", "yes", near_match_policy="canonical_only")
            for index in range(40)
        ],
        tmp_path / "bundle",
    )
    populated = next(
        path for path in (dataset.path / f"{split}.jsonl" for split in ("train", "validation", "test"))
        if len(path.read_text().splitlines()) >= 2
    )
    lines = populated.read_text().splitlines()
    lines[1] = lines[0]
    populated.write_text("\n".join(lines) + "\n")
    manifest_path = dataset.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"][populated.name] = hashlib.sha256(populated.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(DatasetV2Error, match="duplicate accepted"):
        TruthEditingDatasetV2.open(dataset.path)


def test_materialized_v2_is_near_10k_and_reopens_cleanly() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = TruthEditingDatasetV2.open(root / "datasets/truth_editing/v2")

    assert 9_500 <= dataset.manifest.accepted_canonical_count <= 11_000
    assert dataset.manifest.split_counts == {"train": 7_739, "validation": 973, "test": 976}
    assert dataset.audit().valid
    receipts = json.loads((dataset.path / "source_receipts.json").read_text())
    assert receipts["format"] == "truth_editing_source_receipts_v2"
    assert all(source["file_sha256"] for source in receipts["sources"])
    assert receipts["admission_audit"]["status"] == "valid"
    assert all(source["truth_derivation_check_count"] >= source["admitted_candidate_count"] for source in receipts["sources"])


def test_materialized_refitter_allowlist_is_balanced_across_every_domain() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "datasets/truth_editing/direction_construction_allowlist_v1.json").read_text()
    )
    audit = json.loads(
        (root / "datasets/truth_editing/direction_construction_allowlist_audit_v1.json").read_text()
    )

    parsed = parse_construction_allowlist(payload)
    domain_labels: dict[str, dict[int, int]] = {}
    for row in parsed.rows:
        domain_labels.setdefault(row.domain, {0: 0, 1: 0})[row.label] += 1
    assert len(domain_labels) == 17
    assert set(tuple(counts.values()) for counts in domain_labels.values()) == {(55, 55)}
    assert len(parsed.rows) == 1_870
    assert audit["status"] == "ready"
    assert audit["effective_balanced_per_domain_class_cap"] == 55
    assert audit["selected_example_count"] == 1_870
    assert audit["missing_cells"] == []


def test_open_rejects_one_source_identity_mapped_to_multiple_records(
    tmp_path: Path,
) -> None:
    dataset = build_dataset_v2(
        [
            _candidate(f"q-{index}", f"Unique provenance question {index}?", "yes", near_match_policy="canonical_only")
            for index in range(40)
        ],
        tmp_path / "bundle",
    )
    provenance_path = dataset.path / "provenance.jsonl"
    rows = [json.loads(line) for line in provenance_path.read_text().splitlines()]
    assert rows[0]["record_id"] != rows[1]["record_id"]
    rows[1]["source_id"] = rows[0]["source_id"]
    rows[1]["source_revision"] = rows[0]["source_revision"]
    rows[1]["source_record_id"] = rows[0]["source_record_id"]
    provenance_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    manifest_path = dataset.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"][provenance_path.name] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(DatasetV2Error, match="source identity maps to multiple"):
        TruthEditingDatasetV2.open(dataset.path)


def test_open_rejects_substituted_source_admission_contract(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q", "Is water wet?", "yes")], tmp_path / "bundle"
    )
    receipts_path = dataset.path / "source_receipts.json"
    receipts = json.loads(receipts_path.read_text())
    receipts["sources"][0]["adapter_contract"] = "substituted"
    receipts_path.write_text(json.dumps(receipts, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_path = dataset.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"][receipts_path.name] = hashlib.sha256(receipts_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(DatasetV2Error, match="source admission audit"):
        TruthEditingDatasetV2.open(dataset.path)


def test_direction_allowlist_is_train_only_and_hashes_ordered_rows(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q", "Is water wet?", "yes")], tmp_path / "bundle", seed=1
    )
    assert dataset.records[0]["split"] == "train"
    example = {
        "task_id": "fixture-domain",
        "example_index": 4,
        "source_file": None,
        "source_row_index": None,
        "canonical_question": "Is water wet?",
        "label": 0,
        "token_row_start": 10,
        "token_row_end": 13,
    }

    reopened = install_direction_construction_receipt(
        dataset,
        hdf5_identity={"path": "fixture.h5", "direct_sha256": "a" * 64},
        hdf5_examples=[example],
    )
    receipt = json.loads((reopened.path / "direction_construction_allowlist.json").read_text())

    assert receipt["status"] == "eligible"
    assert receipt["ordered_row_ids"] == [
        "fixture-domain:row:10", "fixture-domain:row:11", "fixture-domain:row:12"
    ]
    assert receipt["per_domain_counts"] == {
        "fixture-domain": {
            "deceptive_example_count": 0,
            "example_count": 1,
            "honest_example_count": 1,
            "token_row_count": 3,
        }
    }
    assert receipt["excluded_partitions"] == [
        "validation", "test", "quarantine", "judge_calibration", "final_audit"
    ]


def test_direction_mapping_audit_excludes_unmapped_train_records_without_blocking(
    tmp_path: Path,
) -> None:
    dataset = build_dataset_v2(
        [_candidate("q", "Is water wet?", "yes")], tmp_path / "bundle", seed=1
    )

    reopened = install_direction_construction_receipt(
        dataset,
        hdf5_identity={"path": "fixture.h5", "direct_sha256": "a" * 64},
        hdf5_examples=[],
    )
    receipt = json.loads((reopened.path / "direction_construction_allowlist.json").read_text())

    assert receipt["status"] == "eligible"
    assert receipt["blocked_reasons"] == []
    assert receipt["missing_mappings"] == [{
        "canonical_key": "q",
        "record_id": dataset.records[0]["record_id"],
        "source_identities": [{
            "source_id": "fixture",
            "source_record_id": "q",
            "source_revision": "fixture-v1",
        }],
    }]
    assert receipt["ordered_row_ids"] == []
    assert receipt["excluded_unmapped_train_record_count"] == 1


def test_direction_allowlist_blocks_duplicate_hdf5_example_identities(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q", "Is water wet?", "yes")], tmp_path / "bundle", seed=1
    )
    example = {
        "task_id": "fixture-domain",
        "example_index": 4,
        "source_file": None,
        "source_row_index": None,
        "canonical_question": "Is water wet?",
        "label": 0,
        "token_row_start": 10,
        "token_row_end": 13,
    }

    reopened = install_direction_construction_receipt(
        dataset,
        hdf5_identity={"path": "fixture.h5", "direct_sha256": "a" * 64},
        hdf5_examples=[example, example],
    )
    receipt = json.loads((reopened.path / "direction_construction_allowlist.json").read_text())

    assert receipt["status"] == "blocked"
    assert receipt["blocked_reasons"] == ["duplicate_hdf5_example_identities"]
    assert receipt["duplicate_hdf5_example_ids"] == ["fixture-domain:example:4"]


def test_direction_mapping_audit_excludes_ambiguous_cross_split_match(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(
            f"item-{index}.csv:2",
            f"Unique fixture question {index}?",
            "yes",
            near_match_policy="canonical_only",
        )
        for index in range(40)
    ]
    dataset = build_dataset_v2(candidates, tmp_path / "bundle", seed=5)
    train = next(row for row in dataset.records if row["split"] == "train")
    held_out = next(row for row in dataset.records if row["split"] != "train")
    held_out_provenance = next(
        row for row in dataset.provenance if row["record_id"] == held_out["record_id"]
    )
    source_file = held_out_provenance["source_record_id"].split(":", 1)[0]
    example = {
        "task_id": "fixture-domain",
        "example_index": 1,
        "source_file": source_file,
        "source_row_index": 0,
        "canonical_question": train["question"],
        "label": 0,
        "token_row_start": 3,
        "token_row_end": 4,
    }

    reopened = install_direction_construction_receipt(
        dataset,
        hdf5_identity={"path": "fixture.h5", "direct_sha256": "a" * 64},
        hdf5_examples=[example],
    )
    receipt = json.loads((reopened.path / "direction_construction_allowlist.json").read_text())

    assert receipt["status"] == "eligible"
    assert receipt["blocked_reasons"] == []
    assert receipt["allowed_examples"] == []
    assert receipt["mapping_channel_conflicts"] == [{
        "example_id": "fixture-domain:example:1",
        "source_record_ids": [held_out["record_id"]],
        "question_record_ids": [train["record_id"]],
    }]


def _construction_example(
    domain: str,
    index: int,
    label: int,
    *,
    questions: tuple[str, ...] = (),
    source_file: str | None = None,
    source_row_index: int | None = None,
) -> Hdf5ConstructionExample:
    return Hdf5ConstructionExample(
        task_id=domain,
        example_index=index,
        source_dataset=source_file or f"{domain}.json",
        source_index=index,
        source_file=source_file,
        source_row_index=source_row_index,
        canonical_questions=questions,
        label=label,
        token_row_start=index * 2,
        token_row_end=index * 2 + 2,
        metadata_sha256=f"{index:064x}",
    )


def test_refitter_allowlist_excludes_heldout_quarantine_ambiguous_and_unknown(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(
            f"q-{index}",
            f"Unique heldout candidate {index}?",
            "yes",
            near_match_policy="canonical_only",
        )
        for index in range(40)
    ] + [
        _candidate("bad", "Is the quarantine collision true?", "yes"),
        _candidate("bad", "Is the quarantine collision true?", "no"),
    ]
    dataset = build_dataset_v2(candidates, tmp_path / "bundle", seed=5)
    heldout = next(row for row in dataset.records if row["split"] != "train")
    examples = [
        _construction_example("domain-a", 0, 0),
        _construction_example("domain-a", 1, 1),
        _construction_example("domain-b", 2, 0),
        _construction_example("domain-b", 3, 1),
        _construction_example("domain-a", 4, 0, questions=(heldout["question"],)),
        _construction_example(
            "domain-a", 5, 1, questions=("Is the quarantine collision true?",)
        ),
        _construction_example("domain-a", 6, 0, questions=("one", "two")),
        _construction_example("domain-a", 7, -1),
    ]

    build = build_refitter_construction_allowlist(
        dataset,
        examples,
        activation_direct_sha256="a" * 64,
        required_domains=("domain-a", "domain-b"),
        minimum_per_class=1,
    )

    assert build.ready
    parsed = parse_construction_allowlist(build.allowlist)
    assert {row.domain for row in parsed.rows} == {"domain-a", "domain-b"}
    assert len(parsed.rows) == 4
    assert build.audit["excluded_counts"] == {
        "ambiguous_question_identity": 1,
        "heldout_or_quarantine_question_collision": 2,
        "unknown_label": 1,
    }


def test_refitter_allowlist_blocks_with_exact_undercovered_domain_label_cells(
    tmp_path: Path,
) -> None:
    dataset = build_dataset_v2(
        [_candidate("q", "Is water wet?", "yes")], tmp_path / "bundle", seed=1
    )

    build = build_refitter_construction_allowlist(
        dataset,
        [_construction_example("domain-a", 0, 0)],
        activation_direct_sha256="a" * 64,
        required_domains=("domain-a", "domain-b"),
        minimum_per_class=1,
    )

    assert not build.ready
    assert build.allowlist is None
    assert build.audit["missing_cells"] == [
        {"domain": "domain-a", "label": 1, "observed": 0, "required": 1},
        {"domain": "domain-b", "label": 0, "observed": 0, "required": 1},
        {"domain": "domain-b", "label": 1, "observed": 0, "required": 1},
    ]


def test_refitter_allowlist_caps_paired_source_groups_atomically(tmp_path: Path) -> None:
    dataset = build_dataset_v2(
        [_candidate("q", "Is water wet?", "yes")], tmp_path / "bundle", seed=1
    )
    examples = [
        replace(_construction_example("domain-a", 0, 0), source_index=10),
        replace(_construction_example("domain-a", 1, 1), source_index=10),
        replace(_construction_example("domain-b", 2, 0), source_index=20),
        replace(_construction_example("domain-b", 3, 1), source_index=20),
    ]

    build = build_refitter_construction_allowlist(
        dataset,
        examples,
        activation_direct_sha256="a" * 64,
        required_domains=("domain-a", "domain-b"),
        minimum_per_class=1,
        maximum_per_class=1,
    )

    assert build.ready
    parsed = parse_construction_allowlist(build.allowlist)
    groups = {
        domain: {row.group_id for row in parsed.rows if row.domain == domain}
        for domain in {row.domain for row in parsed.rows}
    }
    assert all(len(group_ids) == 1 for group_ids in groups.values())
    assert all(":paired:" in next(iter(group_ids)) for group_ids in groups.values())
    assert build.audit["selected_atomic_group_count"] == 2
