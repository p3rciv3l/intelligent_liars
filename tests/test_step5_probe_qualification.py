from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intelligent_liars.step5_probe_qualification import (
    compile_probe_qualification,
    validate_probe_qualification,
    write_probe_qualification,
)


SIGN = "positive_logit_points_honest_to_deceptive"
POOLING = "mean_answer_tokens_per_example"


def _artifact(path: Path, direction: list[float]) -> None:
    path.write_text(json.dumps({"result": {"direction": direction}}) + "\n")


def _registry(tmp_path: Path) -> dict[str, object]:
    _artifact(tmp_path / "regularizer.json", [1.0, 2.0, 3.0])
    _artifact(tmp_path / "evaluator.json", [2.0, -1.0, 1.0])
    return {
        "format": "intelligent_liars_step5_probe_registry_v1",
        "qualification": {
            "layer": 19,
            "token_pooling": POOLING,
            "direction_sign_convention": SIGN,
            "orthogonal_controls_per_probe": 2,
        },
        "probes": [
            {
                "probe_id": "regularizer-a",
                "ensemble": "regularizer",
                "artifact_path": "regularizer.json",
                "artifact_direction_path": ["result", "direction"],
                "source_group_ids": ["apollo/train/a"],
                "example_ids": ["a-1", "a-2"],
                "layer": 19,
                "token_pooling": POOLING,
                "direction_sign_convention": SIGN,
            },
            {
                "probe_id": "evaluator-a",
                "ensemble": "evaluator",
                "artifact_path": "evaluator.json",
                "artifact_direction_path": ["result", "direction"],
                "source_group_ids": ["truthspec/heldout/b"],
                "example_ids": ["b-1", "b-2"],
                "layer": 19,
                "token_pooling": POOLING,
                "direction_sign_convention": SIGN,
            },
        ],
    }


def test_compile_emits_disjoint_split_receipts_and_controls(tmp_path: Path):
    registry = _registry(tmp_path)

    manifest = compile_probe_qualification(registry, artifact_root=tmp_path)

    assert manifest["status"] == "qualified"
    assert manifest["qualification"]["split_unit"] == "source_group_id"
    assert [row["probe_id"] for row in manifest["ensembles"]["regularizer"]] == [
        "regularizer-a"
    ]
    evaluator = manifest["ensembles"]["evaluator"][0]
    assert len(evaluator["artifact_sha256"]) == 64
    assert len(evaluator["direction_sha256"]) == 64
    assert [control["kind"] for control in evaluator["controls"]] == [
        "sign_flip",
        "orthogonal",
        "orthogonal",
    ]
    assert evaluator["controls"][0]["vector"] == [-2.0, 1.0, -1.0]
    assert all(
        abs(sum(a * b for a, b in zip([2.0, -1.0, 1.0], row["vector"]))) < 1e-12
        for row in evaluator["controls"][1:]
    )
    assert set(manifest["split_receipts"]) == {"regularizer", "evaluator"}
    assert len(manifest["split_receipts"]["regularizer"]["receipt_sha256"]) == 64
    assert len(manifest["qualification_receipt_sha256"]) == 64


def test_compile_is_deterministic_across_probe_and_identifier_order(tmp_path: Path):
    first_registry = _registry(tmp_path)
    first = compile_probe_qualification(first_registry, artifact_root=tmp_path)
    second_registry = json.loads(json.dumps(first_registry))
    second_registry["probes"].reverse()
    for probe in second_registry["probes"]:
        probe["source_group_ids"].reverse()
        probe["example_ids"].reverse()

    second = compile_probe_qualification(second_registry, artifact_root=tmp_path)

    assert first == second


@pytest.mark.parametrize(
    ("field", "overlap"),
    [
        ("source_group_ids", "apollo/train/a"),
        ("example_ids", "a-1"),
    ],
)
def test_compile_rejects_cross_ensemble_leakage(
    tmp_path: Path, field: str, overlap: str
):
    registry = _registry(tmp_path)
    registry["probes"][1][field].append(overlap)

    with pytest.raises(ValueError, match=f"cross-ensemble {field} overlap"):
        compile_probe_qualification(registry, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("layer", 20),
        ("token_pooling", "last_token"),
        ("direction_sign_convention", "unknown"),
    ],
)
def test_compile_rejects_probe_that_does_not_match_pinned_contract(
    tmp_path: Path, field: str, bad_value: object
):
    registry = _registry(tmp_path)
    registry["probes"][0][field] = bad_value

    with pytest.raises(ValueError, match=f"does not match qualification {field}"):
        compile_probe_qualification(registry, artifact_root=tmp_path)


def test_compile_rejects_missing_provenance_and_duplicate_artifact(tmp_path: Path):
    missing = _registry(tmp_path)
    missing["probes"][0]["example_ids"] = []
    with pytest.raises(ValueError, match="nonempty example_ids"):
        compile_probe_qualification(missing, artifact_root=tmp_path)

    duplicate = _registry(tmp_path)
    duplicate["probes"][1]["artifact_path"] = "regularizer.json"
    with pytest.raises(ValueError, match="duplicate artifact content"):
        compile_probe_qualification(duplicate, artifact_root=tmp_path)

    noncanonical = _registry(tmp_path)
    noncanonical["probes"][0]["source_group_ids"] = [" apollo/train/a "]
    with pytest.raises(ValueError, match="canonical identifier syntax"):
        compile_probe_qualification(noncanonical, artifact_root=tmp_path)


def test_compile_rejects_duplicate_direction_even_if_artifacts_differ(tmp_path: Path):
    registry = _registry(tmp_path)
    (tmp_path / "evaluator.json").write_text(
        json.dumps({"result": {"direction": [1.0, 2.0, 3.0]}, "metadata": "other"})
    )
    with pytest.raises(ValueError, match="duplicate direction vectors"):
        compile_probe_qualification(registry, artifact_root=tmp_path)


def test_compile_rejects_bad_vector_or_unavailable_orthogonal_control(tmp_path: Path):
    registry = _registry(tmp_path)
    _artifact(tmp_path / "regularizer.json", [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="nonzero finite direction vector"):
        compile_probe_qualification(registry, artifact_root=tmp_path)

    registry = _registry(tmp_path)
    _artifact(tmp_path / "regularizer.json", [1.0])
    _artifact(tmp_path / "evaluator.json", [1.0])
    with pytest.raises(ValueError, match="at least two dimensions"):
        compile_probe_qualification(registry, artifact_root=tmp_path)


def test_write_is_atomic_enough_to_refuse_overwrite(tmp_path: Path):
    registry = _registry(tmp_path)
    output = tmp_path / "qualified.json"
    written = write_probe_qualification(
        registry,
        artifact_root=tmp_path,
        output_path=output,
    )
    assert json.loads(output.read_text()) == written

    with pytest.raises(FileExistsError):
        write_probe_qualification(
            registry,
            artifact_root=tmp_path,
            output_path=output,
        )


def test_write_rebases_artifacts_so_default_verification_root_works(tmp_path: Path):
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    output_root = tmp_path / "receipts"
    registry = _registry(registry_root)
    output = output_root / "qualified.json"

    manifest = write_probe_qualification(
        registry, artifact_root=registry_root, output_path=output
    )

    assert validate_probe_qualification(manifest, artifact_root=output.parent)["valid"]
    assert manifest["ensembles"]["regularizer"][0]["artifact_path"].startswith("..")


def test_compile_hashes_the_same_byte_snapshot_it_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = _registry(tmp_path)
    artifact_path = tmp_path / "regularizer.json"
    original_bytes = artifact_path.read_bytes()
    original_read_bytes = Path.read_bytes

    def read_then_mutate(path: Path) -> bytes:
        value = original_read_bytes(path)
        if path == artifact_path:
            path.write_text(json.dumps({"result": {"direction": [9.0, 8.0, 7.0]}}))
        return value

    monkeypatch.setattr(Path, "read_bytes", read_then_mutate)
    manifest = compile_probe_qualification(registry, artifact_root=tmp_path)
    regularizer = manifest["ensembles"]["regularizer"][0]
    assert regularizer["artifact_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert regularizer["controls"][0]["vector"] == [-1.0, -2.0, -3.0]


def test_validate_recompiles_controls_receipts_and_artifact_hashes(tmp_path: Path):
    registry = _registry(tmp_path)
    manifest = compile_probe_qualification(registry, artifact_root=tmp_path)
    assert validate_probe_qualification(manifest, artifact_root=tmp_path)["valid"]

    tampered = json.loads(json.dumps(manifest))
    tampered["ensembles"]["regularizer"][0]["controls"][0]["vector"][0] = 99.0
    result = validate_probe_qualification(tampered, artifact_root=tmp_path)
    assert not result["valid"]
    assert "does not match" in result["issues"][0]

    _artifact(tmp_path / "regularizer.json", [3.0, 2.0, 1.0])
    result = validate_probe_qualification(manifest, artifact_root=tmp_path)
    assert not result["valid"]


@pytest.mark.parametrize("root", [None, [], "bad"])
def test_validate_malformed_root_fails_closed(root: object, tmp_path: Path):
    result = validate_probe_qualification(root, artifact_root=tmp_path)
    assert not result["valid"]
    assert result["issues"] == ["manifest root must be an object"]
