from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from intelligent_liars.truth_editing_contracts import canonical_sha256
from intelligent_liars.truth_editing_directions import (
    DirectionBank,
    DirectionBankError,
    build_direction_bank,
    build_reconstruction_workload,
    compile_control_basis_receipt,
    compile_equal_rank_orthogonal_control,
    compile_shuffled_control,
    vector_sha256,
    promote_reconstructed_direction_bank,
    parse_control_basis_receipt,
)
from intelligent_liars.truth_editing_direction_refit import (
    DirectionArtifactReceipt,
    DirectionRefitReceipt,
    LayerRefitReceipt,
    parse_direction_refit_receipt,
)


def _write_source(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "directions": [
                    {
                        "direction_vector": [3.0, 0.0, 0.0, 0.0],
                        "feature_count": 4,
                        "intercept": -0.25,
                        "layer": 1,
                        "task": "logical truth",
                        "direction_sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                    },
                    {
                        "direction_vector": [1.0, 1.0, 0.0, 0.0],
                        "feature_count": 4,
                        "intercept": 0.5,
                        "layer": 1,
                        "task": "factual truth",
                        "direction_sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                    },
                ],
                "general_domain": {
                    "directions": [
                        {
                            "direction_vector": [0.0, 0.0, 2.0, 0.0],
                            "feature_count": 4,
                            "intercept": 0.0,
                            "layer": 2,
                            "task": "general_domain",
                            "direction_sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                        }
                    ]
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_config(path: Path, *, status: str = "candidate") -> None:
    value = {
        "format": "truth_editing_direction_source_inventory_v1",
        "manifest_id": "fixture-bank",
        "model": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "1" * 40,
            "model_sha256": "2" * 64,
            "tokenizer_sha256": "3" * 64,
            "chat_template_sha256": "4" * 64,
            "decoder_layer_count": 3,
            "hidden_width": 4,
        },
        "sources": [
            {
                "source_id": "fixture",
                "path": "source.json",
                "qualification_status": status,
                "include_domain_directions": True,
                "include_general_directions": True,
                "provenance": {
                    "dataset": "fixture",
                    "dataset_revision": "v1",
                    "split": "direction_construction",
                    "ordered_row_ids_sha256": "5" * 64,
                    "source_code_revision": "6" * 40,
                },
                "leakage": {
                    "evaluation_disjoint": status == "qualified",
                    "heldout_family_disjoint": status == "qualified",
                    "sealed_audit_accessed": False,
                    "audit_receipt_sha256": "7" * 64,
                },
            }
        ],
    }
    value["self_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builder_inventories_general_and_domain_directions_with_exact_coverage(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path / "source.json")
    _write_config(tmp_path / "config.json")

    result = build_direction_bank(tmp_path / "config.json", root=tmp_path)

    assert len(result.manifest.directions) == 3
    assert result.coverage.total == 3
    assert result.coverage.by_family == {"domain_specific": 2, "general": 1}
    assert result.coverage.by_layer == {1: 2, 2: 1}
    assert result.coverage.by_status == {"candidate": 3}
    assert result.coverage.cells == (
        ("domain_specific", "factual truth", 1, "candidate", 1),
        ("domain_specific", "logical truth", 1, "candidate", 1),
        ("general", "general_domain", 2, "candidate", 1),
    )
    assert result.manifest.to_dict()["self_sha256"] == result.manifest.self_sha256


def _open_fixture_bank(tmp_path: Path, *, status: str = "qualified") -> DirectionBank:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write_source(tmp_path / "source.json")
    config_status = "candidate" if status == "qualified" else status
    _write_config(tmp_path / "config.json", status=config_status)
    result = build_direction_bank(tmp_path / "config.json", root=tmp_path)
    manifest = result.manifest.to_dict()
    if status == "qualified":
        for entry in manifest["directions"]:
            entry["qualification"]["status"] = "qualified"
            _, pointer = entry["artifact"]["path"].split("#", 1)
            entry["qualification"]["receipt_sha256"] = canonical_sha256(
                {
                    "format": "truth_editing_direction_qualification_receipt_v1",
                    "status": "qualified",
                    "source_file_sha256": entry["artifact"]["file_sha256"],
                    "json_pointer": pointer,
                    "vector_sha256": entry["artifact"]["vector_sha256"],
                }
            )
            entry["leakage"]["evaluation_disjoint"] = True
            entry["leakage"]["heldout_family_disjoint"] = True
        manifest["self_sha256"] = canonical_sha256(
            {key: value for key, value in manifest.items() if key != "self_sha256"}
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return DirectionBank.open(manifest_path, root=tmp_path)


@pytest.mark.parametrize("method,rank", [("qr", 2), ("svd", 1)])
def test_bank_compiles_deterministic_orthonormal_basis_with_requested_rank(
    tmp_path: Path, method: str, rank: int
) -> None:
    bank = _open_fixture_bank(tmp_path)
    selected = tuple(
        entry.direction_id
        for entry in bank.manifest.directions
        if entry.family == "domain_specific"
    )

    first = bank.compile_basis(selected, method=method, requested_rank=rank)
    second = bank.compile_basis(selected, method=method, requested_rank=rank)

    assert first.basis_sha256 == second.basis_sha256
    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_allclose(first.matrix.T @ first.matrix, np.eye(rank), atol=1e-12)
    if method == "qr":
        inputs = np.column_stack([bank.load_vector(item) for item in selected])
        np.testing.assert_allclose(
            first.matrix @ first.matrix.T @ inputs, inputs, atol=1e-12
        )


def test_bank_fails_closed_on_candidate_hash_layer_and_rank(tmp_path: Path) -> None:
    candidate = _open_fixture_bank(tmp_path / "candidate", status="candidate")
    candidate_id = candidate.manifest.directions[0].direction_id
    with pytest.raises(DirectionBankError, match="not qualified"):
        candidate.load_vector(candidate_id)

    hash_root = tmp_path / "hash"
    bank = _open_fixture_bank(hash_root)
    source = hash_root / "source.json"
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(DirectionBankError, match="file hash mismatch"):
        bank.load_vector(bank.manifest.directions[0].direction_id)

    fresh = _open_fixture_bank(tmp_path / "rank")
    domain_ids = tuple(
        item.direction_id
        for item in fresh.manifest.directions
        if item.family == "domain_specific"
    )
    with pytest.raises(DirectionBankError, match="requested_rank"):
        fresh.compile_basis(domain_ids, method="svd", requested_rank=3)
    cross_layer_ids = (
        domain_ids[0],
        next(
            item.direction_id
            for item in fresh.manifest.directions
            if item.family == "general"
        ),
    )
    with pytest.raises(DirectionBankError, match="same source layer"):
        fresh.compile_basis(cross_layer_ids, method="qr", requested_rank=1)


def test_bank_loads_verified_npy_matrix_rows_and_rejects_bad_locator(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path)
    matrix = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="<f8")
    artifact = tmp_path / "layer-01.npy"
    np.save(artifact, matrix, allow_pickle=False)
    file_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    raw = bank.manifest.to_dict()
    raw["directions"][0]["artifact"] = {
        "path": "layer-01.npy#row/1",
        "file_sha256": file_sha,
        "vector_sha256": vector_sha256(matrix[1]),
    }
    raw["self_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_sha256"}
    )
    manifest_path = tmp_path / "npy-manifest.json"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    npy_bank = DirectionBank.open(manifest_path, root=tmp_path)

    np.testing.assert_array_equal(
        npy_bank.load_vector(raw["directions"][0]["direction_id"]), matrix[1]
    )

    raw["directions"][0]["artifact"]["path"] = "layer-01.npy#row/2"
    raw["self_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_sha256"}
    )
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    bad_bank = DirectionBank.open(manifest_path, root=tmp_path)
    with pytest.raises(DirectionBankError, match="row index is absent"):
        bad_bank.load_vector(raw["directions"][0]["direction_id"])


def test_equal_rank_controls_are_deterministic_orthonormal_and_parent_orthogonal(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path)
    selected = tuple(
        item.direction_id
        for item in bank.manifest.directions
        if item.family == "domain_specific"
    )
    parent = bank.compile_basis(selected, method="qr", requested_rank=2)

    orthogonal = compile_equal_rank_orthogonal_control(parent, seed=17)
    shuffled = compile_shuffled_control(parent, seed=23)

    for control in (orthogonal, shuffled):
        assert control.rank == parent.rank
        np.testing.assert_allclose(
            control.matrix.T @ control.matrix, np.eye(2), atol=1e-12
        )
        np.testing.assert_allclose(
            parent.matrix.T @ control.matrix, np.zeros((2, 2)), atol=1e-12
        )
    assert (
        orthogonal.basis_sha256
        == compile_equal_rank_orthogonal_control(parent, seed=17).basis_sha256
    )
    assert (
        shuffled.basis_sha256 == compile_shuffled_control(parent, seed=23).basis_sha256
    )


def test_per_layer_basis_set_identity_binds_layers_order_and_each_basis(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path)
    selected = tuple(item.direction_id for item in bank.manifest.directions)

    basis_set = bank.compile_basis_set(selected, method="qr", requested_rank=1)

    assert tuple(layer for layer, _ in basis_set.by_layer) == (1, 2)
    assert basis_set.basis_set_sha256 != bank.manifest.self_sha256
    assert (
        basis_set.basis_set_sha256
        == bank.compile_basis_set(
            tuple(reversed(selected)), method="qr", requested_rank=1
        ).basis_set_sha256
    )
    substituted = bank.compile_basis_set(selected[:-1], method="qr", requested_rank=1)
    assert substituted.basis_set_sha256 != basis_set.basis_set_sha256

    with pytest.raises(DirectionBankError, match="sorted unique"):
        replace(basis_set, by_layer=tuple(reversed(basis_set.by_layer))).verify()
    first_layer, first_basis = basis_set.by_layer[0]
    _, second_basis = basis_set.by_layer[1]
    replaced_basis = replace(first_basis, matrix=second_basis.matrix)
    tampered = replace(
        basis_set,
        by_layer=((first_layer, replaced_basis), basis_set.by_layer[1]),
    )
    with pytest.raises(DirectionBankError, match="basis identity mismatch"):
        tampered.verify()


def test_relocated_basis_binds_source_lineage_model_and_destination_layers(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path)
    selected = tuple(
        item.direction_id
        for item in bank.manifest.directions
        if item.family == "domain_specific"
    )

    early = bank.compile_relocated_basis_set(
        selected,
        destination_layers=(0, 1),
        method="qr",
        requested_rank=2,
        expected_model_sha256=bank.manifest.model.model_sha256,
    )
    late = bank.compile_relocated_basis_set(
        selected,
        destination_layers=(1, 2),
        method="qr",
        requested_rank=2,
    )

    early.verify()
    assert tuple(layer for layer, _ in early.by_layer) == (0, 1)
    assert early.source_by_destination == ((0, 1), (1, 1))
    assert early.model_sha256 == bank.manifest.model.model_sha256
    assert tuple(layer for layer, _ in early.destination_basis_sha256s) == (0, 1)
    assert early.by_layer[0][1] is early.by_layer[1][1]
    assert not early.by_layer[0][1].matrix.flags.writeable
    assert early.basis_set_sha256 != late.basis_set_sha256

    reordered_ids = bank.compile_relocated_basis_set(
        tuple(reversed(selected)),
        destination_layers=(0, 1),
        method="qr",
        requested_rank=2,
    )
    assert reordered_ids.basis_set_sha256 == early.basis_set_sha256


def test_relocated_basis_fails_closed_on_invalid_destination_model_and_width(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path)
    selected = tuple(
        item.direction_id
        for item in bank.manifest.directions
        if item.family == "domain_specific"
    )

    for destinations, message in (
        ((), "nonempty"),
        ((1, 1), "sorted unique"),
        ((2, 1), "sorted unique"),
        ((-1,), "outside"),
        ((3,), "outside"),
        ((True,), "integers"),
    ):
        with pytest.raises(DirectionBankError, match=message):
            bank.compile_relocated_basis_set(
                selected,
                destination_layers=destinations,
                method="qr",
                requested_rank=1,
            )
    with pytest.raises(DirectionBankError, match="model identity mismatch"):
        bank.compile_relocated_basis_set(
            selected,
            destination_layers=(0,),
            method="qr",
            requested_rank=1,
            expected_model_sha256="f" * 64,
        )

    incompatible_manifest = replace(
        bank.manifest,
        model=replace(bank.manifest.model, hidden_width=5),
    )
    incompatible = DirectionBank(incompatible_manifest, tmp_path)
    with pytest.raises(DirectionBankError, match="width is incompatible"):
        incompatible.compile_relocated_basis_set(
            selected,
            destination_layers=(0,),
            method="qr",
            requested_rank=1,
        )


def test_relocated_basis_receipt_rejects_lineage_hash_reorder_and_substitution(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path)
    selected = tuple(
        item.direction_id
        for item in bank.manifest.directions
        if item.family == "domain_specific"
    )
    receipt = bank.compile_relocated_basis_set(
        selected,
        destination_layers=(0, 2),
        method="qr",
        requested_rank=2,
    )

    with pytest.raises(DirectionBankError, match="source lineage differs"):
        replace(
            receipt,
            source_by_destination=tuple(reversed(receipt.source_by_destination)),
        ).verify()
    with pytest.raises(DirectionBankError, match="destination hashes differ"):
        replace(
            receipt,
            destination_basis_sha256s=tuple(
                reversed(receipt.destination_basis_sha256s)
            ),
        ).verify()
    destination, destination_hash = receipt.destination_basis_sha256s[0]
    substituted = replace(
        receipt,
        destination_basis_sha256s=(
            (destination, "0" * 64),
            receipt.destination_basis_sha256s[1],
        ),
    )
    with pytest.raises(DirectionBankError, match="destination basis identity"):
        substituted.verify()
    with pytest.raises(DirectionBankError, match="sorted unique"):
        replace(receipt, by_layer=tuple(reversed(receipt.by_layer))).verify()


@pytest.mark.parametrize("kind", ["orthogonal_control", "shuffled_control"])
def test_control_receipt_binds_parent_seed_policy_and_each_layer_matrix(
    tmp_path: Path, kind: str
) -> None:
    bank = _open_fixture_bank(tmp_path)
    parent = bank.compile_basis_set(
        tuple(item.direction_id for item in bank.manifest.directions),
        method="qr",
        requested_rank=1,
    )
    receipt = compile_control_basis_receipt(parent, kind=kind, seed=29)

    receipt.verify(parent)
    assert (
        parse_control_basis_receipt(receipt.to_dict(), parent=parent).self_sha256
        == receipt.self_sha256
    )
    assert receipt.self_sha256 == canonical_sha256(
        {key: value for key, value in receipt.to_dict().items() if key != "self_sha256"}
    )
    for (_, parent_basis), (_, control_basis) in zip(
        parent.by_layer, receipt.by_layer, strict=True
    ):
        np.testing.assert_allclose(
            parent_basis.matrix.T @ control_basis.matrix,
            np.zeros((1, 1)),
            atol=1e-10,
        )

    with pytest.raises(DirectionBankError, match="derived seed mismatch"):
        replace(receipt, seed=30).verify(parent)
    other_parent = bank.compile_basis_set(
        tuple(
            item.direction_id
            for item in bank.manifest.directions
            if item.family == "domain_specific"
        ),
        method="qr",
        requested_rank=1,
    )
    with pytest.raises(DirectionBankError, match="parent basis-set mismatch"):
        receipt.verify(other_parent)
    first_layer, first_control = receipt.by_layer[0]
    _, second_control = receipt.by_layer[1]
    substituted = replace(
        receipt,
        by_layer=(
            (first_layer, replace(first_control, matrix=second_control.matrix)),
            receipt.by_layer[1],
        ),
    )
    with pytest.raises(DirectionBankError, match="basis identity mismatch"):
        substituted.verify(parent)
    persisted = receipt.to_dict()
    persisted["layers"][0]["matrix_sha256"] = "0" * 64
    persisted["self_sha256"] = canonical_sha256(
        {key: value for key, value in persisted.items() if key != "self_sha256"}
    )
    with pytest.raises(DirectionBankError, match="differs from deterministic"):
        parse_control_basis_receipt(persisted, parent=parent)


def test_builder_rejects_unverified_qualified_provenance_and_invalid_source_shape(
    tmp_path: Path,
) -> None:
    _write_source(tmp_path / "source.json")
    _write_config(tmp_path / "config.json")
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["sources"][0]["qualification_status"] = "qualified"
    config["sources"][0]["leakage"]["evaluation_disjoint"] = False
    config["self_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if key != "self_sha256"}
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DirectionBankError, match="cannot self-attest qualification"):
        build_direction_bank(tmp_path / "config.json", root=tmp_path)

    _write_config(tmp_path / "config.json")
    source = json.loads((tmp_path / "source.json").read_text(encoding="utf-8"))
    source["directions"][0]["feature_count"] = 5
    (tmp_path / "source.json").write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(DirectionBankError, match="hidden_width"):
        build_direction_bank(tmp_path / "config.json", root=tmp_path)


def test_glob_source_inventory_is_count_and_order_hash_pinned(tmp_path: Path) -> None:
    probes = tmp_path / "probes"
    probes.mkdir()
    _write_source(probes / "a.json")
    _write_source(probes / "b.json")
    second = json.loads((probes / "b.json").read_text(encoding="utf-8"))
    for item in second["directions"]:
        item["direction_vector"][-1] = 0.25
    second["general_domain"]["directions"][0]["direction_vector"][-1] = 0.25
    (probes / "b.json").write_text(json.dumps(second), encoding="utf-8")
    _write_config(tmp_path / "config.json")
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    source = config["sources"][0]
    source["path_glob"] = "probes/*.json"
    del source["path"]
    source["expected_path_count"] = 2
    source["ordered_paths_sha256"] = canonical_sha256(
        ["probes/a.json", "probes/b.json"]
    )
    config["self_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if key != "self_sha256"}
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    result = build_direction_bank(tmp_path / "config.json", root=tmp_path)
    assert result.coverage.total == 6

    config["sources"][0]["expected_path_count"] = 3
    config["self_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if key != "self_sha256"}
    )
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(DirectionBankError, match="glob match count"):
        build_direction_bank(tmp_path / "config.json", root=tmp_path)


def test_checked_in_inventory_covers_all_existing_direction_records() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = build_direction_bank(
        repository / "configs/truth_editing_direction_sources_v1.json",
        root=repository,
    )

    assert result.coverage.source_records == 3419
    assert result.coverage.duplicate_records == 1474
    assert result.coverage.total == 1945
    assert result.coverage.by_family == {
        "domain_specific": 1754,
        "general": 191,
    }
    assert result.coverage.by_status == {
        "candidate": 1891,
        "diagnostic_only": 54,
    }
    assert set(result.coverage.by_layer) == set(range(36))
    assert not any(
        family == "general" and layer == 0
        for family, _, layer, _, _ in result.coverage.cells
    )
    assert result.manifest.self_sha256 == (
        "32f35372a98e90ecc65895637d2afeb5108a027bf4697a9abb9a7e75b2d03ed1"
    )


def test_reconstruction_workload_covers_every_domain_layer_and_blocks_without_allowlist(
    tmp_path: Path,
) -> None:
    bank = _open_fixture_bank(tmp_path, status="candidate")
    activation = {
        "path": "artifacts/activations/construction.h5",
        "byte_size": 100,
        "direct_sha256": "a" * 64,
        "dvc_md5": "b" * 32,
        "evidence_status": "verified_metadata",
    }

    workload = build_reconstruction_workload(
        bank.manifest,
        activation_input=activation,
        construction_row_allowlist_sha256=None,
        output_root="artifacts/directions/rebuilt-v1",
        maximum_external_spend_usd=15.0,
    )

    assert workload["target_cell_count"] == 9
    assert workload["refit_cell_count"] == 9
    assert workload["blocked_cell_count"] == 9
    assert workload["data_access"]["allowed_partition"] == "direction_construction"
    assert workload["self_sha256"] == canonical_sha256(
        {key: value for key, value in workload.items() if key != "self_sha256"}
    )


def _refit_receipt(
    tmp_path: Path, bank: DirectionBank, *, plan_sha: str = "9" * 64
) -> DirectionRefitReceipt:
    domains = tuple(
        sorted({d for item in bank.manifest.directions for d in item.domains})
    )
    layers = []
    for layer in range(bank.manifest.model.decoder_layer_count):
        shard_id = canonical_sha256({"fixture_layer": layer})
        vectors = []
        for index in range(len(domains)):
            vector = np.asarray([1.0, index + 0.25, layer + 0.5, 0.1], dtype="<f8")
            vectors.append(vector / np.linalg.norm(vector))
        matrix = np.ascontiguousarray(vectors, dtype="<f8")
        relative = f"refit/layer-{layer:02d}.npy"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix, allow_pickle=False)
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cells = []
        for row, domain in enumerate(domains):
            unsigned = {
                "format": "truth_editing_refit_direction_v1",
                "direction_id": f"refit-{domain}-layer-{layer}",
                "domain": domain,
                "source_layer": layer,
                "width": 4,
                "rank": 1,
                "artifact_path": f"{relative}#row/{row}",
                "artifact_file_sha256": file_sha,
                "artifact_row": row,
                "vector_sha256": vector_sha256(matrix[row]),
                "original_norm": 2.0,
                "rescaled_intercept": 0.1,
                "sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                "model_repository": bank.manifest.model.repository,
                "model_revision": bank.manifest.model.revision,
                "model_sha256": bank.manifest.model.model_sha256,
                "activation_direct_sha256": "a" * 64,
                "activation_dvc_md5": "b" * 32,
                "activation_sidecar_sha256": "1" * 64,
                "construction_selector": "direction_construction",
                "construction_row_allowlist_sha256": "c" * 64,
                "ordered_row_ids_sha256": canonical_sha256([domain]),
                "construction_group_manifest_sha256": "d" * 64,
                "dataset_manifest_sha256": "e" * 64,
                "source_code_revision": "f" * 40,
                "probe_config_sha256": "2" * 64,
                "reconstruction_plan_sha256": plan_sha,
                "shard_id": shard_id,
                "finite": True,
                "unit_norm": True,
                "qualified_rank": 1,
            }
            cells.append(
                DirectionArtifactReceipt(
                    **unsigned, self_sha256=canonical_sha256(unsigned)
                )
            )
        unsigned_layer = {
            "format": "truth_editing_direction_refit_layer_receipt_v1",
            "shard_id": shard_id,
            "source_layer": layer,
            "artifact_path": relative,
            "artifact_file_sha256": file_sha,
            "matrix_shape": [len(domains), 4],
            "matrix_dtype": "<f8",
            "ordered_domains_sha256": canonical_sha256(list(domains)),
            "direction_receipts": [asdict(item) for item in cells],
        }
        layers.append(
            LayerRefitReceipt(
                format="truth_editing_direction_refit_layer_receipt_v1",
                shard_id=shard_id,
                source_layer=layer,
                artifact_path=relative,
                artifact_file_sha256=file_sha,
                matrix_shape=(len(domains), 4),
                matrix_dtype="<f8",
                ordered_domains_sha256=canonical_sha256(list(domains)),
                direction_receipts=tuple(cells),
                self_sha256=canonical_sha256(unsigned_layer),
            )
        )
    unsigned_receipt = {
        "format": "truth_editing_direction_refit_receipt_v1",
        "plan_sha256": plan_sha,
        "completed_direction_count": len(domains) * 3,
        "layer_receipts": [asdict(item) for item in layers],
    }
    return DirectionRefitReceipt(
        format="truth_editing_direction_refit_receipt_v1",
        plan_sha256=plan_sha,
        completed_direction_count=len(domains) * 3,
        layer_receipts=tuple(layers),
        self_sha256=canonical_sha256(unsigned_receipt),
    )


def test_only_complete_verified_refit_aggregate_can_promote_qualified_bank(
    tmp_path: Path,
) -> None:
    base = _open_fixture_bank(tmp_path / "base", status="candidate")
    receipt = parse_direction_refit_receipt(_refit_receipt(tmp_path, base).to_dict())

    promoted = promote_reconstructed_direction_bank(
        base.manifest,
        receipt,
        expected_plan_sha256="9" * 64,
        root=tmp_path,
        manifest_id="qualified-refit-fixture",
    )

    assert len(promoted.directions) == 9
    assert {item.qualification.status for item in promoted.directions} == {"qualified"}
    assert {item.qualification.receipt_sha256 for item in promoted.directions} == {
        receipt.self_sha256
    }

    incomplete_unsigned = {
        "format": receipt.format,
        "plan_sha256": receipt.plan_sha256,
        "completed_direction_count": 6,
        "layer_receipts": [asdict(item) for item in receipt.layer_receipts[:-1]],
    }
    incomplete = replace(
        receipt,
        completed_direction_count=6,
        layer_receipts=receipt.layer_receipts[:-1],
        self_sha256=canonical_sha256(incomplete_unsigned),
    )
    with pytest.raises(DirectionBankError, match="cover every layer"):
        promote_reconstructed_direction_bank(
            base.manifest,
            incomplete,
            expected_plan_sha256="9" * 64,
            root=tmp_path,
            manifest_id="must-fail",
        )


def _promotion_cli_fixture(tmp_path: Path) -> dict[str, Path]:
    bank = _open_fixture_bank(tmp_path / "base", status="candidate")
    base_manifest = tmp_path / "base-manifest.json"
    base_manifest.write_text(json.dumps(bank.manifest.to_dict()), encoding="utf-8")
    plan_unsigned = {
        "format": "truth_editing_direction_refit_plan_v1",
        "config_sha256": "8" * 64,
        "target_direction_count": 9,
    }
    plan_sha = canonical_sha256(plan_unsigned)
    plan = dict(plan_unsigned, self_sha256=plan_sha)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    receipt_path = tmp_path / "refit-receipt.json"
    receipt_path.write_text(
        json.dumps(_refit_receipt(tmp_path, bank, plan_sha=plan_sha).to_dict()),
        encoding="utf-8",
    )
    return {
        "base": base_manifest,
        "plan": plan_path,
        "receipt": receipt_path,
        "manifest": tmp_path / "qualified-manifest.json",
        "coverage": tmp_path / "qualified-coverage.json",
    }


def _run_promotion_cli(tmp_path: Path, paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    repository = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/promote_truth_editing_direction_bank.py"),
            "--base-manifest",
            str(paths["base"]),
            "--plan",
            str(paths["plan"]),
            "--refit-receipt",
            str(paths["receipt"]),
            "--artifact-root",
            str(tmp_path),
            "--manifest-output",
            str(paths["manifest"]),
            "--coverage-output",
            str(paths["coverage"]),
            "--manifest-id",
            "qualified-cli-fixture",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )


def test_promotion_cli_atomically_writes_identity_bound_qualified_outputs(
    tmp_path: Path,
) -> None:
    paths = _promotion_cli_fixture(tmp_path)

    first = _run_promotion_cli(tmp_path, paths)
    second = _run_promotion_cli(tmp_path, paths)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    result = json.loads(first.stdout)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    coverage = json.loads(paths["coverage"].read_text(encoding="utf-8"))
    assert result["direction_count"] == result["qualified_count"] == 9
    assert result["manifest_sha256"] == manifest["self_sha256"]
    assert coverage["manifest_sha256"] == manifest["self_sha256"]
    assert coverage["base_manifest"]["self_sha256"] == json.loads(
        paths["base"].read_text(encoding="utf-8")
    )["self_sha256"]
    assert coverage["refit_plan"]["self_sha256"] == result["plan_sha256"]
    assert coverage["refit_receipt"]["self_sha256"] == result[
        "refit_receipt_sha256"
    ]
    assert coverage["artifact_root"] == str(tmp_path.resolve())
    assert coverage["model"] == manifest["model"]
    assert {item["qualification"]["status"] for item in manifest["directions"]} == {
        "qualified"
    }

    paths["manifest"].write_text("{}\n", encoding="utf-8")
    refused = _run_promotion_cli(tmp_path, paths)
    assert refused.returncode == 2
    assert "refusing to overwrite different existing output" in refused.stderr


def test_promotion_cli_rejects_tampered_plan(tmp_path: Path) -> None:
    paths = _promotion_cli_fixture(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan["target_direction_count"] = 10
    paths["plan"].write_text(json.dumps(plan), encoding="utf-8")

    result = _run_promotion_cli(tmp_path, paths)

    assert result.returncode == 2
    assert "refit plan self hash mismatch" in result.stderr
    assert not paths["manifest"].exists()
    assert not paths["coverage"].exists()


def test_promotion_cli_rejects_incomplete_refit_without_partial_outputs(
    tmp_path: Path,
) -> None:
    paths = _promotion_cli_fixture(tmp_path)
    raw = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    raw["layer_receipts"] = raw["layer_receipts"][:-1]
    raw["completed_direction_count"] -= 3
    raw["self_sha256"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "self_sha256"}
    )
    paths["receipt"].write_text(json.dumps(raw), encoding="utf-8")

    result = _run_promotion_cli(tmp_path, paths)

    assert result.returncode == 2
    assert "cover every layer" in result.stderr
    assert not paths["manifest"].exists()
    assert not paths["coverage"].exists()
