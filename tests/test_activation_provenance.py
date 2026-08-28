from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.activation_provenance import (
    ACTIVATION_PROVENANCE_FORMAT,
    INVENTORY_FORMAT,
    ProvenanceContractError,
    canonical_sha256,
    parse_activation_provenance,
    parse_inventory,
    validate_compatibility,
)


SHA = "a" * 64
REVISION = "b" * 40


def _unsigned_sidecar(*, evidence_status: str = "verified_metadata") -> dict:
    return {
        "format": ACTIVATION_PROVENANCE_FORMAT,
        "sidecar_id": "all-text-20260624",
        "artifact": {
            "path": "artifacts/activations/all-text.h5",
            "byte_size": 62394047728,
            "dvc_pointer_path": "artifacts/activations/all-text.h5.dvc",
            "dvc_hash_algorithm": "md5",
            "dvc_hash": "c" * 32,
            "dvc_size": 62394047728,
            "dvc_pointer_sha256": "d" * 64,
            "direct_sha256": "e" * 64,
            "direct_hash_evidence": "historical_validation_receipt",
        },
        "hdf5_inventory": {
            "validator_format": "qwen_answer_token_activations_v2",
            "tasks": ["claims", "geometry"],
            "task_rows": {"claims": 10, "geometry": 12},
            "example_counts": {"claims": 4, "geometry": 5},
            "layers": ["layer_0", "layer_21"],
            "hidden_dim": 4096,
            "storage_dtype": "float16",
            "finite_check": "sample",
            "validator_revision": REVISION,
        },
        "model": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": REVISION,
            "content_sha256": "f" * 64,
        },
        "processor": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": REVISION,
            "content_sha256": None,
            "tokenizer_sha256": None,
            "chat_template_sha256": None,
        },
        "runtime": {
            "python_version": "3.11.9",
            "torch_version": "2.5.1",
            "transformers_version": "4.57.1",
            "backend": "transformers",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "attention_implementation": "flash_attention_2",
            "quantization": "none",
            "batch_size": 8,
            "use_cache": True,
        },
        "source_dataset": {
            "dataset_id": "truth-spec-static",
            "revision": "truth-spec-b95fdc1c5a1670f3c6c013140f1b76467d80cbf4",
            "manifest_sha256": SHA,
            "source_row_ids_sha256": "0" * 64,
        },
        "split_receipt": {
            "format": "truth_editing_split_receipt_v1",
            "status": "verified",
            "split_name": "direction_construction",
            "split_policy": "grouped_by_canonical_question_and_proposition",
            "assignment_seed": 17,
            "dataset_manifest_sha256": SHA,
            "ordered_row_ids_sha256": "1" * 64,
            "group_ids_sha256": "2" * 64,
            "disjoint_from": ["validation", "test"],
            "receipt_sha256": "3" * 64,
        },
        "evidence_status": evidence_status,
    }


def _sidecar(*, evidence_status: str = "verified_metadata") -> dict:
    value = _unsigned_sidecar(evidence_status=evidence_status)
    value["self_sha256"] = canonical_sha256(value)
    return value


def _unsigned_inventory() -> dict:
    return {
        "format": INVENTORY_FORMAT,
        "inventory_id": "activation-provenance-v1",
        "entries": [_sidecar()],
    }


def _inventory() -> dict:
    value = _unsigned_inventory()
    value["self_sha256"] = canonical_sha256(value)
    return value


def test_sidecar_round_trip_preserves_canonical_identity() -> None:
    raw = _sidecar()
    parsed = parse_activation_provenance(raw)

    assert parsed.to_payload() == raw
    assert parsed.self_sha256 == raw["self_sha256"]
    assert parse_activation_provenance(parsed.to_payload()) == parsed


def test_inventory_round_trip_and_entry_identity() -> None:
    raw = _inventory()
    parsed = parse_inventory(raw)

    assert parsed.to_payload() == raw
    assert parsed.entries[0].sidecar_id == "all-text-20260624"
    assert parsed.self_sha256 == raw["self_sha256"]


def test_tampering_path_or_hash_fails_closed() -> None:
    for field, value in (("path", "other.h5"), ("direct_sha256", "f" * 64)):
        raw = _sidecar()
        raw["artifact"][field] = value
        with pytest.raises(ProvenanceContractError, match="self hash mismatch"):
            parse_activation_provenance(raw)


def test_unknown_fields_and_wrong_revision_shape_fail_closed() -> None:
    raw = _sidecar()
    raw["unexpected"] = True
    with pytest.raises(ProvenanceContractError, match="fields differ"):
        parse_activation_provenance(raw)

    raw = _sidecar()
    raw["model"]["revision"] = "not-a-git-revision"
    raw["self_sha256"] = canonical_sha256({k: v for k, v in raw.items() if k != "self_sha256"})
    with pytest.raises(ProvenanceContractError, match="revision"):
        parse_activation_provenance(raw)


def test_unknown_evidence_is_allowed_but_proven_requires_every_binding() -> None:
    raw = _sidecar(evidence_status="unknown")
    raw["artifact"]["direct_sha256"] = None
    raw["artifact"]["direct_hash_evidence"] = "not_checked"
    raw["model"]["revision"] = None
    raw["processor"]["revision"] = None
    raw["source_dataset"]["manifest_sha256"] = None
    raw["split_receipt"]["status"] = "unknown"
    raw["split_receipt"]["receipt_sha256"] = None
    raw["self_sha256"] = canonical_sha256({k: v for k, v in raw.items() if k != "self_sha256"})
    assert parse_activation_provenance(raw).evidence_status == "unknown"

    proven = _sidecar(evidence_status="proven")
    proven["artifact"]["direct_sha256"] = None
    proven["artifact"]["direct_hash_evidence"] = "not_checked"
    proven["self_sha256"] = canonical_sha256({k: v for k, v in proven.items() if k != "self_sha256"})
    with pytest.raises(ProvenanceContractError, match="proven evidence"):
        parse_activation_provenance(proven)


def test_compatibility_binds_model_processor_source_split_and_runtime() -> None:
    sidecar = parse_activation_provenance(_sidecar())
    validate_compatibility(
        sidecar,
        model_revision=REVISION,
        processor_revision=REVISION,
        source_dataset_id="truth-spec-static",
        split_name="direction_construction",
        runtime_backend="transformers",
        runtime_dtype="bfloat16",
    )

    with pytest.raises(ProvenanceContractError, match="model revision"):
        validate_compatibility(sidecar, model_revision="c" * 40)
    with pytest.raises(ProvenanceContractError, match="split"):
        validate_compatibility(sidecar, split_name="test")


def test_compatibility_refuses_unknown_fields_when_requested() -> None:
    raw = _sidecar(evidence_status="unknown")
    raw["model"]["revision"] = None
    raw["processor"]["revision"] = None
    raw["self_sha256"] = canonical_sha256({k: v for k, v in raw.items() if k != "self_sha256"})
    sidecar = parse_activation_provenance(raw)

    with pytest.raises(ProvenanceContractError, match="unknown"):
        validate_compatibility(sidecar, model_revision=REVISION)


def test_schema_fixture_and_generated_inventory_are_json_objects() -> None:
    schema = json.loads(Path("schemas/activation-provenance-v1.schema.json").read_text())
    assert schema["$id"] == "activation-provenance-v1.schema.json"
    inventory = json.loads(Path("configs/activation_provenance_inventory_v1.json").read_text())
    assert inventory["format"] == INVENTORY_FORMAT
    assert parse_inventory(inventory).entries
