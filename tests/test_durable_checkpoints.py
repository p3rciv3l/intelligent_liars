from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from intelligent_liars.durable_checkpoints import (
    CheckpointIdentityMismatch,
    CheckpointIntegrityError,
    ImmutableCheckpointError,
    advance_latest_checkpoint,
    create_checkpoint_generation,
    resolve_latest_checkpoint,
    validate_checkpoint_generation,
)


IDENTITY = {
    "run_id": "step5-screen-v1",
    "candidate_id": "tinylora-64",
    "seed": 20260822,
    "plan_sha256": "a" * 64,
}


def _source(tmp_path: Path, value: bytes) -> Path:
    source = tmp_path / f"source-{value.decode()}"
    source.mkdir()
    (source / "adapter.safetensors").write_bytes(value)
    (source / "trainer").mkdir()
    (source / "trainer" / "state.json").write_text('{"step": 25}\n')
    return source


def test_create_generation_writes_an_immutable_hash_verified_manifest(tmp_path: Path):
    root = tmp_path / "checkpoints"
    generation = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000025",
        source_dir=_source(tmp_path, b"one"),
    )

    manifest = json.loads(generation.manifest_path.read_text())
    assert generation.path == root / "generations" / "step-000025"
    assert manifest["identity"] == IDENTITY
    assert manifest["generation_id"] == "step-000025"
    assert manifest["files"] == [
        {
            "path": "adapter.safetensors",
            "sha256": hashlib.sha256(b"one").hexdigest(),
            "size_bytes": 3,
        },
        {
            "path": "trainer/state.json",
            "sha256": hashlib.sha256(b'{"step": 25}\n').hexdigest(),
            "size_bytes": 13,
        },
    ]
    assert validate_checkpoint_generation(
        generation.path, expected_identity=IDENTITY
    ) == generation

    with pytest.raises(ImmutableCheckpointError, match="already exists"):
        create_checkpoint_generation(
            root,
            identity=IDENTITY,
            generation_id="step-000025",
            source_dir=_source(tmp_path, b"two"),
        )


def test_corruption_blocks_latest_pointer_advance(tmp_path: Path):
    root = tmp_path / "checkpoints"
    generation = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000025",
        source_dir=_source(tmp_path, b"one"),
    )
    (generation.path / "adapter.safetensors").write_bytes(b"two")

    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        advance_latest_checkpoint(
            root,
            generation.generation_id,
            identity=IDENTITY,
            durable_verifier=lambda _generation: True,
        )

    assert not (root / "latest.json").exists()


def test_durable_verifier_must_accept_before_latest_advances(tmp_path: Path):
    root = tmp_path / "checkpoints"
    generation = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000025",
        source_dir=_source(tmp_path, b"one"),
    )
    calls: list[str] = []

    def reject(candidate):
        calls.append(candidate.manifest_sha256)
        return False

    with pytest.raises(CheckpointIntegrityError, match="durability verification"):
        advance_latest_checkpoint(
            root,
            generation.generation_id,
            identity=IDENTITY,
            durable_verifier=reject,
        )

    assert calls == [generation.manifest_sha256]
    assert not (root / "latest.json").exists()
    assert not (root / "verified").exists()
    assert generation.path.exists()


def test_latest_retains_current_and_previous_verified_generation(tmp_path: Path):
    root = tmp_path / "checkpoints"
    for number in range(1, 4):
        generation = create_checkpoint_generation(
            root,
            identity=IDENTITY,
            generation_id=f"step-{number:06d}",
            source_dir=_source(tmp_path, str(number).encode()),
        )
        advance_latest_checkpoint(
            root,
            generation.generation_id,
            identity=IDENTITY,
            durable_verifier=lambda _generation: True,
        )

    assert sorted(path.name for path in (root / "generations").iterdir()) == [
        "step-000002",
        "step-000003",
    ]
    latest = resolve_latest_checkpoint(root, expected_identity=IDENTITY)
    assert latest.generation_id == "step-000003"
    pointer = json.loads((root / "latest.json").read_text())
    assert pointer["generation_id"] == "step-000003"
    assert pointer["previous_generation_id"] == "step-000002"


def test_identity_mismatch_cannot_reuse_root_or_resolve_latest(tmp_path: Path):
    root = tmp_path / "checkpoints"
    generation = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000025",
        source_dir=_source(tmp_path, b"one"),
    )
    advance_latest_checkpoint(
        root,
        generation.generation_id,
        identity=IDENTITY,
        durable_verifier=lambda _generation: True,
    )
    other_identity = {**IDENTITY, "seed": 99}

    with pytest.raises(CheckpointIdentityMismatch, match="identity mismatch"):
        create_checkpoint_generation(
            root,
            identity=other_identity,
            generation_id="step-000050",
            source_dir=_source(tmp_path, b"two"),
        )
    with pytest.raises(CheckpointIdentityMismatch, match="identity mismatch"):
        resolve_latest_checkpoint(root, expected_identity=other_identity)


def test_unverified_generation_is_never_pruned(tmp_path: Path):
    root = tmp_path / "checkpoints"
    for number in range(1, 3):
        generation = create_checkpoint_generation(
            root,
            identity=IDENTITY,
            generation_id=f"step-{number:06d}",
            source_dir=_source(tmp_path, str(number).encode()),
        )
        advance_latest_checkpoint(
            root,
            generation.generation_id,
            identity=IDENTITY,
            durable_verifier=lambda _generation: True,
        )
    unverified = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000003",
        source_dir=_source(tmp_path, b"three"),
    )
    fourth = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000004",
        source_dir=_source(tmp_path, b"four"),
    )

    advance_latest_checkpoint(
        root,
        fourth.generation_id,
        identity=IDENTITY,
        durable_verifier=lambda _generation: True,
    )

    assert unverified.path.exists()
    assert (root / "generations" / "step-000002").exists()
    assert fourth.path.exists()


def test_undeclared_file_and_tampered_latest_pointer_fail_closed(tmp_path: Path):
    root = tmp_path / "checkpoints"
    generation = create_checkpoint_generation(
        root,
        identity=IDENTITY,
        generation_id="step-000025",
        source_dir=_source(tmp_path, b"one"),
    )
    (generation.path / "undeclared.bin").write_bytes(b"surprise")
    with pytest.raises(CheckpointIntegrityError, match="inventory mismatch"):
        advance_latest_checkpoint(
            root,
            generation.generation_id,
            identity=IDENTITY,
            durable_verifier=lambda _generation: True,
        )
    (generation.path / "undeclared.bin").unlink()
    advance_latest_checkpoint(
        root,
        generation.generation_id,
        identity=IDENTITY,
        durable_verifier=lambda _generation: True,
    )

    pointer_path = root / "latest.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["generation_id"] = "step-999999"
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(CheckpointIntegrityError, match="pointer is invalid"):
        resolve_latest_checkpoint(root, expected_identity=IDENTITY)
