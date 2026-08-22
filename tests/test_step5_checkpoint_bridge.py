from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intelligent_liars.durable_checkpoints import create_checkpoint_generation
from intelligent_liars.step5_checkpoint_bridge import (
    BridgeContractError,
    build_checkpoint_archive,
    build_controller_ack,
    build_upload_request,
    controller_key_id,
    verify_checkpoint_archive,
    verify_controller_ack,
)


IDENTITY = {"run_id": "canary", "seed": 20260822, "plan_sha256": "a" * 64}


def _generation(tmp_path: Path, generation_id: str = "step-000025-test"):
    source = tmp_path / "source"
    source.mkdir()
    (source / "step5_state.pt").write_bytes(b"state")
    (source / "adapter_state.safetensors").write_bytes(b"adapter")
    return create_checkpoint_generation(
        tmp_path / "checkpoints",
        identity=IDENTITY,
        generation_id=generation_id,
        source_dir=source,
    )


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    private = tmp_path / "controller-private.pem"
    public = tmp_path / "controller-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    private.chmod(0o600)
    return private, public


def _accepted(tmp_path: Path):
    generation = _generation(tmp_path)
    archive = tmp_path / "generation.tar"
    archive_identity = build_checkpoint_archive(generation, archive)
    request = build_upload_request(
        generation,
        archive_identity=archive_identity,
        request_nonce="b" * 32,
        requested_at="2026-08-22T12:00:00Z",
    )
    private, public = _keys(tmp_path)
    ack = build_controller_ack(
        request,
        object_ref="s3://bucket/checkpoints/step-000025-test/generation.tar",
        object_version="version-1",
        verified_at="2026-08-22T12:01:00Z",
        private_key_path=private,
        public_key_path=public,
    )
    return generation, archive, request, ack, public


def test_archive_is_deterministic_and_round_trip_verified(tmp_path: Path):
    generation = _generation(tmp_path)
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    one = build_checkpoint_archive(generation, first)
    two = build_checkpoint_archive(generation, second)

    assert one == two
    assert first.read_bytes() == second.read_bytes()
    verified = verify_checkpoint_archive(
        first,
        expected_generation_id=generation.generation_id,
        expected_manifest_sha256=generation.manifest_sha256,
    )
    assert verified == one


def test_controller_ack_is_signed_and_bound_to_exact_request(tmp_path: Path):
    generation, _archive, request, ack, public = _accepted(tmp_path)
    receipt = verify_controller_ack(
        ack,
        request=request,
        public_key_path=public,
        now=datetime(2026, 8, 22, 12, 2, tzinfo=UTC),
    )

    assert receipt["generation_id"] == generation.generation_id
    assert receipt["manifest_sha256"] == generation.manifest_sha256
    assert receipt["verified"] is True
    assert receipt["controller_key_id"] == controller_key_id(public)
    assert "signature" not in receipt


@pytest.mark.parametrize("mutation", ["generation", "manifest", "nonce"])
def test_wrong_ack_binding_is_rejected(tmp_path: Path, mutation: str):
    _generation, _archive, request, ack, public = _accepted(tmp_path)
    changed = dict(ack)
    field = {
        "generation": "generation_id",
        "manifest": "manifest_sha256",
        "nonce": "request_nonce",
    }[mutation]
    changed[field] = "wrong" if field != "manifest_sha256" else "f" * 64
    with pytest.raises(BridgeContractError):
        verify_controller_ack(
            changed,
            request=request,
            public_key_path=public,
            now=datetime(2026, 8, 22, 12, 2, tzinfo=UTC),
        )


def test_missing_stale_and_self_attested_ack_are_rejected(tmp_path: Path):
    _generation, _archive, request, ack, public = _accepted(tmp_path)
    with pytest.raises(BridgeContractError, match="acknowledgement is missing"):
        verify_controller_ack(
            None,
            request=request,
            public_key_path=public,
            now=datetime(2026, 8, 22, 12, 2, tzinfo=UTC),
        )

    with pytest.raises(BridgeContractError, match="stale"):
        verify_controller_ack(
            ack,
            request=request,
            public_key_path=public,
            now=datetime(2026, 8, 22, 14, 2, tzinfo=UTC),
            max_ack_age=timedelta(minutes=30),
        )

    forged = dict(ack)
    forged["object_version"] = "self-attested"
    forged["signature"] = ack["signature"]
    with pytest.raises(BridgeContractError, match="signature"):
        verify_controller_ack(
            forged,
            request=request,
            public_key_path=public,
            now=datetime(2026, 8, 22, 12, 2, tzinfo=UTC),
        )


def test_request_and_ack_never_contain_presigned_query_secrets(tmp_path: Path):
    _generation, _archive, request, ack, _public = _accepted(tmp_path)
    serialized = json.dumps({"request": request, "ack": ack}, sort_keys=True)
    assert "X-Amz-" not in serialized
    assert "?" not in ack["object_ref"]
