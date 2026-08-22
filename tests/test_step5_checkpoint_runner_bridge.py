from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intelligent_liars.durable_checkpoints import create_checkpoint_generation
from intelligent_liars.step5_checkpoint_bridge import (
    ENVELOPE_FORMAT,
    build_checkpoint_archive,
    build_controller_ack,
    build_upload_request,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tinylora_step5_screen.py"
SPEC = importlib.util.spec_from_file_location("screen_bridge_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCREEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCREEN)


def _keys(tmp_path: Path) -> tuple[Path, Path]:
    private = tmp_path / "private.pem"
    public = tmp_path / "public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "ED25519", "-out", str(private)],
        check=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
    )
    return private, public


def _generation_and_envelope(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "state.pt").write_bytes(b"state")
    generation = create_checkpoint_generation(
        tmp_path / "store",
        identity={"run_id": "runner-boundary"},
        generation_id="step-000001-boundary",
        source_dir=source,
    )
    archive_identity = build_checkpoint_archive(generation, tmp_path / "archive.tar")
    requested_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    request = build_upload_request(
        generation,
        archive_identity=archive_identity,
        request_nonce="a" * 32,
        requested_at=requested_at,
    )
    private, public = _keys(tmp_path)
    ack = build_controller_ack(
        request,
        object_ref="s3://bucket/run/generation.tar",
        object_version="v1",
        verified_at=requested_at,
        private_key_path=private,
        public_key_path=public,
    )
    return generation, {"format": ENVELOPE_FORMAT, "request": request, "ack": ack}, public


def _emitter(tmp_path: Path, payload: dict) -> str:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    script = tmp_path / "emit.py"
    script.write_text("import pathlib,sys\nsys.stdout.write(pathlib.Path(sys.argv[1]).read_text())\n")
    return " ".join(
        shlex.quote(value) for value in (sys.executable, str(script), str(payload_path))
    )


def test_runner_reverifies_signed_controller_envelope(tmp_path: Path):
    generation, envelope, public = _generation_and_envelope(tmp_path)
    verifier = SCREEN.command_durability_verifier(
        _emitter(tmp_path, envelope), controller_public_key=public
    )

    receipt = verifier(generation)

    assert receipt["generation_id"] == generation.generation_id
    assert receipt["object_version"] == "v1"


def test_forged_v2_stdout_cannot_cross_runner_boundary(tmp_path: Path):
    generation, _envelope, public = _generation_and_envelope(tmp_path)
    forged = {
        "format": "tinylora_step5_checkpoint_durability_receipt_v2",
        "generation_id": generation.generation_id,
        "manifest_sha256": generation.manifest_sha256,
        "archive_sha256": "f" * 64,
        "size_bytes": 1,
        "object_ref": "s3://bucket/forged",
        "object_version": "fake",
        "controller_key_id": "e" * 64,
        "verified_at": datetime.now(UTC).isoformat(),
        "verified": True,
    }
    verifier = SCREEN.command_durability_verifier(
        _emitter(tmp_path, forged), controller_public_key=public
    )

    with pytest.raises(SCREEN.CheckpointIntegrityError, match="signed controller"):
        verifier(generation)
