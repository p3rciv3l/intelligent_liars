from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "safe_fetch_artifact.py"
)
SPEC = importlib.util.spec_from_file_location("safe_fetch_artifact", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
safe_fetch_artifact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = safe_fetch_artifact
SPEC.loader.exec_module(safe_fetch_artifact)


def test_hash_size_and_temp_path(tmp_path: Path) -> None:
    artifact = tmp_path / "activation.h5"
    content = b"not really hdf5, but enough for helper verification"
    artifact.write_bytes(content)

    assert safe_fetch_artifact.file_size(artifact) == len(content)
    assert safe_fetch_artifact.sha256_file(artifact) == hashlib.sha256(
        content
    ).hexdigest()
    assert safe_fetch_artifact.temp_path_for(artifact) == tmp_path / (
        "activation.h5.tmp"
    )


def test_atomic_promote_moves_temp_to_destination(tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"
    destination = tmp_path / "artifact.h5"
    temp_path.write_bytes(b"verified")

    safe_fetch_artifact.atomic_promote(temp_path, destination)

    assert destination.read_bytes() == b"verified"
    assert not temp_path.exists()


def test_atomic_promote_refuses_existing_destination(tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"
    destination = tmp_path / "artifact.h5"
    temp_path.write_bytes(b"new")
    destination.write_bytes(b"old")

    with pytest.raises(safe_fetch_artifact.FetchError, match="already exists"):
        safe_fetch_artifact.atomic_promote(temp_path, destination)

    assert destination.read_bytes() == b"old"
    assert temp_path.read_bytes() == b"new"


def test_atomic_promote_can_overwrite_when_requested(tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"
    destination = tmp_path / "artifact.h5"
    temp_path.write_bytes(b"new")
    destination.write_bytes(b"old")

    safe_fetch_artifact.atomic_promote(
        temp_path,
        destination,
        overwrite=True,
    )

    assert destination.read_bytes() == b"new"
    assert not temp_path.exists()


def test_verify_temp_file_detects_size_mismatch(tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"
    temp_path.write_bytes(b"abc")
    metadata = safe_fetch_artifact.RemoteMetadata(
        byte_count=4,
        sha256=hashlib.sha256(b"abc").hexdigest(),
    )

    with pytest.raises(safe_fetch_artifact.FetchError, match="byte-size"):
        safe_fetch_artifact.verify_temp_file(temp_path, metadata)


def test_verify_temp_file_detects_sha_mismatch(tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"
    temp_path.write_bytes(b"abc")
    metadata = safe_fetch_artifact.RemoteMetadata(
        byte_count=3,
        sha256=hashlib.sha256(b"abcd").hexdigest(),
    )

    with pytest.raises(safe_fetch_artifact.FetchError, match="sha256"):
        safe_fetch_artifact.verify_temp_file(temp_path, metadata)


def test_query_remote_metadata_uses_ssh_and_parses_outputs(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is False
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
        assert kwargs["text"] is True
        if "stat -c %s" in command[-1]:
            stdout = "12\n"
        else:
            stdout = f"{hashlib.sha256(b'content').hexdigest()}  file.h5\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(safe_fetch_artifact.subprocess, "run", fake_run)

    metadata = safe_fetch_artifact.query_remote_metadata(
        safe_fetch_artifact.SshSpec(target="user@example-host", port=2222),
        "/remote/path/file.h5",
    )

    assert metadata.byte_count == 12
    assert metadata.sha256 == hashlib.sha256(b"content").hexdigest()
    assert calls == [
        [
            "ssh",
            "-p",
            "2222",
            "user@example-host",
            "LC_ALL=C stat -c %s -- /remote/path/file.h5",
        ],
        [
            "ssh",
            "-p",
            "2222",
            "user@example-host",
            "LC_ALL=C sha256sum -- /remote/path/file.h5",
        ],
    ]


def test_copy_remote_to_temp_writes_fresh_file(monkeypatch, tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"

    def fake_run(command, **kwargs):
        assert "cat -- /remote/path/file.h5" in command
        kwargs["stdout"].write(b"copied bytes")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(safe_fetch_artifact.subprocess, "run", fake_run)

    safe_fetch_artifact.copy_remote_to_temp(
        safe_fetch_artifact.SshSpec(target="user@example-host"),
        "/remote/path/file.h5",
        temp_path,
    )

    assert temp_path.read_bytes() == b"copied bytes"


def test_prepare_temp_path_requires_explicit_replace(tmp_path: Path) -> None:
    temp_path = tmp_path / "artifact.h5.tmp"
    temp_path.write_bytes(b"stale")

    with pytest.raises(safe_fetch_artifact.FetchError, match="already exists"):
        safe_fetch_artifact.prepare_temp_path(temp_path, replace_temp=False)

    safe_fetch_artifact.prepare_temp_path(temp_path, replace_temp=True)

    assert not temp_path.exists()
