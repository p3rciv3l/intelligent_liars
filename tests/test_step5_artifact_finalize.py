from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from intelligent_liars.step5_artifact_finalize import (
    ArtifactContractError,
    build_deterministic_artifact_archive,
    finalize_artifacts,
    load_expected_inventory,
    publish_presigned_put,
    read_presigned_url,
    stage_inventory,
)
from intelligent_liars.step5_artifact_store import (
    LIFECYCLE_ARTIFACT_MANIFEST_FORMAT,
    validate_lifecycle_artifact_manifest,
)


def _inventory(tmp_path: Path, files: list[str]) -> Path:
    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_expected_artifact_inventory_v1",
                "files": files,
            }
        )
    )
    return path


def test_stages_exact_inventory_and_builds_reproducible_archive(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"loss": 1.5}\n')
    checkpoint = source / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "state.bin").write_bytes(b"state")
    expected = load_expected_inventory(
        _inventory(tmp_path, ["result.json", "checkpoint_store.tar"])
    )
    artifact_root = tmp_path / "artifacts"
    records = stage_inventory(
        artifact_root,
        expected,
        file_mappings=[("result.json", source / "result.json")],
        tree_archive_mappings=[("checkpoint_store.tar", checkpoint)],
    )
    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    first_receipt = build_deterministic_artifact_archive(artifact_root, records, first)
    second_receipt = build_deterministic_artifact_archive(artifact_root, records, second)
    assert first_receipt == second_receipt
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r") as archive:
        assert archive.getnames() == ["checkpoint_store.tar", "result.json"]


def test_rejects_unexpected_files_symlinks_and_nonfinite_json(tmp_path: Path):
    source = tmp_path / "result.json"
    source.write_text('{"loss": NaN}\n')
    inventory = load_expected_inventory(_inventory(tmp_path, ["result.json"]))
    with pytest.raises(ArtifactContractError, match="nonfinite"):
        stage_inventory(
            tmp_path / "artifacts",
            inventory,
            file_mappings=[("result.json", source)],
            tree_archive_mappings=[],
        )

    source.write_text('{"loss": 1}\n')
    artifact_root = tmp_path / "artifacts-two"
    artifact_root.mkdir()
    (artifact_root / "extra").write_text("unexpected")
    with pytest.raises(ArtifactContractError, match="extra"):
        stage_inventory(
            artifact_root,
            inventory,
            file_mappings=[("result.json", source)],
            tree_archive_mappings=[],
        )

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "real").write_text("x")
    (tree / "link").symlink_to(tree / "real")
    inventory_two = load_expected_inventory(_inventory(tmp_path, ["tree.tar"]))
    with pytest.raises(ArtifactContractError, match="symlink"):
        stage_inventory(
            tmp_path / "artifacts-three",
            inventory_two,
            file_mappings=[],
            tree_archive_mappings=[("tree.tar", tree)],
        )


def test_staging_refuses_symlinked_parent_before_writing(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text('{"ok": true}\n')
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "nested").symlink_to(outside)
    inventory = load_expected_inventory(_inventory(tmp_path, ["nested/result.json"]))
    with pytest.raises(ArtifactContractError, match="parent is a symlink"):
        stage_inventory(
            artifact_root,
            inventory,
            file_mappings=[("nested/result.json", source)],
            tree_archive_mappings=[],
        )
    assert not (outside / "result.json").exists()


def test_presigned_url_file_must_be_private(tmp_path: Path):
    path = tmp_path / "url"
    path.write_text("https://example.invalid/signed")
    path.chmod(0o644)
    with pytest.raises(ArtifactContractError, match="0600"):
        read_presigned_url(url_file=path, url_env=None)
    path.chmod(0o600)
    assert read_presigned_url(url_file=path, url_env=None).startswith("https://")


def test_presigned_put_is_no_clobber_and_412_requires_controller(monkeypatch, tmp_path: Path):
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"artifact bytes")
    captured: dict[str, str] = {}

    def fake_put(url, *, data, headers, timeout):
        del data, timeout
        captured["url"] = url
        captured.update(headers)
        return SimpleNamespace(status_code=412)

    monkeypatch.setattr("intelligent_liars.step5_artifact_finalize.requests.put", fake_put)
    result = publish_presigned_put(archive, url="https://example.invalid/private-query")
    assert result == {"uploaded": False, "preexisting": True, "attempts": 1}
    assert captured["If-None-Match"] == "*"
    assert "private-query" not in json.dumps(result)


def test_finalize_emits_canonical_v2_manifest_after_upload(monkeypatch, tmp_path: Path):
    source = tmp_path / "result.json"
    source.write_text('{"ok": true}\n')
    inventory = _inventory(tmp_path, ["result.json"])
    url = tmp_path / "signed-url"
    url.write_text("https://example.invalid/private")
    url.chmod(0o600)
    monkeypatch.setattr(
        "intelligent_liars.step5_artifact_finalize.requests.put",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=200),
    )
    artifact_root = tmp_path / "artifacts"
    receipt = finalize_artifacts(
        artifact_root=artifact_root,
        expected_inventory=inventory,
        run_id="canary-v1",
        durable_uri="s3://bucket/immutable/artifacts.tar",
        archive_path=tmp_path / "durable.tar",
        file_mappings=[("result.json", source)],
        tree_archive_mappings=[],
        presigned_url_file=url,
        presigned_url_env=None,
    )
    manifest = json.loads((artifact_root / "artifact_manifest.json").read_text())
    assert manifest["format"] == LIFECYCLE_ARTIFACT_MANIFEST_FORMAT
    assert validate_lifecycle_artifact_manifest(manifest) == manifest
    assert receipt["controller_verification_required"] is True
    assert receipt["publication"]["uploaded"] is True
    assert manifest["durable_object"]["sha256"] == hashlib.sha256(
        (tmp_path / "durable.tar").read_bytes()
    ).hexdigest()


def test_finalize_can_generate_summary_bound_to_result_and_prerequisite(
    monkeypatch, tmp_path: Path
):
    result = tmp_path / "result.json"
    result.write_text('{"ok": true}\n')
    prerequisite = tmp_path / "prerequisite_receipt.json"
    prerequisite.write_text('{"qualified": true}\n')
    inventory = _inventory(
        tmp_path,
        ["result.json", "prerequisite_receipt.json", "canary_summary.json"],
    )
    url = tmp_path / "signed-url"
    url.write_text("https://example.invalid/private")
    url.chmod(0o600)
    monkeypatch.setattr(
        "intelligent_liars.step5_artifact_finalize.requests.put",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=200),
    )
    artifact_root = tmp_path / "artifacts"
    finalize_artifacts(
        artifact_root=artifact_root,
        expected_inventory=inventory,
        run_id="canary-v1",
        durable_uri="s3://bucket/immutable/artifacts.tar",
        archive_path=tmp_path / "durable.tar",
        file_mappings=[
            ("result.json", result),
            ("prerequisite_receipt.json", prerequisite),
        ],
        tree_archive_mappings=[],
        presigned_url_file=url,
        presigned_url_env=None,
        canary_summary_target="canary_summary.json",
    )
    summary = json.loads((artifact_root / "canary_summary.json").read_text())
    assert summary["run_id"] == "canary-v1"
    assert summary["worker_self_attestation_only"] is True
    assert summary["controller_verification_required"] is True


def test_idempotent_retry_refuses_changed_local_or_remote_bytes(monkeypatch, tmp_path: Path):
    source = tmp_path / "result.json"
    source.write_text('{"ok": true}\n')
    inventory = _inventory(tmp_path, ["result.json"])
    url = tmp_path / "signed-url"
    url.write_text("https://example.invalid/private")
    os.chmod(url, 0o600)
    monkeypatch.setattr(
        "intelligent_liars.step5_artifact_finalize.requests.put",
        lambda *_args, **_kwargs: SimpleNamespace(status_code=412),
    )
    kwargs = dict(
        artifact_root=tmp_path / "artifacts",
        expected_inventory=inventory,
        run_id="canary-v1",
        durable_uri="s3://bucket/immutable/artifacts.tar",
        archive_path=tmp_path / "durable.tar",
        file_mappings=[("result.json", source)],
        tree_archive_mappings=[],
        presigned_url_file=url,
        presigned_url_env=None,
    )
    first = finalize_artifacts(**kwargs)
    second = finalize_artifacts(**kwargs)
    assert first["artifact_set_id"] == second["artifact_set_id"]
    source.write_text('{"ok": false}\n')
    with pytest.raises(ArtifactContractError, match="refusing to replace"):
        finalize_artifacts(**kwargs)
