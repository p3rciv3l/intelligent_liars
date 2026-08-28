from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.model_cache import (
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_SNAPSHOT_FILES,
    CacheValidationError,
    canonical_json_bytes,
    legal_artifact_descriptors,
    verify_huggingface_cache_for_loading,
)
from intelligent_liars.models import ModelLoadConfig, load_processor


def _cache_fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    cache = tmp_path / "cache"
    snapshot = (
        cache / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / MODEL_REVISION
    )
    snapshot.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for index, relative in enumerate(REQUIRED_SNAPSHOT_FILES):
        content = f"fixture-{index}-{relative}\n".encode()
        (snapshot / relative).write_bytes(content)
        files.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    content_payload = {
        "format": "tinylora_model_cache_content_v1",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "files": files,
        "legal": legal_artifact_descriptors(),
    }
    content_sha256 = hashlib.sha256(canonical_json_bytes(content_payload)).hexdigest()
    manifest = {
        "format": "tinylora_model_cache_manifest_v1",
        "complete": True,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "source_plan_sha256": "a" * 64,
        "content_sha256": content_sha256,
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
        "legal": legal_artifact_descriptors(),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return cache, manifest_path, content_sha256, manifest_sha256


def test_verified_cache_receipt_binds_all_runtime_identity_fields(
    tmp_path: Path,
) -> None:
    cache, manifest, content_sha, manifest_sha = _cache_fixture(tmp_path)

    identity = verify_huggingface_cache_for_loading(
        cache_dir=cache,
        manifest_path=manifest,
        expected_model_sha256=content_sha,
        expected_manifest_sha256=manifest_sha,
    )

    assert identity.to_mapping() == {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_sha256": content_sha,
        "snapshot_manifest_sha256": manifest_sha,
    }


def test_verified_cache_reuses_startup_receipt_without_rehashing_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, manifest, content_sha, manifest_sha = _cache_fixture(tmp_path)
    first = verify_huggingface_cache_for_loading(
        cache_dir=cache,
        manifest_path=manifest,
        expected_model_sha256=content_sha,
        expected_manifest_sha256=manifest_sha,
    )

    def fail_rehash(_path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
        del chunk_bytes
        raise AssertionError("verified unchanged shards must not be rehashed")

    monkeypatch.setattr("intelligent_liars.model_cache.sha256_file", fail_rehash)
    second = verify_huggingface_cache_for_loading(
        cache_dir=cache,
        manifest_path=manifest,
        expected_model_sha256=content_sha,
        expected_manifest_sha256=manifest_sha,
    )

    assert second == first


def test_verified_cache_rejects_file_substitution(tmp_path: Path) -> None:
    cache, manifest, content_sha, manifest_sha = _cache_fixture(tmp_path)
    target = (
        cache
        / "models--Qwen--Qwen3-VL-8B-Thinking"
        / "snapshots"
        / MODEL_REVISION
        / REQUIRED_SNAPSHOT_FILES[0]
    )
    target.write_bytes(b"substituted bytes with a different identity\n")

    with pytest.raises(CacheValidationError, match="hash changed"):
        verify_huggingface_cache_for_loading(
            cache_dir=cache,
            manifest_path=manifest,
            expected_model_sha256=content_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_verified_cache_rejects_missing_file(tmp_path: Path) -> None:
    cache, manifest, content_sha, manifest_sha = _cache_fixture(tmp_path)
    target = (
        cache
        / "models--Qwen--Qwen3-VL-8B-Thinking"
        / "snapshots"
        / MODEL_REVISION
        / REQUIRED_SNAPSHOT_FILES[-1]
    )
    target.unlink()

    with pytest.raises(CacheValidationError, match="missing model cache file"):
        verify_huggingface_cache_for_loading(
            cache_dir=cache,
            manifest_path=manifest,
            expected_model_sha256=content_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_verified_cache_rejects_manifest_revision_drift(tmp_path: Path) -> None:
    cache, manifest_path, content_sha, _manifest_sha = _cache_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["revision"] = "main"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    changed_manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()

    with pytest.raises(CacheValidationError, match="revision or repository changed"):
        verify_huggingface_cache_for_loading(
            cache_dir=cache,
            manifest_path=manifest_path,
            expected_model_sha256=content_sha,
            expected_manifest_sha256=changed_manifest_sha,
        )


def test_processor_bundle_carries_independent_snapshot_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, manifest, content_sha, manifest_sha = _cache_fixture(tmp_path)
    tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        eos_token="<eos>",
        padding_side="right",
    )
    processor = types.SimpleNamespace(tokenizer=tokenizer)

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_id: str, **_kwargs: object) -> object:
            return processor

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoProcessor=FakeAutoProcessor),
    )
    bundle = load_processor(
        ModelLoadConfig(
            cache_dir=str(cache),
            snapshot_manifest_path=str(manifest),
            expected_model_sha256=content_sha,
            expected_snapshot_manifest_sha256=manifest_sha,
        )
    )

    assert bundle.verified_snapshot == {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_sha256": content_sha,
        "snapshot_manifest_sha256": manifest_sha,
    }
    assert bundle.model_identity == bundle.verified_snapshot
