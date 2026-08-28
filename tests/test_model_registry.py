from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.model_registry import (
    ARTIFACT_KINDS,
    MODEL_REGISTRY_CONFIG_FORMAT,
    MODEL_REGISTRY_FORMAT,
    MODEL_REGISTRY_RECEIPT_FORMAT,
    RegistryError,
    artifact_key,
    build_artifact_record,
    build_registry,
    build_s3_receipt,
    canonical_json_bytes,
    content_identity,
    load_registry,
    load_registry_config,
    save_registry,
    validate_registry,
    verify_s3_roundtrip,
)


def _content() -> bytes:
    return b"a small immutable model artifact\n"


def _record(*, version_id: str = "version-1") -> dict[str, Any]:
    return build_artifact_record(
        kind="successful_experiment",
        run_id="run-20260827-001",
        filename="metrics.json",
        content=_content(),
        bucket="example-bucket",
        version_id=version_id,
        checked_at="2026-08-27T12:00:00Z",
    )


def test_artifact_keys_are_deterministic_and_use_exact_kind_prefixes() -> None:
    digest = hashlib.sha256(_content()).hexdigest()
    assert artifact_key(
        "failed_experiment",
        run_id="run-1",
        content_sha256=digest,
        filename="result.json",
    ) == (
        "model-registry/v1/experiments/failed/run-1/"
        f"{digest}/result.json"
    )
    assert artifact_key(
        "successful_experiment",
        run_id="run-1",
        content_sha256=digest,
        filename="result.json",
    ).startswith("model-registry/v1/experiments/successful/run-1/")
    assert artifact_key(
        "final_model",
        run_id="run-1",
        model_slug="qwen3-vl-8b-thinking",
        content_sha256=digest,
        filename="model.safetensors",
    ).startswith("model-registry/v1/models/final/qwen3-vl-8b-thinking/")


def test_key_builder_rejects_unknown_kind_and_unsafe_components() -> None:
    assert set(ARTIFACT_KINDS) == {
        "failed_experiment",
        "successful_experiment",
        "final_model",
    }
    with pytest.raises(RegistryError, match="artifact kind"):
        artifact_key(
            "control",
            run_id="run-1",
            content_sha256="a" * 64,
            filename="result.json",
        )
    with pytest.raises(RegistryError, match="unsafe"):
        artifact_key(
            "failed_experiment",
            run_id="../escape",
            content_sha256="a" * 64,
            filename="result.json",
        )


def test_content_identity_is_path_independent_and_exact(tmp_path: Path) -> None:
    content = _content()
    first = tmp_path / "first.bin"
    second = tmp_path / "nested" / "second.bin"
    second.parent.mkdir()
    first.write_bytes(content)
    second.write_bytes(content)

    assert content_identity(first) == content_identity(second)
    assert content_identity(first) == {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_content_identity_rejects_symlinks_even_when_target_is_regular(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(_content())
    link = tmp_path / "artifact.bin"
    link.symlink_to(target)

    with pytest.raises(RegistryError, match="symlink"):
        content_identity(link)


def test_record_and_registry_round_trip_through_canonical_json(tmp_path: Path) -> None:
    record = _record()
    registry = build_registry(bucket="example-bucket", artifacts=[record])
    assert registry["format"] == MODEL_REGISTRY_FORMAT
    assert validate_registry(registry) == registry

    path = tmp_path / "registry.json"
    save_registry(path, registry)
    loaded = load_registry(path)
    assert loaded == registry
    assert json.loads(canonical_json_bytes(registry)) == registry


def test_record_identity_changes_when_content_or_key_changes() -> None:
    first = _record()
    second = build_artifact_record(
        kind="successful_experiment",
        run_id="run-20260827-001",
        filename="metrics.json",
        content=b"a different artifact\n",
        bucket="example-bucket",
        version_id="version-1",
        checked_at="2026-08-27T12:00:00Z",
    )
    assert first["artifact_id"] != second["artifact_id"]

    third = build_artifact_record(
        kind="successful_experiment",
        run_id="run-20260827-001",
        filename="other.json",
        content=_content(),
        bucket="example-bucket",
        version_id="version-1",
        checked_at="2026-08-27T12:00:00Z",
    )
    assert first["artifact_id"] != third["artifact_id"]


def test_receipt_requires_non_null_version_id_and_exact_identity() -> None:
    record = _record()
    receipt = record["receipt"]
    assert receipt["format"] == MODEL_REGISTRY_RECEIPT_FORMAT
    assert receipt["verified"] is True
    assert receipt["version_id"] == "version-1"

    with pytest.raises(RegistryError, match="VersionId"):
        build_s3_receipt(
            bucket="example-bucket",
            key=record["s3_key"],
            version_id="null",
            expected=record["content"],
            observed=record["content"],
            checked_at="2026-08-27T12:00:00Z",
        )

    bad = dict(record)
    bad["receipt"] = dict(receipt)
    bad["receipt"]["version_id"] = None
    with pytest.raises(RegistryError, match="VersionId"):
        validate_registry(build_registry(bucket="example-bucket", artifacts=[bad]))


def test_registry_is_fail_closed_for_unknown_fields_wrong_format_and_mismatch() -> None:
    registry = build_registry(bucket="example-bucket", artifacts=[_record()])

    unknown = dict(registry)
    unknown["unexpected"] = True
    with pytest.raises(RegistryError, match="unknown|additional"):
        validate_registry(unknown)

    wrong_version = dict(registry)
    wrong_version["format"] = "model_registry_v99"
    with pytest.raises(RegistryError, match="format"):
        validate_registry(wrong_version)

    mismatch = json.loads(json.dumps(registry))
    mismatch["artifacts"][0]["receipt"]["key"] += ".tampered"
    with pytest.raises(RegistryError, match="key"):
        validate_registry(mismatch)


def test_s3_roundtrip_uses_head_and_exact_versioned_get_without_listing() -> None:
    record = _record()

    class FakeBody:
        def read(self) -> bytes:
            return _content()

    class FakeS3:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, str]]] = []

        def head_object(self, **kwargs: str) -> dict[str, Any]:
            self.calls.append(("head_object", kwargs))
            return {
                "VersionId": "version-1",
                "ContentLength": len(_content()),
            }

        def get_object(self, **kwargs: str) -> dict[str, FakeBody]:
            self.calls.append(("get_object", kwargs))
            return {"Body": FakeBody()}

    s3 = FakeS3()
    receipt = verify_s3_roundtrip(s3, record)
    assert receipt["verified"] is True
    assert receipt["version_id"] == "version-1"
    assert [name for name, _ in s3.calls] == ["head_object", "get_object"]
    assert s3.calls[1][1]["VersionId"] == "version-1"
    assert all(name != "list_objects_v2" for name, _ in s3.calls)


def test_s3_roundtrip_fails_closed_on_missing_version_or_hash_mismatch() -> None:
    record = _record()

    class FakeBody:
        def read(self) -> bytes:
            return b"tampered"

    class MissingVersionS3:
        def head_object(self, **kwargs: str) -> dict[str, Any]:
            return {"VersionId": "null", "ContentLength": len(_content())}

    with pytest.raises(RegistryError, match="VersionId"):
        verify_s3_roundtrip(MissingVersionS3(), record)

    class HashMismatchS3:
        def head_object(self, **kwargs: str) -> dict[str, Any]:
            return {
                "VersionId": "version-1",
                "ContentLength": len(b"tampered"),
            }

        def get_object(self, **kwargs: str) -> dict[str, FakeBody]:
            return {"Body": FakeBody()}

    with pytest.raises(RegistryError, match="SHA-256|hash|bytes"):
        verify_s3_roundtrip(HashMismatchS3(), record)


def test_registry_config_is_strict_and_loadable() -> None:
    config = load_registry_config(
        Path("configs/model_registry_v1.json")
    )
    assert config["format"] == MODEL_REGISTRY_CONFIG_FORMAT
    assert config["base_model_cache"]["content_sha256"] == "bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8"
    assert config["base_model_cache"]["s3_prefix_status"] == "unresolved_manifest_prefix_mismatch"
