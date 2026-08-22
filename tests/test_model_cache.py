from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import intelligent_liars.model_cache as model_cache_module

from intelligent_liars.model_cache import (
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_SNAPSHOT_FILES,
    CacheValidationError,
    build_cache_manifest,
    build_snapshot_plan,
    completion_marker,
    git_blob_oid,
    legal_artifact_descriptors,
    materialize_huggingface_cache,
    verify_snapshot,
)
from intelligent_liars.models import ModelLoadConfig


def _hub_payload(files: dict[str, bytes], *, revision: str = MODEL_REVISION) -> dict:
    siblings = []
    for path, content in files.items():
        item = {
            "rfilename": path,
            "size": len(content),
            "blobId": git_blob_oid(content),
        }
        if path.endswith(".safetensors"):
            item["lfs"] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        siblings.append(item)
    return {"id": MODEL_ID, "sha": revision, "siblings": siblings}


def _complete_files() -> dict[str, bytes]:
    return {path: f"fixture:{path}\n".encode() for path in REQUIRED_SNAPSHOT_FILES}


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _load_build_script():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_tinylora_model_cache.py"
    )
    spec = importlib.util.spec_from_file_location("model_cache_build_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_snapshot_plan_is_exact_revision_and_narrow_inventory() -> None:
    files = _complete_files()
    payload = _hub_payload({**files, "README.md": b"not needed"})

    plan = build_snapshot_plan(payload)

    assert plan["model"] == {"repo_id": MODEL_ID, "revision": MODEL_REVISION}
    assert plan["allow_patterns"] == list(REQUIRED_SNAPSHOT_FILES)
    assert [entry["path"] for entry in plan["files"]] == list(
        REQUIRED_SNAPSHOT_FILES
    )
    assert plan["expected_download_bytes"] == sum(map(len, files.values()))
    assert "README.md" not in {entry["path"] for entry in plan["files"]}
    assert plan["excluded_repository_files"] == ["README.md"]


def test_snapshot_plan_rejects_revision_drift_and_missing_runtime_file() -> None:
    files = _complete_files()
    with pytest.raises(CacheValidationError, match="revision drift"):
        build_snapshot_plan(_hub_payload(files, revision="0" * 40))

    files.pop("tokenizer.json")
    with pytest.raises(CacheValidationError, match="missing required Hub files"):
        build_snapshot_plan(_hub_payload(files))


def test_snapshot_plan_requires_exact_model_identity() -> None:
    payload = _hub_payload(_complete_files())
    payload.pop("id")
    with pytest.raises(CacheValidationError, match="model identity drift"):
        build_snapshot_plan(payload)

    payload["id"] = "SomeOther/model"
    with pytest.raises(CacheValidationError, match="model identity drift"):
        build_snapshot_plan(payload)


def test_verify_snapshot_checks_lfs_sha_git_oid_size_and_extra_files(
    tmp_path: Path,
) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)

    verified = verify_snapshot(tmp_path, plan)
    assert len(verified) == len(REQUIRED_SNAPSHOT_FILES)
    assert all(len(entry["sha256"]) == 64 for entry in verified)

    (tmp_path / "config.json").write_bytes(b"same-size-wrong"[: len(files["config.json"])])
    with pytest.raises(CacheValidationError, match="config.json"):
        verify_snapshot(tmp_path, plan)

    (tmp_path / "config.json").write_bytes(files["config.json"])
    (tmp_path / "surprise.bin").write_bytes(b"unexpected")
    with pytest.raises(CacheValidationError, match="unexpected snapshot files"):
        verify_snapshot(tmp_path, plan)


def test_huggingface_download_metadata_is_ignored_but_not_user_files(
    tmp_path: Path,
) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)
    _write_files(tmp_path / ".cache" / "huggingface", {"download/config.lock": b""})

    assert len(verify_snapshot(tmp_path, plan)) == len(REQUIRED_SNAPSHOT_FILES)


def test_manifest_and_completion_marker_are_content_addressed(tmp_path: Path) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)
    verified = verify_snapshot(tmp_path, plan)

    manifest = build_cache_manifest(
        plan,
        verified,
        bucket="example-bucket",
        base_prefix="model-cache/v1",
    )
    digest = manifest["content_sha256"]
    expected_prefix = (
        "model-cache/v1/qwen--qwen3-vl-8b-thinking/"
        f"{MODEL_REVISION}/{digest}"
    )
    assert manifest["s3"]["object_prefix"] == expected_prefix
    assert manifest["s3"]["manifest_key"] == f"{expected_prefix}/manifest.json"
    assert manifest["s3"]["completion_key"] == f"{expected_prefix}/_COMPLETE.json"
    assert manifest["s3"]["files_prefix"] == f"{expected_prefix}/files"
    assert manifest["complete"] is True

    marker = completion_marker(manifest)
    assert marker["content_sha256"] == digest
    assert marker["manifest_sha256"] == hashlib.sha256(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def test_manifest_digest_changes_when_any_file_changes(tmp_path: Path) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)
    first = build_cache_manifest(plan, verify_snapshot(tmp_path, plan))

    altered = dict(files)
    altered["config.json"] = b"changed config"
    altered_plan = build_snapshot_plan(_hub_payload(altered))
    _write_files(tmp_path, altered)
    second = build_cache_manifest(
        altered_plan, verify_snapshot(tmp_path, altered_plan)
    )

    assert first["content_sha256"] != second["content_sha256"]


def test_corrupt_external_plan_is_rejected_as_validation_error(tmp_path: Path) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)

    plan["expected_download_bytes"] += 1
    with pytest.raises(CacheValidationError, match="total byte count"):
        verify_snapshot(tmp_path, plan)

    plan = build_snapshot_plan(_hub_payload(files))
    plan["files"][0]["source"]["git_blob_oid"] = "not-a-hash"
    with pytest.raises(CacheValidationError, match="Git blob id"):
        verify_snapshot(tmp_path, plan)


def test_completion_marker_requires_valid_s3_manifest(tmp_path: Path) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)
    manifest = build_cache_manifest(plan, verify_snapshot(tmp_path, plan))

    with pytest.raises(CacheValidationError, match="S3 cache contract"):
        completion_marker(manifest)

    manifest = build_cache_manifest(
        plan, verify_snapshot(tmp_path, plan), bucket="example-bucket"
    )
    manifest["total_bytes"] += 1
    with pytest.raises(CacheValidationError, match="byte count is inconsistent"):
        completion_marker(manifest)

    manifest = build_cache_manifest(
        plan, verify_snapshot(tmp_path, plan), bucket="example-bucket"
    )
    manifest["s3"]["files_prefix"] += "-tampered"
    with pytest.raises(CacheValidationError, match="S3 contract"):
        completion_marker(manifest)

    manifest = build_cache_manifest(
        plan, verify_snapshot(tmp_path, plan), bucket="example-bucket"
    )
    manifest["legal"][-1]["content_utf8"] = "tampered but structurally valid\n"
    manifest["legal"][-1]["bytes"] = len(
        manifest["legal"][-1]["content_utf8"].encode()
    )
    manifest["legal"][-1]["sha256"] = hashlib.sha256(
        manifest["legal"][-1]["content_utf8"].encode()
    ).hexdigest()
    with pytest.raises(CacheValidationError, match="legal attribution is inconsistent"):
        completion_marker(manifest)


def test_verified_snapshot_hydrates_model_load_config_cache_layout(
    tmp_path: Path,
) -> None:
    files = _complete_files()
    snapshot = tmp_path / "standalone"
    cache_dir = tmp_path / "hf-cache"
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(snapshot, files)

    report = materialize_huggingface_cache(snapshot, cache_dir, plan)
    config = ModelLoadConfig(**report["model_load_config"])

    assert config.model_name == MODEL_ID
    assert config.revision == MODEL_REVISION
    assert config.cache_dir == str(cache_dir.resolve())
    assert report["required_environment"] == {"HF_HUB_OFFLINE": "1"}
    revision_root = Path(report["snapshot_path"])
    for path, content in files.items():
        cached = revision_root / path
        assert cached.is_symlink()
        assert cached.read_bytes() == content
    assert materialize_huggingface_cache(snapshot, cache_dir, plan) == report

    with pytest.raises(CacheValidationError, match="must not overlap"):
        materialize_huggingface_cache(snapshot, snapshot / "hf-cache", plan)


def test_legal_artifacts_are_adjacent_and_pinned() -> None:
    descriptors = {item["path"]: item for item in legal_artifact_descriptors()}

    assert set(descriptors) == {
        "LICENSE-APACHE-2.0.txt",
        "UPSTREAM_README.md",
        "ATTRIBUTION.json",
        "MODIFICATIONS.md",
    }
    assert descriptors["LICENSE-APACHE-2.0.txt"]["source_url"].startswith(
        "https://www.apache.org/"
    )
    assert MODEL_REVISION in descriptors["UPSTREAM_README.md"]["source_url"]
    assert "modified works" in descriptors["MODIFICATIONS.md"]["content_utf8"]


def test_legal_change_produces_a_new_content_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _complete_files()
    plan = build_snapshot_plan(_hub_payload(files))
    _write_files(tmp_path, files)
    verified = verify_snapshot(tmp_path, plan)
    first = build_cache_manifest(plan, verified)
    original = legal_artifact_descriptors

    def changed_legal() -> list[dict]:
        changed = original()
        changed[-1] = {**changed[-1], "content_utf8": "new notice\n"}
        changed[-1]["bytes"] = len(changed[-1]["content_utf8"].encode())
        changed[-1]["sha256"] = hashlib.sha256(
            changed[-1]["content_utf8"].encode()
        ).hexdigest()
        return changed

    monkeypatch.setattr(model_cache_module, "legal_artifact_descriptors", changed_legal)
    second = build_cache_manifest(plan, verified)

    assert first["content_sha256"] != second["content_sha256"]


def test_download_command_is_inert_without_explicit_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_build_script()
    called = False

    def unexpected_network() -> dict:
        nonlocal called
        called = True
        raise AssertionError("network should not be called")

    monkeypatch.setattr(script, "fetch_hub_model_info", unexpected_network)
    result = script.main(
        [
            "download",
            "--snapshot-dir",
            str(tmp_path / "snapshot"),
            "--plan-output",
            str(tmp_path / "plan.json"),
            "--manifest-output",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert result == 2
    assert called is False
