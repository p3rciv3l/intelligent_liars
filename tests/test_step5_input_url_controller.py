from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.model_cache import (
    MODEL_ID,
    MODEL_REVISION,
    REQUIRED_SNAPSHOT_FILES,
    _content_sha256,
    canonical_json_bytes,
    completion_marker,
    legal_artifact_descriptors,
)
from intelligent_liars.step5_input_hydration import validate_url_manifest
from intelligent_liars.step5_input_url_controller import prepare_input_urls


BUCKET = "frozen-step5-test"
ACCOUNT = "123456789012"
REGION = "us-west-2"


class Body:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class FakeSTS:
    def __init__(self, account: str = ACCOUNT) -> None:
        self.account = account

    def get_caller_identity(self) -> dict[str, str]:
        return {"Account": self.account}


class FakeS3:
    def __init__(
        self, objects: dict[str, dict[str, Any]], *, region: str = REGION
    ) -> None:
        self.objects = objects
        self.region = region
        self.puts: list[dict[str, Any]] = []

    def get_bucket_location(self, *, Bucket: str) -> dict[str, str]:
        assert Bucket == BUCKET
        return {"LocationConstraint": self.region}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Body]:
        assert Bucket == BUCKET
        return {"Body": Body(self.objects[Key]["payload"])}

    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str
    ) -> dict[str, Any]:
        assert Bucket == BUCKET
        assert ChecksumMode == "ENABLED"
        item = self.objects[Key]
        result: dict[str, Any] = {"ContentLength": item["bytes"]}
        if item.get("head_sha"):
            result["ChecksumSHA256"] = base64.b64encode(
                bytes.fromhex(item["sha256"])
            ).decode()
            result["ChecksumType"] = item.get("checksum_type", "FULL_OBJECT")
        if item.get("metadata_sha", True):
            result["Metadata"] = {"sha256": item["sha256"]}
        return result

    def generate_presigned_url(
        self,
        operation: str,
        *,
        Params: dict[str, str],
        ExpiresIn: int,
        HttpMethod: str,
    ) -> str:
        assert operation == "get_object"
        assert HttpMethod == "GET"
        return (
            f"https://objects.example/{Params['Key']}?expiry={ExpiresIn}&secret=hidden"
        )

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        assert kwargs["IfNoneMatch"] == "*"
        key = kwargs["Key"]
        if key in self.objects:
            raise RuntimeError("precondition failed")
        payload = kwargs["Body"]
        self.objects[key] = _object(payload)
        self.puts.append(kwargs)
        return {"ETag": "immutable"}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _object(
    payload: bytes,
    *,
    size: int | None = None,
    sha256: str | None = None,
    metadata_sha: bool = True,
) -> dict[str, Any]:
    return {
        "bytes": len(payload) if size is None else size,
        "metadata_sha": metadata_sha,
        "payload": payload,
        "sha256": sha256 or _sha(payload),
    }


def _fixture() -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    model_prefix = "model/cache"
    file_payloads = {
        name: f"payload:{name}".encode() for name in REQUIRED_SNAPSHOT_FILES
    }
    files = [
        {
            "bytes": 128 * 1024 * 1024 if index == 4 else len(file_payloads[name]),
            "path": name,
            "sha256": _sha(file_payloads[name]),
        }
        for index, name in enumerate(REQUIRED_SNAPSHOT_FILES)
    ]
    model_content = _content_sha256(files)
    model_manifest = {
        "complete": True,
        "content_sha256": model_content,
        "files": files,
        "format": "tinylora_model_cache_manifest_v1",
        "legal": legal_artifact_descriptors(),
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "s3": {
            "bucket": BUCKET,
            "completion_key": f"{model_prefix}/_COMPLETE.json",
            "files_prefix": f"{model_prefix}/files",
            "legal_prefix": f"{model_prefix}/legal",
            "manifest_key": f"{model_prefix}/manifest.json",
            "object_prefix": model_prefix,
            "publication_order": [
                "files",
                "legal",
                "manifest",
                "completion_marker_last",
            ],
            "schema": "tinylora_model_cache_s3_v1",
        },
        "source_plan_sha256": "0" * 64,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    # The shared validator requires the production suffix even in a unit fixture.
    model_prefix = (
        f"model-cache/v1/qwen--qwen3-vl-8b-thinking/{MODEL_REVISION}/{model_content}"
    )
    model_manifest["s3"].update(
        {
            "completion_key": f"{model_prefix}/_COMPLETE.json",
            "files_prefix": f"{model_prefix}/files",
            "legal_prefix": f"{model_prefix}/legal",
            "manifest_key": f"{model_prefix}/manifest.json",
            "object_prefix": model_prefix,
        }
    )
    model_manifest_bytes = canonical_json_bytes(model_manifest)
    model_complete_bytes = canonical_json_bytes(completion_marker(model_manifest))

    frozen_prefix = "step5-inputs/v2/plan/archive"
    frozen_archive = b"frozen archive"
    plan_sha = "1" * 64
    probe_receipt_sha = "2" * 64
    frozen_complete = {
        "archive_bytes": len(frozen_archive),
        "archive_sha256": _sha(frozen_archive),
        "plan_sha256": plan_sha,
        "probe_qualification_receipt_sha256": probe_receipt_sha,
    }
    frozen_complete_bytes = canonical_json_bytes(frozen_complete)

    pixmo_prefix = "step5-assets/v1/pixmo/content"
    pixmo_archive = b"pixmo archive"
    pixmo_content = "3" * 64
    pixmo_manifest_bytes = canonical_json_bytes({"content_sha256": pixmo_content})
    pixmo_complete = {
        "archive_bytes": len(pixmo_archive),
        "archive_name": "pixmo.tar.gz",
        "archive_sha256": _sha(pixmo_archive),
        "manifest_commitment": pixmo_content,
    }
    pixmo_complete_bytes = canonical_json_bytes(pixmo_complete)

    objects: dict[str, dict[str, Any]] = {
        f"{model_prefix}/manifest.json": _object(model_manifest_bytes),
        f"{model_prefix}/_COMPLETE.json": _object(model_complete_bytes),
        frozen_prefix: _object(frozen_archive),
        "step5-inputs/v2/plan/_COMPLETE.json": _object(frozen_complete_bytes),
        f"{pixmo_prefix}/manifest.json": _object(pixmo_manifest_bytes),
        f"{pixmo_prefix}/_COMPLETE.json": _object(pixmo_complete_bytes),
        f"{pixmo_prefix}/pixmo.tar.gz": _object(pixmo_archive),
    }
    for item in files:
        objects[f"{model_prefix}/files/{item['path']}"] = _object(
            file_payloads[item["path"]], size=item["bytes"], sha256=item["sha256"]
        )

    packet = {
        "execution": {"enabled": False},
        "format": "tinylora_step5_canary_launch_packet_v1",
        "identity": {
            "model_content_sha256": model_content,
            "model_revision": MODEL_REVISION,
            "pixmo_content_sha256": pixmo_content,
            "plan_sha256": plan_sha,
            "probe_qualification_receipt_sha256": probe_receipt_sha,
        },
        "remote_inputs": {
            "frozen_inputs_completion_sha256": _sha(frozen_complete_bytes),
            "frozen_inputs_tar_sha256": _sha(frozen_archive),
            "model_completion_sha256": _sha(model_complete_bytes),
            "model_manifest_sha256": _sha(model_manifest_bytes),
            "model_s3_prefix": f"s3://{BUCKET}/{model_prefix}/",
            "pixmo_completion_sha256": _sha(pixmo_complete_bytes),
            "pixmo_manifest_sha256": _sha(pixmo_manifest_bytes),
            "pixmo_s3_prefix": f"s3://{BUCKET}/{pixmo_prefix}/",
            "pixmo_tar_sha256": _sha(pixmo_archive),
            "plan_s3_uri": f"s3://{BUCKET}/{frozen_prefix}",
        },
    }
    return packet, objects, f"{model_prefix}/files/{REQUIRED_SNAPSHOT_FILES[4]}"


def _run(tmp_path: Path, packet: dict[str, Any], s3: FakeS3) -> dict[str, Any]:
    return prepare_input_urls(
        packet,
        s3=s3,
        sts=FakeSTS(),
        account_id=ACCOUNT,
        region=REGION,
        manifest_bucket=BUCKET,
        manifest_key="controller/attempt-1/input-url-manifest.json",
        manifest_output=tmp_path / "input-url-manifest.json",
        url_file=tmp_path / "input-url-manifest.url",
        host_gate_url_file=tmp_path / "host-gate.url",
        receipt_path=tmp_path / "receipt.json",
        expiry_seconds=3600,
        now=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )


def test_prepares_private_manifest_bootstrap_host_gate_and_receipt(
    tmp_path: Path,
) -> None:
    packet, objects, largest_key = _fixture()
    s3 = FakeS3(objects)

    receipt = _run(tmp_path, packet, s3)

    manifest = json.loads((tmp_path / "input-url-manifest.json").read_text())
    validate_url_manifest(manifest)
    assert manifest["controller"] == {
        "account_id": ACCOUNT,
        "bucket": BUCKET,
        "created_at": "2026-08-22T00:00:00Z",
        "expires_at": "2026-08-22T01:00:00Z",
        "expiry_seconds": 3600,
        "manifest_key": "controller/attempt-1/input-url-manifest.json",
        "region": REGION,
    }
    assert largest_key in (tmp_path / "host-gate.url").read_text()
    assert "https://" not in json.dumps(receipt)
    assert "secret=" not in (tmp_path / "receipt.json").read_text()
    for name in (
        "input-url-manifest.json",
        "input-url-manifest.url",
        "host-gate.url",
        "receipt.json",
    ):
        assert os.stat(tmp_path / name).st_mode & 0o777 == 0o600
    assert s3.puts[0]["ServerSideEncryption"] == "AES256"
    assert s3.puts[0]["Metadata"]["sha256"] == receipt["manifest"]["sha256"]


def test_rejects_wrong_account_region_and_frozen_hash(tmp_path: Path) -> None:
    packet, objects, _ = _fixture()
    with pytest.raises(ValueError, match="approved account"):
        prepare_input_urls(
            packet,
            s3=FakeS3(objects),
            sts=FakeSTS("999999999999"),
            account_id=ACCOUNT,
            region=REGION,
            manifest_bucket=BUCKET,
            manifest_key="controller/m.json",
            manifest_output=tmp_path / "m",
            url_file=tmp_path / "u",
            host_gate_url_file=tmp_path / "h",
            receipt_path=tmp_path / "r",
            expiry_seconds=3600,
        )
    with pytest.raises(ValueError, match="approved AWS region"):
        _run(tmp_path, packet, FakeS3(objects, region="us-east-1"))
    packet["remote_inputs"]["frozen_inputs_tar_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Frozen input completion"):
        _run(tmp_path, packet, FakeS3(objects))


def test_large_object_requires_exact_head_sha256(tmp_path: Path) -> None:
    packet, objects, largest_key = _fixture()
    objects[largest_key]["metadata_sha"] = False
    objects[largest_key]["head_sha"] = False

    with pytest.raises(ValueError, match="exact SHA-256 commitment"):
        _run(tmp_path, packet, FakeS3(objects))


def test_small_object_can_be_stream_verified(tmp_path: Path) -> None:
    packet, objects, _ = _fixture()
    key = next(key for key in objects if key.endswith("manifest.json"))
    objects[key]["metadata_sha"] = False

    receipt = _run(tmp_path, packet, FakeS3(objects))

    verified = next(item for item in receipt["objects"] if item["key"] == key)
    assert verified["verification"] == "stream_sha256"


def test_outputs_are_no_clobber_and_reject_symlink_parent(tmp_path: Path) -> None:
    packet, objects, _ = _fixture()
    existing = tmp_path / "input-url-manifest.url"
    existing.write_text("preserve")
    s3 = FakeS3(objects)
    with pytest.raises(FileExistsError):
        _run(tmp_path, packet, s3)
    assert existing.read_text() == "preserve"
    assert s3.puts == []

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_input_urls(
            packet,
            s3=FakeS3(objects),
            sts=FakeSTS(),
            account_id=ACCOUNT,
            region=REGION,
            manifest_bucket=BUCKET,
            manifest_key="controller/m.json",
            manifest_output=linked / "m",
            url_file=tmp_path / "u2",
            host_gate_url_file=tmp_path / "h2",
            receipt_path=tmp_path / "r2",
            expiry_seconds=3600,
        )


def test_local_promotion_failure_is_clean_and_remote_manifest_is_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, objects, _ = _fixture()
    s3 = FakeS3(objects)
    real_link = os.link

    with monkeypatch.context() as context:
        context.setattr(
            os, "link", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk"))
        )
        with pytest.raises(OSError, match="disk"):
            _run(tmp_path, packet, s3)
    assert not any(tmp_path.iterdir())
    remote_key = "controller/attempt-1/input-url-manifest.json"
    assert remote_key in objects

    monkeypatch.setattr(os, "link", real_link)
    receipt = _run(tmp_path, packet, s3)
    assert receipt["manifest"]["key"] == remote_key
    assert len(s3.puts) == 1


def test_promotion_collision_preserves_other_writer_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, objects, _ = _fixture()
    s3 = FakeS3(objects)
    real_link = os.link
    links = 0

    def collide_after_first_link(*args: Any, **kwargs: Any) -> None:
        nonlocal links
        real_link(*args, **kwargs)
        links += 1
        if links == 1:
            (tmp_path / "input-url-manifest.url").write_text("other writer")

    monkeypatch.setattr(os, "link", collide_after_first_link)
    with pytest.raises(FileExistsError):
        _run(tmp_path, packet, s3)

    assert not (tmp_path / "input-url-manifest.json").exists()
    assert (tmp_path / "input-url-manifest.url").read_text() == "other writer"
    assert not (tmp_path / "host-gate.url").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_composite_checksum_is_not_treated_as_exact(tmp_path: Path) -> None:
    packet, objects, largest_key = _fixture()
    objects[largest_key]["metadata_sha"] = False
    objects[largest_key]["head_sha"] = True
    objects[largest_key]["checksum_type"] = "COMPOSITE"

    with pytest.raises(ValueError, match="exact SHA-256 commitment"):
        _run(tmp_path, packet, FakeS3(objects))


def test_url_manifest_rejects_inconsistent_expiry(tmp_path: Path) -> None:
    packet, objects, _ = _fixture()
    s3 = FakeS3(objects)
    # Produce a real manifest once, then prove the worker rejects timestamp drift.
    _run(tmp_path, packet, s3)
    manifest = json.loads((tmp_path / "input-url-manifest.json").read_text())
    manifest["controller"]["expires_at"] = "2026-08-22T02:00:00Z"
    with pytest.raises(ValueError, match="expiry binding"):
        validate_url_manifest(manifest)
