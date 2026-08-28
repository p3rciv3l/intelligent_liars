from __future__ import annotations

import hashlib
import ast
import io
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_final_checkpoint_publication import (
    FinalCheckpointPublicationError,
    FilesystemFinalCheckpointStore,
    S3FinalCheckpointStore,
    build_final_checkpoint_target,
    open_final_checkpoint_publication_receipt,
    publish_final_checkpoint,
    retire_verified_local_checkpoint_weights,
    validate_compact_vast_output_contract,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "checkpoint-publication"
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"qwen3_vl"}\n')
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"weights-a")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"weights-b")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 18},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    files = []
    for path in sorted(checkpoint.iterdir()):
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    manifest = {
        "format": "truth_editing_finalist_checkpoint_manifest_v1",
        "trial_id": "trial-final",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "self_sha256": "a" * 64,
    }
    return root, {"manifest": manifest}


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "format": "intelligent_liars_model_registry_config_v1",
                "schema_version": 1,
                "registry": {
                    "bucket": "private-models-example",
                    "base_prefix": "model-registry/v1",
                },
                "base_model_cache": {},
            }
        )
    )
    return path


def test_filesystem_publication_is_sharded_versioned_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    checkpoint_root, verified = _checkpoint(tmp_path)
    evidence = tmp_path / "adaptive-finalization-audit.json"
    evidence.write_text('{"format":"audit-v1"}\n')
    target = build_final_checkpoint_target(
        _registry(tmp_path), model_slug="qwen3-vl-truth-editing-r10"
    )
    store = FilesystemFinalCheckpointStore(tmp_path / "remote")
    receipt_path = tmp_path / "compact/final-model-publication-receipt.json"

    first = publish_final_checkpoint(
        checkpoint_root,
        verified_checkpoint=verified,
        evidence_paths=(evidence,),
        target=target,
        store=store,
        receipt_path=receipt_path,
    )
    second = publish_final_checkpoint(
        checkpoint_root,
        verified_checkpoint=verified,
        evidence_paths=(evidence,),
        target=target,
        store=store,
        receipt_path=receipt_path,
    )

    assert first == second
    assert first["status"] == "remote_roundtrip_verified"
    assert first["checkpoint_file_count"] == 4
    assert len(first["objects"]) == 5
    assert all(item["version_id"] for item in first["objects"])
    assert all(item["head_verified"] and item["roundtrip_verified"] for item in first["objects"])
    assert first["offhost_finalization_state"]["version_id"]
    assert first["offhost_finalization_state"]["roundtrip_verified"] is True
    assert store.upload_count == 7  # five files, one manifest, one finalization state
    assert open_final_checkpoint_publication_receipt(receipt_path) == first
    assert not any(path.suffix == ".safetensors" for path in receipt_path.parent.rglob("*"))

    retire_verified_local_checkpoint_weights(
        checkpoint_root,
        verified_checkpoint=verified,
        publication_receipt=first,
    )
    assert not (checkpoint_root / "checkpoint").exists()
    assert (checkpoint_root / "checkpoint-manifest.json").exists() is False
    assert receipt_path.is_file()


def test_publication_preserves_candidate_qualified_evidence_paths_and_replays(
    tmp_path: Path,
) -> None:
    checkpoint_root, verified = _checkpoint(tmp_path)
    output_root = tmp_path / "run-output"
    evidence_paths = []
    for trial_id in ("trial-0007", "trial-0019", "trial-0042"):
        candidate_root = output_root / "causal-finalization" / trial_id
        candidate_root.mkdir(parents=True)
        for basename in ("receipt.json", "backend-config.json", "plan.json"):
            path = candidate_root / basename
            path.write_text(f'{trial_id}:{basename}\n')
            evidence_paths.append(path)
    target = build_final_checkpoint_target(_registry(tmp_path), model_slug="model")
    store = FilesystemFinalCheckpointStore(tmp_path / "remote")
    receipt_path = tmp_path / "compact/final-model-publication-receipt.json"

    first = publish_final_checkpoint(
        checkpoint_root,
        verified_checkpoint=verified,
        evidence_paths=tuple(reversed(evidence_paths)),
        evidence_root=output_root,
        target=target,
        store=store,
        receipt_path=receipt_path,
    )
    second = publish_final_checkpoint(
        checkpoint_root,
        verified_checkpoint=verified,
        evidence_paths=tuple(evidence_paths),
        evidence_root=output_root,
        target=target,
        store=store,
        receipt_path=receipt_path,
    )

    assert first == second
    published_evidence = {
        item["logical_path"]: item["sha256"]
        for item in first["objects"]
        if item["logical_path"].startswith("evidence/")
    }
    assert published_evidence == {
        f"evidence/causal-finalization/{path.parent.name}/{path.name}": _sha(path)
        for path in evidence_paths
    }


def test_publication_rejects_evidence_outside_root_or_through_symlink(
    tmp_path: Path,
) -> None:
    checkpoint_root, verified = _checkpoint(tmp_path)
    output_root = tmp_path / "run-output"
    output_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n")
    target = build_final_checkpoint_target(_registry(tmp_path), model_slug="model")

    with pytest.raises(FinalCheckpointPublicationError, match="evidence root"):
        publish_final_checkpoint(
            checkpoint_root,
            verified_checkpoint=verified,
            evidence_paths=(outside,),
            evidence_root=output_root,
            target=target,
            store=FilesystemFinalCheckpointStore(tmp_path / "remote-outside"),
            receipt_path=tmp_path / "outside-receipt.json",
        )

    linked_parent = output_root / "linked-candidate"
    linked_parent.symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(FinalCheckpointPublicationError, match="symlink"):
        publish_final_checkpoint(
            checkpoint_root,
            verified_checkpoint=verified,
            evidence_paths=(linked_parent / outside.name,),
            evidence_root=output_root,
            target=target,
            store=FilesystemFinalCheckpointStore(tmp_path / "remote-symlink"),
            receipt_path=tmp_path / "symlink-receipt.json",
        )


def test_strict_open_rejects_duplicate_candidate_logical_paths(
    tmp_path: Path,
) -> None:
    checkpoint_root, verified = _checkpoint(tmp_path)
    output_root = tmp_path / "run-output"
    candidate_a = output_root / "causal-finalization/trial-a/receipt.json"
    candidate_b = output_root / "causal-finalization/trial-b/receipt.json"
    candidate_a.parent.mkdir(parents=True)
    candidate_b.parent.mkdir(parents=True)
    candidate_a.write_text("candidate-a\n")
    candidate_b.write_text("candidate-b\n")
    receipt_path = tmp_path / "publication-receipt.json"
    receipt = publish_final_checkpoint(
        checkpoint_root,
        verified_checkpoint=verified,
        evidence_paths=(candidate_a, candidate_b),
        evidence_root=output_root,
        target=build_final_checkpoint_target(_registry(tmp_path), model_slug="model"),
        store=FilesystemFinalCheckpointStore(tmp_path / "remote"),
        receipt_path=receipt_path,
    )
    evidence = [
        item for item in receipt["objects"] if item["logical_path"].startswith("evidence/")
    ]
    evidence[1]["logical_path"] = evidence[0]["logical_path"]
    unsigned = dict(receipt)
    unsigned.pop("self_sha256")
    receipt["self_sha256"] = _json_sha(unsigned)
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(FinalCheckpointPublicationError, match="unique and sorted"):
        open_final_checkpoint_publication_receipt(receipt_path)


class _Body:
    def __init__(self, data: bytes) -> None:
        self._stream = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class _FakeS3:
    def __init__(self) -> None:
        self.version = 0
        self.objects: dict[tuple[str, str], bytes] = {}
        self.current: dict[str, str] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.multipart_created = 0
        self.uploaded_parts: list[int] = []

    def get_bucket_versioning(self, **kwargs):
        return {"Status": "Enabled"}

    def head_object(self, **kwargs):
        key = kwargs["Key"]
        version = kwargs.get("VersionId", self.current[key])
        data = self.objects[(key, version)]
        return {
            "VersionId": version,
            "ContentLength": len(data),
            "Metadata": {"sha256": hashlib.sha256(data).hexdigest()},
        }

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        version = kwargs.get("VersionId", self.current[key])
        return {"VersionId": version, "Body": _Body(self.objects[(key, version)])}

    def put_object(self, **kwargs):
        data = kwargs["Body"].read()
        return self._save(kwargs["Key"], data)

    def create_multipart_upload(self, **kwargs):
        self.multipart_created += 1
        upload_id = f"upload-{self.multipart_created}"
        self.uploads[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs):
        data = kwargs["Body"]
        self.uploads[kwargs["UploadId"]][kwargs["PartNumber"]] = data
        self.uploaded_parts.append(kwargs["PartNumber"])
        return {"ETag": f'"part-{kwargs["PartNumber"]}"'}

    def list_parts(self, **kwargs):
        return {
            "Parts": [
                {"PartNumber": number, "ETag": f'"part-{number}"'}
                for number in sorted(self.uploads[kwargs["UploadId"]])
            ],
            "IsTruncated": False,
        }

    def complete_multipart_upload(self, **kwargs):
        data = b"".join(
            self.uploads[kwargs["UploadId"]][part["PartNumber"]]
            for part in kwargs["MultipartUpload"]["Parts"]
        )
        return self._save(kwargs["Key"], data)

    def _save(self, key: str, data: bytes):
        self.version += 1
        version = f"version-{self.version}"
        self.objects[(key, version)] = data
        self.current[key] = version
        return {"VersionId": version}


def test_s3_adapter_uses_multipart_and_version_pinned_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "shard.safetensors"
    source.write_bytes(b"0123456789")
    fake = _FakeS3()
    store = S3FinalCheckpointStore(
        fake, bucket="private-models-example", multipart_threshold_bytes=5, part_size_bytes=5
    )

    result = store.publish_file(
        "model-registry/v1/models/final/model/content/objects/hash.blob", source
    )

    assert fake.multipart_created == 1
    assert fake.uploaded_parts == [1, 2]
    assert result["version_id"] == "version-1"
    assert result["head_verified"] is True
    assert result["roundtrip_verified"] is True


def test_s3_multipart_resume_reuses_already_uploaded_parts(tmp_path: Path) -> None:
    source = tmp_path / "shard.safetensors"
    source.write_bytes(b"0123456789")
    key = "model-registry/v1/models/final/model/content/objects/hash.blob"
    fake = _FakeS3()
    fake.uploads["upload-existing"] = {1: b"01234"}
    resume = tmp_path / "resume.json"
    resume.write_text(
        json.dumps(
            {
                "format": "truth_editing_s3_multipart_resume_v1",
                "key": key,
                "bytes": 10,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "upload_id": "upload-existing",
                "parts": [{"PartNumber": 1, "ETag": '"part-1"'}],
            }
        )
    )
    store = S3FinalCheckpointStore(
        fake,
        bucket="private-models-example",
        multipart_threshold_bytes=5,
        part_size_bytes=5,
    )

    result = store.publish_file(key, source, resume_path=resume)

    assert fake.multipart_created == 0
    assert fake.uploaded_parts == [2]
    assert result["roundtrip_verified"] is True
    assert not resume.exists()


def test_publication_fails_closed_on_manifest_or_remote_identity_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_root, verified = _checkpoint(tmp_path)
    verified["manifest"]["files"][0]["sha256"] = "0" * 64
    target = build_final_checkpoint_target(_registry(tmp_path), model_slug="model")
    with pytest.raises(FinalCheckpointPublicationError, match="checkpoint inventory"):
        publish_final_checkpoint(
            checkpoint_root,
            verified_checkpoint=verified,
            evidence_paths=(),
            target=target,
            store=FilesystemFinalCheckpointStore(tmp_path / "remote"),
            receipt_path=tmp_path / "receipt.json",
        )


def test_local_weights_survive_until_matching_remote_receipt_exists(tmp_path: Path) -> None:
    checkpoint_root, verified = _checkpoint(tmp_path)
    with pytest.raises(FinalCheckpointPublicationError, match="remote verification"):
        retire_verified_local_checkpoint_weights(
            checkpoint_root,
            verified_checkpoint=verified,
            publication_receipt={
                "format": "wrong",
                "status": "pending",
                "checkpoint_manifest_sha256": "a" * 64,
            },
        )
    assert (checkpoint_root / "checkpoint/model-00001-of-00002.safetensors").is_file()


def test_compact_vast_output_contract_excludes_weights_and_stays_below_one_gib() -> None:
    validate_compact_vast_output_contract(
        expected_outputs=(
            "finalization/final-model-publication-receipt.json",
            "finalization/adaptive-finalization-receipt.json",
        ),
        maximum_upload_gib=1.0,
    )
    with pytest.raises(FinalCheckpointPublicationError, match="model weights"):
        validate_compact_vast_output_contract(
            expected_outputs=(
                "finalization/checkpoint-publication/checkpoint/model.safetensors",
            ),
            maximum_upload_gib=1.0,
        )
    with pytest.raises(FinalCheckpointPublicationError, match="1 GiB"):
        validate_compact_vast_output_contract(
            expected_outputs=("finalization/final-model-publication-receipt.json",),
            maximum_upload_gib=1.01,
        )


def test_production_controller_cannot_complete_before_remote_publication() -> None:
    script = Path(__file__).parents[1] / "scripts/run_truth_editing_cuda_fleet_controller.py"
    tree = ast.parse(script.read_text())
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    publication = next(node for node in calls if node.func.id == "publish_final_checkpoint")
    controller_write = next(
        node for node in calls
        if node.func.id == "_write_json_immutable"
        and any(
            isinstance(child, ast.Constant)
            and child.value == "adaptive-controller-result.json"
            for child in ast.walk(node)
        )
    )
    assert publication.lineno < controller_write.lineno
