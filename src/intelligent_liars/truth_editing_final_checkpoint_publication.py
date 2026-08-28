"""Durable, content-addressed publication of a full finalist checkpoint.

The local finalist exporter owns model mutation and ``save_pretrained``.  This
module starts only after that checkpoint has been strictly reopened.  Its one
public publication interface streams every checkpoint shard and finalization
evidence file to a private, versioned object store, verifies an exact
version-pinned HEAD and streamed GET hash, and writes a metadata-only receipt.

Weights are deliberately never placed in W&B or the compact Vast output tar.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


RECEIPT_FORMAT = "truth_editing_final_checkpoint_remote_publication_v1"
REMOTE_MANIFEST_FORMAT = "truth_editing_final_checkpoint_remote_manifest_v1"
FINALIZATION_STATE_FORMAT = "truth_editing_offhost_finalization_state_v1"
DEFAULT_MULTIPART_THRESHOLD_BYTES = 64 * 1024 * 1024
DEFAULT_PART_SIZE_BYTES = 64 * 1024 * 1024
MAX_COMPACT_OUTPUT_GIB = 1.0
_SHA = re.compile(r"^[0-9a-f]{64}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


class FinalCheckpointPublicationError(RuntimeError):
    """The final checkpoint cannot be trusted as durably published."""


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise FinalCheckpointPublicationError("value is not canonical JSON") from error


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise FinalCheckpointPublicationError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FinalCheckpointPublicationError(f"{label} must be nonempty trimmed text")
    return value


def _safe_key(value: Any, label: str = "object key") -> str:
    text = _text(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text or "//" in text:
        raise FinalCheckpointPublicationError(f"{label} is unsafe")
    return text


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FinalCheckpointPublicationError(f"{label} must be a regular file")
    return path


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise FinalCheckpointPublicationError(f"{label} must be a real directory")
    return path


def _write_metadata_only(path: Path, value: Mapping[str, Any]) -> None:
    data = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise FinalCheckpointPublicationError(f"immutable receipt differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FinalCheckpointTarget:
    bucket: str
    key_prefix: str
    registry_config_sha256: str

    def __post_init__(self) -> None:
        if _BUCKET.fullmatch(self.bucket) is None:
            raise FinalCheckpointPublicationError("registry bucket is invalid")
        _safe_key(self.key_prefix, "final-model registry prefix")
        _digest(self.registry_config_sha256, "registry config identity")


def build_final_checkpoint_target(
    registry_config_path: Path | str, *, model_slug: str
) -> FinalCheckpointTarget:
    """Resolve the private final-model namespace from the checked-in registry."""

    path = _regular(Path(registry_config_path), "model registry config")
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinalCheckpointPublicationError("model registry config is invalid JSON") from error
    if not isinstance(raw, Mapping) or raw.get("format") != (
        "intelligent_liars_model_registry_config_v1"
    ) or raw.get("schema_version") != 1:
        raise FinalCheckpointPublicationError("model registry config format differs")
    registry = raw.get("registry")
    if not isinstance(registry, Mapping) or set(registry) != {"bucket", "base_prefix"}:
        raise FinalCheckpointPublicationError("model registry namespace differs")
    slug = _text(model_slug, "model slug")
    if _SLUG.fullmatch(slug) is None:
        raise FinalCheckpointPublicationError("model slug is unsafe")
    base = _safe_key(registry.get("base_prefix"), "model registry base prefix").rstrip("/")
    return FinalCheckpointTarget(
        bucket=_text(registry.get("bucket"), "model registry bucket"),
        key_prefix=f"{base}/models/final/{slug}",
        registry_config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


class FinalCheckpointStore(Protocol):
    def ensure_versioning(self) -> None: ...

    def publish_file(
        self, key: str, path: Path, *, resume_path: Path | None = None
    ) -> dict[str, Any]: ...


class FilesystemFinalCheckpointStore:
    """Versioned filesystem adapter used for deterministic end-to-end tests."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.upload_count = 0

    def ensure_versioning(self) -> None:
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise FinalCheckpointPublicationError("filesystem store root is unsafe")
        self.root.mkdir(parents=True, exist_ok=True)

    def _current_path(self, key: str) -> Path:
        return self.root / "current" / (_safe_key(key) + ".json")

    def _version_path(self, key: str, version_id: str) -> Path:
        return self.root / "versions" / _safe_key(key) / f"{_text(version_id, 'version ID')}.blob"

    def _verify(self, key: str, version_id: str, expected: Mapping[str, Any]) -> dict[str, Any]:
        path = _regular(self._version_path(key, version_id), "versioned object")
        observed = _file_identity(path)
        if observed != dict(expected):
            raise FinalCheckpointPublicationError("versioned object round-trip identity differs")
        return {
            "key": key,
            "version_id": version_id,
            "bytes": observed["bytes"],
            "sha256": observed["sha256"],
            "head_verified": True,
            "roundtrip_verified": True,
        }

    def publish_file(
        self, key: str, path: Path, *, resume_path: Path | None = None
    ) -> dict[str, Any]:
        del resume_path
        self.ensure_versioning()
        key = _safe_key(key)
        source = _regular(path, "publication source")
        identity = _file_identity(source)
        current_path = self._current_path(key)
        if current_path.exists():
            try:
                current = json.loads(_regular(current_path, "current object metadata").read_text())
            except (UnicodeError, json.JSONDecodeError) as error:
                raise FinalCheckpointPublicationError("current object metadata is invalid") from error
            if not isinstance(current, Mapping) or set(current) != {
                "version_id", "bytes", "sha256"
            }:
                raise FinalCheckpointPublicationError("current object metadata differs")
            if current["bytes"] == identity["bytes"] and current["sha256"] == identity["sha256"]:
                return self._verify(key, _text(current["version_id"], "version ID"), identity)
            raise FinalCheckpointPublicationError("content-addressed object already differs")
        version_id = uuid.uuid4().hex
        destination = self._version_path(key, version_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        self.upload_count += 1
        _write_metadata_only(
            current_path,
            {"version_id": version_id, "bytes": identity["bytes"], "sha256": identity["sha256"]},
        )
        return self._verify(key, version_id, identity)


class S3FinalCheckpointStore:
    """Versioned S3 adapter with resumable multipart upload and exact verification."""

    def __init__(
        self,
        client: Any,
        *,
        bucket: str,
        multipart_threshold_bytes: int = DEFAULT_MULTIPART_THRESHOLD_BYTES,
        part_size_bytes: int = DEFAULT_PART_SIZE_BYTES,
    ) -> None:
        if _BUCKET.fullmatch(bucket) is None:
            raise FinalCheckpointPublicationError("S3 bucket is invalid")
        if multipart_threshold_bytes <= 0 or part_size_bytes <= 0:
            raise FinalCheckpointPublicationError("multipart sizes must be positive")
        self.client = client
        self.bucket = bucket
        self.multipart_threshold_bytes = multipart_threshold_bytes
        self.part_size_bytes = part_size_bytes

    def ensure_versioning(self) -> None:
        response = self.client.get_bucket_versioning(Bucket=self.bucket)
        if not isinstance(response, Mapping) or response.get("Status") != "Enabled":
            raise FinalCheckpointPublicationError("final checkpoint bucket versioning is not enabled")

    @staticmethod
    def _missing(error: Exception) -> bool:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        return code in {"404", "NoSuchKey", "NotFound"} or isinstance(error, KeyError)

    def _head(self, key: str, version_id: str | None = None) -> Mapping[str, Any] | None:
        arguments = {"Bucket": self.bucket, "Key": key}
        if version_id is not None:
            arguments["VersionId"] = version_id
        try:
            response = self.client.head_object(**arguments)
        except Exception as error:
            if version_id is None and self._missing(error):
                return None
            raise FinalCheckpointPublicationError("S3 HEAD failed") from error
        if not isinstance(response, Mapping):
            raise FinalCheckpointPublicationError("S3 HEAD response is invalid")
        return response

    def _verify(
        self, key: str, version_id: str, expected: Mapping[str, Any]
    ) -> dict[str, Any]:
        head = self._head(key, version_id)
        assert head is not None
        if head.get("VersionId") != version_id or head.get("ContentLength") != expected["bytes"]:
            raise FinalCheckpointPublicationError("S3 HEAD identity differs")
        metadata = head.get("Metadata")
        if not isinstance(metadata, Mapping) or metadata.get("sha256") != expected["sha256"]:
            raise FinalCheckpointPublicationError("S3 HEAD content hash metadata differs")
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=key, VersionId=version_id
            )
            body = response["Body"]
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = body.read(8 * 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        except Exception as error:
            raise FinalCheckpointPublicationError("S3 version-pinned round trip failed") from error
        if {"bytes": size, "sha256": digest.hexdigest()} != dict(expected):
            raise FinalCheckpointPublicationError("S3 round-trip identity differs")
        return {
            "key": key,
            "version_id": version_id,
            "bytes": size,
            "sha256": digest.hexdigest(),
            "head_verified": True,
            "roundtrip_verified": True,
        }

    def _existing(self, key: str, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        head = self._head(key)
        if head is None:
            return None
        version = head.get("VersionId")
        metadata = head.get("Metadata")
        if (
            not isinstance(version, str)
            or not version
            or version == "null"
            or head.get("ContentLength") != identity["bytes"]
            or not isinstance(metadata, Mapping)
            or metadata.get("sha256") != identity["sha256"]
        ):
            raise FinalCheckpointPublicationError("content-addressed S3 object already differs")
        return self._verify(key, version, identity)

    def _resume_state(self, path: Path | None, *, key: str, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        if path is None or not path.exists():
            return None
        try:
            value = json.loads(_regular(path, "multipart resume state").read_text())
        except (UnicodeError, json.JSONDecodeError) as error:
            raise FinalCheckpointPublicationError("multipart resume state is invalid") from error
        if not isinstance(value, Mapping) or set(value) != {
            "format", "key", "bytes", "sha256", "upload_id", "parts"
        } or value.get("format") != "truth_editing_s3_multipart_resume_v1":
            raise FinalCheckpointPublicationError("multipart resume state fields differ")
        if value.get("key") != key or value.get("bytes") != identity["bytes"] or value.get("sha256") != identity["sha256"]:
            raise FinalCheckpointPublicationError("multipart resume state identity differs")
        return dict(value)

    def _multipart(self, key: str, source: Path, identity: Mapping[str, Any], resume_path: Path | None) -> str:
        state = self._resume_state(resume_path, key=key, identity=identity)
        if state is None:
            response = self.client.create_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                Metadata={"sha256": identity["sha256"]},
            )
            upload_id = _text(response.get("UploadId"), "multipart upload ID")
            state = {
                "format": "truth_editing_s3_multipart_resume_v1",
                "key": key,
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
                "upload_id": upload_id,
                "parts": [],
            }
            if resume_path is not None:
                _write_metadata_only(resume_path, state)
        upload_id = _text(state["upload_id"], "multipart upload ID")
        listed = self.client.list_parts(
            Bucket=self.bucket, Key=key, UploadId=upload_id
        )
        if listed.get("IsTruncated") is True:
            raise FinalCheckpointPublicationError("multipart part listing is truncated")
        remote_parts = {
            int(item["PartNumber"]): _text(item["ETag"], "multipart ETag")
            for item in listed.get("Parts", [])
        }
        expected_parts = (identity["bytes"] + self.part_size_bytes - 1) // self.part_size_bytes
        completed: list[dict[str, Any]] = []
        with source.open("rb") as handle:
            for part_number in range(1, expected_parts + 1):
                data = handle.read(self.part_size_bytes)
                etag = remote_parts.get(part_number)
                if etag is None:
                    response = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=data,
                    )
                    etag = _text(response.get("ETag"), "multipart ETag")
                completed.append({"PartNumber": part_number, "ETag": etag})
                state["parts"] = completed
                if resume_path is not None:
                    # Resume state is intentionally mutable operational state.
                    resume_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = resume_path.with_name(f".{resume_path.name}.tmp")
                    temporary.write_bytes(_canonical(state))
                    temporary.replace(resume_path)
        response = self.client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": completed},
        )
        if resume_path is not None:
            resume_path.unlink(missing_ok=True)
        return _text(response.get("VersionId"), "S3 VersionId")

    def publish_file(
        self, key: str, path: Path, *, resume_path: Path | None = None
    ) -> dict[str, Any]:
        self.ensure_versioning()
        key = _safe_key(key)
        source = _regular(path, "publication source")
        identity = _file_identity(source)
        existing = self._existing(key, identity)
        if existing is not None:
            return existing
        if identity["bytes"] >= self.multipart_threshold_bytes:
            version_id = self._multipart(key, source, identity, resume_path)
        else:
            with source.open("rb") as body:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=body,
                    Metadata={"sha256": identity["sha256"]},
                )
            version_id = _text(response.get("VersionId"), "S3 VersionId")
        if version_id == "null":
            raise FinalCheckpointPublicationError("S3 object lacks immutable VersionId")
        return self._verify(key, version_id, identity)


def _checkpoint_inventory(
    checkpoint_root: Path, verified_checkpoint: Mapping[str, Any]
) -> tuple[str, list[tuple[str, Path, dict[str, Any]]]]:
    manifest = verified_checkpoint.get("manifest")
    if not isinstance(manifest, Mapping):
        raise FinalCheckpointPublicationError("verified checkpoint manifest is missing")
    manifest_sha = _digest(manifest.get("self_sha256"), "checkpoint manifest identity")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FinalCheckpointPublicationError("checkpoint inventory is empty")
    checkpoint_dir = checkpoint_root / "checkpoint"
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise FinalCheckpointPublicationError("checkpoint directory is unsafe")
    expected: list[tuple[str, Path, dict[str, Any]]] = []
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise FinalCheckpointPublicationError("checkpoint inventory fields differ")
        relative = _safe_key(item["path"], "checkpoint relative path")
        path = _regular(checkpoint_dir / relative, "checkpoint file")
        identity = _file_identity(path)
        if identity != {"bytes": item["bytes"], "sha256": item["sha256"]}:
            raise FinalCheckpointPublicationError("checkpoint inventory identity differs")
        expected.append((f"checkpoint/{relative}", path, identity))
    actual = {
        path.relative_to(checkpoint_dir).as_posix()
        for path in checkpoint_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != {item[0].removeprefix("checkpoint/") for item in expected}:
        raise FinalCheckpointPublicationError("checkpoint inventory differs from disk")
    if manifest.get("file_count") != len(expected) or manifest.get("total_bytes") != sum(
        item[2]["bytes"] for item in expected
    ):
        raise FinalCheckpointPublicationError("checkpoint inventory totals differ")
    return manifest_sha, expected


def _temporary_json(value: Mapping[str, Any]) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    directory = tempfile.TemporaryDirectory(prefix="truth-editing-final-publication-")
    path = Path(directory.name) / "object.json"
    path.write_bytes(_canonical(value))
    return directory, path


def _evidence_inventory(
    evidence_paths: Sequence[Path | str], *, evidence_root: Path | str | None
) -> list[tuple[str, Path, dict[str, Any]]]:
    """Bind evidence to stable, root-relative logical paths.

    Basenames are not identities: every relative directory is retained so the
    same receipt/config/plan names from several finalists remain distinct.  A
    caller-supplied root is the production seam and is checked before any
    upload.  Root inference remains only for backwards-compatible one-off use.
    """

    if not evidence_paths:
        return []
    raw_paths = [Path(raw) for raw in evidence_paths]
    if evidence_root is None:
        regular_paths = [_regular(path, "finalization evidence") for path in raw_paths]
        resolved_paths = [path.resolve(strict=True) for path in regular_paths]
        common_parent = Path(
            os.path.commonpath([str(path.parent) for path in resolved_paths])
        )
        root = _directory(common_parent, "inferred finalization evidence root")
        source_paths = resolved_paths
    else:
        lexical_root = Path(evidence_root)
        root = _directory(lexical_root, "finalization evidence root").resolve(strict=True)
        source_paths = []
        for raw_path in raw_paths:
            if ".." in raw_path.parts:
                raise FinalCheckpointPublicationError(
                    "finalization evidence path is unsafe"
                )
            candidate = raw_path if raw_path.is_absolute() else root / raw_path
            source_paths.append(candidate)

    inventory: list[tuple[str, Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for source in source_paths:
        if ".." in source.parts:
            raise FinalCheckpointPublicationError("finalization evidence path is unsafe")
        try:
            relative = source.relative_to(root)
        except ValueError as error:
            raise FinalCheckpointPublicationError(
                "finalization evidence is outside its evidence root"
            ) from error
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise FinalCheckpointPublicationError(
                    "finalization evidence path contains a symlink"
                )
        path = _regular(source, "finalization evidence")
        resolved = path.resolve(strict=True)
        try:
            resolved_relative = resolved.relative_to(root)
        except ValueError as error:
            raise FinalCheckpointPublicationError(
                "finalization evidence is outside its evidence root"
            ) from error
        relative_key = _safe_key(
            resolved_relative.as_posix(), "finalization evidence relative path"
        )
        logical = f"evidence/{relative_key}"
        if logical in seen:
            raise FinalCheckpointPublicationError(
                "finalization evidence logical paths collide"
            )
        seen.add(logical)
        inventory.append((logical, resolved, _file_identity(resolved)))
    return sorted(inventory, key=lambda item: item[0])


def publish_final_checkpoint(
    checkpoint_publication_dir: Path | str,
    *,
    verified_checkpoint: Mapping[str, Any],
    evidence_paths: Sequence[Path | str],
    evidence_root: Path | str | None = None,
    target: FinalCheckpointTarget,
    store: FinalCheckpointStore,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Publish and verify a full multi-file checkpoint plus finalization evidence."""

    root = Path(checkpoint_publication_dir)
    if root.is_symlink() or not root.is_dir():
        raise FinalCheckpointPublicationError("checkpoint publication root is unsafe")
    if not isinstance(verified_checkpoint, Mapping):
        raise FinalCheckpointPublicationError("verified checkpoint must be an object")
    manifest_sha, sources = _checkpoint_inventory(root, verified_checkpoint)
    sources.extend(_evidence_inventory(evidence_paths, evidence_root=evidence_root))
    sources.sort(key=lambda item: item[0])
    store.ensure_versioning()
    receipt = Path(receipt_path)
    resume_root = receipt.parent / ".final-model-multipart-resume"
    uploaded_by_identity: dict[str, dict[str, Any]] = {}
    objects: list[dict[str, Any]] = []
    for logical, path, identity in sources:
        object_key = (
            f"{target.key_prefix}/content/{manifest_sha}/objects/"
            f"{identity['sha256']}.blob"
        )
        remote = uploaded_by_identity.get(identity["sha256"])
        if remote is None:
            remote = store.publish_file(
                object_key,
                path,
                resume_path=resume_root / f"{identity['sha256']}.json",
            )
            uploaded_by_identity[identity["sha256"]] = remote
        objects.append({"logical_path": logical, **remote})
    remote_manifest_unsigned = {
        "format": REMOTE_MANIFEST_FORMAT,
        "bucket": target.bucket,
        "registry_config_sha256": target.registry_config_sha256,
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_file_count": sum(
            1 for item in objects if item["logical_path"].startswith("checkpoint/")
        ),
        "objects": objects,
    }
    remote_manifest = {
        **remote_manifest_unsigned,
        "self_sha256": _json_sha(remote_manifest_unsigned),
    }
    manifest_temp, manifest_path = _temporary_json(remote_manifest)
    try:
        manifest_remote = store.publish_file(
            f"{target.key_prefix}/content/{remote_manifest['self_sha256']}/manifest.json",
            manifest_path,
            resume_path=resume_root / "remote-manifest.json",
        )
    finally:
        manifest_temp.cleanup()
    state_unsigned = {
        "format": FINALIZATION_STATE_FORMAT,
        "bucket": target.bucket,
        "checkpoint_manifest_sha256": manifest_sha,
        "remote_manifest_sha256": remote_manifest["self_sha256"],
        "remote_manifest_object": manifest_remote,
        "all_checkpoint_objects_roundtrip_verified": all(
            item["head_verified"] is True and item["roundtrip_verified"] is True
            for item in objects
        ),
    }
    state = {**state_unsigned, "self_sha256": _json_sha(state_unsigned)}
    state_temp, state_path = _temporary_json(state)
    try:
        state_remote = store.publish_file(
            f"{target.key_prefix}/finalization/{state['self_sha256']}/state.json",
            state_path,
            resume_path=resume_root / "finalization-state.json",
        )
    finally:
        state_temp.cleanup()
    if resume_root.exists() and not any(resume_root.iterdir()):
        resume_root.rmdir()
    receipt_unsigned = {
        "format": RECEIPT_FORMAT,
        "status": "remote_roundtrip_verified",
        "bucket": target.bucket,
        "key_prefix": target.key_prefix,
        "registry_config_sha256": target.registry_config_sha256,
        "checkpoint_manifest_sha256": manifest_sha,
        "checkpoint_file_count": remote_manifest["checkpoint_file_count"],
        "objects": objects,
        "remote_manifest": {
            "self_sha256": remote_manifest["self_sha256"],
            **manifest_remote,
        },
        "offhost_finalization_state": {
            "self_sha256": state["self_sha256"],
            **state_remote,
        },
    }
    result = {**receipt_unsigned, "self_sha256": _json_sha(receipt_unsigned)}
    _write_metadata_only(receipt, result)
    return open_final_checkpoint_publication_receipt(receipt)


def open_final_checkpoint_publication_receipt(path: Path | str) -> dict[str, Any]:
    """Strict-open the metadata-only proof required by the final controller."""

    source = _regular(Path(path), "final checkpoint publication receipt")
    try:
        value = json.loads(source.read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinalCheckpointPublicationError("publication receipt is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise FinalCheckpointPublicationError("publication receipt must be an object")
    expected = {
        "format", "status", "bucket", "key_prefix", "registry_config_sha256",
        "checkpoint_manifest_sha256", "checkpoint_file_count", "objects",
        "remote_manifest", "offhost_finalization_state", "self_sha256",
    }
    if set(value) != expected or value.get("format") != RECEIPT_FORMAT or value.get("status") != "remote_roundtrip_verified":
        raise FinalCheckpointPublicationError("publication receipt fields or status differ")
    unsigned = dict(value)
    claimed = _digest(unsigned.pop("self_sha256"), "publication receipt identity")
    if claimed != _json_sha(unsigned):
        raise FinalCheckpointPublicationError("publication receipt identity differs")
    _digest(value.get("registry_config_sha256"), "registry config identity")
    _digest(value.get("checkpoint_manifest_sha256"), "checkpoint manifest identity")
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects:
        raise FinalCheckpointPublicationError("publication object inventory is empty")
    logical_paths: list[str] = []
    for item in objects:
        if not isinstance(item, Mapping) or set(item) != {
            "logical_path", "key", "version_id", "bytes", "sha256",
            "head_verified", "roundtrip_verified",
        }:
            raise FinalCheckpointPublicationError("publication object fields differ")
        logical_path = _safe_key(item["logical_path"], "logical path")
        logical_paths.append(logical_path)
        key = _safe_key(item["key"])
        _text(item["version_id"], "provider VersionId")
        object_sha = _digest(item["sha256"], "object identity")
        expected_key = (
            f"{value['key_prefix']}/content/{value['checkpoint_manifest_sha256']}"
            f"/objects/{object_sha}.blob"
        )
        if key != expected_key:
            raise FinalCheckpointPublicationError("publication object key differs")
        if item["head_verified"] is not True or item["roundtrip_verified"] is not True:
            raise FinalCheckpointPublicationError("publication object is not verified")
    if logical_paths != sorted(logical_paths) or len(logical_paths) != len(
        set(logical_paths)
    ):
        raise FinalCheckpointPublicationError(
            "publication logical paths must be unique and sorted"
        )
    for label in ("remote_manifest", "offhost_finalization_state"):
        item = value.get(label)
        if not isinstance(item, Mapping) or set(item) != {
            "self_sha256", "key", "version_id", "bytes", "sha256",
            "head_verified", "roundtrip_verified",
        }:
            raise FinalCheckpointPublicationError(f"{label} fields differ")
        _digest(item["self_sha256"], f"{label} identity")
        _safe_key(item["key"])
        _text(item["version_id"], f"{label} VersionId")
        _digest(item["sha256"], f"{label} object SHA-256")
        if item["head_verified"] is not True or item["roundtrip_verified"] is not True:
            raise FinalCheckpointPublicationError(f"{label} is not verified")
    checkpoint_count = value.get("checkpoint_file_count")
    observed_count = sum(
        1 for item in objects if item["logical_path"].startswith("checkpoint/")
    )
    if isinstance(checkpoint_count, bool) or not isinstance(checkpoint_count, int) or checkpoint_count != observed_count:
        raise FinalCheckpointPublicationError("checkpoint file count differs")
    return dict(value)


def retire_verified_local_checkpoint_weights(
    checkpoint_publication_dir: Path | str,
    *,
    verified_checkpoint: Mapping[str, Any],
    publication_receipt: Mapping[str, Any],
) -> None:
    """Remove only local weight shards after their remote proof is strict-opened.

    The surrounding checkpoint metadata directory remains available to the
    compact Vast archive.  This operation intentionally refuses an absent or
    mismatched remote receipt, an unexpected local inventory, or symlinks.
    """

    root = Path(checkpoint_publication_dir)
    if root.is_symlink() or not root.is_dir():
        raise FinalCheckpointPublicationError("checkpoint publication root is unsafe")
    manifest_sha, _sources = _checkpoint_inventory(root, verified_checkpoint)
    receipt = dict(publication_receipt)
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("status") != "remote_roundtrip_verified"
        or receipt.get("checkpoint_manifest_sha256") != manifest_sha
    ):
        raise FinalCheckpointPublicationError(
            "local weights cannot be retired before matching remote verification"
        )
    unsigned = dict(receipt)
    claimed = _digest(unsigned.pop("self_sha256", None), "publication receipt identity")
    if claimed != _json_sha(unsigned):
        raise FinalCheckpointPublicationError("publication receipt identity differs")
    checkpoint = root / "checkpoint"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FinalCheckpointPublicationError("local checkpoint weight directory is unsafe")
    shutil.rmtree(checkpoint)
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FinalCheckpointPublicationError("local checkpoint weights were not retired")


def validate_compact_vast_output_contract(
    *, expected_outputs: Sequence[str], maximum_upload_gib: float
) -> None:
    """Fail closed if the compact fetch could include model weights or exceed 1 GiB."""

    if isinstance(maximum_upload_gib, bool) or not isinstance(maximum_upload_gib, (int, float)) or maximum_upload_gib > MAX_COMPACT_OUTPUT_GIB:
        raise FinalCheckpointPublicationError("compact Vast output must stay at or below 1 GiB")
    for raw in expected_outputs:
        path = _safe_key(raw, "Vast expected output")
        lower = path.lower()
        if "/checkpoint/" in f"/{lower}" or lower.endswith(_WEIGHT_SUFFIXES):
            raise FinalCheckpointPublicationError("compact Vast output must not contain model weights")


__all__ = [
    "FinalCheckpointPublicationError",
    "FinalCheckpointTarget",
    "FilesystemFinalCheckpointStore",
    "S3FinalCheckpointStore",
    "build_final_checkpoint_target",
    "open_final_checkpoint_publication_receipt",
    "publish_final_checkpoint",
    "retire_verified_local_checkpoint_weights",
    "validate_compact_vast_output_contract",
]
