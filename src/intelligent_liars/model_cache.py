"""Pinned, checksum-addressed model-cache contracts for the TinyLoRA runs.

This module deliberately separates three states:

* a source plan derived from an exact Hugging Face commit;
* a locally verified, self-contained snapshot; and
* an immutable S3 layout whose completion marker is written last.

Planning never downloads model weights.  Verification never trusts a filename alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
MODEL_REVISION_URL = f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}"
PINNED_README_URL = (
    f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/README.md"
)
PINNED_README_SHA256 = (
    "1422c370346c2d6a1f382db769c6d600995ceead1c064f88220f9f9e7bb39cb6"
)
APACHE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"
APACHE_LICENSE_SHA256 = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
)

# Exact Transformers/Qwen-VL runtime inventory.  Repository prose and Git metadata
# are intentionally omitted; using exact paths prevents an accidental broad mirror.
REQUIRED_SNAPSHOT_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


class CacheValidationError(ValueError):
    """Raised when a model cache cannot be proven to match its source plan."""


@dataclass(frozen=True)
class VerifiedSnapshotIdentity:
    """Immutable identity established from a manifest and local snapshot bytes."""

    model_id: str
    revision: str
    model_sha256: str
    snapshot_manifest_sha256: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_sha256": self.model_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
        }


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used by cache digests."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(content: bytes) -> str:
    """Return the Git SHA-1 blob id used by the Hub for non-LFS files."""

    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324 - source identity


def git_blob_oid_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha1()  # noqa: S324 - verifies a Git object id, not security
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise CacheValidationError(f"unsafe snapshot path: {value!r}")
    return value


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def build_snapshot_plan(hub_model_info: Mapping[str, Any]) -> dict[str, Any]:
    """Build the narrow source plan from a pinned Hub model-info response.

    ``files_metadata=True`` (or ``?blobs=true`` in the HTTP API) is required so
    every selected object has a size and source object identity.
    """

    resolved_revision = hub_model_info.get("sha")
    if resolved_revision != MODEL_REVISION:
        raise CacheValidationError(
            f"revision drift: expected {MODEL_REVISION}, got {resolved_revision!r}"
        )
    reported_id = hub_model_info.get("id")
    if reported_id != MODEL_ID:
        raise CacheValidationError(
            f"model identity drift: expected {MODEL_ID}, got {reported_id!r}"
        )

    siblings = hub_model_info.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes)):
        raise CacheValidationError("Hub response has no file metadata")
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw in siblings:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("rfilename"), str):
            continue
        path = _safe_relative_path(str(raw["rfilename"]))
        if path in by_path:
            raise CacheValidationError(f"duplicate Hub file metadata: {path}")
        by_path[path] = raw

    missing = sorted(set(REQUIRED_SNAPSHOT_FILES) - set(by_path))
    if missing:
        raise CacheValidationError(f"missing required Hub files: {', '.join(missing)}")

    planned_files: list[dict[str, Any]] = []
    for path in REQUIRED_SNAPSHOT_FILES:
        item = by_path[path]
        size = item.get("size")
        blob_oid = item.get("blobId")
        if not isinstance(size, int) or size < 0:
            raise CacheValidationError(f"missing or invalid Hub size: {path}")
        if not _is_lower_hex(blob_oid, 40):
            raise CacheValidationError(f"missing or invalid Hub blob id: {path}")
        source: dict[str, Any] = {"git_blob_oid": blob_oid}
        lfs = item.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, Mapping):
                raise CacheValidationError(f"invalid LFS metadata: {path}")
            lfs_size = lfs.get("size")
            lfs_sha256 = lfs.get("sha256")
            if lfs_size != size:
                raise CacheValidationError(f"inconsistent LFS size: {path}")
            if not _is_lower_hex(lfs_sha256, 64):
                raise CacheValidationError(f"missing LFS SHA-256: {path}")
            source["lfs_sha256"] = lfs_sha256
        planned_files.append({"path": path, "bytes": size, "source": source})

    return {
        "format": "tinylora_qwen_snapshot_plan_v1",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "allow_patterns": list(REQUIRED_SNAPSHOT_FILES),
        "files": planned_files,
        "expected_download_bytes": sum(entry["bytes"] for entry in planned_files),
        "excluded_repository_files": sorted(
            set(by_path) - set(REQUIRED_SNAPSHOT_FILES)
        ),
    }


def _validate_plan(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if plan.get("format") != "tinylora_qwen_snapshot_plan_v1":
        raise CacheValidationError("unsupported snapshot-plan format")
    if plan.get("model") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise CacheValidationError(
            "snapshot plan does not pin the approved model revision"
        )
    if plan.get("allow_patterns") != list(REQUIRED_SNAPSHOT_FILES):
        raise CacheValidationError(
            "snapshot plan does not use the exact narrow inventory"
        )
    files = plan.get("files")
    if not isinstance(files, list):
        raise CacheValidationError("snapshot plan has no files")
    if not all(isinstance(item, Mapping) for item in files):
        raise CacheValidationError("snapshot-plan file entries must be objects")
    if [item.get("path") for item in files] != list(REQUIRED_SNAPSHOT_FILES):
        raise CacheValidationError("snapshot-plan file inventory or order changed")
    total_bytes = 0
    for item in files:
        path = str(item["path"])
        size = item.get("bytes")
        source = item.get("source")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CacheValidationError(f"invalid planned byte count: {path}")
        if not isinstance(source, Mapping):
            raise CacheValidationError(f"invalid planned source metadata: {path}")
        if not _is_lower_hex(source.get("git_blob_oid"), 40):
            raise CacheValidationError(f"invalid planned Git blob id: {path}")
        if "lfs_sha256" in source and not _is_lower_hex(source["lfs_sha256"], 64):
            raise CacheValidationError(f"invalid planned LFS SHA-256: {path}")
        total_bytes += size
    if plan.get("expected_download_bytes") != total_bytes:
        raise CacheValidationError("snapshot-plan total byte count is inconsistent")
    return files


def _visible_snapshot_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[:2] == (".cache", "huggingface"):
            continue
        if path.is_symlink():
            raise CacheValidationError(
                f"snapshot must be self-contained, not symlinked: {relative.as_posix()}"
            )
        if path.is_file():
            files.add(relative.as_posix())
    return files


def verify_snapshot(
    snapshot_root: Path, plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verify exact inventory, sizes, Hub identities, and local SHA-256 hashes."""

    planned_files = _validate_plan(plan)
    snapshot_root = snapshot_root.resolve()
    if not snapshot_root.is_dir():
        raise CacheValidationError(
            f"snapshot directory does not exist: {snapshot_root}"
        )

    visible = _visible_snapshot_files(snapshot_root)
    expected = set(REQUIRED_SNAPSHOT_FILES)
    missing = sorted(expected - visible)
    extra = sorted(visible - expected)
    if missing:
        raise CacheValidationError(f"missing snapshot files: {', '.join(missing)}")
    if extra:
        raise CacheValidationError(f"unexpected snapshot files: {', '.join(extra)}")

    verified: list[dict[str, Any]] = []
    for item in planned_files:
        path = snapshot_root / str(item["path"])
        if path.is_symlink() or not path.is_file():
            raise CacheValidationError(
                f"not a self-contained regular file: {item['path']}"
            )
        actual_size = path.stat().st_size
        if actual_size != item["bytes"]:
            raise CacheValidationError(
                f"size mismatch for {item['path']}: "
                f"expected {item['bytes']}, got {actual_size}"
            )
        source = item["source"]
        actual_sha256 = sha256_file(path)
        lfs_sha256 = source.get("lfs_sha256")
        if lfs_sha256 is not None:
            if actual_sha256 != lfs_sha256:
                raise CacheValidationError(f"LFS SHA-256 mismatch for {item['path']}")
        elif git_blob_oid_file(path) != source["git_blob_oid"]:
            raise CacheValidationError(f"Git blob mismatch for {item['path']}")
        verified.append(
            {"path": item["path"], "bytes": actual_size, "sha256": actual_sha256}
        )
    return verified


def verify_huggingface_cache_for_loading(
    *,
    cache_dir: Path,
    manifest_path: Path,
    expected_model_sha256: str,
    expected_manifest_sha256: str,
) -> VerifiedSnapshotIdentity:
    """Verify the exact local snapshot consumed by Transformers.

    Large shard hashes are computed once per process and cached against the
    manifest plus every resolved file's device, inode, size, and nanosecond
    modification time. Subsequent worker trials only compare the returned
    immutable receipt; they never rehash model weights.
    """

    resolved_manifest = manifest_path.resolve(strict=True)
    if manifest_path.is_symlink() or not resolved_manifest.is_file():
        raise CacheValidationError("model cache manifest must be a regular file")
    if not _is_lower_hex(expected_model_sha256, 64):
        raise CacheValidationError("expected model SHA-256 is invalid")
    if not _is_lower_hex(expected_manifest_sha256, 64):
        raise CacheValidationError("expected manifest SHA-256 is invalid")

    manifest = _load_strict_json_object(resolved_manifest)
    _validate_load_manifest(
        manifest,
        expected_model_sha256=expected_model_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    snapshot_root, file_signature = _resolve_load_snapshot(
        cache_dir.resolve(), manifest
    )
    manifest_stat = resolved_manifest.stat()
    return _verify_load_snapshot_cached(
        str(resolved_manifest),
        manifest_stat.st_dev,
        manifest_stat.st_ino,
        manifest_stat.st_size,
        manifest_stat.st_mtime_ns,
        str(snapshot_root),
        file_signature,
        expected_model_sha256,
        expected_manifest_sha256,
    )


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CacheValidationError(f"duplicate manifest field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheValidationError("model cache manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise CacheValidationError("model cache manifest must be an object")
    return value


def _validate_load_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_model_sha256: str,
    expected_manifest_sha256: str,
) -> list[Mapping[str, Any]]:
    if manifest.get("format") != "tinylora_model_cache_manifest_v1":
        raise CacheValidationError("unsupported model cache manifest format")
    if manifest.get("complete") is not True:
        raise CacheValidationError("model cache manifest is incomplete")
    if manifest.get("model") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise CacheValidationError(
            "model cache manifest revision or repository changed"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not all(
        isinstance(item, Mapping) for item in files
    ):
        raise CacheValidationError("model cache manifest has no file inventory")
    typed_files = [item for item in files if isinstance(item, Mapping)]
    if [item.get("path") for item in typed_files] != list(REQUIRED_SNAPSHOT_FILES):
        raise CacheValidationError("model cache manifest file inventory changed")
    for item in typed_files:
        if (
            not isinstance(item.get("bytes"), int)
            or isinstance(item.get("bytes"), bool)
            or item["bytes"] < 0
            or not _is_lower_hex(item.get("sha256"), 64)
        ):
            raise CacheValidationError("model cache manifest has an invalid file entry")
    if manifest.get("total_bytes") != sum(int(item["bytes"]) for item in typed_files):
        raise CacheValidationError("model cache manifest total byte count changed")
    if manifest.get("content_sha256") != _content_sha256(typed_files):
        raise CacheValidationError("model cache content identity is inconsistent")
    if manifest["content_sha256"] != expected_model_sha256:
        raise CacheValidationError(
            "model cache content identity differs from expectation"
        )
    if not _is_lower_hex(manifest.get("source_plan_sha256"), 64):
        raise CacheValidationError("model cache source-plan identity is invalid")
    if manifest.get("legal") != legal_artifact_descriptors():
        raise CacheValidationError("model cache legal inventory changed")
    actual_manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise CacheValidationError(
            "model cache manifest identity differs from expectation"
        )
    return typed_files


def _resolve_load_snapshot(
    cache_dir: Path, manifest: Mapping[str, Any]
) -> tuple[Path, tuple[tuple[str, int, int, int, int], ...]]:
    storage_name = "models--Qwen--Qwen3-VL-8B-Thinking"
    candidates = (
        cache_dir / storage_name / "snapshots" / MODEL_REVISION,
        cache_dir / "hub" / storage_name / "snapshots" / MODEL_REVISION,
    )
    roots = [candidate for candidate in candidates if candidate.is_dir()]
    if len(roots) != 1:
        raise CacheValidationError(
            "cache_dir must resolve exactly one pinned Qwen snapshot directory"
        )
    snapshot_root = roots[0]
    storage_root = snapshot_root.parents[1].resolve(strict=True)
    files = manifest["files"]
    signature: list[tuple[str, int, int, int, int]] = []
    for item in files:
        relative = _safe_relative_path(str(item["path"]))
        visible_path = snapshot_root / relative
        try:
            resolved_path = visible_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise CacheValidationError(
                f"missing model cache file: {relative}"
            ) from error
        if storage_root not in resolved_path.parents or not resolved_path.is_file():
            raise CacheValidationError(
                f"model cache file is not a regular file inside model storage: {relative}"
            )
        stat = resolved_path.stat()
        signature.append(
            (relative, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        )
    return snapshot_root, tuple(signature)


@lru_cache(maxsize=8)
def _verify_load_snapshot_cached(
    manifest_path: str,
    manifest_device: int,
    manifest_inode: int,
    manifest_size: int,
    manifest_mtime_ns: int,
    snapshot_root: str,
    file_signature: tuple[tuple[str, int, int, int, int], ...],
    expected_model_sha256: str,
    expected_manifest_sha256: str,
) -> VerifiedSnapshotIdentity:
    del manifest_device, manifest_inode, manifest_size, manifest_mtime_ns
    manifest = _load_strict_json_object(Path(manifest_path))
    files = _validate_load_manifest(
        manifest,
        expected_model_sha256=expected_model_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    by_path = {str(item["path"]): item for item in files}
    for relative, _device, _inode, size, _mtime_ns in file_signature:
        item = by_path[relative]
        path = (Path(snapshot_root) / relative).resolve(strict=True)
        if size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise CacheValidationError(f"model cache file hash changed: {relative}")
    return VerifiedSnapshotIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        model_sha256=expected_model_sha256,
        snapshot_manifest_sha256=expected_manifest_sha256,
    )


def materialize_huggingface_cache(
    snapshot_root: Path,
    cache_dir: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Hydrate the exact on-disk layout consumed by ``ModelLoadConfig``.

    The verified standalone snapshot remains the durable source. Blob hardlinks
    (or copies when hardlinks are unavailable) and revision symlinks make the
    cache consumable through the approved Hub model ID; no unsupported local
    model path is passed to ``resolve_model_id``.
    """

    resolved_snapshot = snapshot_root.resolve()
    resolved_cache = cache_dir.resolve()
    if (
        resolved_snapshot == resolved_cache
        or resolved_snapshot in resolved_cache.parents
        or resolved_cache in resolved_snapshot.parents
    ):
        raise CacheValidationError(
            "standalone snapshot and Hugging Face cache roots must not overlap"
        )
    verified = verify_snapshot(resolved_snapshot, plan)
    planned = _validate_plan(plan)
    storage = resolved_cache / "models--Qwen--Qwen3-VL-8B-Thinking"
    blobs = storage / "blobs"
    revision_root = storage / "snapshots" / MODEL_REVISION
    blobs.mkdir(parents=True, exist_ok=True)
    revision_root.mkdir(parents=True, exist_ok=True)

    for source_entry, verified_entry in zip(planned, verified, strict=True):
        source_path = resolved_snapshot / str(source_entry["path"])
        source_identity = source_entry["source"]
        blob_name = source_identity.get("lfs_sha256") or source_identity["git_blob_oid"]
        blob_path = blobs / str(blob_name)
        if blob_path.exists():
            if (
                not blob_path.is_file()
                or sha256_file(blob_path) != verified_entry["sha256"]
            ):
                raise CacheValidationError(
                    f"corrupt existing Hugging Face blob: {blob_path}"
                )
        else:
            temporary = blobs / f".{blob_name}.{os.getpid()}.tmp"
            try:
                try:
                    os.link(source_path, temporary)
                except OSError:
                    shutil.copyfile(source_path, temporary)
                if sha256_file(temporary) != verified_entry["sha256"]:
                    raise CacheValidationError(
                        f"materialized blob hash mismatch: {source_entry['path']}"
                    )
                temporary.replace(blob_path)
            finally:
                temporary.unlink(missing_ok=True)

        snapshot_path = revision_root / str(source_entry["path"])
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        expected_target = os.path.relpath(blob_path, start=snapshot_path.parent)
        if snapshot_path.is_symlink():
            if os.readlink(snapshot_path) != expected_target:
                raise CacheValidationError(
                    f"wrong existing Hugging Face snapshot link: {snapshot_path}"
                )
        elif snapshot_path.exists():
            raise CacheValidationError(
                f"unexpected non-link in Hugging Face snapshot: {snapshot_path}"
            )
        else:
            snapshot_path.symlink_to(expected_target)

    return {
        "format": "tinylora_huggingface_cache_materialization_v1",
        "model_load_config": {
            "model_name": MODEL_ID,
            "cache_dir": str(resolved_cache),
            "revision": MODEL_REVISION,
        },
        "required_environment": {"HF_HUB_OFFLINE": "1"},
        "snapshot_path": str(revision_root),
        "files": len(verified),
        "total_bytes": sum(int(item["bytes"]) for item in verified),
    }


def legal_artifact_descriptors() -> list[dict[str, Any]]:
    """Describe legal/attribution objects adjacent to, never inside, the loader tree."""

    attribution = {
        "format": "tinylora_model_attribution_v1",
        "upstream_model": MODEL_ID,
        "upstream_revision": MODEL_REVISION,
        "source_revision_url": MODEL_REVISION_URL,
        "pinned_model_card_url": PINNED_README_URL,
        "upstream_license": "Apache-2.0",
        "upstream_snapshot_modified": False,
    }
    modifications = (
        "Upstream cache: no modifications; files are byte-for-byte from the pinned "
        f"{MODEL_ID} revision {MODEL_REVISION}.\n\n"
        "Derivatives: TinyLoRA/LoRA adapters or merged checkpoints produced from this "
        "cache are modified works. Each published derivative must add its artifact "
        "identifier, training configuration, creation time, and a description of its "
        "changes to its own modification notice.\n"
    )
    generated = {
        "ATTRIBUTION.json": canonical_json_bytes(attribution),
        "MODIFICATIONS.md": modifications.encode(),
    }
    descriptors = [
        {
            "path": "LICENSE-APACHE-2.0.txt",
            "bytes": 11358,
            "sha256": APACHE_LICENSE_SHA256,
            "source_url": APACHE_LICENSE_URL,
        },
        {
            "path": "UPSTREAM_README.md",
            "bytes": 7201,
            "sha256": PINNED_README_SHA256,
            "source_url": PINNED_README_URL,
        },
    ]
    descriptors.extend(
        {
            "path": path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "content_utf8": content.decode(),
        }
        for path, content in generated.items()
    )
    return descriptors


def _content_sha256(files: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "format": "tinylora_model_cache_content_v1",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "files": list(files),
        "legal": legal_artifact_descriptors(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_cache_manifest(
    plan: Mapping[str, Any],
    verified_files: Sequence[Mapping[str, Any]],
    *,
    bucket: str | None = None,
    base_prefix: str = "model-cache/v1",
) -> dict[str, Any]:
    """Create an immutable cache manifest; this function performs no upload."""

    planned_files = _validate_plan(plan)
    files = [dict(item) for item in verified_files]
    if [item.get("path") for item in files] != list(REQUIRED_SNAPSHOT_FILES):
        raise CacheValidationError("verified inventory does not match the source plan")
    for item, planned in zip(files, planned_files, strict=True):
        if (
            not isinstance(item.get("bytes"), int)
            or isinstance(item.get("bytes"), bool)
            or not _is_lower_hex(item.get("sha256"), 64)
        ):
            raise CacheValidationError(f"invalid verified file entry: {item!r}")
        if item["bytes"] != planned["bytes"]:
            raise CacheValidationError(f"verified byte count changed: {item['path']}")
        source_sha256 = planned["source"].get("lfs_sha256")
        if source_sha256 is not None and item["sha256"] != source_sha256:
            raise CacheValidationError(f"verified source hash changed: {item['path']}")
    content_sha256 = _content_sha256(files)
    plan_sha256 = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    manifest: dict[str, Any] = {
        "format": "tinylora_model_cache_manifest_v1",
        "complete": True,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "source_plan_sha256": plan_sha256,
        "content_sha256": content_sha256,
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
        "legal": legal_artifact_descriptors(),
    }
    if bucket is not None:
        if not bucket or "/" in bucket:
            raise CacheValidationError("S3 bucket must be a non-empty bucket name")
        prefix = base_prefix.strip("/")
        if not prefix:
            raise CacheValidationError("S3 base prefix must not be empty")
        model_slug = MODEL_ID.lower().replace("/", "--")
        object_prefix = f"{prefix}/{model_slug}/{MODEL_REVISION}/{content_sha256}"
        manifest["s3"] = {
            "schema": "tinylora_model_cache_s3_v1",
            "bucket": bucket,
            "object_prefix": object_prefix,
            "files_prefix": f"{object_prefix}/files",
            "legal_prefix": f"{object_prefix}/legal",
            "manifest_key": f"{object_prefix}/manifest.json",
            "completion_key": f"{object_prefix}/_COMPLETE.json",
            "publication_order": [
                "files",
                "legal",
                "manifest",
                "completion_marker_last",
            ],
        }
    return manifest


def completion_marker(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the small object consumers require before accepting an S3 cache."""

    if manifest.get("format") != "tinylora_model_cache_manifest_v1":
        raise CacheValidationError("unsupported cache-manifest format")
    if manifest.get("complete") is not True:
        raise CacheValidationError("cannot mark an incomplete cache complete")
    if manifest.get("model") != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        raise CacheValidationError("cache manifest does not identify the pinned model")
    if not _is_lower_hex(manifest.get("content_sha256"), 64):
        raise CacheValidationError("cache manifest has an invalid content digest")
    if not isinstance(manifest.get("total_bytes"), int) or manifest["total_bytes"] < 0:
        raise CacheValidationError("cache manifest has an invalid total byte count")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CacheValidationError("cache manifest has no file inventory")
    if [item.get("path") for item in files if isinstance(item, Mapping)] != list(
        REQUIRED_SNAPSHOT_FILES
    ):
        raise CacheValidationError("cache manifest has the wrong file inventory")
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("bytes"), int)
        or isinstance(item.get("bytes"), bool)
        or item["bytes"] < 0
        or not _is_lower_hex(item.get("sha256"), 64)
        for item in files
    ):
        raise CacheValidationError("cache manifest has an invalid file entry")
    if manifest["content_sha256"] != _content_sha256(files):
        raise CacheValidationError("cache manifest content digest is inconsistent")
    if manifest["total_bytes"] != sum(int(item["bytes"]) for item in files):
        raise CacheValidationError("cache manifest byte count is inconsistent")
    s3 = manifest.get("s3")
    if not isinstance(s3, Mapping) or s3.get("schema") != "tinylora_model_cache_s3_v1":
        raise CacheValidationError("completion marker requires the S3 cache contract")
    expected_suffix = (
        f"/qwen--qwen3-vl-8b-thinking/{MODEL_REVISION}/{manifest['content_sha256']}"
    )
    object_prefix = s3.get("object_prefix")
    if not isinstance(object_prefix, str) or not (
        f"/{object_prefix}".endswith(expected_suffix)
    ):
        raise CacheValidationError(
            "cache manifest has an inconsistent S3 object prefix"
        )
    expected_s3 = {
        "schema": "tinylora_model_cache_s3_v1",
        "bucket": s3.get("bucket"),
        "object_prefix": object_prefix,
        "files_prefix": f"{object_prefix}/files",
        "legal_prefix": f"{object_prefix}/legal",
        "manifest_key": f"{object_prefix}/manifest.json",
        "completion_key": f"{object_prefix}/_COMPLETE.json",
        "publication_order": [
            "files",
            "legal",
            "manifest",
            "completion_marker_last",
        ],
    }
    if (
        not isinstance(s3.get("bucket"), str)
        or not s3["bucket"]
        or "/" in s3["bucket"]
        or dict(s3) != expected_s3
    ):
        raise CacheValidationError("cache manifest has an inconsistent S3 contract")
    legal = manifest.get("legal")
    if not isinstance(legal, list) or [
        item.get("path") for item in legal if isinstance(item, Mapping)
    ] != [
        "LICENSE-APACHE-2.0.txt",
        "UPSTREAM_README.md",
        "ATTRIBUTION.json",
        "MODIFICATIONS.md",
    ]:
        raise CacheValidationError("cache manifest lacks required legal attribution")
    if any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("bytes"), int)
        or item["bytes"] < 0
        or not _is_lower_hex(item.get("sha256"), 64)
        for item in legal
    ):
        raise CacheValidationError("cache manifest has invalid legal attribution")
    if legal != legal_artifact_descriptors():
        raise CacheValidationError("cache manifest legal attribution is inconsistent")
    return {
        "format": "tinylora_model_cache_complete_v1",
        "model": manifest["model"],
        "content_sha256": manifest["content_sha256"],
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "total_bytes": manifest["total_bytes"],
    }
