"""Strict, offline contracts for the model-artifact registry.

The registry is intentionally a small local module.  It does not upload, delete,
or enumerate objects.  A publisher can use :func:`artifact_key` to compute an
exact destination and persist the observed S3 ``VersionId`` in a receipt after
the provider has accepted the object.  Consumers then use
:func:`verify_s3_roundtrip`, which performs only exact-key ``HEAD`` and
version-pinned ``GET`` calls; no bucket listing is required.

Only immutable, content-addressed records are accepted in a registry.  A
missing or ``"null"`` version is never a receipt, and a hash or byte mismatch
fails closed instead of producing a partially trusted record.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol


MODEL_REGISTRY_FORMAT = "intelligent_liars_model_registry_v1"
MODEL_REGISTRY_CONFIG_FORMAT = "intelligent_liars_model_registry_config_v1"
MODEL_REGISTRY_RECEIPT_FORMAT = "intelligent_liars_model_registry_s3_receipt_v1"
MODEL_REGISTRY_SCHEMA_VERSION = 1
DEFAULT_REGISTRY_PREFIX = "model-registry/v1"
ARTIFACT_KINDS = (
    "failed_experiment",
    "successful_experiment",
    "final_model",
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
MODEL_CONTENT_SHA256 = (
    "bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8"
)
MODEL_TOTAL_BYTES = 17_545_907_058
MODEL_FILE_COUNT = 14

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class RegistryError(ValueError):
    """Raised when a registry, artifact, or provider receipt is not trusted."""


class _S3Client(Protocol):
    def head_object(self, **kwargs: str) -> Mapping[str, Any]: ...

    def get_object(self, **kwargs: str) -> Mapping[str, Any]: ...


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one deterministic UTF-8 JSON encoding used for identities."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_identity(value: Path | bytes | bytearray | memoryview) -> dict[str, Any]:
    """Return a path-independent byte count and SHA-256 content identity."""

    if isinstance(value, Path):
        if value.is_symlink():
            raise RegistryError(f"artifact content must not be a symlink: {value}")
        path = value.resolve(strict=True)
        if not path.is_file():
            raise RegistryError(f"artifact content must be a regular file: {value}")
        return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    raise RegistryError("artifact content must be bytes or a regular file")


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RegistryError(f"{label} must be a lowercase SHA-256")
    return value


def _require_component(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RegistryError(f"{label} must be a safe non-empty component")
    if allow_empty and value == "":
        return value
    if _SAFE_COMPONENT.fullmatch(value) is None:
        raise RegistryError(f"unsafe {label}: {value!r}")
    return value


def _require_bucket(value: Any) -> str:
    if not isinstance(value, str) or _SAFE_BUCKET.fullmatch(value) is None:
        raise RegistryError("bucket must be a valid S3 bucket name")
    return value


def _require_key(value: Any, label: str = "key") -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise RegistryError(f"{label} must be a safe relative S3 key")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value or "//" in value:
        raise RegistryError(f"unsafe {label}: {value!r}")
    return value


def _require_prefix(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise RegistryError("registry prefix must be a safe non-empty key prefix")
    path = PurePosixPath(value.rstrip("/"))
    if ".." in path.parts or str(path) != value.rstrip("/"):
        raise RegistryError(f"unsafe registry prefix: {value!r}")
    return value.rstrip("/")


def _require_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{label} must be an object")
    if set(value) != {"bytes", "sha256"}:
        raise RegistryError(f"{label} has unknown or missing fields")
    size = value["bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise RegistryError(f"{label}.bytes must be a non-negative integer")
    digest = _require_hash(value["sha256"], f"{label}.sha256")
    return {"bytes": size, "sha256": digest}


def _require_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RegistryError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RegistryError(f"{label} must include a timezone")
    return value


def _require_version_id(value: Any) -> str:
    if not isinstance(value, str) or not value or value == "null":
        raise RegistryError("S3 receipt requires a non-null immutable VersionId")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise RegistryError(f"{label} has invalid fields ({'; '.join(detail)})")


def artifact_key(
    kind: str,
    *,
    run_id: str,
    content_sha256: str,
    filename: str,
    model_slug: str = "",
    base_prefix: str = DEFAULT_REGISTRY_PREFIX,
) -> str:
    """Compute the deterministic S3 key for one published artifact.

    ``failed_experiment`` and ``successful_experiment`` records are grouped by
    run.  ``final_model`` records are grouped by a human-readable model slug.
    The digest remains in every path so accidental overwrite is detectable.
    """

    if kind not in ARTIFACT_KINDS:
        raise RegistryError(f"unknown artifact kind: {kind!r}")
    prefix = _require_prefix(base_prefix)
    run = _require_component(run_id, "run_id")
    digest = _require_hash(content_sha256, "content_sha256")
    name = _require_component(filename, "filename")
    if "/" in filename:
        raise RegistryError("unsafe filename: nested paths are not allowed")
    if kind == "final_model":
        slug = _require_component(model_slug, "model_slug")
        return f"{prefix}/models/final/{slug}/{digest}/{name}"
    if model_slug:
        raise RegistryError("model_slug is only valid for final_model artifacts")
    category = "failed" if kind == "failed_experiment" else "successful"
    return f"{prefix}/experiments/{category}/{run}/{digest}/{name}"


def build_s3_receipt(
    *,
    bucket: str,
    key: str,
    version_id: str,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    checked_at: str,
) -> dict[str, Any]:
    """Build a strict provider receipt; false or unversioned receipts are not trusted."""

    bucket = _require_bucket(bucket)
    key = _require_key(key)
    version_id = _require_version_id(version_id)
    expected_identity = _require_identity(expected, "expected")
    observed_identity = _require_identity(observed, "observed")
    checked_at = _require_timestamp(checked_at, "checked_at")
    mismatches = [
        field
        for field in ("bytes", "sha256")
        if expected_identity[field] != observed_identity[field]
    ]
    return {
        "format": MODEL_REGISTRY_RECEIPT_FORMAT,
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "expected": expected_identity,
        "observed": observed_identity,
        "checked_at": checked_at,
        "verified": not mismatches,
        "mismatches": mismatches,
    }


def _validate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError("artifact receipt must be an object")
    expected_fields = {
        "format",
        "schema_version",
        "bucket",
        "key",
        "version_id",
        "expected",
        "observed",
        "checked_at",
        "verified",
        "mismatches",
    }
    _exact_fields(value, expected_fields, "receipt")
    if value["format"] != MODEL_REGISTRY_RECEIPT_FORMAT:
        raise RegistryError("unsupported model registry receipt format")
    if value["schema_version"] != MODEL_REGISTRY_SCHEMA_VERSION:
        raise RegistryError("unsupported model registry receipt schema version")
    bucket = _require_bucket(value["bucket"])
    key = _require_key(value["key"])
    version_id = _require_version_id(value["version_id"])
    expected = _require_identity(value["expected"], "receipt.expected")
    observed = _require_identity(value["observed"], "receipt.observed")
    checked_at = _require_timestamp(value["checked_at"], "receipt.checked_at")
    verified = value["verified"]
    mismatches = value["mismatches"]
    if not isinstance(verified, bool) or not isinstance(mismatches, list):
        raise RegistryError("receipt verification fields have invalid types")
    if mismatches != [
        field
        for field in ("bytes", "sha256")
        if expected[field] != observed[field]
    ]:
        raise RegistryError("receipt mismatch list is inconsistent")
    if verified != (not mismatches):
        raise RegistryError("receipt verified flag is inconsistent")
    if any(item not in ("bytes", "sha256") for item in mismatches) or len(
        set(mismatches)
    ) != len(mismatches):
        raise RegistryError("receipt mismatch list is invalid")
    return {
        "format": value["format"],
        "schema_version": value["schema_version"],
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "expected": expected,
        "observed": observed,
        "checked_at": checked_at,
        "verified": verified,
        "mismatches": list(mismatches),
    }


def build_artifact_record(
    *,
    kind: str,
    run_id: str,
    filename: str,
    content: Path | bytes | bytearray | memoryview,
    bucket: str,
    version_id: str,
    checked_at: str,
    model_slug: str = "",
    base_prefix: str = DEFAULT_REGISTRY_PREFIX,
) -> dict[str, Any]:
    """Build one immutable registry artifact record from bytes or a local file."""

    if kind not in ARTIFACT_KINDS:
        raise RegistryError(f"unknown artifact kind: {kind!r}")
    identity = content_identity(content)
    key = artifact_key(
        kind,
        run_id=run_id,
        content_sha256=identity["sha256"],
        filename=filename,
        model_slug=model_slug,
        base_prefix=base_prefix,
    )
    receipt = build_s3_receipt(
        bucket=bucket,
        key=key,
        version_id=version_id,
        expected=identity,
        observed=identity,
        checked_at=checked_at,
    )
    identity_payload = {
        "kind": kind,
        "run_id": run_id,
        "model_slug": model_slug,
        "filename": filename,
        "s3_key": key,
        "content": identity,
    }
    artifact_id = _sha256_bytes(canonical_json_bytes(identity_payload))
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "run_id": _require_component(run_id, "run_id"),
        "model_slug": (
            _require_component(model_slug, "model_slug") if model_slug else ""
        ),
        "filename": _require_component(filename, "filename"),
        "s3_key": key,
        "content": identity,
        "receipt": receipt,
    }


def _validate_artifact_record(value: Any, *, bucket: str, base_prefix: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError("artifact record must be an object")
    fields = {
        "artifact_id",
        "kind",
        "run_id",
        "model_slug",
        "filename",
        "s3_key",
        "content",
        "receipt",
    }
    _exact_fields(value, fields, "artifact record")
    artifact_id = _require_hash(value["artifact_id"], "artifact_id")
    kind = value["kind"]
    if kind not in ARTIFACT_KINDS:
        raise RegistryError(f"unknown artifact kind: {kind!r}")
    run_id = _require_component(value["run_id"], "run_id")
    model_slug = value["model_slug"]
    if not isinstance(model_slug, str):
        raise RegistryError("model_slug must be a string")
    if model_slug:
        model_slug = _require_component(model_slug, "model_slug")
    if kind != "final_model" and model_slug:
        raise RegistryError("non-final artifact must not have a model_slug")
    if kind == "final_model" and not model_slug:
        raise RegistryError("final_model artifact requires a model_slug")
    filename = _require_component(value["filename"], "filename")
    if "/" in filename:
        raise RegistryError("unsafe filename: nested paths are not allowed")
    content = _require_identity(value["content"], "content")
    key = _require_key(value["s3_key"], "s3_key")
    expected_key = artifact_key(
        kind,
        run_id=run_id,
        content_sha256=content["sha256"],
        filename=filename,
        model_slug=model_slug,
        base_prefix=base_prefix,
    )
    if key != expected_key:
        raise RegistryError("s3_key does not match artifact identity")
    receipt = _validate_receipt(value["receipt"])
    if receipt["bucket"] != bucket:
        raise RegistryError("artifact receipt bucket differs from registry bucket")
    if receipt["key"] != key:
        raise RegistryError("artifact receipt key differs from artifact key")
    if receipt["expected"] != content:
        raise RegistryError("receipt expected identity differs from artifact content")
    if receipt["verified"] is not True:
        raise RegistryError("registry accepts only verified artifact receipts")
    identity_payload = {
        "kind": kind,
        "run_id": run_id,
        "model_slug": model_slug,
        "filename": filename,
        "s3_key": key,
        "content": content,
    }
    expected_id = _sha256_bytes(canonical_json_bytes(identity_payload))
    if artifact_id != expected_id:
        raise RegistryError("artifact_id does not match artifact identity")
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "run_id": run_id,
        "model_slug": model_slug,
        "filename": filename,
        "s3_key": key,
        "content": content,
        "receipt": receipt,
    }


def build_registry(
    *,
    bucket: str,
    artifacts: Sequence[Mapping[str, Any]] = (),
    base_prefix: str = DEFAULT_REGISTRY_PREFIX,
) -> dict[str, Any]:
    """Build and validate a deterministic registry manifest."""

    bucket = _require_bucket(bucket)
    base_prefix = _require_prefix(base_prefix)
    validated = [
        _validate_artifact_record(item, bucket=bucket, base_prefix=base_prefix)
        for item in artifacts
    ]
    validated.sort(key=lambda item: item["artifact_id"])
    if len({item["artifact_id"] for item in validated}) != len(validated):
        raise RegistryError("registry contains duplicate artifact_id values")
    unsigned = {
        "format": MODEL_REGISTRY_FORMAT,
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "bucket": bucket,
        "base_prefix": base_prefix,
        "artifacts": validated,
    }
    return {
        **unsigned,
        "registry_id": _sha256_bytes(canonical_json_bytes(unsigned)),
    }


def validate_registry(value: Any) -> dict[str, Any]:
    """Validate a registry with exact fields and all provider receipts pinned."""

    if not isinstance(value, Mapping):
        raise RegistryError("registry must be a JSON object")
    fields = {
        "format",
        "schema_version",
        "registry_id",
        "bucket",
        "base_prefix",
        "artifacts",
    }
    _exact_fields(value, fields, "registry")
    if value["format"] != MODEL_REGISTRY_FORMAT:
        raise RegistryError("unsupported model registry format")
    if value["schema_version"] != MODEL_REGISTRY_SCHEMA_VERSION:
        raise RegistryError("unsupported model registry schema version")
    registry_id = _require_hash(value["registry_id"], "registry_id")
    bucket = _require_bucket(value["bucket"])
    base_prefix = _require_prefix(value["base_prefix"])
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list):
        raise RegistryError("registry artifacts must be an array")
    validated = [
        _validate_artifact_record(item, bucket=bucket, base_prefix=base_prefix)
        for item in artifacts
    ]
    if [item["artifact_id"] for item in validated] != sorted(
        item["artifact_id"] for item in validated
    ):
        raise RegistryError("registry artifacts must be sorted by artifact_id")
    if len({item["artifact_id"] for item in validated}) != len(validated):
        raise RegistryError("registry contains duplicate artifact_id values")
    unsigned = {
        "format": value["format"],
        "schema_version": value["schema_version"],
        "bucket": bucket,
        "base_prefix": base_prefix,
        "artifacts": validated,
    }
    if registry_id != _sha256_bytes(canonical_json_bytes(unsigned)):
        raise RegistryError("registry_id does not match canonical registry content")
    return {
        **unsigned,
        "registry_id": registry_id,
    }


def save_registry(path: Path, registry: Mapping[str, Any]) -> None:
    """Validate and atomically save a local registry manifest."""

    validated = validate_registry(registry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(validated))
    temporary.replace(path)


def load_registry(path: Path) -> dict[str, Any]:
    """Load a local registry and fail closed on malformed or unsupported JSON."""

    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry: {path}") from exc
    return validate_registry(value)


def verify_s3_roundtrip(
    s3: _S3Client,
    record: Mapping[str, Any],
    *,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """HEAD and GET one exact, version-pinned object and return its receipt.

    The adapter must expose only the ordinary boto3-shaped ``head_object`` and
    ``get_object`` methods.  This function never calls ``list_objects`` or any
    prefix enumeration operation.
    """

    # The registry bucket and prefix are recovered from the record itself by
    # validating its existing receipt.  This function is intentionally useful
    # before the record is inserted into a registry.
    if not isinstance(record, Mapping):
        raise RegistryError("artifact record must be an object")
    receipt = _validate_receipt(record.get("receipt"))
    if receipt["key"] != record.get("s3_key"):
        raise RegistryError("record receipt key differs from artifact key")
    content = _require_identity(record.get("content"), "content")
    bucket = receipt["bucket"]
    key = receipt["key"]
    version_id = receipt["version_id"]
    try:
        head = s3.head_object(Bucket=bucket, Key=key, VersionId=version_id)
    except Exception as exc:  # provider exceptions must fail closed at this seam
        raise RegistryError(f"S3 HEAD failed for exact object {key}") from exc
    if not isinstance(head, Mapping):
        raise RegistryError("S3 HEAD response is not an object")
    observed_version = _require_version_id(head.get("VersionId"))
    if observed_version != version_id:
        raise RegistryError("S3 HEAD returned a different VersionId")
    content_length = head.get("ContentLength")
    if content_length != content["bytes"]:
        raise RegistryError("S3 HEAD ContentLength does not match artifact bytes")
    try:
        response = s3.get_object(Bucket=bucket, Key=key, VersionId=version_id)
    except Exception as exc:
        raise RegistryError(f"S3 GET failed for exact object {key}") from exc
    if not isinstance(response, Mapping) or "Body" not in response:
        raise RegistryError("S3 GET response lacks a body")
    body = response["Body"]
    try:
        if hasattr(body, "read"):
            downloaded = body.read()
        else:
            downloaded = b"".join(body)
    except Exception as exc:
        raise RegistryError("S3 object body could not be read") from exc
    if not isinstance(downloaded, bytes):
        raise RegistryError("S3 object body was not bytes")
    observed = {"bytes": len(downloaded), "sha256": _sha256_bytes(downloaded)}
    if observed != content:
        raise RegistryError("S3 round-trip SHA-256 or byte count mismatch")
    return build_s3_receipt(
        bucket=bucket,
        key=key,
        version_id=version_id,
        expected=content,
        observed=observed,
        checked_at=checked_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _validate_registry_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegistryError("registry config must be a JSON object")
    fields = {"format", "schema_version", "registry", "base_model_cache"}
    _exact_fields(value, fields, "registry config")
    if value["format"] != MODEL_REGISTRY_CONFIG_FORMAT:
        raise RegistryError("unsupported model registry config format")
    if value["schema_version"] != MODEL_REGISTRY_SCHEMA_VERSION:
        raise RegistryError("unsupported model registry config schema version")
    registry = value["registry"]
    if not isinstance(registry, Mapping):
        raise RegistryError("registry config.registry must be an object")
    _exact_fields(registry, {"bucket", "base_prefix"}, "registry config.registry")
    registry_validated = {
        "bucket": _require_bucket(registry["bucket"]),
        "base_prefix": _require_prefix(registry["base_prefix"]),
    }
    cache = value["base_model_cache"]
    if not isinstance(cache, Mapping):
        raise RegistryError("registry config.base_model_cache must be an object")
    cache_fields = {
        "repo_id",
        "revision",
        "content_sha256",
        "total_bytes",
        "file_count",
        "s3_prefix",
        "s3_prefix_status",
        "verified_receipt_path",
        "notes",
    }
    _exact_fields(cache, cache_fields, "registry config.base_model_cache")
    if cache["repo_id"] != MODEL_ID or cache["revision"] != MODEL_REVISION:
        raise RegistryError("base model cache is not the exact target checkpoint")
    digest = _require_hash(cache["content_sha256"], "base_model_cache.content_sha256")
    if digest != MODEL_CONTENT_SHA256:
        raise RegistryError("base model cache content identity is not verified")
    for field in ("total_bytes", "file_count"):
        if not isinstance(cache[field], int) or isinstance(cache[field], bool) or cache[field] < 0:
            raise RegistryError(f"base_model_cache.{field} must be a non-negative integer")
    if cache["total_bytes"] != MODEL_TOTAL_BYTES or cache["file_count"] != MODEL_FILE_COUNT:
        raise RegistryError("base model cache size or file count is inconsistent")
    _require_key(cache["s3_prefix"], "base_model_cache.s3_prefix")
    if cache["s3_prefix_status"] != "unresolved_manifest_prefix_mismatch":
        raise RegistryError("base model cache S3 prefix status must remain unresolved")
    _require_key(cache["verified_receipt_path"], "base_model_cache.verified_receipt_path")
    if not isinstance(cache["notes"], str) or not cache["notes"]:
        raise RegistryError("base_model_cache.notes must be non-empty")
    return {
        "format": value["format"],
        "schema_version": value["schema_version"],
        "registry": registry_validated,
        "base_model_cache": {
            "repo_id": cache["repo_id"],
            "revision": cache["revision"],
            "content_sha256": digest,
            "total_bytes": cache["total_bytes"],
            "file_count": cache["file_count"],
            "s3_prefix": cache["s3_prefix"],
            "s3_prefix_status": cache["s3_prefix_status"],
            "verified_receipt_path": cache["verified_receipt_path"],
            "notes": cache["notes"],
        },
    }


def load_registry_config(path: Path) -> dict[str, Any]:
    """Load the checked-in registry configuration with fail-closed validation."""

    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry config: {path}") from exc
    return _validate_registry_config(value)


__all__ = [
    "ARTIFACT_KINDS",
    "DEFAULT_REGISTRY_PREFIX",
    "MODEL_CONTENT_SHA256",
    "MODEL_FILE_COUNT",
    "MODEL_ID",
    "MODEL_REGISTRY_CONFIG_FORMAT",
    "MODEL_REGISTRY_FORMAT",
    "MODEL_REGISTRY_RECEIPT_FORMAT",
    "MODEL_REGISTRY_SCHEMA_VERSION",
    "MODEL_REVISION",
    "MODEL_TOTAL_BYTES",
    "RegistryError",
    "artifact_key",
    "build_artifact_record",
    "build_registry",
    "build_s3_receipt",
    "canonical_json_bytes",
    "content_identity",
    "load_registry",
    "load_registry_config",
    "save_registry",
    "validate_registry",
    "verify_s3_roundtrip",
]
