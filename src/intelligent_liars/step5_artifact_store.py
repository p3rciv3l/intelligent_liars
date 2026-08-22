"""Provider-neutral artifact identity and publication receipts for Step 5.

This module deliberately performs no network or object-store operations.  A caller
may publish the bytes under any bucket/key layout, then pass the observed metadata
and downloaded files back here for deterministic verification.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ARTIFACT_MANIFEST_FORMAT = "tinylora_step5_artifact_manifest_v1"
LIFECYCLE_ARTIFACT_MANIFEST_FORMAT = "tinylora_step5_lifecycle_artifact_manifest_v2"
HEAD_RECEIPT_FORMAT = "tinylora_step5_head_receipt_v1"
ROUNDTRIP_RECEIPT_FORMAT = "tinylora_step5_roundtrip_receipt_v1"
FINAL_RECEIPT_FORMAT = "tinylora_step5_final_receipt_v1"

PAYLOAD_ROLES = ("inputs", "logs", "checkpoints", "results")
MANIFEST_LOGICAL_PATH = "manifest.json"
FINAL_RECEIPT_LOGICAL_PATH = "final_receipt.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


_HASH_PROPERTY = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_EXPECTED_PROPERTIES = {
    "type": "object",
    "required": ["size_bytes", "sha256"],
    "properties": {
        "size_bytes": {"type": "integer", "minimum": 0},
        "sha256": _HASH_PROPERTY,
    },
    "additionalProperties": False,
}
_RECEIPT_PROPERTIES = {
    "artifact_set_id": _HASH_PROPERTY,
    "logical_path": {"type": "string", "minLength": 1},
    "object_ref": {"type": "string", "minLength": 1},
    "expected": _EXPECTED_PROPERTIES,
    "observed_at": {"type": "string", "format": "date-time"},
    "verified": {"type": "boolean"},
    "mismatches": {
        "type": "array",
        "items": {"enum": ["size_bytes", "sha256"]},
        "uniqueItems": True,
    },
}

# Public schemas make the on-disk receipt boundary explicit without introducing a
# runtime dependency on a JSON Schema implementation. The validators below also
# enforce manifest-relative semantic invariants that JSON Schema cannot express.
HEAD_RECEIPT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "format",
        "artifact_set_id",
        "logical_path",
        "object_ref",
        "expected",
        "observed",
        "observed_at",
        "verified",
        "mismatches",
    ],
    "properties": {
        **_RECEIPT_PROPERTIES,
        "format": {"const": HEAD_RECEIPT_FORMAT},
        "observed": {
            "type": "object",
            "required": ["size_bytes", "sha256", "etag"],
            "properties": {
                **_EXPECTED_PROPERTIES["properties"],
                "etag": {"type": ["string", "null"], "minLength": 1},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

ROUNDTRIP_RECEIPT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "format",
        "artifact_set_id",
        "logical_path",
        "object_ref",
        "expected",
        "downloaded",
        "observed_at",
        "verified",
        "mismatches",
    ],
    "properties": {
        **_RECEIPT_PROPERTIES,
        "format": {"const": ROUNDTRIP_RECEIPT_FORMAT},
        "downloaded": _EXPECTED_PROPERTIES,
    },
    "additionalProperties": False,
}


class ArtifactContractError(ValueError):
    """An artifact inventory or receipt violates the Step 5 contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return the one byte representation used to publish ``manifest.json``."""
    return _canonical_bytes(validate_artifact_manifest(manifest))


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactContractError("artifact path must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ArtifactContractError(f"Unsafe artifact path: {value!r}")
    return value


def _lifecycle_file_record(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        {"path", "bytes", "sha256"},
        label="lifecycle artifact record",
    )
    return {
        "path": _flat_logical_path(value["path"]),
        "bytes": _require_nonnegative_integer(
            value["bytes"], label="artifact bytes"
        ),
        "sha256": _require_sha256(value["sha256"], label="artifact sha256"),
    }


def build_lifecycle_artifact_manifest(
    *,
    run_id: str,
    files: Iterable[Mapping[str, Any]],
    durable_uri: str,
    durable_bytes: int,
    durable_sha256: str,
) -> dict[str, Any]:
    """Build the controller/worker lifecycle manifest with explicit v2 semantics."""
    normalized_files = sorted(
        (_lifecycle_file_record(item) for item in files),
        key=lambda item: item["path"],
    )
    identity = {
        "format": LIFECYCLE_ARTIFACT_MANIFEST_FORMAT,
        "run_id": run_id,
        "files": normalized_files,
        "durable_object": {
            "uri": durable_uri,
            "bytes": durable_bytes,
            "sha256": durable_sha256,
        },
    }
    return validate_lifecycle_artifact_manifest(
        {**identity, "artifact_set_id": _stable_sha256(identity)}
    )


def validate_lifecycle_artifact_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sole schema accepted by the Vast lifecycle wrapper.

    The older provider-neutral manifest remains deliberately named v1.  This v2
    format is not backwards-compatible and must never be silently interpreted as
    v1.
    """
    _require_exact_keys(
        value,
        {"format", "run_id", "artifact_set_id", "files", "durable_object"},
        label="lifecycle artifact manifest",
    )
    if value["format"] != LIFECYCLE_ARTIFACT_MANIFEST_FORMAT:
        raise ArtifactContractError("Unsupported lifecycle artifact manifest format")
    run_id = value["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ArtifactContractError("run_id must be nonempty")
    raw_files = value["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ArtifactContractError("lifecycle files must be a nonempty list")
    files = [_lifecycle_file_record(item) for item in raw_files]
    paths = [item["path"] for item in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ArtifactContractError("lifecycle file inventory must be sorted and unique")
    durable = value["durable_object"]
    if not isinstance(durable, Mapping):
        raise ArtifactContractError("durable_object must be an object")
    _require_exact_keys(
        durable,
        {"uri", "bytes", "sha256"},
        label="durable_object",
    )
    uri = durable["uri"]
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ArtifactContractError("durable_object uri must be an S3 URI")
    normalized_durable = {
        "uri": uri,
        "bytes": _require_nonnegative_integer(
            durable["bytes"], label="durable_object bytes"
        ),
        "sha256": _require_sha256(
            durable["sha256"], label="durable_object sha256"
        ),
    }
    if normalized_durable["bytes"] == 0:
        raise ArtifactContractError("durable_object bytes must be positive")
    identity = {
        "format": LIFECYCLE_ARTIFACT_MANIFEST_FORMAT,
        "run_id": run_id,
        "files": files,
        "durable_object": normalized_durable,
    }
    artifact_set_id = _require_sha256(
        value["artifact_set_id"], label="artifact_set_id"
    )
    if artifact_set_id != _stable_sha256(identity):
        raise ArtifactContractError("artifact_set_id does not match lifecycle contents")
    return {**identity, "artifact_set_id": artifact_set_id}


def _require_exact_keys(
    value: Mapping[str, Any],
    keys: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ArtifactContractError(
            f"{label} keys differ; missing={missing}, extra={extra}"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactContractError(f"{label} must be a nonnegative integer")
    return value


def _require_timestamp(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ArtifactContractError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ArtifactContractError(f"{label} must include a timezone")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _logical_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactContractError("logical_path must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ArtifactContractError(f"Unsafe logical_path: {value!r}")
    if len(path.parts) < 2 or path.parts[0] not in PAYLOAD_ROLES:
        raise ArtifactContractError(
            f"logical_path must begin with one of {PAYLOAD_ROLES}: {value!r}"
        )
    return value


def _artifact_record(value: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        value,
        {"logical_path", "role", "size_bytes", "sha256"},
        label="artifact record",
    )
    logical_path = _logical_path(value["logical_path"])
    role = value["role"]
    if role not in PAYLOAD_ROLES or PurePosixPath(logical_path).parts[0] != role:
        raise ArtifactContractError("Artifact role does not match its logical_path")
    return {
        "logical_path": logical_path,
        "role": role,
        "size_bytes": _require_nonnegative_integer(
            value["size_bytes"], label="artifact size_bytes"
        ),
        "sha256": _require_sha256(value["sha256"], label="artifact sha256"),
    }


def build_artifact_manifest(root: Path, *, run_id: str) -> dict[str, Any]:
    """Inventory the four required payload trees under ``root``.

    The resulting identity contains only logical paths and byte properties.  It is
    therefore unchanged when the same files are stored under a different S3 prefix
    or another object store entirely.
    """
    root = root.resolve()
    if not isinstance(run_id, str) or not run_id.strip():
        raise ArtifactContractError("run_id must be nonempty")
    artifacts: list[dict[str, Any]] = []
    for role in PAYLOAD_ROLES:
        role_root = root / role
        if not role_root.is_dir():
            raise ArtifactContractError(f"Required artifact role is missing: {role}")
        role_files: list[Path] = []
        for path in role_root.rglob("*"):
            if path.is_symlink():
                raise ArtifactContractError(f"Artifact inventory refuses symlink: {path}")
            if path.is_file():
                role_files.append(path)
        if not role_files:
            raise ArtifactContractError(f"Required artifact role is empty: {role}")
        for path in role_files:
            logical_path = path.relative_to(root).as_posix()
            artifacts.append(
                {
                    "logical_path": logical_path,
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    artifacts.sort(key=lambda item: item["logical_path"])
    identity = {
        "format": ARTIFACT_MANIFEST_FORMAT,
        "run_id": run_id,
        "artifacts": artifacts,
    }
    manifest = {**identity, "artifact_set_id": _stable_sha256(identity)}
    return validate_artifact_manifest(manifest)


def validate_artifact_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a Step 5 artifact manifest."""
    _require_exact_keys(
        value,
        {"format", "run_id", "artifact_set_id", "artifacts"},
        label="artifact manifest",
    )
    if value["format"] != ARTIFACT_MANIFEST_FORMAT:
        raise ArtifactContractError("Unsupported artifact manifest format")
    run_id = value["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ArtifactContractError("run_id must be nonempty")
    raw_artifacts = value["artifacts"]
    if not isinstance(raw_artifacts, list):
        raise ArtifactContractError("artifacts must be a list")
    artifacts = [_artifact_record(item) for item in raw_artifacts]
    paths = [item["logical_path"] for item in artifacts]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ArtifactContractError("Artifact logical_path inventory must be sorted and unique")
    observed_roles = {item["role"] for item in artifacts}
    missing_roles = set(PAYLOAD_ROLES) - observed_roles
    if missing_roles:
        raise ArtifactContractError(
            f"Artifact inventory is missing required roles: {sorted(missing_roles)}"
        )
    identity = {
        "format": ARTIFACT_MANIFEST_FORMAT,
        "run_id": run_id,
        "artifacts": artifacts,
    }
    artifact_set_id = _require_sha256(
        value["artifact_set_id"], label="artifact_set_id"
    )
    if artifact_set_id != _stable_sha256(identity):
        raise ArtifactContractError("artifact_set_id does not match manifest contents")
    return {**identity, "artifact_set_id": artifact_set_id}


def _manifest_artifact(
    manifest: Mapping[str, Any], logical_path: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = validate_artifact_manifest(manifest)
    matches = [
        item
        for item in _publication_artifacts(normalized)
        if item["logical_path"] == logical_path
    ]
    if len(matches) != 1:
        raise ArtifactContractError(f"Unknown artifact logical_path: {logical_path!r}")
    return normalized, matches[0]


def _publication_artifacts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical manifest followed by its declared payload artifacts."""
    normalized = validate_artifact_manifest(manifest)
    manifest_content = _canonical_bytes(normalized)
    return [
        {
            "logical_path": MANIFEST_LOGICAL_PATH,
            "role": "manifest",
            "size_bytes": len(manifest_content),
            "sha256": hashlib.sha256(manifest_content).hexdigest(),
        },
        *normalized["artifacts"],
    ]


def _expected(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "size_bytes": artifact["size_bytes"],
        "sha256": artifact["sha256"],
    }


def _mismatches(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> list[str]:
    return [
        field for field in ("size_bytes", "sha256") if expected[field] != observed[field]
    ]


def build_head_receipt(
    manifest: Mapping[str, Any],
    *,
    logical_path: str,
    object_ref: str,
    observed_size_bytes: int,
    observed_sha256: str,
    etag: str | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Record object metadata observed after publication.

    ``object_ref`` is intentionally opaque: it may be an S3 URI, version ID,
    provider locator, or an internal publication identifier.
    """
    normalized, artifact = _manifest_artifact(manifest, logical_path)
    if not isinstance(object_ref, str) or not object_ref:
        raise ArtifactContractError("object_ref must be nonempty")
    if etag is not None and (not isinstance(etag, str) or not etag):
        raise ArtifactContractError("etag must be null or nonempty")
    expected = _expected(artifact)
    observed = {
        "size_bytes": _require_nonnegative_integer(
            observed_size_bytes, label="observed size_bytes"
        ),
        "sha256": _require_sha256(observed_sha256, label="observed sha256"),
        "etag": etag,
    }
    mismatches = _mismatches(expected, observed)
    receipt = {
        "format": HEAD_RECEIPT_FORMAT,
        "artifact_set_id": normalized["artifact_set_id"],
        "logical_path": logical_path,
        "object_ref": object_ref,
        "expected": expected,
        "observed": observed,
        "observed_at": observed_at or _utc_now(),
        "verified": not mismatches,
        "mismatches": mismatches,
    }
    return validate_head_receipt(receipt, normalized)


def validate_head_receipt(
    value: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a local record of provider HEAD-equivalent metadata."""
    required = set(HEAD_RECEIPT_SCHEMA["required"])
    _require_exact_keys(value, required, label="HEAD receipt")
    if value["format"] != HEAD_RECEIPT_FORMAT:
        raise ArtifactContractError("Unsupported HEAD receipt format")
    normalized, artifact = _manifest_artifact(manifest, value["logical_path"])
    if value["artifact_set_id"] != normalized["artifact_set_id"]:
        raise ArtifactContractError("HEAD receipt artifact_set_id differs")
    if not isinstance(value["object_ref"], str) or not value["object_ref"]:
        raise ArtifactContractError("object_ref must be nonempty")
    expected = value["expected"]
    _require_exact_keys(expected, {"size_bytes", "sha256"}, label="HEAD expected")
    if expected != _expected(artifact):
        raise ArtifactContractError("HEAD receipt expected metadata differs from manifest")
    observed = value["observed"]
    _require_exact_keys(
        observed, {"size_bytes", "sha256", "etag"}, label="HEAD observed"
    )
    normalized_observed = {
        "size_bytes": _require_nonnegative_integer(
            observed["size_bytes"], label="observed size_bytes"
        ),
        "sha256": _require_sha256(observed["sha256"], label="observed sha256"),
        "etag": observed["etag"],
    }
    if normalized_observed["etag"] is not None and (
        not isinstance(normalized_observed["etag"], str)
        or not normalized_observed["etag"]
    ):
        raise ArtifactContractError("etag must be null or nonempty")
    mismatches = _mismatches(expected, normalized_observed)
    if value["mismatches"] != mismatches or value["verified"] is not (not mismatches):
        raise ArtifactContractError("HEAD receipt verification fields are inconsistent")
    _require_timestamp(value["observed_at"], label="observed_at")
    return dict(value)


def build_roundtrip_receipt(
    manifest: Mapping[str, Any],
    *,
    logical_path: str,
    object_ref: str,
    downloaded_path: Path,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Hash downloaded bytes and compare them to one manifest artifact."""
    normalized, artifact = _manifest_artifact(manifest, logical_path)
    if not isinstance(object_ref, str) or not object_ref:
        raise ArtifactContractError("object_ref must be nonempty")
    if downloaded_path.is_symlink() or not downloaded_path.is_file():
        raise ArtifactContractError("downloaded_path must be a regular non-symlink file")
    expected = _expected(artifact)
    downloaded = {
        "size_bytes": downloaded_path.stat().st_size,
        "sha256": sha256_file(downloaded_path),
    }
    mismatches = _mismatches(expected, downloaded)
    receipt = {
        "format": ROUNDTRIP_RECEIPT_FORMAT,
        "artifact_set_id": normalized["artifact_set_id"],
        "logical_path": logical_path,
        "object_ref": object_ref,
        "expected": expected,
        "downloaded": downloaded,
        "observed_at": observed_at or _utc_now(),
        "verified": not mismatches,
        "mismatches": mismatches,
    }
    return validate_roundtrip_receipt(receipt, normalized)


def validate_roundtrip_receipt(
    value: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a local record of a downloaded-byte round trip."""
    required = set(ROUNDTRIP_RECEIPT_SCHEMA["required"])
    _require_exact_keys(value, required, label="roundtrip receipt")
    if value["format"] != ROUNDTRIP_RECEIPT_FORMAT:
        raise ArtifactContractError("Unsupported roundtrip receipt format")
    normalized, artifact = _manifest_artifact(manifest, value["logical_path"])
    if value["artifact_set_id"] != normalized["artifact_set_id"]:
        raise ArtifactContractError("Roundtrip receipt artifact_set_id differs")
    if not isinstance(value["object_ref"], str) or not value["object_ref"]:
        raise ArtifactContractError("object_ref must be nonempty")
    expected = value["expected"]
    _require_exact_keys(
        expected, {"size_bytes", "sha256"}, label="roundtrip expected"
    )
    if expected != _expected(artifact):
        raise ArtifactContractError(
            "Roundtrip receipt expected metadata differs from manifest"
        )
    downloaded = value["downloaded"]
    _require_exact_keys(
        downloaded, {"size_bytes", "sha256"}, label="roundtrip downloaded"
    )
    normalized_downloaded = {
        "size_bytes": _require_nonnegative_integer(
            downloaded["size_bytes"], label="downloaded size_bytes"
        ),
        "sha256": _require_sha256(downloaded["sha256"], label="downloaded sha256"),
    }
    mismatches = _mismatches(expected, normalized_downloaded)
    if value["mismatches"] != mismatches or value["verified"] is not (not mismatches):
        raise ArtifactContractError(
            "Roundtrip receipt verification fields are inconsistent"
        )
    _require_timestamp(value["observed_at"], label="observed_at")
    return dict(value)


def _receipt_inventory(
    receipts: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    validator: Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw in receipts:
        receipt = validator(raw, manifest)
        logical_path = receipt["logical_path"]
        if logical_path in normalized:
            raise ArtifactContractError(f"Duplicate {label} receipt: {logical_path}")
        normalized[logical_path] = receipt
    expected = {item["logical_path"] for item in _publication_artifacts(manifest)}
    if set(normalized) != expected:
        raise ArtifactContractError(f"{label} receipt inventory is incomplete")
    return normalized


def build_final_receipt(
    manifest: Mapping[str, Any],
    *,
    head_receipts: Iterable[Mapping[str, Any]],
    roundtrip_receipts: Iterable[Mapping[str, Any]],
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Seal complete, verified HEAD and round-trip evidence for every artifact."""
    normalized = validate_artifact_manifest(manifest)
    heads = _receipt_inventory(
        head_receipts,
        normalized,
        validator=validate_head_receipt,
        label="HEAD",
    )
    roundtrips = _receipt_inventory(
        roundtrip_receipts,
        normalized,
        validator=validate_roundtrip_receipt,
        label="roundtrip",
    )
    verifications: list[dict[str, Any]] = []
    for artifact in _publication_artifacts(normalized):
        logical_path = artifact["logical_path"]
        head = heads[logical_path]
        roundtrip = roundtrips[logical_path]
        if not head["verified"] or not roundtrip["verified"]:
            raise ArtifactContractError(f"Artifact did not verify: {logical_path}")
        if head["object_ref"] != roundtrip["object_ref"]:
            raise ArtifactContractError(
                f"HEAD and roundtrip object_ref differ: {logical_path}"
            )
        verifications.append(
            {"logical_path": logical_path, "head": head, "roundtrip": roundtrip}
        )
    core = {
        "format": FINAL_RECEIPT_FORMAT,
        "logical_path": FINAL_RECEIPT_LOGICAL_PATH,
        "artifact_set_id": normalized["artifact_set_id"],
        "run_id": normalized["run_id"],
        "manifest_canonical_sha256": _stable_sha256(normalized),
        "completed_at": completed_at or _utc_now(),
        "verifications": verifications,
        "verified": True,
    }
    _require_timestamp(core["completed_at"], label="completed_at")
    receipt = {**core, "receipt_id": _stable_sha256(core)}
    return validate_final_receipt(receipt, normalized)


def validate_final_receipt(
    value: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a final receipt and all embedded publication evidence."""
    keys = {
        "format",
        "logical_path",
        "artifact_set_id",
        "run_id",
        "manifest_canonical_sha256",
        "completed_at",
        "verifications",
        "verified",
        "receipt_id",
    }
    _require_exact_keys(value, keys, label="final receipt")
    normalized = validate_artifact_manifest(manifest)
    if value["format"] != FINAL_RECEIPT_FORMAT:
        raise ArtifactContractError("Unsupported final receipt format")
    if value["logical_path"] != FINAL_RECEIPT_LOGICAL_PATH:
        raise ArtifactContractError("Final receipt logical_path differs")
    if value["artifact_set_id"] != normalized["artifact_set_id"]:
        raise ArtifactContractError("Final receipt artifact_set_id differs")
    if value["run_id"] != normalized["run_id"]:
        raise ArtifactContractError("Final receipt run_id differs")
    if value["manifest_canonical_sha256"] != _stable_sha256(normalized):
        raise ArtifactContractError("Final receipt manifest hash differs")
    _require_timestamp(value["completed_at"], label="completed_at")
    verifications = value["verifications"]
    if not isinstance(verifications, list):
        raise ArtifactContractError("Final receipt verifications must be a list")
    heads: list[Mapping[str, Any]] = []
    roundtrips: list[Mapping[str, Any]] = []
    observed_paths: list[str] = []
    for verification in verifications:
        _require_exact_keys(
            verification,
            {"logical_path", "head", "roundtrip"},
            label="final verification",
        )
        if verification["head"].get("logical_path") != verification["logical_path"]:
            raise ArtifactContractError("Final verification HEAD path differs")
        if verification["roundtrip"].get("logical_path") != verification["logical_path"]:
            raise ArtifactContractError("Final verification roundtrip path differs")
        heads.append(verification["head"])
        roundtrips.append(verification["roundtrip"])
        observed_paths.append(verification["logical_path"])
    expected_paths = [
        item["logical_path"] for item in _publication_artifacts(normalized)
    ]
    if observed_paths != expected_paths:
        raise ArtifactContractError("Final verification inventory differs from manifest")
    normalized_heads = _receipt_inventory(
        heads,
        normalized,
        validator=validate_head_receipt,
        label="HEAD",
    )
    normalized_roundtrips = _receipt_inventory(
        roundtrips,
        normalized,
        validator=validate_roundtrip_receipt,
        label="roundtrip",
    )
    for logical_path in expected_paths:
        head = normalized_heads[logical_path]
        roundtrip = normalized_roundtrips[logical_path]
        if (
            not head["verified"]
            or not roundtrip["verified"]
            or head["object_ref"] != roundtrip["object_ref"]
        ):
            raise ArtifactContractError(
                f"Final receipt contains unverified evidence: {logical_path}"
            )
    if value["verified"] is not True:
        raise ArtifactContractError("Final receipt must be verified")
    core = {key: value[key] for key in keys if key != "receipt_id"}
    receipt_id = _require_sha256(value["receipt_id"], label="receipt_id")
    if receipt_id != _stable_sha256(core):
        raise ArtifactContractError("receipt_id does not match final receipt contents")
    return dict(value)
