"""Durable off-host snapshots for the adaptive eight-trial batch barrier.

The local adaptive checkpoint remains authoritative.  This module packages its
verified state together with the files that make paid evaluation replay-safe,
publishes one immutable content-addressed archive, verifies a version-pinned
round trip, and only then conditionally advances a small ``latest.json``.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from uuid import uuid4


SNAPSHOT_FORMAT = "truth_editing_offhost_checkpoint_snapshot_v1"
POINTER_FORMAT = "truth_editing_offhost_checkpoint_latest_v1"
RECEIPT_FORMAT = "truth_editing_offhost_checkpoint_receipt_v1"
RESTORE_FORMAT = "truth_editing_offhost_checkpoint_restore_v1"
HYDRATION_FORMAT = "truth_editing_offhost_checkpoint_hydration_v1"
PARTIAL_SNAPSHOT_FORMAT = "truth_editing_offhost_partial_snapshot_v1"
PARTIAL_POINTER_FORMAT = "truth_editing_offhost_partial_latest_v1"
PARTIAL_RECEIPT_FORMAT = "truth_editing_offhost_partial_receipt_v1"
PARTIAL_RESTORE_FORMAT = "truth_editing_offhost_partial_restore_v1"
PARTIAL_HYDRATION_FORMAT = "truth_editing_offhost_partial_hydration_v1"
_REQUIRED_ROOTS = (
    "adaptive-state",
    "fleet-receipts",
    "runtime",
    "judge-cache",
    "judge-budget-ledger",
)
_ADAPTIVE_STATE_FILES = (
    "study/study-journal.json",
    "study/study-journal.json.optuna.log",
    "study/adaptive-run-checkpoint.json",
    "monitoring/wandb-run.json",
    "monitoring/adaptive-progress.json",
    "monitoring/rolling-capacity-receipt.json",
)
_SAFE_TELEMETRY_FIELDS = frozenset(
    {
        "evaluation_seconds",
        "generated_tokens",
        "generated_tokens_per_second",
        "cuda_peak_allocated_bytes",
        "projection_max_residual_ratio",
        "projection_max_error_ratio",
        "projection_total_weight_delta_norm",
        "projection_edited_writer_count",
        "projection_restoration_verified",
        "judge_calls",
        "judge_failures",
        "judge_latency_seconds",
        "judge_cost_usd",
    }
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_SECRET_MARKERS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-or-v1-[A-Za-z0-9_-]+\b", re.IGNORECASE),
    re.compile(
        rb"\b(?:OPENROUTER_API_KEY|WANDB_API_KEY|AWS_SECRET_ACCESS_KEY|"
        rb"AWS_SESSION_TOKEN|VAST_API_KEY|AUTHORIZATION)\s*[\"']?\s*[:=]",
        re.IGNORECASE,
    ),
)


class OffHostCheckpointError(RuntimeError):
    """The snapshot cannot be trusted as a durable batch barrier."""


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
        raise OffHostCheckpointError("checkpoint value is not canonical JSON") from error


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _compact_json_sha(value: Any) -> str:
    """Identity used by existing fleet receipts (no trailing newline)."""

    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise OffHostCheckpointError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OffHostCheckpointError(f"{label} must be a nonempty trimmed string")
    return value


def _key(value: Any, label: str = "object key") -> str:
    result = _text(value, label)
    pure = PurePosixPath(result)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != result or "//" in result:
        raise OffHostCheckpointError(f"{label} is unsafe")
    return result


def _read_json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OffHostCheckpointError(f"{label} is invalid JSON") from error
    if not isinstance(raw, dict):
        raise OffHostCheckpointError(f"{label} must be an object")
    return raw


def _reject_secrets(value: bytes) -> None:
    if any(pattern.search(value) for pattern in _SECRET_MARKERS):
        raise OffHostCheckpointError("checkpoint snapshot contains secret-like material")


@dataclass(frozen=True)
class SnapshotBinding:
    study_identity_sha256: str
    study_config_sha256: str
    fleet_config_sha256: str
    optuna_study_name: str
    wandb_run_id: str
    completed_trials: int
    trial_number_start: int = 0

    def __post_init__(self) -> None:
        _digest(self.study_identity_sha256, "study identity")
        _digest(self.study_config_sha256, "study config identity")
        _digest(self.fleet_config_sha256, "fleet config identity")
        _text(self.optuna_study_name, "Optuna study name")
        _text(self.wandb_run_id, "W&B run ID")
        if (
            isinstance(self.completed_trials, bool)
            or not isinstance(self.completed_trials, int)
            or not 0 <= self.completed_trials <= 800
            or self.completed_trials % 8
        ):
            raise OffHostCheckpointError(
                "completed trials must be an eight-trial barrier in [0, 800]"
            )
        if self.trial_number_start not in {0, 1}:
            raise OffHostCheckpointError("trial number start must be zero or one")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "study_identity_sha256": self.study_identity_sha256,
            "study_config_sha256": self.study_config_sha256,
            "fleet_config_sha256": self.fleet_config_sha256,
            "optuna_study_name": self.optuna_study_name,
            "wandb_run_id": self.wandb_run_id,
            "completed_trials": self.completed_trials,
            "trial_number_start": self.trial_number_start,
        }


@dataclass(frozen=True)
class PartialTrialReceiptBinding:
    trial_id: str
    ordinal: int
    proposal_sha256: str
    request_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or self.trial_id
            not in {f"trial-{self.ordinal:04d}", f"trial-{self.ordinal + 1:04d}"}
        ):
            raise OffHostCheckpointError("partial trial receipt identity is malformed")
        _digest(self.proposal_sha256, "partial proposal identity")
        _digest(self.request_sha256, "partial request identity")
        _digest(self.receipt_sha256, "partial receipt identity")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "ordinal": self.ordinal,
            "proposal_sha256": self.proposal_sha256,
            "request_sha256": self.request_sha256,
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True)
class PartialBatchBinding:
    committed: SnapshotBinding
    batch_ordinal: int
    batch_size: int
    durable_receipts: tuple[PartialTrialReceiptBinding, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_ordinal, bool)
            or not isinstance(self.batch_ordinal, int)
            or isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size != 8
            or self.batch_ordinal != self.committed.completed_trials // 8
        ):
            raise OffHostCheckpointError("partial batch identity differs from committed barrier")
        if not 1 <= len(self.durable_receipts) <= self.batch_size:
            raise OffHostCheckpointError("partial batch must bind one through eight receipts")
        ordered = tuple(sorted(self.durable_receipts, key=lambda item: item.ordinal))
        if ordered != self.durable_receipts or len({item.ordinal for item in ordered}) != len(ordered):
            raise OffHostCheckpointError("partial receipt set must be unique and ordered")
        allowed = set(
            range(
                self.committed.completed_trials,
                self.committed.completed_trials + self.batch_size,
            )
        )
        if {item.ordinal for item in ordered} - allowed:
            raise OffHostCheckpointError("partial receipt ordinal is outside current batch")
        if any(
            item.trial_id
            != f"trial-{item.ordinal + self.committed.trial_number_start:04d}"
            for item in ordered
        ):
            raise OffHostCheckpointError("partial receipt numbering differs from study")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "committed": self.committed.to_mapping(),
            "batch_ordinal": self.batch_ordinal,
            "batch_size": self.batch_size,
            "durable_receipts": [item.to_mapping() for item in self.durable_receipts],
        }

    @property
    def receipt_ordinals(self) -> frozenset[int]:
        return frozenset(item.ordinal for item in self.durable_receipts)


def _partial_binding_from_mapping(value: Any) -> PartialBatchBinding:
    if not isinstance(value, Mapping) or set(value) != {
        "committed",
        "batch_ordinal",
        "batch_size",
        "durable_receipts",
    }:
        raise OffHostCheckpointError("partial batch binding fields differ")
    committed_raw = value["committed"]
    binding_fields = set(SnapshotBinding.__dataclass_fields__)
    legacy_binding_fields = binding_fields - {"trial_number_start"}
    if not isinstance(committed_raw, Mapping) or set(committed_raw) not in (
        binding_fields,
        legacy_binding_fields,
    ):
        raise OffHostCheckpointError("partial committed binding fields differ")
    receipts_raw = value["durable_receipts"]
    if not isinstance(receipts_raw, list):
        raise OffHostCheckpointError("partial durable receipt set must be an array")
    receipts: list[PartialTrialReceiptBinding] = []
    for raw in receipts_raw:
        if not isinstance(raw, Mapping) or set(raw) != set(
            PartialTrialReceiptBinding.__dataclass_fields__
        ):
            raise OffHostCheckpointError("partial durable receipt fields differ")
        receipts.append(PartialTrialReceiptBinding(**dict(raw)))
    return PartialBatchBinding(
        committed=SnapshotBinding(
            **{"trial_number_start": 0, **dict(committed_raw)}
        ),
        batch_ordinal=value["batch_ordinal"],
        batch_size=value["batch_size"],
        durable_receipts=tuple(receipts),
    )


@dataclass(frozen=True)
class HydratedRuntimePaths:
    output_root: Path
    fleet_receipt_dir: Path
    runtime_output_dir: Path
    judge_cache_dir: Path
    judge_budget_ledger_dir: Path
    hydration_receipt_path: Path

    @classmethod
    def under(cls, output_root: Path | str) -> "HydratedRuntimePaths":
        root = Path(output_root)
        return cls(
            output_root=root,
            fleet_receipt_dir=root / "fleet-receipts",
            runtime_output_dir=root / "study/runtime",
            judge_cache_dir=root / "providers/judge-cache",
            judge_budget_ledger_dir=root / "providers/production-judge-budget",
            hydration_receipt_path=root / "offhost-hydration-receipt.json",
        )


@dataclass(frozen=True)
class OffHostCheckpointTarget:
    bucket: str
    region: str
    key_prefix: str
    registry_config_sha256: str

    @classmethod
    def from_model_registry_config(
        cls,
        path: Path | str,
        *,
        key_prefix: str,
        region: str = "us-east-1",
    ) -> "OffHostCheckpointTarget":
        config_path = Path(path)
        if config_path.is_symlink() or not config_path.is_file():
            raise OffHostCheckpointError("model registry config must be a regular file")
        raw_bytes = config_path.read_bytes()
        raw = _read_json_bytes(raw_bytes, "model registry config")
        registry = raw.get("registry")
        if (
            raw.get("format") != "intelligent_liars_model_registry_config_v1"
            or not isinstance(registry, Mapping)
            or set(registry) != {"bucket", "base_prefix"}
        ):
            raise OffHostCheckpointError("model registry configuration is incompatible")
        bucket = registry["bucket"]
        if not isinstance(bucket, str) or _SAFE_BUCKET.fullmatch(bucket) is None:
            raise OffHostCheckpointError("configured registry bucket is invalid")
        base = _key(registry["base_prefix"], "configured registry namespace")
        prefix = _key(key_prefix, "checkpoint key prefix").rstrip("/")
        if prefix == base or not prefix.startswith(base + "/"):
            raise OffHostCheckpointError(
                "checkpoint prefix must be inside the configured registry namespace"
            )
        return cls(
            bucket=bucket,
            region=_text(region, "AWS region"),
            key_prefix=prefix,
            registry_config_sha256=_bytes_sha(raw_bytes),
        )


@dataclass(frozen=True)
class StoredObject:
    key: str
    data: bytes
    version_id: str
    etag: str


class VersionedObjectStore(Protocol):
    def ensure_versioning(self) -> None: ...

    def read_current(self, key: str) -> StoredObject | None: ...

    def read_version(self, key: str, version_id: str) -> StoredObject: ...

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match_etag: str | None = None,
    ) -> StoredObject: ...


class FilesystemVersionedObjectStore:
    """Filesystem adapter with S3-like immutable versions and conditional puts."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def ensure_versioning(self) -> None:
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise OffHostCheckpointError("filesystem object store root is unsafe")
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def object_count(self) -> int:
        current = self.root / "current"
        return 0 if not current.exists() else sum(1 for path in current.rglob("*.json"))

    def _meta(self, key: str) -> Path:
        return self.root / "current" / (_key(key) + ".json")

    def _version(self, key: str, version: str) -> Path:
        return self.root / "versions" / _key(key) / f"{_text(version, 'version ID')}.bin"

    def read_current(self, key: str) -> StoredObject | None:
        path = self._meta(key)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise OffHostCheckpointError("filesystem object metadata is unsafe")
        raw = _read_json_bytes(path.read_bytes(), "filesystem object metadata")
        if set(raw) != {"version_id", "etag", "sha256", "size_bytes"}:
            raise OffHostCheckpointError("filesystem object metadata fields differ")
        result = self.read_version(key, _text(raw["version_id"], "version ID"))
        if (
            result.etag != raw["etag"]
            or _bytes_sha(result.data) != raw["sha256"]
            or len(result.data) != raw["size_bytes"]
        ):
            raise OffHostCheckpointError("filesystem object metadata identity differs")
        return result

    def read_version(self, key: str, version_id: str) -> StoredObject:
        path = self._version(key, version_id)
        if path.is_symlink() or not path.is_file():
            raise OffHostCheckpointError("requested object version is missing")
        data = path.read_bytes()
        return StoredObject(_key(key), data, version_id, _bytes_sha(data))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match_etag: str | None = None,
    ) -> StoredObject:
        self.ensure_versioning()
        lock_path = self.root / ".lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = self.read_current(key)
            if if_none_match and current is not None:
                raise OffHostCheckpointError("conditional object creation lost a lineage race")
            if if_match_etag is not None and (
                current is None or current.etag != if_match_etag
            ):
                raise OffHostCheckpointError("conditional object update lost a lineage race")
            version = uuid4().hex
            version_path = self._version(key, version)
            version_path.parent.mkdir(parents=True, exist_ok=True)
            _write_new(version_path, data)
            etag = _bytes_sha(data)
            metadata = {
                "version_id": version,
                "etag": etag,
                "sha256": etag,
                "size_bytes": len(data),
            }
            meta = self._meta(key)
            meta.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(meta, _canonical(metadata))
            return StoredObject(key, data, version, etag)


class S3VersionedObjectStore:
    """S3 adapter using version-pinned verification and conditional pointer puts."""

    def __init__(self, client: Any, *, bucket: str) -> None:
        if not isinstance(bucket, str) or _SAFE_BUCKET.fullmatch(bucket) is None:
            raise OffHostCheckpointError("S3 bucket is invalid")
        self.client = client
        self.bucket = bucket

    def ensure_versioning(self) -> None:
        response = self.client.get_bucket_versioning(Bucket=self.bucket)
        if response.get("Status") != "Enabled":
            raise OffHostCheckpointError("checkpoint bucket versioning is not enabled")

    def _read(self, key: str, version_id: str | None) -> StoredObject:
        arguments = {"Bucket": self.bucket, "Key": _key(key)}
        if version_id is not None:
            arguments["VersionId"] = _text(version_id, "S3 VersionId")
        response = self.client.get_object(**arguments)
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise OffHostCheckpointError("S3 object body is missing")
        data = body.read()
        version = response.get("VersionId")
        etag = response.get("ETag")
        if not isinstance(version, str) or not version or version == "null":
            raise OffHostCheckpointError("S3 object lacks an immutable VersionId")
        if not isinstance(etag, str) or not etag:
            raise OffHostCheckpointError("S3 object lacks an ETag")
        # Preserve the exact quoted ETag for a later S3 ``If-Match`` header.
        return StoredObject(key, data, version, etag)

    def read_current(self, key: str) -> StoredObject | None:
        try:
            return self._read(key, None)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def read_version(self, key: str, version_id: str) -> StoredObject:
        return self._read(key, version_id)

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match_etag: str | None = None,
    ) -> StoredObject:
        arguments: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": _key(key),
            "Body": data,
            "ChecksumAlgorithm": "SHA256",
            "ChecksumSHA256": base64.b64encode(bytes.fromhex(_bytes_sha(data))).decode(),
        }
        if if_none_match:
            arguments["IfNoneMatch"] = "*"
        if if_match_etag is not None:
            arguments["IfMatch"] = if_match_etag
        try:
            response = self.client.put_object(**arguments)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
                raise OffHostCheckpointError("conditional S3 put lost a lineage race") from error
            raise
        version = response.get("VersionId")
        if not isinstance(version, str) or not version or version == "null":
            raise OffHostCheckpointError("S3 put lacks an immutable VersionId")
        return self.read_version(key, version)


class OffHostCheckpointRepository:
    """One publish/restore interface for a paid-work-complete batch snapshot."""

    def __init__(
        self, *, store: VersionedObjectStore, target: OffHostCheckpointTarget
    ) -> None:
        self.store = store
        self.target = target
        self._partial_publish_lock = threading.Lock()

    @property
    def _latest_key(self) -> str:
        return f"{self.target.key_prefix}/latest.json"

    @property
    def _partial_latest_key(self) -> str:
        return f"{self.target.key_prefix}/partial/latest.json"

    def read_latest(self, expected: SnapshotBinding) -> StoredObject:
        self.store.ensure_versioning()
        current = self.store.read_current(self._latest_key)
        if current is None:
            raise OffHostCheckpointError("off-host latest pointer is missing")
        self._validate_pointer(current.data, expected=None)
        pointer = _read_json_bytes(current.data, "latest pointer")
        if any(pointer[name] != value for name, value in expected.to_mapping().items()):
            raise OffHostCheckpointError("off-host latest pointer identity differs")
        return current

    def read_latest_binding(
        self,
        expected_study_identity_sha256: str,
        expected_study_config_sha256: str,
        expected_fleet_config_sha256: str,
    ) -> SnapshotBinding:
        """Resolve dynamic resume identity from an immutable validated pointer.

        A clean host supplies only the three identities frozen before the run.
        The completed barrier, Optuna study name, and W&B run ID are recovered
        from the version-pinned latest pointer and remain exact inputs to the
        existing restore path.
        """

        binding = self.read_latest_binding_if_present(
            expected_study_identity_sha256,
            expected_study_config_sha256,
            expected_fleet_config_sha256,
        )
        if binding is None:
            raise OffHostCheckpointError("off-host latest pointer is missing")
        return binding

    def read_latest_binding_if_present(
        self,
        expected_study_identity_sha256: str,
        expected_study_config_sha256: str,
        expected_fleet_config_sha256: str,
    ) -> SnapshotBinding | None:
        """Return ``None`` only when a versioned store has no latest pointer."""

        expected_static = {
            "study_identity_sha256": _digest(
                expected_study_identity_sha256, "expected study identity"
            ),
            "study_config_sha256": _digest(
                expected_study_config_sha256, "expected study config identity"
            ),
            "fleet_config_sha256": _digest(
                expected_fleet_config_sha256, "expected fleet config identity"
            ),
        }
        self.store.ensure_versioning()
        current = self.store.read_current(self._latest_key)
        if current is None:
            return None
        pinned = self.store.read_version(self._latest_key, current.version_id)
        if pinned.data != current.data:
            raise OffHostCheckpointError("latest pointer version identity differs")
        pointer = self._validate_pointer(pinned.data, expected=None)
        if any(pointer[name] != value for name, value in expected_static.items()):
            raise OffHostCheckpointError("latest pointer static identity differs")
        return SnapshotBinding(
            **{
                name: pointer[name]
                for name in SnapshotBinding.__dataclass_fields__
            }
        )

    def publish(
        self,
        snapshot_root: Path | str,
        binding: SnapshotBinding,
        *,
        expected_latest_etag: str | None = None,
    ) -> dict[str, Any]:
        self.store.ensure_versioning()
        manifest, archive = _build_archive(Path(snapshot_root), binding)
        manifest_sha = manifest["manifest_sha256"]
        archive_sha = _bytes_sha(archive)
        archive_key = (
            f"{self.target.key_prefix}/generations/"
            f"{binding.completed_trials:04d}-{manifest_sha}/{archive_sha}.tar"
        )
        current = self.store.read_current(self._latest_key)
        if expected_latest_etag is not None and (
            current is None or current.etag != expected_latest_etag
        ):
            raise OffHostCheckpointError("latest pointer lineage race detected")
        previous_pointer_sha: str | None = None
        if current is not None:
            previous = self._validate_pointer(current.data, expected=None)
            if previous["study_identity_sha256"] != binding.study_identity_sha256:
                raise OffHostCheckpointError("latest pointer study identity differs")
            if previous["study_config_sha256"] != binding.study_config_sha256:
                raise OffHostCheckpointError("latest pointer study config differs")
            if previous["fleet_config_sha256"] != binding.fleet_config_sha256:
                raise OffHostCheckpointError("latest pointer fleet identity differs")
            previous_pointer_sha = previous["pointer_sha256"]
            if previous["completed_trials"] == binding.completed_trials:
                if (
                    previous["archive_sha256"] != archive_sha
                    or previous["snapshot_manifest_sha256"] != manifest_sha
                ):
                    raise OffHostCheckpointError("completed batch has conflicting bytes")
                return _receipt(previous, current.version_id)
            if previous["completed_trials"] + 8 != binding.completed_trials:
                raise OffHostCheckpointError("off-host checkpoint batch lineage is not contiguous")
        elif binding.completed_trials != 0:
            raise OffHostCheckpointError("first off-host checkpoint must be trial barrier zero")

        existing_archive = self.store.read_current(archive_key)
        if existing_archive is None:
            uploaded = self.store.put(archive_key, archive, if_none_match=True)
        else:
            uploaded = existing_archive
        verified_archive = self.store.read_version(archive_key, uploaded.version_id)
        if verified_archive.data != archive or _bytes_sha(verified_archive.data) != archive_sha:
            raise OffHostCheckpointError("off-host archive round-trip identity differs")
        _verify_archive(verified_archive.data, manifest)

        unsigned_pointer = {
            "format": POINTER_FORMAT,
            **binding.to_mapping(),
            "registry_config_sha256": self.target.registry_config_sha256,
            "bucket": self.target.bucket,
            "key_prefix": self.target.key_prefix,
            "archive_key": archive_key,
            "archive_version_id": verified_archive.version_id,
            "archive_sha256": archive_sha,
            "archive_size_bytes": len(archive),
            "snapshot_manifest_sha256": manifest_sha,
            "previous_pointer_sha256": previous_pointer_sha,
        }
        pointer = {
            **unsigned_pointer,
            "pointer_sha256": _json_sha(unsigned_pointer),
        }
        pointer_bytes = _canonical(pointer)
        pointer_object = self.store.put(
            self._latest_key,
            pointer_bytes,
            if_none_match=current is None,
            if_match_etag=None if current is None else current.etag,
        )
        verified_pointer = self.store.read_version(
            self._latest_key, pointer_object.version_id
        )
        if verified_pointer.data != pointer_bytes:
            raise OffHostCheckpointError("latest pointer round-trip identity differs")
        self._validate_pointer(verified_pointer.data, expected=binding)
        return _receipt(pointer, verified_pointer.version_id)

    def publish_partial(
        self,
        snapshot_root: Path | str,
        binding: PartialBatchBinding,
        *,
        expected_latest_etag: str | None = None,
    ) -> dict[str, Any]:
        """Advance the separate monotone in-flight receipt frontier."""

        # A partial generation is never allowed to invent its committed base.
        self.read_latest(binding.committed)
        manifest, archive = _build_partial_archive(Path(snapshot_root), binding)
        manifest_sha = manifest["manifest_sha256"]
        archive_sha = _bytes_sha(archive)
        binding_sha = _json_sha(binding.to_mapping())
        archive_key = (
            f"{self.target.key_prefix}/partial/generations/"
            f"{binding.committed.completed_trials:04d}-{binding_sha}/{archive_sha}.tar"
        )
        current = self.store.read_current(self._partial_latest_key)
        if expected_latest_etag is not None and (
            current is None or current.etag != expected_latest_etag
        ):
            raise OffHostCheckpointError("partial latest pointer lineage race detected")
        previous_pointer_sha: str | None = None
        if current is not None:
            previous = self._validate_partial_pointer(current.data, expected=None)
            previous_binding = _partial_binding_from_mapping(previous["binding"])
            previous_pointer_sha = previous["pointer_sha256"]
            static_names = (
                "study_identity_sha256",
                "study_config_sha256",
                "fleet_config_sha256",
                "optuna_study_name",
                "wandb_run_id",
            )
            if any(
                getattr(previous_binding.committed, name)
                != getattr(binding.committed, name)
                for name in static_names
            ):
                raise OffHostCheckpointError("partial latest static identity differs")
            previous_completed = previous_binding.committed.completed_trials
            completed = binding.committed.completed_trials
            if previous_completed == completed:
                if binding.receipt_ordinals <= previous_binding.receipt_ordinals:
                    if (
                        binding.receipt_ordinals == previous_binding.receipt_ordinals
                        and (
                            previous["archive_sha256"] != archive_sha
                            or previous["snapshot_manifest_sha256"] != manifest_sha
                        )
                    ):
                        raise OffHostCheckpointError(
                            "partial receipt frontier has conflicting bytes"
                        )
                    return _partial_receipt(previous, current.version_id)
                if not previous_binding.receipt_ordinals < binding.receipt_ordinals:
                    raise OffHostCheckpointError(
                        "partial receipt frontier is not a monotone superset"
                    )
            elif previous_completed + 8 != completed:
                raise OffHostCheckpointError("partial batch lineage is not contiguous")

        existing_archive = self.store.read_current(archive_key)
        uploaded = (
            self.store.put(archive_key, archive, if_none_match=True)
            if existing_archive is None
            else existing_archive
        )
        verified_archive = self.store.read_version(archive_key, uploaded.version_id)
        if verified_archive.data != archive or _bytes_sha(verified_archive.data) != archive_sha:
            raise OffHostCheckpointError("partial archive round-trip identity differs")
        _verify_archive(verified_archive.data, manifest)
        unsigned_pointer = {
            "format": PARTIAL_POINTER_FORMAT,
            "binding": binding.to_mapping(),
            "registry_config_sha256": self.target.registry_config_sha256,
            "bucket": self.target.bucket,
            "key_prefix": self.target.key_prefix,
            "archive_key": archive_key,
            "archive_version_id": verified_archive.version_id,
            "archive_sha256": archive_sha,
            "archive_size_bytes": len(archive),
            "snapshot_manifest_sha256": manifest_sha,
            "previous_pointer_sha256": previous_pointer_sha,
        }
        pointer = {
            **unsigned_pointer,
            "pointer_sha256": _json_sha(unsigned_pointer),
        }
        pointer_bytes = _canonical(pointer)
        pointer_object = self.store.put(
            self._partial_latest_key,
            pointer_bytes,
            if_none_match=current is None,
            if_match_etag=None if current is None else current.etag,
        )
        verified_pointer = self.store.read_version(
            self._partial_latest_key, pointer_object.version_id
        )
        if verified_pointer.data != pointer_bytes:
            raise OffHostCheckpointError("partial latest pointer round-trip identity differs")
        self._validate_partial_pointer(verified_pointer.data, expected=binding)
        return _partial_receipt(pointer, verified_pointer.version_id)

    def publish_partial_from_runtime(
        self,
        staging_parent: Path | str,
        *,
        committed_binding: SnapshotBinding,
        durable_event: Mapping[str, Any],
        adaptive_state_root: Path | str,
        fleet_receipt_dir: Path | str,
        runtime_output_dir: Path | str,
        judge_cache_dir: Path | str,
        judge_budget_ledger_dir: Path | str,
    ) -> tuple[dict[str, Any], PartialBatchBinding, Path]:
        """Serialize concurrent worker callbacks through verified publication."""

        with self._partial_publish_lock:
            parent = Path(staging_parent)
            parent.mkdir(parents=True, exist_ok=True)
            target = parent / f"partial-{uuid4().hex}"
            snapshot, binding = materialize_offhost_partial_snapshot(
                target,
                committed_binding=committed_binding,
                durable_event=durable_event,
                adaptive_state_root=adaptive_state_root,
                fleet_receipt_dir=fleet_receipt_dir,
                runtime_output_dir=runtime_output_dir,
                judge_cache_dir=judge_cache_dir,
                judge_budget_ledger_dir=judge_budget_ledger_dir,
            )
            return self.publish_partial(snapshot, binding), binding, snapshot

    def read_latest_partial_binding_if_present(
        self, expected_committed: SnapshotBinding
    ) -> PartialBatchBinding | None:
        """Resolve an in-flight frontier only when based on the current full barrier."""

        self.read_latest(expected_committed)
        current = self.store.read_current(self._partial_latest_key)
        if current is None:
            return None
        pinned = self.store.read_version(self._partial_latest_key, current.version_id)
        if pinned.data != current.data:
            raise OffHostCheckpointError("partial latest pointer version identity differs")
        pointer = self._validate_partial_pointer(pinned.data, expected=None)
        binding = _partial_binding_from_mapping(pointer["binding"])
        if binding.committed == expected_committed:
            return binding
        if (
            binding.committed.completed_trials < expected_committed.completed_trials
            and all(
                getattr(binding.committed, name) == getattr(expected_committed, name)
                for name in (
                    "study_identity_sha256",
                    "study_config_sha256",
                    "fleet_config_sha256",
                    "optuna_study_name",
                    "wandb_run_id",
                )
            )
        ):
            return None
        raise OffHostCheckpointError("partial latest committed identity differs")

    def restore_latest_partial(
        self,
        target_root: Path | str,
        expected: PartialBatchBinding,
    ) -> dict[str, Any]:
        self.read_latest(expected.committed)
        current = self.store.read_current(self._partial_latest_key)
        if current is None:
            raise OffHostCheckpointError("off-host partial latest pointer is missing")
        pointer = self._validate_partial_pointer(current.data, expected=expected)
        archive = self.store.read_version(
            pointer["archive_key"], pointer["archive_version_id"]
        )
        if (
            len(archive.data) != pointer["archive_size_bytes"]
            or _bytes_sha(archive.data) != pointer["archive_sha256"]
        ):
            raise OffHostCheckpointError("partial restore archive identity differs")
        manifest = _verify_archive(archive.data)
        if manifest["manifest_sha256"] != pointer["snapshot_manifest_sha256"]:
            raise OffHostCheckpointError("partial restore manifest identity differs")
        target = Path(target_root)
        if target.exists() or target.is_symlink():
            raise OffHostCheckpointError("partial restore target must not already exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
        staging.mkdir()
        try:
            _extract_archive(archive.data, staging)
            _validate_partial_snapshot(staging, expected)
            os.rename(staging, target)
            _fsync_directory(target.parent)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        unsigned = {
            "format": PARTIAL_RESTORE_FORMAT,
            "binding": expected.to_mapping(),
            "pointer_sha256": pointer["pointer_sha256"],
            "pointer_version_id": current.version_id,
            "archive_key": pointer["archive_key"],
            "archive_version_id": pointer["archive_version_id"],
            "archive_sha256": pointer["archive_sha256"],
            "snapshot_manifest_sha256": pointer["snapshot_manifest_sha256"],
        }
        return {**unsigned, "receipt_sha256": _json_sha(unsigned)}

    def _validate_partial_pointer(
        self, data: bytes, expected: PartialBatchBinding | None
    ) -> dict[str, Any]:
        raw = _read_json_bytes(data, "partial latest pointer")
        fields = {
            "format",
            "binding",
            "registry_config_sha256",
            "bucket",
            "key_prefix",
            "archive_key",
            "archive_version_id",
            "archive_sha256",
            "archive_size_bytes",
            "snapshot_manifest_sha256",
            "previous_pointer_sha256",
            "pointer_sha256",
        }
        if set(raw) != fields or raw["format"] != PARTIAL_POINTER_FORMAT:
            raise OffHostCheckpointError("partial latest pointer fields or format differ")
        binding = _partial_binding_from_mapping(raw["binding"])
        unsigned = dict(raw)
        claimed = _digest(unsigned.pop("pointer_sha256"), "partial pointer self hash")
        if claimed != _json_sha(unsigned):
            raise OffHostCheckpointError("partial latest pointer self hash differs")
        if (
            raw["registry_config_sha256"] != self.target.registry_config_sha256
            or raw["bucket"] != self.target.bucket
            or raw["key_prefix"] != self.target.key_prefix
        ):
            raise OffHostCheckpointError("partial latest pointer storage target differs")
        archive_key = _key(raw["archive_key"], "partial archive object key")
        if not archive_key.startswith(self.target.key_prefix + "/partial/generations/"):
            raise OffHostCheckpointError("partial archive key escapes its namespace")
        _text(raw["archive_version_id"], "partial archive VersionId")
        _digest(raw["archive_sha256"], "partial archive SHA-256")
        _digest(raw["snapshot_manifest_sha256"], "partial manifest SHA-256")
        if (
            isinstance(raw["archive_size_bytes"], bool)
            or not isinstance(raw["archive_size_bytes"], int)
            or raw["archive_size_bytes"] <= 0
        ):
            raise OffHostCheckpointError("partial archive size is invalid")
        if raw["previous_pointer_sha256"] is not None:
            _digest(raw["previous_pointer_sha256"], "previous partial pointer SHA-256")
        if expected is not None and binding != expected:
            raise OffHostCheckpointError("partial latest expected identity differs")
        return raw

    def restore_latest(
        self,
        target_root: Path | str,
        expected: SnapshotBinding,
    ) -> dict[str, Any]:
        current = self.read_latest(expected)
        pointer = self._validate_pointer(current.data, expected=expected)
        archive = self.store.read_version(
            pointer["archive_key"], pointer["archive_version_id"]
        )
        if (
            len(archive.data) != pointer["archive_size_bytes"]
            or _bytes_sha(archive.data) != pointer["archive_sha256"]
        ):
            raise OffHostCheckpointError("restore archive identity differs")
        manifest = _verify_archive(archive.data)
        if manifest["manifest_sha256"] != pointer["snapshot_manifest_sha256"]:
            raise OffHostCheckpointError("restore manifest identity differs")
        target = Path(target_root)
        if target.exists() or target.is_symlink():
            raise OffHostCheckpointError("restore target must not already exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
        staging.mkdir()
        try:
            _extract_archive(archive.data, staging)
            _validate_snapshot(staging, expected)
            os.rename(staging, target)
            _fsync_directory(target.parent)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        unsigned = {
            "format": RESTORE_FORMAT,
            **expected.to_mapping(),
            "pointer_sha256": pointer["pointer_sha256"],
            "pointer_version_id": current.version_id,
            "archive_key": pointer["archive_key"],
            "archive_version_id": pointer["archive_version_id"],
            "archive_sha256": pointer["archive_sha256"],
            "snapshot_manifest_sha256": pointer["snapshot_manifest_sha256"],
        }
        return {**unsigned, "receipt_sha256": _json_sha(unsigned)}

    def _validate_pointer(
        self, data: bytes, expected: SnapshotBinding | None
    ) -> dict[str, Any]:
        raw = _read_json_bytes(data, "latest pointer")
        fields = {
            "format",
            *SnapshotBinding.__dataclass_fields__,
            "registry_config_sha256",
            "bucket",
            "key_prefix",
            "archive_key",
            "archive_version_id",
            "archive_sha256",
            "archive_size_bytes",
            "snapshot_manifest_sha256",
            "previous_pointer_sha256",
            "pointer_sha256",
        }
        legacy_fields = fields - {"trial_number_start"}
        if set(raw) not in (fields, legacy_fields) or raw["format"] != POINTER_FORMAT:
            raise OffHostCheckpointError("latest pointer fields or format differ")
        parsed_binding = SnapshotBinding(
            **{
                name: raw.get(name, 0)
                for name in SnapshotBinding.__dataclass_fields__
            }
        )
        unsigned = dict(raw)
        claimed = _digest(unsigned.pop("pointer_sha256"), "pointer self hash")
        if claimed != _json_sha(unsigned):
            raise OffHostCheckpointError("latest pointer self hash differs")
        if (
            raw["registry_config_sha256"] != self.target.registry_config_sha256
            or raw["bucket"] != self.target.bucket
            or raw["key_prefix"] != self.target.key_prefix
        ):
            raise OffHostCheckpointError("latest pointer storage target differs")
        _digest(raw["archive_sha256"], "archive SHA-256")
        _digest(raw["snapshot_manifest_sha256"], "snapshot manifest SHA-256")
        archive_key = _key(raw["archive_key"], "archive object key")
        if not archive_key.startswith(self.target.key_prefix + "/generations/"):
            raise OffHostCheckpointError("latest pointer archive key escapes its namespace")
        _text(raw["archive_version_id"], "archive VersionId")
        if (
            isinstance(raw["archive_size_bytes"], bool)
            or not isinstance(raw["archive_size_bytes"], int)
            or raw["archive_size_bytes"] <= 0
        ):
            raise OffHostCheckpointError("latest pointer archive size is invalid")
        if raw["previous_pointer_sha256"] is not None:
            _digest(raw["previous_pointer_sha256"], "previous pointer SHA-256")
        if expected is not None and parsed_binding != expected:
            raise OffHostCheckpointError("latest pointer expected identity differs")
        if "trial_number_start" not in raw:
            raw = {**raw, "trial_number_start": 0}
        return raw


def _receipt(pointer: Mapping[str, Any], pointer_version_id: str) -> dict[str, Any]:
    unsigned = {
        "format": RECEIPT_FORMAT,
        **{name: pointer[name] for name in SnapshotBinding.__dataclass_fields__},
        "pointer_sha256": pointer["pointer_sha256"],
        "latest_pointer_version_id": pointer_version_id,
        "archive_key": pointer["archive_key"],
        "archive_version_id": pointer["archive_version_id"],
        "archive_sha256": pointer["archive_sha256"],
        "snapshot_manifest_sha256": pointer["snapshot_manifest_sha256"],
        "previous_pointer_sha256": pointer["previous_pointer_sha256"],
    }
    return {**unsigned, "receipt_sha256": _json_sha(unsigned)}


def _partial_receipt(
    pointer: Mapping[str, Any], pointer_version_id: str
) -> dict[str, Any]:
    unsigned = {
        "format": PARTIAL_RECEIPT_FORMAT,
        "binding": pointer["binding"],
        "pointer_sha256": pointer["pointer_sha256"],
        "latest_pointer_version_id": pointer_version_id,
        "archive_key": pointer["archive_key"],
        "archive_version_id": pointer["archive_version_id"],
        "archive_sha256": pointer["archive_sha256"],
        "snapshot_manifest_sha256": pointer["snapshot_manifest_sha256"],
        "previous_pointer_sha256": pointer["previous_pointer_sha256"],
    }
    return {**unsigned, "receipt_sha256": _json_sha(unsigned)}


def _snapshot_files(
    root: Path, *, allow_empty_roots: frozenset[str] = frozenset()
) -> list[tuple[str, bytes]]:
    if root.is_symlink() or not root.is_dir():
        raise OffHostCheckpointError("snapshot root must be a regular directory")
    actual_roots = {path.name for path in root.iterdir()}
    if actual_roots != set(_REQUIRED_ROOTS):
        raise OffHostCheckpointError("snapshot root inventory differs")
    files: list[tuple[str, bytes]] = []
    for name in _REQUIRED_ROOTS:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise OffHostCheckpointError(f"snapshot directory is unsafe: {name}")
        found = False
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise OffHostCheckpointError("snapshot refuses symlinks")
            if path.is_dir():
                continue
            if not path.is_file():
                raise OffHostCheckpointError("snapshot contains a non-regular file")
            found = True
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            _reject_secrets(data)
            files.append((relative, data))
        if not found and name not in allow_empty_roots:
            raise OffHostCheckpointError(f"snapshot directory is empty: {name}")
    return files


_TRANSIENT_ENTRY_MARKERS = (".staging-", ".tmp-", ".partial-", ".part")


def _is_transient_entry(name: str) -> bool:
    """Identify writer-owned temporary entries that are never resume state."""

    return name.startswith(".") and any(
        marker in name for marker in _TRANSIENT_ENTRY_MARKERS
    )


def _ignore_transient_entries(
    _directory: str, names: list[str]
) -> list[str]:
    """Keep concurrent atomic-writer staging paths out of copied snapshots."""

    return [name for name in names if _is_transient_entry(name)]


def _copy_resume_tree(
    source: Path, destination: Path, *, source_label: str
) -> None:
    """Copy a mutable resume tree while excluding ephemeral writer staging."""

    if source.is_symlink() or not source.is_dir():
        raise OffHostCheckpointError(
            f"{source_label} source is missing or unsafe: {source.name}"
        )
    for directory, dirnames, filenames in os.walk(source, topdown=True):
        # Prune temporary directories before checking symlinks. They may be
        # created and removed by a concurrent worker while the snapshot runs.
        dirnames[:] = [
            name for name in dirnames if not _is_transient_entry(name)
        ]
        for name in (*dirnames, *filenames):
            if (Path(directory) / name).is_symlink():
                raise OffHostCheckpointError(
                    f"{source_label} source contains a symlink: {source.name}"
                )
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        ignore=_ignore_transient_entries,
    )


def _validate_fleet_receipts(root: Path, binding: SnapshotBinding) -> None:
    directory = root / "fleet-receipts"
    expected = {
        f"trial-{ordinal + binding.trial_number_start:04d}.json"
        for ordinal in range(binding.completed_trials)
    }
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != expected:
        raise OffHostCheckpointError("snapshot must contain exact completed trial receipts")
    for ordinal in range(binding.completed_trials):
        _read_fleet_receipt(
            directory
            / f"trial-{ordinal + binding.trial_number_start:04d}.json",
            ordinal=ordinal,
            fleet_config_sha256=binding.fleet_config_sha256,
            trial_number_start=binding.trial_number_start,
        )


def _read_fleet_receipt(
    path: Path,
    *,
    ordinal: int,
    fleet_config_sha256: str,
    trial_number_start: int = 0,
) -> dict[str, Any]:
    raw = _read_json_bytes(path.read_bytes(), "fleet receipt")
    unsigned = dict(raw)
    claimed = unsigned.pop("receipt_sha256", None)
    receipt_format = raw.get("format")
    expected_fields = {
        "format",
        "fleet_config_sha256",
        "trial_id",
        "ordinal",
        "request_sha256",
        "worker_slot",
        "result",
        "receipt_sha256",
    }
    if receipt_format == "truth_editing_vast_fleet_trial_receipt_v2":
        expected_fields.add("telemetry")
        telemetry = raw.get("telemetry")
        if not isinstance(telemetry, Mapping) or not set(telemetry) <= _SAFE_TELEMETRY_FIELDS:
            raise OffHostCheckpointError("fleet receipt telemetry is not privacy-safe")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in telemetry.values()
        ):
            raise OffHostCheckpointError("fleet receipt telemetry is malformed")
    elif receipt_format != "truth_editing_vast_fleet_trial_receipt_v1":
        raise OffHostCheckpointError("fleet receipt format differs")
    if (
        set(raw) != expected_fields
        or raw.get("fleet_config_sha256") != fleet_config_sha256
        or raw.get("trial_id") != f"trial-{ordinal + trial_number_start:04d}"
        or raw.get("ordinal") != ordinal
        or claimed != _compact_json_sha(unsigned)
    ):
        raise OffHostCheckpointError("fleet receipt identity differs")
    _digest(raw.get("request_sha256"), "fleet request identity")
    return raw


def _validate_snapshot(root: Path, binding: SnapshotBinding) -> list[tuple[str, bytes]]:
    files = _snapshot_files(
        root,
        allow_empty_roots=(
            frozenset({"fleet-receipts", "runtime", "judge-cache"})
            if binding.completed_trials == 0
            else frozenset()
        ),
    )
    adaptive_files = {
        path.relative_to(root / "adaptive-state").as_posix()
        for path in (root / "adaptive-state").rglob("*")
        if path.is_file()
    }
    if adaptive_files != set(_ADAPTIVE_STATE_FILES):
        raise OffHostCheckpointError("adaptive state file inventory differs")
    _validate_fleet_receipts(root, binding)
    _validate_adaptive_barrier(root, binding)
    return files


def _validate_adaptive_barrier(root: Path, binding: SnapshotBinding) -> None:
    scheduler = _read_json_bytes(
        (root / "adaptive-state/study/adaptive-run-checkpoint.json").read_bytes(),
        "adaptive scheduler checkpoint",
    )
    rolling = _read_json_bytes(
        (
            root
            / "adaptive-state/monitoring/rolling-capacity-receipt.json"
        ).read_bytes(),
        "rolling capacity receipt",
    )
    rolling_sha = rolling.get("receipt_sha256")
    if (
        scheduler.get("completed_trials") != binding.completed_trials
        or scheduler.get("current_capacity_receipt_sha256") != rolling_sha
        or rolling.get("completed_through_trial") != binding.completed_trials
        or _SHA.fullmatch(str(rolling_sha)) is None
    ):
        raise OffHostCheckpointError(
            "adaptive scheduler is not bound to the rolling capacity receipt"
        )
    authorized = scheduler.get("authorized_through_trial")
    allowed_authorized = {
        binding.completed_trials,
        min(800, binding.completed_trials + 8),
    }
    if (
        isinstance(authorized, bool)
        or not isinstance(authorized, int)
        or authorized not in allowed_authorized
        or (binding.completed_trials == 0 and authorized != 8)
    ):
        raise OffHostCheckpointError(
            "adaptive scheduler authorization is not a durable batch barrier"
        )
    if "study_identity_sha256" in scheduler and (
        scheduler["study_identity_sha256"] != binding.study_identity_sha256
    ):
        raise OffHostCheckpointError("adaptive scheduler study identity differs")
    if "wandb_run_id" in scheduler and scheduler["wandb_run_id"] != binding.wandb_run_id:
        raise OffHostCheckpointError("adaptive scheduler W&B identity differs")
    wandb_path = root / "adaptive-state/monitoring/wandb-run.json"
    if wandb_path.is_file():
        raw = _read_json_bytes(wandb_path.read_bytes(), "W&B run checkpoint")
        if raw.get("run_id") != binding.wandb_run_id:
            raise OffHostCheckpointError("snapshot W&B run identity differs")


def _pending_batch_entries(root: Path, committed: SnapshotBinding) -> dict[int, Mapping[str, Any]]:
    journal = _read_json_bytes(
        (root / "adaptive-state/study/study-journal.json").read_bytes(),
        "partial study journal",
    )
    if set(journal) != {
        "format",
        "study_identity_sha256",
        "identity_inputs",
        "batches",
        "journal_sha256",
    } or journal.get("format") != "truth_editing_study_journal_v1":
        raise OffHostCheckpointError("partial study journal fields or format differ")
    unsigned = dict(journal)
    claimed = unsigned.pop("journal_sha256")
    if (
        claimed != _compact_json_sha(unsigned)
        or journal["study_identity_sha256"] != committed.study_identity_sha256
    ):
        raise OffHostCheckpointError("partial study journal identity differs")
    batches = journal["batches"]
    if not isinstance(batches, list) or len(batches) != committed.completed_trials // 8 + 1:
        raise OffHostCheckpointError("partial journal must contain one current batch")
    for index, batch in enumerate(batches):
        if (
            not isinstance(batch, Mapping)
            or set(batch) != {"ordinal", "trials"}
            or batch.get("ordinal") != index
            or not isinstance(batch.get("trials"), list)
            or len(batch["trials"]) != 8
        ):
            raise OffHostCheckpointError("partial journal batch structure differs")
        for entry in batch["trials"]:
            if not isinstance(entry, Mapping) or set(entry) != {
                "trial_id",
                "ordinal",
                "tier_name",
                "evaluation_record_ids",
                "proposal",
                "result",
            }:
                raise OffHostCheckpointError("partial journal trial fields differ")
            if index < len(batches) - 1 and entry["result"] is None:
                raise OffHostCheckpointError("committed journal batch contains a null result")
            if index == len(batches) - 1 and entry["result"] is not None:
                raise OffHostCheckpointError(
                    "partial batch results must remain invisible in the study journal"
                )
    pending = batches[-1]["trials"]
    expected_ordinals = range(committed.completed_trials, committed.completed_trials + 8)
    by_ordinal: dict[int, Mapping[str, Any]] = {}
    for expected, entry in zip(expected_ordinals, pending, strict=True):
        if (
                entry["ordinal"] != expected
                or entry["trial_id"]
                != f"trial-{expected + committed.trial_number_start:04d}"
            or not isinstance(entry["proposal"], Mapping)
            or not isinstance(entry["evaluation_record_ids"], list)
            or not isinstance(entry["tier_name"], str)
            or not entry["tier_name"]
        ):
            raise OffHostCheckpointError("partial journal pending proposal identity differs")
        by_ordinal[expected] = entry
    return by_ordinal


def _derive_partial_snapshot(
    root: Path, committed: SnapshotBinding
) -> tuple[list[tuple[str, bytes]], PartialBatchBinding]:
    files = _snapshot_files(
        root,
        allow_empty_roots=frozenset({"runtime", "judge-cache"}),
    )
    adaptive_files = {
        path.relative_to(root / "adaptive-state").as_posix()
        for path in (root / "adaptive-state").rglob("*")
        if path.is_file()
    }
    if adaptive_files != set(_ADAPTIVE_STATE_FILES):
        raise OffHostCheckpointError("adaptive state file inventory differs")
    _validate_adaptive_barrier(root, committed)
    entries = _pending_batch_entries(root, committed)
    directory = root / "fleet-receipts"
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    committed_names = {
        f"trial-{ordinal + committed.trial_number_start:04d}.json"
        for ordinal in range(committed.completed_trials)
    }
    partial_names = actual_names - committed_names
    if not partial_names or len(partial_names) > 8:
        raise OffHostCheckpointError("partial snapshot must contain current-batch receipts")
    durable: list[PartialTrialReceiptBinding] = []
    for ordinal in range(committed.completed_trials):
        _read_fleet_receipt(
            directory
            / f"trial-{ordinal + committed.trial_number_start:04d}.json",
            ordinal=ordinal,
            fleet_config_sha256=committed.fleet_config_sha256,
            trial_number_start=committed.trial_number_start,
        )
    for name in sorted(partial_names):
        if not re.fullmatch(r"trial-\d{4}\.json", name):
            raise OffHostCheckpointError("partial receipt filename is invalid")
        ordinal = int(name[6:10]) - committed.trial_number_start
        if ordinal not in entries:
            raise OffHostCheckpointError("partial receipt is outside current batch")
        raw = _read_fleet_receipt(
            directory / name,
            ordinal=ordinal,
            fleet_config_sha256=committed.fleet_config_sha256,
            trial_number_start=committed.trial_number_start,
        )
        durable.append(
            PartialTrialReceiptBinding(
                trial_id=raw["trial_id"],
                ordinal=ordinal,
                proposal_sha256=_compact_json_sha(entries[ordinal]["proposal"]),
                request_sha256=raw["request_sha256"],
                receipt_sha256=raw["receipt_sha256"],
            )
        )
    derived = PartialBatchBinding(
        committed=committed,
        batch_ordinal=committed.completed_trials // 8,
        batch_size=8,
        durable_receipts=tuple(durable),
    )
    return files, derived


def _validate_partial_snapshot(
    root: Path, binding: PartialBatchBinding
) -> list[tuple[str, bytes]]:
    files, derived = _derive_partial_snapshot(root, binding.committed)
    if derived != binding:
        raise OffHostCheckpointError("partial snapshot binding differs")
    return files


def materialize_offhost_snapshot(
    destination: Path | str,
    *,
    binding: SnapshotBinding,
    adaptive_state_root: Path | str,
    fleet_receipt_dir: Path | str,
    runtime_output_dir: Path | str,
    judge_cache_dir: Path | str,
    judge_budget_ledger_dir: Path | str,
) -> Path:
    """Atomically assemble only the mutable state required for exact resume."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise OffHostCheckpointError("snapshot destination must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        adaptive = Path(adaptive_state_root)
        (staging / "adaptive-state").mkdir()
        for relative in _ADAPTIVE_STATE_FILES:
            source = adaptive / relative
            if source.is_symlink() or not source.is_file():
                raise OffHostCheckpointError(
                    f"adaptive state file is missing or unsafe: {relative}"
                )
            destination_path = staging / "adaptive-state" / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination_path, source.read_bytes())
        for name, source_value in (
            ("fleet-receipts", fleet_receipt_dir),
            ("runtime", runtime_output_dir),
            ("judge-cache", judge_cache_dir),
            ("judge-budget-ledger", judge_budget_ledger_dir),
        ):
            source = Path(source_value)
            _copy_resume_tree(source, staging / name, source_label="snapshot")
        _validate_snapshot(staging, binding)
        os.rename(staging, target)
        _fsync_directory(target.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def materialize_offhost_partial_snapshot(
    destination: Path | str,
    *,
    committed_binding: SnapshotBinding,
    durable_event: Mapping[str, Any],
    adaptive_state_root: Path | str,
    fleet_receipt_dir: Path | str,
    runtime_output_dir: Path | str,
    judge_cache_dir: Path | str,
    judge_budget_ledger_dir: Path | str,
) -> tuple[Path, PartialBatchBinding]:
    """Atomically capture one privacy-minimal in-flight receipt frontier."""

    expected_event_fields = {
        "format",
        "fleet_config_sha256",
        "trial_id",
        "ordinal",
        "request_sha256",
        "receipt_path",
        "receipt_sha256",
    }
    if (
        set(durable_event) != expected_event_fields
        or durable_event.get("format")
        != "truth_editing_vast_fleet_receipt_durable_event_v1"
        or durable_event.get("fleet_config_sha256")
        != committed_binding.fleet_config_sha256
        or isinstance(durable_event.get("ordinal"), bool)
        or not isinstance(durable_event.get("ordinal"), int)
    ):
        raise OffHostCheckpointError("durable fleet receipt event identity differs")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise OffHostCheckpointError("partial snapshot destination must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid4().hex}"
    staging.mkdir()
    try:
        adaptive = Path(adaptive_state_root)
        (staging / "adaptive-state").mkdir()
        for relative in _ADAPTIVE_STATE_FILES:
            source = adaptive / relative
            if source.is_symlink() or not source.is_file():
                raise OffHostCheckpointError(
                    f"adaptive state file is missing or unsafe: {relative}"
                )
            destination_path = staging / "adaptive-state" / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination_path, source.read_bytes())
        for name, source_value in (
            ("fleet-receipts", fleet_receipt_dir),
            ("runtime", runtime_output_dir),
            ("judge-cache", judge_cache_dir),
            ("judge-budget-ledger", judge_budget_ledger_dir),
        ):
            source = Path(source_value)
            _copy_resume_tree(source, staging / name, source_label="partial")
        _, binding = _derive_partial_snapshot(staging, committed_binding)
        ordinal = durable_event.get("ordinal")
        event_receipt = next(
            (item for item in binding.durable_receipts if item.ordinal == ordinal),
            None,
        )
        expected_path = (
            Path(fleet_receipt_dir)
            / f"trial-{ordinal + committed_binding.trial_number_start:04d}.json"
        )
        if (
            event_receipt is None
            or Path(str(durable_event.get("receipt_path"))).resolve()
            != expected_path.resolve()
            or event_receipt.trial_id != durable_event.get("trial_id")
            or event_receipt.request_sha256 != durable_event.get("request_sha256")
            or event_receipt.receipt_sha256 != durable_event.get("receipt_sha256")
        ):
            raise OffHostCheckpointError("partial snapshot does not contain durable event")
        os.rename(staging, target)
        _fsync_directory(target.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target, binding


def hydrate_offhost_snapshot(
    snapshot_root: Path | str,
    output_root: Path | str,
    *,
    binding: SnapshotBinding,
) -> dict[str, Any]:
    """Atomically reconstruct the canonical controller tree on a clean host.

    The interface intentionally owns the layout.  Controller code must point
    its mutable paths at ``fleet-receipts``, ``study/runtime``,
    ``providers/judge-cache``, and ``providers/production-judge-budget`` under
    the supplied output root rather than copying categories itself.
    """

    source = Path(snapshot_root)
    source_files = dict(_validate_snapshot(source, binding))
    target = Path(output_root)
    paths = HydratedRuntimePaths.under(target)
    if target.exists() or target.is_symlink():
        raise OffHostCheckpointError("hydration output root must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.hydrate-{uuid4().hex}"
    staging.mkdir()
    mapping = {
        "adaptive-state": Path("."),
        "fleet-receipts": Path("fleet-receipts"),
        "runtime": Path("study/runtime"),
        "judge-cache": Path("providers/judge-cache"),
        "judge-budget-ledger": Path("providers/production-judge-budget"),
    }
    try:
        for directory in (
            Path("fleet-receipts"),
            Path("study/runtime"),
            Path("providers/judge-cache"),
            Path("providers/production-judge-budget"),
        ):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        hydrated_rows: list[dict[str, Any]] = []
        for snapshot_name, data in source_files.items():
            category, separator, relative = snapshot_name.partition("/")
            if not separator or category not in mapping:
                raise OffHostCheckpointError("snapshot hydration path is invalid")
            destination_relative = (mapping[category] / relative).as_posix()
            destination = staging / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination, data)
            hydrated_rows.append(
                {
                    "path": destination_relative,
                    "size_bytes": len(data),
                    "sha256": _bytes_sha(data),
                }
            )
        if len({row["path"] for row in hydrated_rows}) != len(hydrated_rows):
            raise OffHostCheckpointError("hydration layout contains a path collision")
        for row in hydrated_rows:
            path = staging / str(row["path"])
            data = path.read_bytes()
            if len(data) != row["size_bytes"] or _bytes_sha(data) != row["sha256"]:
                raise OffHostCheckpointError("hydrated checkpoint identity differs")
        unsigned = {
            "format": HYDRATION_FORMAT,
            **binding.to_mapping(),
            "files": sorted(hydrated_rows, key=lambda row: str(row["path"])),
        }
        receipt = {**unsigned, "receipt_sha256": _json_sha(unsigned)}
        receipt_relative = paths.hydration_receipt_path.relative_to(target)
        _write_new(staging / receipt_relative, _canonical(receipt))
        _fsync_tree(staging)
        os.rename(staging, target)
        _fsync_directory(target.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return receipt


def hydrate_offhost_partial_snapshot(
    snapshot_root: Path | str,
    output_root: Path | str,
    *,
    binding: PartialBatchBinding,
) -> dict[str, Any]:
    """Atomically hydrate an in-flight generation onto a clean host."""

    source = Path(snapshot_root)
    source_files = dict(_validate_partial_snapshot(source, binding))
    target = Path(output_root)
    if target.exists() or target.is_symlink():
        raise OffHostCheckpointError("partial hydration output root must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.partial-hydrate-{uuid4().hex}"
    staging.mkdir()
    mapping = {
        "adaptive-state": Path("."),
        "fleet-receipts": Path("fleet-receipts"),
        "runtime": Path("study/runtime"),
        "judge-cache": Path("providers/judge-cache"),
        "judge-budget-ledger": Path("providers/production-judge-budget"),
    }
    try:
        for directory in (
            Path("fleet-receipts"),
            Path("study/runtime"),
            Path("providers/judge-cache"),
            Path("providers/production-judge-budget"),
        ):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for snapshot_name, data in source_files.items():
            category, separator, relative = snapshot_name.partition("/")
            if not separator or category not in mapping:
                raise OffHostCheckpointError("partial hydration path is invalid")
            destination_relative = (mapping[category] / relative).as_posix()
            destination = staging / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination, data)
            rows.append(
                {
                    "path": destination_relative,
                    "size_bytes": len(data),
                    "sha256": _bytes_sha(data),
                }
            )
        if len({row["path"] for row in rows}) != len(rows):
            raise OffHostCheckpointError("partial hydration contains a path collision")
        unsigned = {
            "format": PARTIAL_HYDRATION_FORMAT,
            "binding": binding.to_mapping(),
            "files": sorted(rows, key=lambda row: str(row["path"])),
        }
        receipt = {**unsigned, "receipt_sha256": _json_sha(unsigned)}
        _write_new(
            staging / "offhost-partial-hydration-receipt.json",
            _canonical(receipt),
        )
        _fsync_tree(staging)
        os.rename(staging, target)
        _fsync_directory(target.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return receipt


def _build_archive(root: Path, binding: SnapshotBinding) -> tuple[dict[str, Any], bytes]:
    files = _validate_snapshot(root, binding)
    rows = [
        {"path": name, "size_bytes": len(data), "sha256": _bytes_sha(data)}
        for name, data in files
    ]
    unsigned = {
        "format": SNAPSHOT_FORMAT,
        **binding.to_mapping(),
        "files": rows,
    }
    manifest = {**unsigned, "manifest_sha256": _json_sha(unsigned)}
    manifest_bytes = _canonical(manifest)
    with tempfile.NamedTemporaryFile() as handle:
        with tarfile.open(fileobj=handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
            _tar_bytes(archive, "snapshot-manifest.json", manifest_bytes)
            for name, data in files:
                _tar_bytes(archive, name, data)
        handle.flush()
        archive_bytes = Path(handle.name).read_bytes()
    _verify_archive(archive_bytes, manifest)
    return manifest, archive_bytes


def _build_partial_archive(
    root: Path, binding: PartialBatchBinding
) -> tuple[dict[str, Any], bytes]:
    files = _validate_partial_snapshot(root, binding)
    rows = [
        {"path": name, "size_bytes": len(data), "sha256": _bytes_sha(data)}
        for name, data in files
    ]
    unsigned = {
        "format": PARTIAL_SNAPSHOT_FORMAT,
        "binding": binding.to_mapping(),
        "files": rows,
    }
    manifest = {**unsigned, "manifest_sha256": _json_sha(unsigned)}
    manifest_bytes = _canonical(manifest)
    with tempfile.NamedTemporaryFile() as handle:
        with tarfile.open(fileobj=handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
            _tar_bytes(archive, "snapshot-manifest.json", manifest_bytes)
            for name, data in files:
                _tar_bytes(archive, name, data)
        handle.flush()
        archive_bytes = Path(handle.name).read_bytes()
    _verify_archive(archive_bytes, manifest)
    return manifest, archive_bytes


def _tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    import io

    archive.addfile(info, io.BytesIO(data))


def _verify_archive(
    archive_bytes: bytes, expected_manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    import io

    try:
        archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:")
    except tarfile.TarError as error:
        raise OffHostCheckpointError("checkpoint archive is invalid") from error
    with archive:
        members = archive.getmembers()
        if len({item.name for item in members}) != len(members):
            raise OffHostCheckpointError("checkpoint archive contains duplicate paths")
        for member in members:
            pure = PurePosixPath(member.name)
            if not member.isfile() or pure.is_absolute() or ".." in pure.parts:
                raise OffHostCheckpointError("checkpoint archive contains an unsafe member")
        manifest_file = archive.extractfile("snapshot-manifest.json")
        if manifest_file is None:
            raise OffHostCheckpointError("checkpoint archive lacks its manifest")
        manifest = _read_json_bytes(manifest_file.read(), "snapshot manifest")
        manifest_format = manifest.get("format")
        manifest_fields = (
            {
                "format",
                *SnapshotBinding.__dataclass_fields__,
                "files",
                "manifest_sha256",
            }
            if manifest_format == SNAPSHOT_FORMAT
            else {"format", "binding", "files", "manifest_sha256"}
            if manifest_format == PARTIAL_SNAPSHOT_FORMAT
            else set()
        )
        legacy_manifest_fields = manifest_fields - {"trial_number_start"}
        if set(manifest) not in (manifest_fields, legacy_manifest_fields):
            raise OffHostCheckpointError("snapshot manifest fields differ")
        if manifest_format == SNAPSHOT_FORMAT:
            SnapshotBinding(
                **{
                    name: manifest.get(name, 0)
                    for name in SnapshotBinding.__dataclass_fields__
                }
            )
        else:
            _partial_binding_from_mapping(manifest["binding"])
        unsigned = dict(manifest)
        claimed = _digest(unsigned.pop("manifest_sha256", None), "manifest self hash")
        if claimed != _json_sha(unsigned) or manifest_format not in {
            SNAPSHOT_FORMAT,
            PARTIAL_SNAPSHOT_FORMAT,
        }:
            raise OffHostCheckpointError("snapshot manifest identity differs")
        if expected_manifest is not None and dict(expected_manifest) != manifest:
            raise OffHostCheckpointError("snapshot manifest differs from expected bytes")
        declared = manifest.get("files")
        if not isinstance(declared, list):
            raise OffHostCheckpointError("snapshot manifest file inventory is invalid")
        expected_names = {"snapshot-manifest.json"}
        for row in declared:
            if not isinstance(row, Mapping) or set(row) != {"path", "size_bytes", "sha256"}:
                raise OffHostCheckpointError("snapshot manifest file row is invalid")
            name = _key(row["path"], "snapshot file path")
            if name in expected_names:
                raise OffHostCheckpointError("snapshot manifest contains duplicate paths")
            expected_names.add(name)
            if (
                isinstance(row["size_bytes"], bool)
                or not isinstance(row["size_bytes"], int)
                or row["size_bytes"] < 0
            ):
                raise OffHostCheckpointError("snapshot manifest file size is invalid")
            _digest(row["sha256"], "snapshot file SHA-256")
            member = archive.extractfile(name)
            if member is None:
                raise OffHostCheckpointError("snapshot archive file is missing")
            data = member.read()
            if len(data) != row["size_bytes"] or _bytes_sha(data) != row["sha256"]:
                raise OffHostCheckpointError("snapshot archive file identity differs")
            _reject_secrets(data)
        if {item.name for item in members} != expected_names:
            raise OffHostCheckpointError("snapshot archive contains undeclared files")
        return manifest


def _extract_archive(archive_bytes: bytes, target: Path) -> None:
    import io

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            if member.name == "snapshot-manifest.json":
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OffHostCheckpointError("snapshot archive member is unreadable")
            path = target / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_new(path, source.read())
    for name in _REQUIRED_ROOTS:
        (target / name).mkdir(parents=True, exist_ok=True)


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    _write_new(temporary, data)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


__all__ = [
    "FilesystemVersionedObjectStore",
    "HydratedRuntimePaths",
    "OffHostCheckpointError",
    "OffHostCheckpointRepository",
    "OffHostCheckpointTarget",
    "PartialBatchBinding",
    "PartialTrialReceiptBinding",
    "S3VersionedObjectStore",
    "SnapshotBinding",
    "StoredObject",
    "VersionedObjectStore",
    "hydrate_offhost_snapshot",
    "hydrate_offhost_partial_snapshot",
    "materialize_offhost_partial_snapshot",
    "materialize_offhost_snapshot",
]
