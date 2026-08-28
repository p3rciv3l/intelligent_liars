"""Versioned, compact off-host progress for adaptive finalization.

Search checkpoints advance in trial batches.  Finalization advances through
smaller immutable commits: one causal candidate, repeat, matched control, or
selection at a time.  This repository gives those commits an independent,
ordinal pointer lineage while preserving the frozen study, W&B, and judge
ledger identities.  Only an explicit evidence-file allowlist is archived;
model weights are never admitted.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .truth_editing_offhost_checkpoint import (
    OffHostCheckpointTarget,
    StoredObject,
    VersionedObjectStore,
)


SNAPSHOT_FORMAT = "truth_editing_finalization_progress_snapshot_v1"
POINTER_FORMAT = "truth_editing_finalization_progress_latest_v1"
RECEIPT_FORMAT = "truth_editing_finalization_progress_receipt_v1"
RESTORE_FORMAT = "truth_editing_finalization_progress_restore_v1"

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_STAGE_KINDS = frozenset(
    {
        "causal_candidate",
        "repeat_evaluation",
        "matched_control",
        "final_selection",
        "checkpoint_export",
        "complete",
    }
)
_WEIGHT_SUFFIXES = frozenset(
    {
        ".bin",
        ".safetensors",
        ".pt",
        ".pth",
        ".ckpt",
        ".gguf",
        ".onnx",
        ".h5",
        ".hdf5",
    }
)
_MAX_FILES = 256
_MAX_FILE_BYTES = 8 * 1024 * 1024
_MAX_ARCHIVE_INPUT_BYTES = 32 * 1024 * 1024
_FIXED_IDENTITY_FIELDS = frozenset(
    {
        "study_identity_sha256",
        "study_config_sha256",
        "fleet_config_sha256",
        "finalization_identity_sha256",
        "judge_ledger_root_sha256",
        "optuna_study_name",
        "wandb_run_id",
    }
)
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


class FinalizationProgressError(RuntimeError):
    """A finalization progress generation cannot be trusted or restored."""


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
        raise FinalizationProgressError("value is not canonical JSON") from error


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha(value: Any) -> str:
    return _bytes_sha(_canonical(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise FinalizationProgressError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise FinalizationProgressError(f"{label} must be a nonempty trimmed string")
    return value


def _slug(value: Any, label: str) -> str:
    result = _text(value, label)
    if _SLUG.fullmatch(result) is None:
        raise FinalizationProgressError(f"{label} is invalid")
    return result


def _relative_path(value: str | Path) -> str:
    raw = str(value)
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != raw
        or "//" in raw
        or raw == "."
    ):
        raise FinalizationProgressError("evidence path is unsafe")
    return raw


def _object(value: bytes, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalizationProgressError(f"{label} is invalid JSON") from error
    if not isinstance(raw, dict):
        raise FinalizationProgressError(f"{label} must be an object")
    return raw


@dataclass(frozen=True)
class FinalizationProgressBinding:
    """Identity and ledger transition for one immutable finalization commit."""

    study_identity_sha256: str
    study_config_sha256: str
    fleet_config_sha256: str
    finalization_identity_sha256: str
    judge_ledger_root_sha256: str
    judge_ledger_before_sha256: str
    judge_ledger_after_sha256: str
    optuna_study_name: str
    wandb_run_id: str
    stage_ordinal: int
    stage_kind: str
    commit_id: str

    def __post_init__(self) -> None:
        for field in (
            "study_identity_sha256",
            "study_config_sha256",
            "fleet_config_sha256",
            "finalization_identity_sha256",
            "judge_ledger_root_sha256",
            "judge_ledger_before_sha256",
            "judge_ledger_after_sha256",
        ):
            _digest(getattr(self, field), field.replace("_", " "))
        _text(self.optuna_study_name, "Optuna study name")
        _text(self.wandb_run_id, "W&B run ID")
        if (
            isinstance(self.stage_ordinal, bool)
            or not isinstance(self.stage_ordinal, int)
            or not 0 <= self.stage_ordinal <= 4096
        ):
            raise FinalizationProgressError("stage ordinal must be in [0, 4096]")
        if self.stage_kind not in _STAGE_KINDS:
            raise FinalizationProgressError("stage kind is unsupported")
        _slug(self.commit_id, "commit ID")
        if (
            self.stage_ordinal == 0
            and self.judge_ledger_before_sha256 != self.judge_ledger_root_sha256
        ):
            raise FinalizationProgressError(
                "first stage must begin at the judge ledger root"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "study_identity_sha256": self.study_identity_sha256,
            "study_config_sha256": self.study_config_sha256,
            "fleet_config_sha256": self.fleet_config_sha256,
            "finalization_identity_sha256": self.finalization_identity_sha256,
            "judge_ledger_root_sha256": self.judge_ledger_root_sha256,
            "judge_ledger_before_sha256": self.judge_ledger_before_sha256,
            "judge_ledger_after_sha256": self.judge_ledger_after_sha256,
            "optuna_study_name": self.optuna_study_name,
            "wandb_run_id": self.wandb_run_id,
            "stage_ordinal": self.stage_ordinal,
            "stage_kind": self.stage_kind,
            "commit_id": self.commit_id,
        }


class FinalizationProgressRepository:
    """Publish and restore exact compact evidence paths for finalization."""

    def __init__(
        self, *, store: VersionedObjectStore, target: OffHostCheckpointTarget
    ) -> None:
        self.store = store
        self.target = target

    @property
    def _latest_key(self) -> str:
        return f"{self.target.key_prefix}/latest.json"

    def read_latest(self, expected: FinalizationProgressBinding) -> StoredObject:
        current = self._read_current_object()
        pointer = self._validate_pointer(current.data)
        if any(pointer[name] != value for name, value in expected.to_mapping().items()):
            raise FinalizationProgressError("finalization latest pointer identity differs")
        return current

    def read_current(self, expected_fixed_identity: Mapping[str, Any]) -> dict[str, Any]:
        """Discover the current stage without knowing its ordinal on a clean host."""

        expected = _fixed_identity(expected_fixed_identity)
        current = self._read_current_object()
        pointer = self._validate_pointer(current.data)
        if any(pointer[name] != value for name, value in expected.items()):
            raise FinalizationProgressError("finalization latest pointer identity differs")
        binding = {
            name: pointer[name]
            for name in FinalizationProgressBinding.__dataclass_fields__
        }
        return {
            "binding": binding,
            "etag": current.etag,
            "latest_pointer_version_id": current.version_id,
            "pointer_sha256": pointer["pointer_sha256"],
        }

    def publish(
        self,
        source_root: Path | str,
        binding: FinalizationProgressBinding,
        *,
        evidence_paths: Sequence[str | Path],
        expected_latest_etag: str | None = None,
    ) -> dict[str, Any]:
        manifest, archive = _build_archive(Path(source_root), binding, evidence_paths)
        self.store.ensure_versioning()
        current = self.store.read_current(self._latest_key)
        if expected_latest_etag is not None and (
            current is None or current.etag != expected_latest_etag
        ):
            raise FinalizationProgressError("latest pointer lineage race detected")

        previous_sha: str | None = None
        if current is None:
            if binding.stage_ordinal != 0:
                raise FinalizationProgressError("first finalization stage must be ordinal zero")
        else:
            previous = self._validate_pointer(current.data)
            _validate_fixed_identity(previous, binding)
            previous_sha = str(previous["pointer_sha256"])
            if previous["stage_ordinal"] == binding.stage_ordinal:
                if (
                    previous["snapshot_manifest_sha256"]
                    != manifest["manifest_sha256"]
                    or previous["archive_sha256"] != _bytes_sha(archive)
                ):
                    raise FinalizationProgressError("completed stage has conflicting bytes")
                return _receipt(previous, current.version_id)
            if previous["stage_ordinal"] + 1 != binding.stage_ordinal:
                raise FinalizationProgressError("finalization stage lineage is not contiguous")
            if (
                previous["judge_ledger_after_sha256"]
                != binding.judge_ledger_before_sha256
            ):
                raise FinalizationProgressError("finalization judge ledger lineage differs")

        archive_sha = _bytes_sha(archive)
        manifest_sha = str(manifest["manifest_sha256"])
        archive_key = (
            f"{self.target.key_prefix}/generations/"
            f"{binding.stage_ordinal:04d}-{manifest_sha}/{archive_sha}.tar"
        )
        existing = self.store.read_current(archive_key)
        uploaded = (
            self.store.put(archive_key, archive, if_none_match=True)
            if existing is None
            else existing
        )
        verified = self.store.read_version(archive_key, uploaded.version_id)
        if verified.data != archive or _bytes_sha(verified.data) != archive_sha:
            raise FinalizationProgressError("finalization archive round-trip identity differs")
        _verify_archive(verified.data, expected_manifest=manifest)

        unsigned = {
            "format": POINTER_FORMAT,
            **binding.to_mapping(),
            "registry_config_sha256": self.target.registry_config_sha256,
            "bucket": self.target.bucket,
            "key_prefix": self.target.key_prefix,
            "archive_key": archive_key,
            "archive_version_id": verified.version_id,
            "archive_sha256": archive_sha,
            "archive_size_bytes": len(archive),
            "snapshot_manifest_sha256": manifest_sha,
            "file_count": len(manifest["files"]),
            "previous_pointer_sha256": previous_sha,
        }
        pointer = {**unsigned, "pointer_sha256": _json_sha(unsigned)}
        pointer_bytes = _canonical(pointer)
        written = self.store.put(
            self._latest_key,
            pointer_bytes,
            if_none_match=current is None,
            if_match_etag=None if current is None else current.etag,
        )
        roundtrip = self.store.read_version(self._latest_key, written.version_id)
        if roundtrip.data != pointer_bytes:
            raise FinalizationProgressError("latest pointer round-trip identity differs")
        self._validate_pointer(roundtrip.data)
        return _receipt(pointer, roundtrip.version_id)

    def restore_latest(
        self, target_root: Path | str, expected: FinalizationProgressBinding
    ) -> dict[str, Any]:
        pointer_object = self.read_latest(expected)
        pointer = self._validate_pointer(pointer_object.data)
        return self._restore_pointer(target_root, pointer_object, pointer)

    def restore_current(
        self,
        target_root: Path | str,
        expected_fixed_identity: Mapping[str, Any],
        *,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        """Discover and restore the latest exact paths for clean-host resume.

        ``replace_existing`` is intentionally opt-in.  It atomically replaces
        only differing regular files explicitly listed in the verified archive;
        this lets newer finalization ledger/cache evidence supersede an older
        batch snapshot without touching any unlisted restored state.
        """

        expected = _fixed_identity(expected_fixed_identity)
        pointer_object = self._read_current_object()
        pointer = self._validate_pointer(pointer_object.data)
        if any(pointer[name] != value for name, value in expected.items()):
            raise FinalizationProgressError("finalization latest pointer identity differs")
        restored = self._restore_pointer(
            target_root,
            pointer_object,
            pointer,
            replace_existing=replace_existing,
        )
        restored["binding"] = {
            name: pointer[name]
            for name in FinalizationProgressBinding.__dataclass_fields__
        }
        restored["latest_pointer_etag"] = pointer_object.etag
        restored["latest_pointer_version_id"] = pointer_object.version_id
        return restored

    def _restore_pointer(
        self,
        target_root: Path | str,
        pointer_object: StoredObject,
        pointer: Mapping[str, Any],
        *,
        replace_existing: bool = False,
    ) -> dict[str, Any]:
        archive = self.store.read_version(
            str(pointer["archive_key"]), str(pointer["archive_version_id"])
        )
        if (
            len(archive.data) != pointer["archive_size_bytes"]
            or _bytes_sha(archive.data) != pointer["archive_sha256"]
        ):
            raise FinalizationProgressError("restore archive identity differs")
        manifest, files = _verify_archive(archive.data)
        if manifest["manifest_sha256"] != pointer["snapshot_manifest_sha256"]:
            raise FinalizationProgressError("restore manifest identity differs")
        root = Path(target_root)
        if root.exists() and (root.is_symlink() or not root.is_dir()):
            raise FinalizationProgressError("restore root is unsafe")
        root.mkdir(parents=True, exist_ok=True)
        replaced_paths: list[str] = []
        for relative, data in files.items():
            destination = _prepare_restore_destination(root, relative)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise FinalizationProgressError("restore destination is unsafe")
                if destination.read_bytes() != data:
                    if not replace_existing:
                        raise FinalizationProgressError("restore destination conflicts")
                    _write_replace(destination, data)
                    replaced_paths.append(relative)
                continue
            _write_new(destination, data)
        return {
            "format": RESTORE_FORMAT,
            "stage_ordinal": pointer["stage_ordinal"],
            "stage_kind": pointer["stage_kind"],
            "commit_id": pointer["commit_id"],
            "pointer_sha256": pointer["pointer_sha256"],
            "archive_sha256": pointer["archive_sha256"],
            "restored_paths": sorted(files),
            "replaced_paths": sorted(replaced_paths),
        }

    def _read_current_object(self) -> StoredObject:
        self.store.ensure_versioning()
        current = self.store.read_current(self._latest_key)
        if current is None:
            raise FinalizationProgressError("finalization latest pointer is missing")
        return current

    def _validate_pointer(self, data: bytes) -> dict[str, Any]:
        pointer = _object(data, "finalization latest pointer")
        expected_fields = {
            "format",
            *FinalizationProgressBinding.__dataclass_fields__,
            "registry_config_sha256",
            "bucket",
            "key_prefix",
            "archive_key",
            "archive_version_id",
            "archive_sha256",
            "archive_size_bytes",
            "snapshot_manifest_sha256",
            "file_count",
            "previous_pointer_sha256",
            "pointer_sha256",
        }
        if set(pointer) != expected_fields or pointer.get("format") != POINTER_FORMAT:
            raise FinalizationProgressError("finalization latest pointer fields differ")
        unsigned = dict(pointer)
        claimed = unsigned.pop("pointer_sha256")
        if claimed != _json_sha(unsigned):
            raise FinalizationProgressError("finalization latest pointer checksum differs")
        binding = FinalizationProgressBinding(
            **{
                name: pointer[name]
                for name in FinalizationProgressBinding.__dataclass_fields__
            }
        )
        if pointer["registry_config_sha256"] != self.target.registry_config_sha256:
            raise FinalizationProgressError("finalization registry identity differs")
        if pointer["bucket"] != self.target.bucket or pointer["key_prefix"] != self.target.key_prefix:
            raise FinalizationProgressError("finalization storage identity differs")
        _digest(pointer["archive_sha256"], "archive identity")
        _digest(pointer["snapshot_manifest_sha256"], "manifest identity")
        if pointer["previous_pointer_sha256"] is not None:
            _digest(pointer["previous_pointer_sha256"], "previous pointer identity")
        if not isinstance(pointer["archive_size_bytes"], int) or pointer["archive_size_bytes"] <= 0:
            raise FinalizationProgressError("archive size is invalid")
        if not isinstance(pointer["file_count"], int) or not 1 <= pointer["file_count"] <= _MAX_FILES:
            raise FinalizationProgressError("file count is invalid")
        _text(pointer["archive_key"], "archive key")
        _text(pointer["archive_version_id"], "archive version ID")
        if binding.stage_ordinal == 0 and pointer["previous_pointer_sha256"] is not None:
            raise FinalizationProgressError("first pointer cannot have a predecessor")
        return pointer


def _validate_fixed_identity(
    previous: Mapping[str, Any], binding: FinalizationProgressBinding
) -> None:
    for field in (
        "study_identity_sha256",
        "study_config_sha256",
        "fleet_config_sha256",
        "finalization_identity_sha256",
        "judge_ledger_root_sha256",
        "optuna_study_name",
        "wandb_run_id",
    ):
        if previous[field] != getattr(binding, field):
            raise FinalizationProgressError("finalization pointer identity differs")


def _fixed_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    if set(raw) != _FIXED_IDENTITY_FIELDS:
        raise FinalizationProgressError("fixed finalization identity fields differ")
    for field in (
        "study_identity_sha256",
        "study_config_sha256",
        "fleet_config_sha256",
        "finalization_identity_sha256",
        "judge_ledger_root_sha256",
    ):
        _digest(raw[field], field.replace("_", " "))
    _text(raw["optuna_study_name"], "Optuna study name")
    _text(raw["wandb_run_id"], "W&B run ID")
    return raw


def _evidence_files(
    source_root: Path, evidence_paths: Sequence[str | Path]
) -> list[tuple[str, bytes]]:
    if source_root.is_symlink() or not source_root.is_dir():
        raise FinalizationProgressError("evidence source root must be a regular directory")
    relative_paths = [_relative_path(path) for path in evidence_paths]
    if not relative_paths or len(relative_paths) > _MAX_FILES:
        raise FinalizationProgressError("evidence file count is outside the compact limit")
    if len(set(relative_paths)) != len(relative_paths):
        raise FinalizationProgressError("evidence paths must be unique")
    result: list[tuple[str, bytes]] = []
    total = 0
    resolved_root = source_root.resolve(strict=True)
    for relative in sorted(relative_paths):
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise FinalizationProgressError("evidence path must be a regular file")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise FinalizationProgressError(
                "evidence path must remain inside the source root"
            ) from error
        cursor = path.parent
        while cursor != source_root:
            if cursor.is_symlink():
                raise FinalizationProgressError(
                    "evidence path must not traverse a symlink"
                )
            cursor = cursor.parent
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            raise FinalizationProgressError("finalization progress cannot contain model weights")
        data = path.read_bytes()
        if len(data) > _MAX_FILE_BYTES:
            raise FinalizationProgressError("evidence file exceeds the compact size limit")
        total += len(data)
        if total > _MAX_ARCHIVE_INPUT_BYTES:
            raise FinalizationProgressError("evidence exceeds the compact archive size limit")
        if any(marker.search(data) for marker in _SECRET_MARKERS):
            raise FinalizationProgressError("evidence contains secret-like material")
        result.append((relative, data))
    return result


def _build_archive(
    source_root: Path,
    binding: FinalizationProgressBinding,
    evidence_paths: Sequence[str | Path],
) -> tuple[dict[str, Any], bytes]:
    files = _evidence_files(source_root, evidence_paths)
    unsigned = {
        "format": SNAPSHOT_FORMAT,
        "binding": binding.to_mapping(),
        "files": [
            {"path": path, "size_bytes": len(data), "sha256": _bytes_sha(data)}
            for path, data in files
        ],
    }
    manifest = {**unsigned, "manifest_sha256": _json_sha(unsigned)}
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        _tar_add(archive, "manifest.json", _canonical(manifest))
        for path, data in files:
            _tar_add(archive, f"evidence/{path}", data)
    return manifest, output.getvalue()


def _tar_add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _verify_archive(
    archive_bytes: bytes, *, expected_manifest: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, bytes]]:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            members = archive.getmembers()
            if any(not member.isfile() for member in members):
                raise FinalizationProgressError("archive contains a non-file member")
            names = [member.name for member in members]
            if not names or names[0] != "manifest.json" or len(set(names)) != len(names):
                raise FinalizationProgressError("archive member inventory differs")
            manifest_file = archive.extractfile(members[0])
            if manifest_file is None:
                raise FinalizationProgressError("archive manifest is missing")
            manifest = _object(manifest_file.read(), "archive manifest")
            files: dict[str, bytes] = {}
            for member in members[1:]:
                if not member.name.startswith("evidence/"):
                    raise FinalizationProgressError("archive evidence path differs")
                relative = _relative_path(member.name.removeprefix("evidence/"))
                handle = archive.extractfile(member)
                if handle is None:
                    raise FinalizationProgressError("archive evidence is missing")
                files[relative] = handle.read()
    except (tarfile.TarError, OSError) as error:
        raise FinalizationProgressError("finalization archive is invalid") from error

    if set(manifest) != {"format", "binding", "files", "manifest_sha256"} or manifest.get("format") != SNAPSHOT_FORMAT:
        raise FinalizationProgressError("archive manifest fields differ")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256")
    if claimed != _json_sha(unsigned):
        raise FinalizationProgressError("archive manifest checksum differs")
    if expected_manifest is not None and dict(expected_manifest) != manifest:
        raise FinalizationProgressError("archive manifest round-trip identity differs")
    raw_binding = manifest["binding"]
    if not isinstance(raw_binding, Mapping):
        raise FinalizationProgressError("archive binding is invalid")
    FinalizationProgressBinding(**dict(raw_binding))
    inventory = manifest["files"]
    if not isinstance(inventory, list) or not inventory:
        raise FinalizationProgressError("archive file inventory is invalid")
    expected_paths: list[str] = []
    for item in inventory:
        if not isinstance(item, Mapping) or set(item) != {"path", "size_bytes", "sha256"}:
            raise FinalizationProgressError("archive file entry fields differ")
        path = _relative_path(item["path"])
        data = files.get(path)
        if (
            data is None
            or item["size_bytes"] != len(data)
            or item["sha256"] != _bytes_sha(data)
        ):
            raise FinalizationProgressError("archive file identity differs")
        expected_paths.append(path)
    if expected_paths != sorted(expected_paths) or set(expected_paths) != set(files):
        raise FinalizationProgressError("archive file inventory differs")
    return manifest, files


def _receipt(pointer: Mapping[str, Any], pointer_version_id: str) -> dict[str, Any]:
    return {
        "format": RECEIPT_FORMAT,
        "stage_ordinal": pointer["stage_ordinal"],
        "stage_kind": pointer["stage_kind"],
        "commit_id": pointer["commit_id"],
        "judge_ledger_after_sha256": pointer["judge_ledger_after_sha256"],
        "archive_key": pointer["archive_key"],
        "archive_version_id": pointer["archive_version_id"],
        "archive_sha256": pointer["archive_sha256"],
        "file_count": pointer["file_count"],
        "pointer_sha256": pointer["pointer_sha256"],
        "previous_pointer_sha256": pointer["previous_pointer_sha256"],
        "latest_pointer_version_id": pointer_version_id,
    }


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FinalizationProgressError("restore destination appeared concurrently") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_replace(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _prepare_restore_destination(root: Path, relative: str) -> Path:
    cursor = root
    for part in PurePosixPath(relative).parts[:-1]:
        cursor = cursor / part
        if cursor.exists():
            if cursor.is_symlink() or not cursor.is_dir():
                raise FinalizationProgressError("restore destination is unsafe")
        else:
            cursor.mkdir()
    return root / relative


__all__ = [
    "FinalizationProgressBinding",
    "FinalizationProgressError",
    "FinalizationProgressRepository",
    "POINTER_FORMAT",
    "RECEIPT_FORMAT",
    "RESTORE_FORMAT",
    "SNAPSHOT_FORMAT",
]
