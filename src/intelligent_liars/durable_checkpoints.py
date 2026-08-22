"""Transport-neutral, crash-safe checkpoint generation publication.

A generation is first copied into an immutable directory and verified locally.  A
separate operation advances ``latest.json`` only after an optional durable-store
verifier accepts the generation.  The pointer remembers which generations were
previously accepted, allowing safe two-generation retention without deleting an
unverified generation that happens to share the same checkpoint root.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

GENERATION_MANIFEST_FORMAT = "intelligent_liars_checkpoint_generation_v1"
LATEST_POINTER_FORMAT = "intelligent_liars_checkpoint_latest_v1"
ROOT_IDENTITY_FORMAT = "intelligent_liars_checkpoint_root_identity_v1"
VERIFICATION_MARKER_FORMAT = "intelligent_liars_checkpoint_verified_v1"
MANIFEST_NAME = "manifest.json"


class CheckpointError(RuntimeError):
    """Base class for durable checkpoint errors."""


class CheckpointIdentityMismatch(CheckpointError):
    """Raised when a checkpoint root or generation belongs to another run."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when checkpoint bytes do not match their declared identities."""


class ImmutableCheckpointError(CheckpointError):
    """Raised when code attempts to replace an existing generation."""


@dataclass(frozen=True)
class CheckpointGeneration:
    """A locally verified immutable checkpoint generation."""

    generation_id: str
    path: Path
    manifest_path: Path
    manifest_sha256: str
    identity: dict[str, Any]


DurableVerifier = Callable[[CheckpointGeneration], bool]


def create_checkpoint_generation(
    root: Path,
    *,
    identity: Mapping[str, Any],
    generation_id: str,
    source_dir: Path,
) -> CheckpointGeneration:
    """Atomically create and locally verify one immutable generation.

    This operation deliberately does not update ``latest.json``.  Call
    :func:`advance_latest_checkpoint` after any external upload has been checked.
    """

    root = Path(root)
    source_dir = Path(source_dir)
    normalized_identity = _normalize_identity(identity)
    _validate_generation_id(generation_id)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint source directory is missing: {source_dir}")
    try:
        source_dir.resolve().relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Checkpoint source directory cannot be inside its store root")

    _ensure_root_identity(root, normalized_identity)
    generations_dir = root / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    final_path = generations_dir / generation_id
    if final_path.exists():
        raise ImmutableCheckpointError(
            f"Checkpoint generation already exists: {generation_id}"
        )

    staging_path = generations_dir / f".{generation_id}.tmp-{uuid4().hex}"
    staging_path.mkdir()
    try:
        files = _copy_checkpoint_files(source_dir, staging_path)
        if not files:
            raise ValueError("Checkpoint source directory contains no files")
        manifest: dict[str, Any] = {
            "format": GENERATION_MANIFEST_FORMAT,
            "generation_id": generation_id,
            "identity": normalized_identity,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "files": files,
        }
        manifest["manifest_sha256"] = _document_sha256(manifest)
        _write_new_file(staging_path / MANIFEST_NAME, manifest)
        _fsync_directory_tree(staging_path)
        generation = validate_checkpoint_generation(
            staging_path,
            expected_identity=normalized_identity,
            expected_generation_id=generation_id,
            _require_directory_name=False,
        )
        try:
            os.rename(staging_path, final_path)
        except FileExistsError as exc:
            raise ImmutableCheckpointError(
                f"Checkpoint generation already exists: {generation_id}"
            ) from exc
        _fsync_directory(generations_dir)
        return CheckpointGeneration(
            generation_id=generation.generation_id,
            path=final_path,
            manifest_path=final_path / MANIFEST_NAME,
            manifest_sha256=generation.manifest_sha256,
            identity=generation.identity,
        )
    finally:
        shutil.rmtree(staging_path, ignore_errors=True)


def validate_checkpoint_generation(
    generation_path: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_generation_id: str | None = None,
    _require_directory_name: bool = True,
) -> CheckpointGeneration:
    """Verify manifest identity, inventory, sizes, and SHA-256 hashes."""

    generation_path = Path(generation_path)
    manifest_path = generation_path / MANIFEST_NAME
    if generation_path.is_symlink() or not generation_path.is_dir():
        raise CheckpointIntegrityError(
            f"Checkpoint generation directory is invalid: {generation_path}"
        )
    manifest = _read_json(manifest_path, "generation manifest")
    if manifest.get("format") != GENERATION_MANIFEST_FORMAT:
        raise CheckpointIntegrityError(
            f"Unsupported checkpoint manifest format at {manifest_path}"
        )
    manifest_sha256 = manifest.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or (
        manifest_sha256 != _document_sha256(manifest, omit="manifest_sha256")
    ):
        raise CheckpointIntegrityError(
            f"Checkpoint manifest SHA-256 mismatch: {manifest_path}"
        )
    identity = _normalize_identity_document(manifest.get("identity"), manifest_path)
    if expected_identity is not None and identity != _normalize_identity(
        expected_identity
    ):
        raise CheckpointIdentityMismatch(
            f"Checkpoint identity mismatch at {manifest_path}"
        )
    generation_id = manifest.get("generation_id")
    if not isinstance(generation_id, str):
        raise CheckpointIntegrityError(
            f"Checkpoint generation ID is invalid at {manifest_path}"
        )
    _validate_generation_id(generation_id)
    if (_require_directory_name and generation_path.name != generation_id) or (
        expected_generation_id is not None and generation_id != expected_generation_id
    ):
        raise CheckpointIntegrityError(
            f"Checkpoint generation identity mismatch at {manifest_path}"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise CheckpointIntegrityError(
            f"Checkpoint manifest has no file inventory: {manifest_path}"
        )
    declared_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CheckpointIntegrityError(
                f"Checkpoint file inventory is invalid: {manifest_path}"
            )
        relative = _validate_relative_path(entry.get("path"))
        relative_text = relative.as_posix()
        if relative_text in declared_paths or relative_text == MANIFEST_NAME:
            raise CheckpointIntegrityError(
                f"Checkpoint file inventory contains an invalid duplicate: {relative_text}"
            )
        declared_paths.add(relative_text)
        artifact = generation_path.joinpath(*relative.parts)
        if artifact.is_symlink() or not artifact.is_file():
            raise CheckpointIntegrityError(f"Checkpoint artifact is missing: {artifact}")
        expected_size = entry.get("size_bytes")
        if not isinstance(expected_size, int) or artifact.stat().st_size != expected_size:
            raise CheckpointIntegrityError(f"Checkpoint size mismatch: {artifact}")
        expected_sha = entry.get("sha256")
        if not isinstance(expected_sha, str) or _file_sha256(artifact) != expected_sha:
            raise CheckpointIntegrityError(f"Checkpoint SHA-256 mismatch: {artifact}")

    actual_paths = {
        path.relative_to(generation_path).as_posix()
        for path in generation_path.rglob("*")
        if path.is_file()
        and path.relative_to(generation_path).as_posix() != MANIFEST_NAME
    }
    if actual_paths != declared_paths:
        raise CheckpointIntegrityError(
            f"Checkpoint inventory mismatch at {generation_path}"
        )
    return CheckpointGeneration(
        generation_id=generation_id,
        path=generation_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        identity=identity,
    )


def advance_latest_checkpoint(
    root: Path,
    generation_id: str,
    *,
    identity: Mapping[str, Any],
    durable_verifier: DurableVerifier,
) -> CheckpointGeneration:
    """Verify durability, atomically advance latest, then retain two generations."""

    root = Path(root)
    normalized_identity = _normalize_identity(identity)
    _validate_root_identity(root, normalized_identity)
    generation = validate_checkpoint_generation(
        root / "generations" / generation_id,
        expected_identity=normalized_identity,
        expected_generation_id=generation_id,
    )
    if durable_verifier(generation) is not True:
        raise CheckpointIntegrityError(
            f"Checkpoint durability verification rejected generation {generation_id}"
        )

    with _exclusive_pointer_lock(root):
        generation = validate_checkpoint_generation(
            root / "generations" / generation_id,
            expected_identity=normalized_identity,
            expected_generation_id=generation_id,
        )
        _ensure_verification_marker(root, generation)
        pointer_path = root / "latest.json"
        previous_pointer = (
            _validate_latest_pointer(pointer_path, normalized_identity)
            if pointer_path.exists()
            else None
        )
        previous_generation_id = (
            str(previous_pointer["generation_id"])
            if previous_pointer is not None
            and previous_pointer["generation_id"] != generation_id
            else (
                previous_pointer.get("previous_generation_id")
                if previous_pointer is not None
                else None
            )
        )
        retained = [generation_id]
        if isinstance(previous_generation_id, str):
            retained.append(previous_generation_id)
        pointer: dict[str, Any] = {
            "format": LATEST_POINTER_FORMAT,
            "identity": normalized_identity,
            "generation_id": generation_id,
            "manifest_sha256": generation.manifest_sha256,
            "previous_generation_id": previous_generation_id,
            "retained_generation_ids": retained,
        }
        pointer["pointer_sha256"] = _document_sha256(pointer)
        _write_json_atomic(pointer_path, pointer)
        _prune_verified_generations(root, normalized_identity, retain=set(retained))
    return generation


def resolve_latest_checkpoint(
    root: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> CheckpointGeneration:
    """Resolve and fully verify the current checkpoint pointer."""

    root = Path(root)
    pointer_path = root / "latest.json"
    identity = (
        _normalize_identity(expected_identity)
        if expected_identity is not None
        else None
    )
    pointer = _validate_latest_pointer(pointer_path, identity)
    generation = validate_checkpoint_generation(
        root / "generations" / str(pointer["generation_id"]),
        expected_identity=pointer["identity"],
        expected_generation_id=str(pointer["generation_id"]),
    )
    if generation.manifest_sha256 != pointer.get("manifest_sha256"):
        raise CheckpointIntegrityError(
            f"Latest checkpoint manifest identity mismatch: {pointer_path}"
        )
    marker = _validate_verification_marker(
        root / "verified" / f"{generation.generation_id}.json",
        generation.identity,
    )
    if marker.get("manifest_sha256") != generation.manifest_sha256:
        raise CheckpointIntegrityError(
            f"Latest checkpoint verification identity mismatch: {pointer_path}"
        )
    return generation


def _copy_checkpoint_files(source_dir: Path, staging_path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for source in sorted(source_dir.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"Checkpoint source cannot contain symlinks: {source}")
        if source.is_dir():
            continue
        if not source.is_file():
            raise ValueError(f"Checkpoint source contains a non-file entry: {source}")
        relative = source.relative_to(source_dir)
        if relative.as_posix() == MANIFEST_NAME:
            raise ValueError(f"Checkpoint source reserves the name {MANIFEST_NAME}")
        destination = staging_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": destination.stat().st_size,
                "sha256": _file_sha256(destination),
            }
        )
    return files


def _ensure_root_identity(root: Path, identity: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "identity.json"
    document: dict[str, Any] = {"format": ROOT_IDENTITY_FORMAT, "identity": identity}
    document["identity_sha256"] = _document_sha256(document)
    if path.exists():
        _validate_root_identity(root, identity)
        return
    _install_immutable_document(
        path,
        document,
        validate_existing=lambda _path: _validate_root_identity(root, identity),
    )


def _ensure_verification_marker(
    root: Path, generation: CheckpointGeneration
) -> None:
    verified_dir = root / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    path = verified_dir / f"{generation.generation_id}.json"
    document: dict[str, Any] = {
        "format": VERIFICATION_MARKER_FORMAT,
        "generation_id": generation.generation_id,
        "identity": generation.identity,
        "manifest_sha256": generation.manifest_sha256,
    }
    document["verification_sha256"] = _document_sha256(document)
    def validate_existing(existing_path: Path) -> None:
        existing = _validate_verification_marker(existing_path, generation.identity)
        if existing.get("manifest_sha256") != generation.manifest_sha256:
            raise CheckpointIntegrityError(
                f"Checkpoint verification manifest mismatch: {existing_path}"
            )

    if path.exists():
        validate_existing(path)
        return
    _install_immutable_document(path, document, validate_existing=validate_existing)


def _validate_verification_marker(
    path: Path, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    marker = _read_json(path, "checkpoint verification marker")
    if marker.get("format") != VERIFICATION_MARKER_FORMAT or marker.get(
        "verification_sha256"
    ) != _document_sha256(marker, omit="verification_sha256"):
        raise CheckpointIntegrityError(
            f"Checkpoint verification marker is invalid: {path}"
        )
    identity = _normalize_identity_document(marker.get("identity"), path)
    if identity != expected_identity:
        raise CheckpointIdentityMismatch(
            f"Checkpoint verification identity mismatch: {path}"
        )
    generation_id = marker.get("generation_id")
    if not isinstance(generation_id, str) or path.name != f"{generation_id}.json":
        raise CheckpointIntegrityError(
            f"Checkpoint verification generation mismatch: {path}"
        )
    _validate_generation_id(generation_id)
    return marker


def _prune_verified_generations(
    root: Path,
    identity: dict[str, Any],
    *,
    retain: set[str],
) -> None:
    verified_dir = root / "verified"
    if not verified_dir.exists():
        return
    for marker_path in sorted(verified_dir.glob("*.json")):
        try:
            marker = _validate_verification_marker(marker_path, identity)
        except CheckpointError:
            continue
        generation_id = str(marker["generation_id"])
        if generation_id in retain:
            continue
        generation_path = root / "generations" / generation_id
        try:
            generation = validate_checkpoint_generation(
                generation_path,
                expected_identity=identity,
                expected_generation_id=generation_id,
            )
        except CheckpointError:
            continue
        if generation.manifest_sha256 != marker.get("manifest_sha256"):
            continue
        shutil.rmtree(generation_path)
        marker_path.unlink()


def _validate_root_identity(root: Path, expected: dict[str, Any]) -> None:
    path = root / "identity.json"
    document = _read_json(path, "checkpoint root identity")
    if document.get("format") != ROOT_IDENTITY_FORMAT or document.get(
        "identity_sha256"
    ) != _document_sha256(document, omit="identity_sha256"):
        raise CheckpointIntegrityError(f"Checkpoint root identity is invalid: {path}")
    actual = _normalize_identity_document(document.get("identity"), path)
    if actual != expected:
        raise CheckpointIdentityMismatch(f"Checkpoint root identity mismatch: {path}")


def _validate_latest_pointer(
    path: Path, expected_identity: dict[str, Any] | None
) -> dict[str, Any]:
    pointer = _read_json(path, "latest checkpoint pointer")
    if pointer.get("format") != LATEST_POINTER_FORMAT or pointer.get(
        "pointer_sha256"
    ) != _document_sha256(pointer, omit="pointer_sha256"):
        raise CheckpointIntegrityError(f"Latest checkpoint pointer is invalid: {path}")
    identity = _normalize_identity_document(pointer.get("identity"), path)
    if expected_identity is not None and identity != expected_identity:
        raise CheckpointIdentityMismatch(f"Checkpoint identity mismatch: {path}")
    retained = pointer.get("retained_generation_ids")
    if (
        not isinstance(retained, list)
        or not retained
        or len(retained) > 2
        or retained[0] != pointer.get("generation_id")
        or len(set(retained)) != len(retained)
        or any(not isinstance(item, str) for item in retained)
        or pointer.get("previous_generation_id")
        != (retained[1] if len(retained) == 2 else None)
    ):
        raise CheckpointIntegrityError(
            f"Latest checkpoint retention record is invalid: {path}"
        )
    return pointer


def _normalize_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("Checkpoint identity must be a nonempty mapping")
    normalized = dict(identity)
    if any(not isinstance(key, str) or not key for key in normalized):
        raise ValueError("Checkpoint identity keys must be nonempty strings")
    try:
        encoded = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint identity must be JSON serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("Checkpoint identity must encode as an object")
    return decoded


def _normalize_identity_document(value: object, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"Checkpoint identity is invalid: {path}")
    try:
        return _normalize_identity(value)
    except ValueError as exc:
        raise CheckpointIntegrityError(f"Checkpoint identity is invalid: {path}") from exc


def _validate_generation_id(generation_id: str) -> None:
    if (
        not generation_id
        or generation_id in {".", ".."}
        or Path(generation_id).name != generation_id
        or "/" in generation_id
        or "\\" in generation_id
    ):
        raise ValueError(f"Invalid checkpoint generation ID: {generation_id!r}")


def _validate_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise CheckpointIntegrityError("Checkpoint manifest contains an invalid path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CheckpointIntegrityError(f"Checkpoint manifest path is unsafe: {value!r}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_sha256(document: Mapping[str, Any], omit: str | None = None) -> str:
    payload = {key: value for key, value in document.items() if key != omit}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"Cannot read {label}: {path}") from exc
    if not isinstance(document, dict):
        raise CheckpointIntegrityError(f"Invalid {label}: {path}")
    return document


def _write_new_file(path: Path, document: Mapping[str, Any]) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        _write_new_file(temporary, document)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _install_immutable_document(
    path: Path,
    document: Mapping[str, Any],
    *,
    validate_existing: Callable[[Path], None],
) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    _write_new_file(temporary, document)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            validate_existing(path)
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
        directories,
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


@contextmanager
def _exclusive_pointer_lock(root: Path):
    import fcntl

    lock_path = root / ".latest.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
