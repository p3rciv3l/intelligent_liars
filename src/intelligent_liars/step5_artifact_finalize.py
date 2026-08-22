"""Fail-closed Step 5 artifact staging, packaging, and credentialless upload."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from intelligent_liars.step5_artifact_store import (
    ArtifactContractError,
    build_lifecycle_artifact_manifest,
    sha256_file,
    validate_lifecycle_artifact_manifest,
)


EXPECTED_INVENTORY_FORMAT = "tinylora_step5_expected_artifact_inventory_v1"
MANIFEST_NAME = "artifact_manifest.json"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _safe_relative(value: str) -> str:
    if not value or "\\" in value:
        raise ArtifactContractError("artifact path must be a nonempty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ArtifactContractError(f"Unsafe artifact path: {value!r}")
    if value == MANIFEST_NAME:
        raise ArtifactContractError(f"{MANIFEST_NAME} is a reserved control file")
    return value


def load_expected_inventory(path: Path) -> tuple[str, ...]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError("expected inventory must be a regular file")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"format", "files"}:
        raise ArtifactContractError("expected inventory has unsupported keys")
    if payload["format"] != EXPECTED_INVENTORY_FORMAT:
        raise ArtifactContractError("expected inventory has unsupported format")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ArtifactContractError("expected inventory files must be nonempty")
    normalized = tuple(_safe_relative(value) for value in files)
    if len(normalized) != len(set(normalized)):
        raise ArtifactContractError("expected inventory contains duplicate paths")
    return tuple(sorted(normalized))


def parse_mapping(value: str) -> tuple[str, Path]:
    target, separator, source = value.partition("=")
    if not separator or not source:
        raise ArtifactContractError("mapping must be TARGET=SOURCE")
    return _safe_relative(target), Path(source)


def _reject_nonfinite_json(path: Path) -> None:
    def reject(token: str) -> None:
        raise ArtifactContractError(f"nonfinite JSON value {token} in {path}")

    if path.suffix == ".json":
        json.loads(path.read_text(), parse_constant=reject)
    elif path.suffix == ".jsonl":
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                raise ArtifactContractError(f"blank JSONL line {line_number} in {path}")
            json.loads(line, parse_constant=reject)


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError(f"{label} must be a regular non-symlink file: {path}")
    _reject_nonfinite_json(path)
    return path


def _write_idempotent(source: Path, destination: Path) -> None:
    source = _regular_file(source, label="source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        existing = _regular_file(destination, label="existing staged artifact")
        if existing.stat().st_size != source.stat().st_size or sha256_file(existing) != sha256_file(source):
            raise ArtifactContractError(f"refusing to replace different staged artifact: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_destination(root: Path, relative: str) -> Path:
    """Create target parents without ever traversing an existing symlink."""
    root = root.resolve()
    current = root
    parts = PurePosixPath(_safe_relative(relative)).parts
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ArtifactContractError(f"artifact target parent is a symlink: {current}")
        if current.exists() and not current.is_dir():
            raise ArtifactContractError(f"artifact target parent is not a directory: {current}")
        current.mkdir(exist_ok=True)
        try:
            current.resolve().relative_to(root)
        except ValueError as error:
            raise ArtifactContractError("artifact target escapes its root") from error
    return root.joinpath(*parts)


def _tree_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactContractError(f"archive source must be a non-symlink directory: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactContractError(f"archive source contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactContractError(f"archive source contains non-file: {path}")
        _reject_nonfinite_json(path)
        files.append(path)
    if not files:
        raise ArtifactContractError("archive source must contain at least one file")
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_deterministic_tar_from_tree(root: Path, destination: Path) -> None:
    files = _tree_files(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w", format=tarfile.GNU_FORMAT) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        if destination.exists() or destination.is_symlink():
            existing = _regular_file(destination, label="existing tree archive")
            if existing.stat().st_size != temporary.stat().st_size or sha256_file(existing) != sha256_file(temporary):
                raise ArtifactContractError(f"refusing to replace different tree archive: {destination}")
        else:
            os.chmod(temporary, 0o644)
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_inventory(
    artifact_root: Path,
    expected: Iterable[str],
    *,
    file_mappings: Iterable[tuple[str, Path]],
    tree_archive_mappings: Iterable[tuple[str, Path]],
) -> list[dict[str, Any]]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ArtifactContractError("artifact root must be a non-symlink directory")
    targets: set[str] = set()
    for target, source in file_mappings:
        if target in targets:
            raise ArtifactContractError(f"duplicate staging target: {target}")
        targets.add(target)
        _write_idempotent(source, _safe_destination(artifact_root, target))
    for target, source in tree_archive_mappings:
        if target in targets:
            raise ArtifactContractError(f"duplicate staging target: {target}")
        targets.add(target)
        build_deterministic_tar_from_tree(
            source, _safe_destination(artifact_root, target)
        )
    expected_set = set(expected)
    if targets != expected_set:
        raise ArtifactContractError(
            f"staging mappings differ from frozen inventory; missing={sorted(expected_set - targets)}, extra={sorted(targets - expected_set)}"
        )
    observed: set[str] = set()
    records: list[dict[str, Any]] = []
    for path in artifact_root.rglob("*"):
        if path.is_symlink():
            raise ArtifactContractError(f"artifact tree contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactContractError(f"artifact tree contains non-file: {path}")
        relative = path.relative_to(artifact_root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        observed.add(relative)
        _reject_nonfinite_json(path)
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    if observed != expected_set:
        raise ArtifactContractError(
            f"artifact tree differs from frozen inventory; missing={sorted(expected_set - observed)}, extra={sorted(observed - expected_set)}"
        )
    return sorted(records, key=lambda item: item["path"])


def build_deterministic_artifact_archive(
    artifact_root: Path,
    records: Iterable[Mapping[str, Any]],
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if artifact_root.resolve() in destination.resolve().parents:
        raise ArtifactContractError("durable archive must be outside the artifact root")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    ordered = sorted(records, key=lambda item: str(item["path"]))
    try:
        with tarfile.open(temporary, "w", format=tarfile.GNU_FORMAT) as archive:
            for record in ordered:
                relative = _safe_relative(str(record["path"]))
                path = _regular_file(artifact_root / relative, label="artifact")
                if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                    raise ArtifactContractError(f"artifact changed during packaging: {relative}")
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.mode = 0o644
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        if destination.exists() or destination.is_symlink():
            existing = _regular_file(destination, label="existing durable archive")
            if existing.stat().st_size != temporary.stat().st_size or sha256_file(existing) != sha256_file(temporary):
                raise ArtifactContractError(f"refusing to replace different durable archive: {destination}")
        else:
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def read_presigned_url(*, url_file: Path | None, url_env: str | None) -> str:
    if (url_file is None) == (url_env is None):
        raise ArtifactContractError("provide exactly one presigned URL source")
    if url_file is not None:
        if url_file.is_symlink() or not url_file.is_file():
            raise ArtifactContractError("presigned URL file must be regular and non-symlink")
        if url_file.stat().st_mode & 0o077:
            raise ArtifactContractError("presigned URL file permissions must be 0600 or stricter")
        value = url_file.read_text().strip()
    else:
        assert url_env is not None
        value = os.environ.get(url_env, "").strip()
    if not value.startswith("https://"):
        raise ArtifactContractError("presigned PUT URL must use HTTPS")
    return value


def publish_presigned_put(
    archive: Path,
    *,
    url: str,
    attempts: int = 3,
    timeout_seconds: float = 900,
) -> dict[str, Any]:
    """Upload without credentials and without exposing the signed URL.

    ``If-None-Match: *`` makes the key no-clobber. A 412 is a safe idempotent
    outcome after an ambiguous prior request, but never counts as verification;
    only the controller's exact-version check does that.
    """
    archive = _regular_file(archive, label="durable archive")
    headers = {
        "Content-Length": str(archive.stat().st_size),
        "If-None-Match": "*",
    }
    last_status: int | None = None
    for attempt in range(1, attempts + 1):
        try:
            with archive.open("rb") as handle:
                response = requests.put(
                    url,
                    data=handle,
                    headers=headers,
                    timeout=timeout_seconds,
                )
        except requests.RequestException:
            if attempt == attempts:
                raise ArtifactContractError("presigned PUT failed after retries") from None
            time.sleep(min(2 ** (attempt - 1), 4))
            continue
        last_status = response.status_code
        if response.status_code in {200, 201, 204}:
            return {"uploaded": True, "preexisting": False, "attempts": attempt}
        if response.status_code == 412:
            return {"uploaded": False, "preexisting": True, "attempts": attempt}
        if response.status_code >= 500 and attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 4))
            continue
        break
    raise ArtifactContractError(f"presigned PUT failed with HTTP status {last_status}")


def write_lifecycle_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    normalized = validate_lifecycle_artifact_manifest(manifest)
    content = _canonical_json(normalized)
    if path.exists() or path.is_symlink():
        existing = _regular_file(path, label="existing lifecycle manifest")
        if existing.read_bytes() != content:
            raise ArtifactContractError("refusing to replace different lifecycle manifest")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_bytes(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_artifacts(
    *,
    artifact_root: Path,
    expected_inventory: Path,
    run_id: str,
    durable_uri: str,
    archive_path: Path,
    file_mappings: Iterable[tuple[str, Path]],
    tree_archive_mappings: Iterable[tuple[str, Path]],
    presigned_url_file: Path | None,
    presigned_url_env: str | None,
    canary_summary_target: str | None = None,
) -> dict[str, Any]:
    expected = load_expected_inventory(expected_inventory)
    staged_files = list(file_mappings)
    summary_path: Path | None = None
    if canary_summary_target is not None:
        target = _safe_relative(canary_summary_target)
        sources = {name: path for name, path in staged_files}
        required = ("result.json", "prerequisite_receipt.json")
        if any(name not in sources for name in required):
            raise ArtifactContractError(
                "generated canary summary requires result.json and prerequisite_receipt.json mappings"
            )
        descriptor, summary_name = tempfile.mkstemp(prefix="step5-canary-summary-", suffix=".json")
        os.close(descriptor)
        summary_path = Path(summary_name)
        summary = {
            "format": "tinylora_step5_canary_summary_v1",
            "run_id": run_id,
            "inputs": {
                name: {
                    "bytes": _regular_file(sources[name], label=name).stat().st_size,
                    "sha256": sha256_file(sources[name]),
                }
                for name in required
            },
            "worker_self_attestation_only": True,
            "controller_verification_required": True,
        }
        summary_path.write_bytes(_canonical_json(summary))
        staged_files.append((target, summary_path))
    try:
        records = stage_inventory(
            artifact_root,
            expected,
            file_mappings=staged_files,
            tree_archive_mappings=tree_archive_mappings,
        )
    finally:
        if summary_path is not None:
            summary_path.unlink(missing_ok=True)
    archive = build_deterministic_artifact_archive(artifact_root, records, archive_path)
    manifest = build_lifecycle_artifact_manifest(
        run_id=run_id,
        files=records,
        durable_uri=durable_uri,
        durable_bytes=archive["bytes"],
        durable_sha256=archive["sha256"],
    )
    url = read_presigned_url(url_file=presigned_url_file, url_env=presigned_url_env)
    publication = publish_presigned_put(archive_path, url=url)
    write_lifecycle_manifest(artifact_root / MANIFEST_NAME, manifest)
    return {
        "format": "tinylora_step5_worker_finalization_receipt_v1",
        "run_id": run_id,
        "artifact_set_id": manifest["artifact_set_id"],
        "manifest_sha256": sha256_file(artifact_root / MANIFEST_NAME),
        "durable_object": manifest["durable_object"],
        "publication": publication,
        "controller_verification_required": True,
    }
