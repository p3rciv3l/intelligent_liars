"""Immutable promotion of recovered truth-editing prerequisite artifacts.

The public seam intentionally accepts paths, not pre-parsed objects.  Promotion
therefore cannot bypass source-byte identity, archive/tree equivalence, semantic
artifact validation, or no-clobber publication by supplying trusted in-memory data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import ctypes
import re
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .truth_editing_refusal_directions import (
    canonical_json_bytes,
    canonical_sha256,
    parse_refusal_direction_bank,
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
)
from .truth_editing_refusal_extraction import parse_run_receipt
from .truth_editing_vast_prerequisites import RECEIPT_FORMAT


PROMOTION_FORMAT = "truth_editing_recovered_artifact_promotion_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 16 * 1024**3
_MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024**3
_HISTORICAL_RECOVERY_BINDINGS = {
    "d93ad884abed9a932e1a8dee2b0f565034a7188dfd9905d9762058377907e1ef": (
        "1bdfa47b00db24574861d2c225bfcfbe0581b337c140140e7017baf18797944f",
        35_559_377,
    )
}
_LEGACY_LIFECYCLE_FIELDS = {
    "format",
    "offer",
    "image",
    "label",
    "instance_id",
    "events",
    "elapsed_seconds",
    "estimated_cost_usd",
    "maximum_network_cost_usd",
    "projected_all_in_max_cost_usd",
    "exit_code",
    "error",
    "destroyed",
    "destroy_error",
    "self_sha256",
}
_CURRENT_LIFECYCLE_FIELDS = _LEGACY_LIFECYCLE_FIELDS | {"artifact_archive"}


class ArtifactPromotionError(RuntimeError):
    """Recovered artifacts cannot be published without weakening provenance."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ArtifactPromotionError(f"cannot read required file: {path}") from error
    return digest.hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ArtifactPromotionError(f"{name} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ArtifactPromotionError(f"{name} contains non-finite JSON: {token}")
            ),
        )
    except ArtifactPromotionError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactPromotionError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ArtifactPromotionError(f"{name} must be a JSON object")
    return value


def _validate_lifecycle_receipt(raw: dict[str, Any]) -> None:
    if set(raw) not in (_LEGACY_LIFECYCLE_FIELDS, _CURRENT_LIFECYCLE_FIELDS):
        raise ArtifactPromotionError("lifecycle receipt fields changed")
    if raw.get("format") != RECEIPT_FORMAT:
        raise ArtifactPromotionError("lifecycle receipt format changed")
    unsigned = dict(raw)
    claimed = unsigned.pop("self_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(unsigned) != claimed:
        raise ArtifactPromotionError("lifecycle receipt self hash mismatch")
    if raw.get("exit_code") != 0:
        raise ArtifactPromotionError("lifecycle workload did not succeed")
    if raw.get("destroyed") is not True or raw.get("destroy_error") is not None:
        raise ArtifactPromotionError("lifecycle instance was not cleanly destroyed")
    error = raw.get("error")
    if error is not None and not (
        isinstance(error, str) and re.fullmatch(r"KeyboardInterrupt:\s*", error)
    ):
        raise ArtifactPromotionError("lifecycle contains an inadmissible execution error")
    instance_id = raw.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.isdigit():
        raise ArtifactPromotionError("lifecycle instance identity is invalid")
    events = raw.get("events")
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        raise ArtifactPromotionError("lifecycle events are invalid")
    successful = [
        index
        for index, event in enumerate(events)
        if event.get("event") == "workload_finished" and event.get("exit_code") == 0
    ]
    destroyed = [
        index for index, event in enumerate(events) if event.get("event") == "destroyed"
    ]
    if len(successful) != 1 or len(destroyed) != 1 or successful[0] >= destroyed[0]:
        raise ArtifactPromotionError("lifecycle event sequence is not promotable")
    for name in (
        "elapsed_seconds",
        "estimated_cost_usd",
        "maximum_network_cost_usd",
        "projected_all_in_max_cost_usd",
    ):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArtifactPromotionError(f"lifecycle {name} is invalid")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ArtifactPromotionError(f"lifecycle {name} is invalid")


def _safe_relative(path: str) -> str:
    normalized = path.removeprefix("./").rstrip("/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in normalized
    ):
        raise ArtifactPromotionError("artifact inventory contains an unsafe path")
    return pure.as_posix()


def _archive_inventory(archive_path: Path) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 100_000:
                raise ArtifactPromotionError("output archive has too many members")
            total_size = 0
            for member in members:
                relative = member.name.removeprefix("./").rstrip("/")
                if not relative:
                    continue
                relative = _safe_relative(relative)
                if relative in seen:
                    raise ArtifactPromotionError("output archive contains duplicate paths")
                seen.add(relative)
                if member.isdir():
                    continue
                if not member.isfile() or member.size < 0:
                    raise ArtifactPromotionError(
                        "output archive contains a non-regular member"
                    )
                if member.size > _MAX_ARCHIVE_MEMBER_BYTES:
                    raise ArtifactPromotionError("output archive member exceeds byte bound")
                total_size += member.size
                if total_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                    raise ArtifactPromotionError("output archive exceeds uncompressed byte bound")
                source = archive.extractfile(member)
                if source is None:
                    raise ArtifactPromotionError("output archive member is unreadable")
                digest = hashlib.sha256()
                size = 0
                with source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                if size != member.size:
                    raise ArtifactPromotionError("output archive member size changed")
                entries.append(
                    {"path": relative, "size_bytes": size, "sha256": digest.hexdigest()}
                )
    except ArtifactPromotionError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise ArtifactPromotionError("output archive is unreadable") from error
    return tuple(sorted(entries, key=lambda item: item["path"]))


def _tree_inventory(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.is_dir():
        raise ArtifactPromotionError("extracted outputs directory is missing")
    entries: list[dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            status = path.lstat()
            if stat.S_ISDIR(status.st_mode):
                continue
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
                raise ArtifactPromotionError("extracted tree contains a non-regular file")
            entries.append(
                {
                    "path": _safe_relative(relative),
                    "size_bytes": status.st_size,
                    "sha256": _file_sha256(path),
                }
            )
    except ArtifactPromotionError:
        raise
    except OSError as error:
        raise ArtifactPromotionError("extracted tree is unreadable") from error
    return tuple(entries)


def _inventory_at_logical_root(
    archived: tuple[dict[str, Any], ...],
    extracted: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Accept a tar rooted at ``.`` or at exactly one enclosing directory."""

    if archived == extracted:
        return archived
    first_parts = {PurePosixPath(item["path"]).parts[0] for item in archived}
    if len(first_parts) != 1:
        raise ArtifactPromotionError("output archive and extracted tree inventories differ")
    prefix = next(iter(first_parts)) + "/"
    stripped = tuple(
        {**item, "path": item["path"].removeprefix(prefix)} for item in archived
    )
    if any(not item["path"] for item in stripped) or stripped != extracted:
        raise ArtifactPromotionError("output archive and extracted tree inventories differ")
    return stripped


def _validate_refusal_artifact(
    refusal_root: Path, config_path: Path, prompt_path: Path
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        config = parse_refusal_direction_config(_read_json(config_path, "refusal config"))
        prompts = parse_refusal_prompt_manifest(
            _read_json(prompt_path, "refusal prompt manifest"), config
        )
        bank = parse_refusal_direction_bank(
            _read_json(refusal_root / "direction_bank.json", "refusal bank"),
            config,
            prompts,
        )
        run = parse_run_receipt(
            _read_json(refusal_root / "run_receipt.json", "refusal run receipt")
        )
        run.runtime_identity.verify_for(config)
    except ArtifactPromotionError:
        raise
    except (ValueError, RuntimeError) as error:
        raise ArtifactPromotionError("refusal artifact contract validation failed") from error
    if (
        run.config_sha256 != config.self_sha256
        or run.prompt_manifest_sha256 != prompts.self_sha256
        or run.direction_bank_sha256 != bank.self_sha256
    ):
        raise ArtifactPromotionError("refusal run receipt input identity mismatch")
    allowed_paths = {"direction_bank.json", "run_receipt.json"}
    for receipt in bank.per_layer_receipts:
        receipt_path = refusal_root / "receipts" / f"layer-{receipt.source_layer:02d}.json"
        allowed_paths.add(f"receipts/layer-{receipt.source_layer:02d}.json")
        receipt_raw = _read_json(receipt_path, "refusal layer receipt")
        if receipt_raw != {
            **receipt.__dict__,
        }:
            raise ArtifactPromotionError("standalone refusal layer receipt differs from bank")
        relative = _safe_relative(receipt.vector_path)
        allowed_paths.add(relative)
        vector_path = refusal_root.joinpath(*PurePosixPath(relative).parts)
        if _file_sha256(vector_path) != receipt.vector_file_sha256:
            raise ArtifactPromotionError("refusal vector file hash mismatch")
        try:
            vector = np.load(vector_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ArtifactPromotionError("refusal vector is unreadable") from error
        if vector.dtype != np.float64 or vector.shape != (receipt.width,):
            raise ArtifactPromotionError("refusal vector dtype or shape mismatch")
        if not np.isfinite(vector).all() or not np.isclose(
            np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-10
        ):
            raise ArtifactPromotionError("refusal vector is not a finite unit vector")
        if hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest() != receipt.vector_sha256:
            raise ArtifactPromotionError("refusal vector value hash mismatch")
    return (
        {
            "refusal_config_sha256": config.self_sha256,
            "refusal_prompt_manifest_sha256": prompts.self_sha256,
            "refusal_bank_sha256": bank.self_sha256,
            "refusal_run_receipt_sha256": run.self_sha256,
            "refusal_direction_count": len(bank.per_layer_receipts),
        },
        tuple(sorted(allowed_paths)),
    )


def _validate_archive_binding(
    lifecycle: dict[str, Any],
    archive_sha: str,
    archive_size: int,
    extracted_paths: frozenset[str],
) -> None:
    artifact_archive = lifecycle.get("artifact_archive")
    if artifact_archive is None:
        expected = _HISTORICAL_RECOVERY_BINDINGS.get(lifecycle["self_sha256"])
        if expected != (archive_sha, archive_size):
            raise ArtifactPromotionError(
                "legacy lifecycle is not bound to this recovered archive"
            )
        return
    fields = {
        "format",
        "archive_sha256",
        "size_bytes",
        "expected_outputs",
        "published_directory",
    }
    if not isinstance(artifact_archive, dict) or set(artifact_archive) != fields:
        raise ArtifactPromotionError("lifecycle artifact archive fields changed")
    if artifact_archive.get("format") != "truth_editing_vast_output_archive_v1":
        raise ArtifactPromotionError("lifecycle artifact archive format changed")
    expected_outputs = artifact_archive.get("expected_outputs")
    if (
        not isinstance(expected_outputs, list)
        or not expected_outputs
        or any(not isinstance(item, str) for item in expected_outputs)
        or len(set(expected_outputs)) != len(expected_outputs)
    ):
        raise ArtifactPromotionError("lifecycle expected outputs are invalid")
    for item in expected_outputs:
        normalized = _safe_relative(item)
        if normalized not in extracted_paths:
            raise ArtifactPromotionError("lifecycle expected output is missing")
    published = artifact_archive.get("published_directory")
    if not isinstance(published, str) or not published.strip():
        raise ArtifactPromotionError("lifecycle published directory is invalid")
    if (
        artifact_archive.get("archive_sha256") != archive_sha
        or artifact_archive.get("size_bytes") != archive_size
    ):
        raise ArtifactPromotionError("lifecycle artifact archive identity mismatch")


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if hasattr(library, "renameatx_np"):  # macOS
        function = library.renameatx_np
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-2, source_bytes, -2, destination_bytes, 0x00000004)
    elif hasattr(library, "renameat2"):  # Linux
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(-100, source_bytes, -100, destination_bytes, 1)
    else:
        raise ArtifactPromotionError(
            "platform lacks an atomic no-replace directory rename"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {17, 39}:  # EEXIST or ENOTEMPTY
            raise ArtifactPromotionError("destination already exists")
        raise OSError(error_number, os.strerror(error_number), str(destination))


def promote_recovered_refusal_bank(
    *,
    lifecycle_receipt_path: Path,
    output_archive_path: Path,
    expected_output_archive_sha256: str,
    extracted_outputs_dir: Path,
    refusal_config_path: Path,
    refusal_prompt_manifest_path: Path,
    destination: Path,
) -> dict[str, Any]:
    """Validate and atomically publish one recovered refusal-direction bank.

    ``destination`` is immutable and must not exist.  The expected archive hash
    comes from the remote pack output, so a manually recovered archive is not
    allowed to establish its own identity after download.
    """

    if not isinstance(expected_output_archive_sha256, str) or _SHA.fullmatch(
        expected_output_archive_sha256
    ) is None:
        raise ArtifactPromotionError("expected output archive SHA-256 is invalid")
    if destination.exists() or destination.is_symlink():
        raise ArtifactPromotionError("destination already exists")
    lifecycle = _read_json(lifecycle_receipt_path, "lifecycle receipt")
    _validate_lifecycle_receipt(lifecycle)
    archive_sha = _file_sha256(output_archive_path)
    if archive_sha != expected_output_archive_sha256:
        raise ArtifactPromotionError("output archive hash differs from remote identity")
    try:
        archive_size = output_archive_path.stat().st_size
    except OSError as error:
        raise ArtifactPromotionError("output archive is unreadable") from error
    archived = _archive_inventory(output_archive_path)
    extracted = _tree_inventory(extracted_outputs_dir)
    _inventory_at_logical_root(archived, extracted)
    _validate_archive_binding(
        lifecycle,
        archive_sha,
        archive_size,
        frozenset(item["path"] for item in extracted),
    )
    refusal_root = extracted_outputs_dir / "refusal"
    semantic, allowed_paths = _validate_refusal_artifact(
        refusal_root, refusal_config_path, refusal_prompt_manifest_path
    )
    observed_final_paths = {
        item["path"].removeprefix("refusal/")
        for item in extracted
        if item["path"].startswith("refusal/")
        and not item["path"].startswith("refusal/batches/")
    }
    if observed_final_paths != set(allowed_paths):
        raise ArtifactPromotionError("refusal final artifact paths differ from allowlist")
    refusal_inventory = tuple(
        {**item, "path": item["path"].removeprefix("refusal/")}
        for item in extracted
        if item["path"].removeprefix("refusal/") in set(allowed_paths)
        and item["path"].startswith("refusal/")
    )
    if not refusal_inventory:
        raise ArtifactPromotionError("recovered outputs contain no refusal artifact")
    unsigned: dict[str, Any] = {
        "format": PROMOTION_FORMAT,
        "artifact_kind": "refusal_direction_bank",
        "source_relative_path": "refusal",
        "lifecycle_receipt_file_sha256": _file_sha256(lifecycle_receipt_path),
        "lifecycle_receipt_sha256": lifecycle["self_sha256"],
        "output_archive_sha256": archive_sha,
        "output_archive_size_bytes": archive_size,
        "extracted_tree_inventory": list(extracted),
        "extracted_tree_inventory_sha256": canonical_sha256(list(extracted)),
        "promoted_artifact_inventory": list(refusal_inventory),
        "promoted_artifact_inventory_sha256": canonical_sha256(
            list(refusal_inventory)
        ),
        "validation_scope": "final_bank_run_layer_receipts_and_vectors_v1",
        "excluded_source_evidence_prefixes": ["refusal/batches/"],
        **semantic,
    }
    receipt = {**unsigned, "self_sha256": canonical_sha256(unsigned)}

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactPromotionError("destination parent is unavailable") from error
    claim = destination.with_name(f".{destination.name}.promotion-claim")
    staging: Path | None = None
    claim_fd: int | None = None
    try:
        try:
            claim_fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ArtifactPromotionError("another promotion owns this destination") from error
        if destination.exists() or destination.is_symlink():
            raise ArtifactPromotionError("destination already exists")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
        )
        artifact_staging = staging / "artifact"
        artifact_staging.mkdir()
        for item in refusal_inventory:
            relative = PurePosixPath(item["path"])
            source = refusal_root.joinpath(*relative.parts)
            target = artifact_staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target, follow_symlinks=False)
        copied = _tree_inventory(staging / "artifact")
        if copied != refusal_inventory:
            raise ArtifactPromotionError("source artifact changed during staging")
        _write_private_json(staging / "artifact" / "promotion_receipt.json", receipt)
        if destination.exists() or destination.is_symlink():
            raise ArtifactPromotionError("destination already exists")
        _rename_noreplace(staging / "artifact", destination)
        staging.rmdir()
        staging = None
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except ArtifactPromotionError:
        raise
    except OSError as error:
        raise ArtifactPromotionError("atomic artifact promotion failed") from error
    finally:
        if claim_fd is not None:
            os.close(claim_fd)
        try:
            claim.unlink(missing_ok=True)
        except OSError:
            pass
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return receipt


__all__ = [
    "ArtifactPromotionError",
    "PROMOTION_FORMAT",
    "promote_recovered_refusal_bank",
]
