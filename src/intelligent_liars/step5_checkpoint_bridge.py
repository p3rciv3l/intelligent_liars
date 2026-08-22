"""Credentialless, controller-attested publication of checkpoint generations.

The worker can upload exactly one deterministic archive using a short-lived PUT
URL.  It cannot mint URLs, inspect AWS credentials, or attest its own upload.
Only an acknowledgement signed by the trusted controller advances ``latest``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from intelligent_liars.durable_checkpoints import (
    MANIFEST_NAME,
    CheckpointGeneration,
)

REQUEST_FORMAT = "tinylora_step5_checkpoint_upload_request_v1"
ACK_FORMAT = "tinylora_step5_checkpoint_controller_ack_v1"
DURABILITY_RECEIPT_FORMAT = "tinylora_step5_checkpoint_durability_receipt_v2"
ARCHIVE_NAME = "generation.tar"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BridgeContractError(ValueError):
    """The checkpoint publication protocol failed closed."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise BridgeContractError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BridgeContractError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise BridgeContractError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BridgeContractError(f"{label} must be a lowercase SHA-256")
    return value


def _require_generation_id(value: Any) -> str:
    if not isinstance(value, str) or _GENERATION_ID.fullmatch(value) is None:
        raise BridgeContractError("generation_id is invalid")
    return value


def build_checkpoint_archive(
    generation: CheckpointGeneration, destination: Path
) -> dict[str, Any]:
    """Write a byte-deterministic, regular-file-only generation archive."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(path for path in generation.path.rglob("*") if path.is_file())
    if not paths or generation.manifest_path not in paths:
        raise BridgeContractError("generation archive lacks its manifest")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for path in paths:
                if path.is_symlink():
                    raise BridgeContractError("generation archive refuses symlinks")
                relative = path.relative_to(generation.path).as_posix()
                info = tarfile.TarInfo(relative)
                info.size = path.stat().st_size
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive_sha256": _hash_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def verify_checkpoint_archive(
    archive_path: Path,
    *,
    expected_generation_id: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Round-trip every archived byte and its embedded generation manifest."""
    expected_generation_id = _require_generation_id(expected_generation_id)
    expected_manifest_sha256 = _require_sha(
        expected_manifest_sha256, "manifest_sha256"
    )
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or MANIFEST_NAME not in names:
            raise BridgeContractError("archive inventory is invalid")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or ".." in pure.parts
                or member.name != pure.as_posix()
            ):
                raise BridgeContractError("archive contains an unsafe member")
        manifest_file = archive.extractfile(MANIFEST_NAME)
        if manifest_file is None:
            raise BridgeContractError("archive manifest is unreadable")
        try:
            manifest = json.load(manifest_file)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BridgeContractError("archive manifest is invalid JSON") from error
        if manifest.get("generation_id") != expected_generation_id:
            raise BridgeContractError("archive generation_id does not match request")
        if manifest.get("manifest_sha256") != expected_manifest_sha256:
            raise BridgeContractError("archive manifest_sha256 does not match request")
        manifest_identity = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        calculated_manifest_sha = hashlib.sha256(
            json.dumps(
                manifest_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if calculated_manifest_sha != expected_manifest_sha256:
            raise BridgeContractError("archive manifest has an invalid self-hash")
        declared = manifest.get("files")
        if not isinstance(declared, list):
            raise BridgeContractError("archive manifest file inventory is invalid")
        expected_names = {MANIFEST_NAME}
        for entry in declared:
            if not isinstance(entry, dict):
                raise BridgeContractError("archive manifest entry is invalid")
            name = str(entry.get("path", ""))
            expected_names.add(name)
            extracted = archive.extractfile(name)
            if extracted is None:
                raise BridgeContractError(f"archive member is missing: {name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            if size != entry.get("size_bytes") or digest.hexdigest() != entry.get(
                "sha256"
            ):
                raise BridgeContractError(f"archive member hash mismatch: {name}")
        if set(names) != expected_names:
            raise BridgeContractError("archive inventory differs from manifest")
    return {
        "archive_sha256": _hash_file(Path(archive_path)),
        "size_bytes": Path(archive_path).stat().st_size,
    }


def build_upload_request(
    generation: CheckpointGeneration,
    *,
    archive_identity: Mapping[str, Any],
    request_nonce: str,
    requested_at: str,
) -> dict[str, Any]:
    """Bind a dynamic generation to one immutable upload request."""
    if not isinstance(request_nonce, str) or re.fullmatch(
        r"[0-9a-f]{32,64}", request_nonce
    ) is None:
        raise BridgeContractError("request_nonce must be 16-32 random bytes in hex")
    _timestamp(requested_at, "requested_at")
    request = {
        "format": REQUEST_FORMAT,
        "generation_id": _require_generation_id(generation.generation_id),
        "manifest_sha256": _require_sha(
            generation.manifest_sha256, "manifest_sha256"
        ),
        "archive_sha256": _require_sha(
            archive_identity.get("archive_sha256"), "archive_sha256"
        ),
        "size_bytes": archive_identity.get("size_bytes"),
        "request_nonce": request_nonce,
        "requested_at": requested_at,
    }
    if (
        isinstance(request["size_bytes"], bool)
        or not isinstance(request["size_bytes"], int)
        or request["size_bytes"] <= 0
    ):
        raise BridgeContractError("size_bytes must be a positive integer")
    request["request_sha256"] = hashlib.sha256(_canonical_bytes(request)).hexdigest()
    return request


def validate_upload_request(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "format",
        "generation_id",
        "manifest_sha256",
        "archive_sha256",
        "size_bytes",
        "request_nonce",
        "requested_at",
        "request_sha256",
    }
    if set(value) != required or value.get("format") != REQUEST_FORMAT:
        raise BridgeContractError("upload request fields or format are invalid")
    normalized = dict(value)
    generation_id = _require_generation_id(normalized["generation_id"])
    synthetic = type("Generation", (), {})()
    synthetic.generation_id = generation_id
    synthetic.manifest_sha256 = _require_sha(
        normalized["manifest_sha256"], "manifest_sha256"
    )
    rebuilt = build_upload_request(
        synthetic,
        archive_identity={
            "archive_sha256": normalized["archive_sha256"],
            "size_bytes": normalized["size_bytes"],
        },
        request_nonce=normalized["request_nonce"],
        requested_at=normalized["requested_at"],
    )
    if rebuilt != normalized:
        raise BridgeContractError("upload request identity is invalid")
    return normalized


def controller_key_id(public_key_path: Path) -> str:
    completed = subprocess.run(
        ["openssl", "pkey", "-pubin", "-in", str(public_key_path), "-outform", "DER"],
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _sign(payload: bytes, private_key_path: Path) -> str:
    with tempfile.NamedTemporaryFile() as message_file:
        message_file.write(payload)
        message_file.flush()
        completed = subprocess.run(
            [
                "openssl", "pkeyutl", "-sign", "-rawin", "-inkey",
                str(private_key_path), "-in", message_file.name,
            ],
            check=True,
            capture_output=True,
        )
    return base64.b64encode(completed.stdout).decode()


def _verify_signature(payload: bytes, signature: str, public_key_path: Path) -> None:
    try:
        decoded = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError) as error:
        raise BridgeContractError("controller signature is invalid") from error
    with tempfile.NamedTemporaryFile() as signature_file, tempfile.NamedTemporaryFile() as message_file:
        signature_file.write(decoded)
        signature_file.flush()
        message_file.write(payload)
        message_file.flush()
        completed = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey",
                str(public_key_path),
                "-sigfile",
                signature_file.name,
                "-in",
                message_file.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if completed.returncode != 0:
        raise BridgeContractError("controller signature verification failed")


def build_controller_ack(
    request: Mapping[str, Any],
    *,
    object_ref: str,
    object_version: str,
    verified_at: str,
    private_key_path: Path,
    public_key_path: Path,
) -> dict[str, Any]:
    request = validate_upload_request(request)
    if not isinstance(object_ref, str) or not object_ref.startswith("s3://") or "?" in object_ref:
        raise BridgeContractError("object_ref must be an S3 origin without a query")
    if not isinstance(object_version, str) or not object_version:
        raise BridgeContractError("object_version must be nonempty")
    _timestamp(verified_at, "verified_at")
    ack = {
        "format": ACK_FORMAT,
        **{
            key: request[key]
            for key in (
                "generation_id",
                "manifest_sha256",
                "archive_sha256",
                "size_bytes",
                "request_nonce",
                "request_sha256",
            )
        },
        "object_ref": object_ref,
        "object_version": object_version,
        "verified_at": verified_at,
        "controller_key_id": controller_key_id(public_key_path),
        "verified": True,
    }
    ack["signature"] = _sign(_canonical_bytes(ack), private_key_path)
    return ack


def verify_controller_ack(
    value: Mapping[str, Any] | None,
    *,
    request: Mapping[str, Any],
    public_key_path: Path,
    now: datetime | None = None,
    max_ack_age: timedelta = timedelta(hours=1),
) -> dict[str, Any]:
    """Reject missing, stale, mismatched, unsigned, or self-attested acks."""
    if value is None:
        raise BridgeContractError("controller acknowledgement is missing")
    request = validate_upload_request(request)
    required = {
        "format", "generation_id", "manifest_sha256", "archive_sha256",
        "size_bytes", "request_nonce", "request_sha256", "object_ref",
        "object_version", "verified_at", "controller_key_id", "verified", "signature",
    }
    if set(value) != required or value.get("format") != ACK_FORMAT:
        raise BridgeContractError("controller acknowledgement fields or format are invalid")
    ack = dict(value)
    for key in (
        "generation_id", "manifest_sha256", "archive_sha256", "size_bytes",
        "request_nonce", "request_sha256",
    ):
        if ack.get(key) != request[key]:
            raise BridgeContractError(f"controller acknowledgement {key} mismatch")
    if ack.get("verified") is not True:
        raise BridgeContractError("controller acknowledgement rejected the upload")
    object_ref = ack.get("object_ref")
    if not isinstance(object_ref, str) or not object_ref.startswith("s3://") or "?" in object_ref:
        raise BridgeContractError("controller acknowledgement object_ref is invalid")
    if not isinstance(ack.get("object_version"), str) or not ack["object_version"]:
        raise BridgeContractError("controller acknowledgement version is invalid")
    if ack.get("controller_key_id") != controller_key_id(public_key_path):
        raise BridgeContractError("controller acknowledgement key identity mismatch")
    observed = _timestamp(ack.get("verified_at"), "verified_at")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    requested = _timestamp(request["requested_at"], "requested_at")
    if observed < requested or current - observed > max_ack_age or observed > current + timedelta(minutes=5):
        raise BridgeContractError("controller acknowledgement is stale or future-dated")
    signature = ack.pop("signature")
    _verify_signature(_canonical_bytes(ack), signature, public_key_path)
    return {
        "format": DURABILITY_RECEIPT_FORMAT,
        "generation_id": ack["generation_id"],
        "manifest_sha256": ack["manifest_sha256"],
        "archive_sha256": ack["archive_sha256"],
        "size_bytes": ack["size_bytes"],
        "object_ref": ack["object_ref"],
        "object_version": ack["object_version"],
        "controller_key_id": ack["controller_key_id"],
        "verified_at": ack["verified_at"],
        "verified": True,
    }
