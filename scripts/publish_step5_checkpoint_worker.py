#!/usr/bin/env python3
"""Worker half of the credentialless Step 5 checkpoint publication bridge."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import shutil
import sys
import time
import urllib.parse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from intelligent_liars.durable_checkpoints import validate_checkpoint_generation
from intelligent_liars.step5_checkpoint_bridge import (
    ARCHIVE_NAME,
    ENVELOPE_FORMAT,
    BridgeContractError,
    build_checkpoint_archive,
    build_upload_request,
    verify_controller_ack,
    validate_upload_request,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-dir", type=Path, required=True)
    parser.add_argument("--controller-public-key", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _wait(path: Path, deadline: float, poll_seconds: float) -> None:
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"checkpoint controller did not answer {path.name}")
        time.sleep(poll_seconds)


def _wait_for_ack_or_secret(
    ack_path: Path, secret_path: Path, deadline: float, poll_seconds: float
) -> str:
    while True:
        if ack_path.is_file():
            return "ack"
        if secret_path.is_file():
            return "secret"
        if time.monotonic() >= deadline:
            raise TimeoutError("checkpoint controller did not provide an ack or PUT URL")
        time.sleep(poll_seconds)


def _upload(archive: Path, secret_path: Path) -> None:
    descriptor = os.open(secret_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        mode = os.fstat(descriptor).st_mode & 0o777
        if mode != 0o600:
            raise BridgeContractError("presigned URL file must have mode 0600")
        with os.fdopen(descriptor) as handle:
            descriptor = -1
            envelope = json.load(handle)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if set(envelope) != {"url", "headers"}:
        raise BridgeContractError("presigned PUT envelope is invalid")
    parsed = urllib.parse.urlsplit(envelope["url"])
    if parsed.scheme != "https" or not parsed.netloc or not parsed.query:
        raise BridgeContractError("presigned PUT URL must be scoped HTTPS URL")
    headers = envelope["headers"]
    if not isinstance(headers, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
    ):
        raise BridgeContractError("presigned PUT headers are invalid")
    headers = {**headers, "Content-Length": str(archive.stat().st_size)}
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=300)
    try:
        with archive.open("rb") as handle:
            connection.request(
                "PUT",
                urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, "")),
                body=handle,
                headers=headers,
            )
            response = connection.getresponse()
            response.read()
            if response.status not in (200, 201):
                raise BridgeContractError(
                    f"generation-specific PUT failed with HTTP {response.status}"
                )
    finally:
        connection.close()
        secret_path.unlink(missing_ok=True)


def _cleanup_superseded_exchange(root: Path, generation_id: str) -> None:
    """Keep one retryable large outbox and remove obsolete URL/marker state."""
    for role in ("outbox", "secrets", "uploaded"):
        directory = root / role
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in directory.iterdir():
            name = path.name.removesuffix(".json")
            if name == generation_id:
                continue
            if path.is_symlink():
                raise BridgeContractError("exchange cleanup refuses symlinks")
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()


def main() -> int:
    args = parse_args()
    generation_dir = Path(os.environ["STEP5_CHECKPOINT_GENERATION_DIR"])
    generation_id = os.environ["STEP5_CHECKPOINT_GENERATION_ID"]
    manifest_sha256 = os.environ["STEP5_CHECKPOINT_MANIFEST_SHA256"]
    generation = validate_checkpoint_generation(
        generation_dir,
        expected_generation_id=generation_id,
    )
    if generation.manifest_sha256 != manifest_sha256:
        raise BridgeContractError("runner manifest identity does not match generation")

    root = args.exchange_dir
    _cleanup_superseded_exchange(root, generation_id)
    archive = root / "outbox" / generation_id / ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_identity = build_checkpoint_archive(generation, archive)
    request_path = root / "requests" / f"{generation_id}.json"
    if request_path.exists():
        request = validate_upload_request(json.loads(request_path.read_text()))
        if (
            request.get("generation_id") != generation_id
            or request.get("manifest_sha256") != manifest_sha256
            or request.get("archive_sha256") != archive_identity["archive_sha256"]
            or request.get("size_bytes") != archive_identity["size_bytes"]
        ):
            raise BridgeContractError("existing request conflicts with this generation")
    else:
        request = build_upload_request(
            generation,
            archive_identity=archive_identity,
            request_nonce=secrets.token_hex(16),
            requested_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        _atomic_json(request_path, request)

    deadline = time.monotonic() + args.timeout_seconds
    ack_path = root / "acks" / f"{generation_id}.json"
    uploaded_path = root / "uploaded" / f"{generation_id}.json"
    secret_path = root / "secrets" / f"{generation_id}.json"
    while True:
        if _wait_for_ack_or_secret(
            ack_path, secret_path, deadline, args.poll_seconds
        ) == "secret":
            _upload(archive, secret_path)
            _atomic_json(
                uploaded_path,
                {
                    "generation_id": generation_id,
                    "request_sha256": request["request_sha256"],
                    "archive_sha256": request["archive_sha256"],
                },
            )
            _wait(ack_path, deadline, args.poll_seconds)
        ack = json.loads(ack_path.read_text())
        try:
            verify_controller_ack(
                ack,
                request=request,
                public_key_path=args.controller_public_key,
                max_ack_age=timedelta(hours=1),
            )
        except (BridgeContractError, json.JSONDecodeError):
            # Invalid acknowledgements never advance latest. Removing them lets the
            # trusted controller replace them; absent a controller, timeout is closed.
            ack_path.unlink(missing_ok=True)
            continue
        break
    archive.unlink(missing_ok=True)
    uploaded_path.unlink(missing_ok=True)
    envelope = {"format": ENVELOPE_FORMAT, "request": request, "ack": ack}
    sys.stdout.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
