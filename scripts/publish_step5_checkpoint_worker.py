#!/usr/bin/env python3
"""Worker half of the credentialless Step 5 checkpoint publication bridge."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import sys
import time
import urllib.parse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from intelligent_liars.durable_checkpoints import validate_checkpoint_generation
from intelligent_liars.step5_checkpoint_bridge import (
    ARCHIVE_NAME,
    BridgeContractError,
    build_checkpoint_archive,
    build_upload_request,
    verify_controller_ack,
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


def _upload(archive: Path, secret_path: Path) -> None:
    descriptor = os.open(secret_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        mode = os.fstat(descriptor).st_mode & 0o777
        if mode & 0o077:
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
    archive = root / "outbox" / generation_id / ARCHIVE_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_identity = build_checkpoint_archive(generation, archive)
    request_path = root / "requests" / f"{generation_id}.json"
    if request_path.exists():
        request = json.loads(request_path.read_text())
        if (
            request.get("generation_id") != generation_id
            or request.get("manifest_sha256") != manifest_sha256
            or request.get("archive_sha256") != archive_identity["archive_sha256"]
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
    if not ack_path.exists() and not uploaded_path.exists():
        secret_path = root / "secrets" / f"{generation_id}.json"
        _wait(secret_path, deadline, args.poll_seconds)
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
    receipt = verify_controller_ack(
        ack,
        request=request,
        public_key_path=args.controller_public_key,
        max_ack_age=timedelta(hours=1),
    )
    sys.stdout.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
