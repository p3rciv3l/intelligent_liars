#!/usr/bin/env python3
"""Trusted controller for credentialless Step 5 checkpoint publication.

AWS credentials and the signing key remain on this controller.  The remote GPU
worker receives only one generation-specific PUT URL in a mode-0600 file.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from intelligent_liars.step5_checkpoint_bridge import (
    build_controller_ack,
    validate_upload_request,
    verify_checkpoint_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--remote-exchange-dir", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--controller-private-key", type=Path, required=True)
    parser.add_argument("--controller-public-key", type=Path, required=True)
    parser.add_argument("--region")
    parser.add_argument("--url-expiry-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--idle-timeout-seconds", type=int, default=180)
    return parser.parse_args()


def _remote_root(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or ".." in pure.parts
        or re.fullmatch(r"/workspace/[A-Za-z0-9._/-]+", pure.as_posix()) is None
    ):
        raise ValueError("remote exchange dir must be traversal-free under /workspace")
    return pure.as_posix().rstrip("/")


class SshExchange:
    def __init__(self, args: argparse.Namespace):
        self.root = _remote_root(args.remote_exchange_dir)
        self.prefix = [
            "ssh", "-o", "StrictHostKeyChecking=yes", "-o",
            f"UserKnownHostsFile={args.known_hosts}", "-o",
            "GlobalKnownHostsFile=/dev/null", "-p", str(args.ssh_port),
            f"root@{args.ssh_host}",
        ]

    def _path(self, role: str, generation_id: str) -> str:
        if role not in {"requests", "secrets", "uploaded", "acks"}:
            raise ValueError("invalid exchange role")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", generation_id) is None:
            raise ValueError("invalid generation id")
        return f"{self.root}/{role}/{generation_id}.json"

    def list_requests(self) -> list[str]:
        command = (
            f"find {shlex.quote(self.root + '/requests')} -maxdepth 1 -type f "
            "-name '*.json' -print 2>/dev/null || true"
        )
        result = subprocess.run(
            [*self.prefix, command], check=True, capture_output=True, text=True
        )
        values = []
        for line in result.stdout.splitlines():
            name = PurePosixPath(line).name
            if name.endswith(".json"):
                values.append(name[:-5])
        return sorted(set(values))

    def read(self, role: str, generation_id: str) -> dict[str, Any] | None:
        path = self._path(role, generation_id)
        result = subprocess.run(
            [*self.prefix, f"cat -- {shlex.quote(path)}"],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            raise ValueError("remote exchange document must be an object")
        return value

    def write_secret(self, generation_id: str, value: dict[str, Any]) -> None:
        self._write("secrets", generation_id, value, secret=True)

    def write_ack(self, generation_id: str, value: dict[str, Any]) -> None:
        self._write("acks", generation_id, value, secret=False)

    def delete(self, role: str, generation_id: str) -> None:
        path = self._path(role, generation_id)
        subprocess.run(
            [*self.prefix, f"rm -f -- {shlex.quote(path)}"],
            check=True,
            capture_output=True,
            text=True,
        )

    def _write(
        self, role: str, generation_id: str, value: dict[str, Any], *, secret: bool
    ) -> None:
        path = self._path(role, generation_id)
        directory = str(PurePosixPath(path).parent)
        temporary = f"{path}.tmp-controller"
        mode = "600" if secret else "644"
        command = (
            f"umask 077; mkdir -p -- {shlex.quote(directory)}; "
            f"cat > {shlex.quote(temporary)}; chmod {mode} {shlex.quote(temporary)}; "
            f"mv -f -- {shlex.quote(temporary)} {shlex.quote(path)}"
        )
        subprocess.run(
            [*self.prefix, command],
            input=json.dumps(value, sort_keys=True, separators=(",", ":")),
            text=True,
            check=True,
            capture_output=True,
        )


def _object_key(prefix: str, request: dict[str, Any]) -> str:
    normalized = prefix.strip("/")
    if not normalized or ".." in PurePosixPath(normalized).parts:
        raise ValueError("prefix must be a safe nonempty S3 key prefix")
    return (
        f"{normalized}/generations/{request['generation_id']}/"
        f"{request['archive_sha256']}.tar"
    )


def _verify_s3(client: Any, bucket: str, key: str, request: dict[str, Any]) -> str:
    head = client.head_object(
        Bucket=bucket, Key=key, ChecksumMode="ENABLED"
    )
    version = head.get("VersionId")
    if not isinstance(version, str) or not version or version == "null":
        raise ValueError("checkpoint object lacks an immutable S3 version")
    if head.get("ContentLength") != request["size_bytes"]:
        raise ValueError("checkpoint HEAD size differs from request")
    expected_checksum = base64.b64encode(
        bytes.fromhex(request["archive_sha256"])
    ).decode()
    if head.get("ChecksumSHA256") != expected_checksum:
        raise ValueError("checkpoint HEAD checksum differs from request")
    response = client.get_object(Bucket=bucket, Key=key, VersionId=version)
    with tempfile.NamedTemporaryFile() as downloaded:
        digest = hashlib.sha256()
        size = 0
        body = response["Body"]
        while chunk := body.read(1024 * 1024):
            downloaded.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        downloaded.flush()
        if size != request["size_bytes"] or digest.hexdigest() != request["archive_sha256"]:
            raise ValueError("checkpoint round-trip bytes differ from request")
        verify_checkpoint_archive(
            Path(downloaded.name),
            expected_generation_id=request["generation_id"],
            expected_manifest_sha256=request["manifest_sha256"],
        )
    return version


def process_request(
    exchange: SshExchange, client: Any, args: argparse.Namespace, generation_id: str
) -> bool:
    raw_request = exchange.read("requests", generation_id)
    if raw_request is None:
        return False
    request = validate_upload_request(raw_request)
    if request["generation_id"] != generation_id:
        raise ValueError("request filename does not match generation identity")
    key = _object_key(args.prefix, request)
    checksum = base64.b64encode(bytes.fromhex(request["archive_sha256"])).decode()
    try:
        version = _verify_s3(client, args.bucket, key, request)
    except client.exceptions.NoSuchKey:
        version = ""
    except Exception as error:
        response = getattr(error, "response", {})
        if response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
            raise
        version = ""
    if not version:
        exchange.delete("uploaded", generation_id)
        exchange.delete("secrets", generation_id)
        url = client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": args.bucket,
                "Key": key,
                "ChecksumAlgorithm": "SHA256",
                "ChecksumSHA256": checksum,
                "IfNoneMatch": "*",
            },
            ExpiresIn=args.url_expiry_seconds,
            HttpMethod="PUT",
        )
        exchange.write_secret(
            generation_id,
            {
                "url": url,
                "headers": {
                    "x-amz-checksum-sha256": checksum,
                    "x-amz-sdk-checksum-algorithm": "SHA256",
                    "If-None-Match": "*",
                },
            },
        )
        deadline = time.monotonic() + args.url_expiry_seconds
        while True:
            marker = exchange.read("uploaded", generation_id)
            if marker is not None and marker == {
                "generation_id": generation_id,
                "request_sha256": request["request_sha256"],
                "archive_sha256": request["archive_sha256"],
            }:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("worker did not complete generation-specific PUT")
            time.sleep(args.poll_seconds)
        version = _verify_s3(client, args.bucket, key, request)
    ack = build_controller_ack(
        request,
        object_ref=f"s3://{args.bucket}/{key}",
        object_version=version,
        verified_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        private_key_path=args.controller_private_key,
        public_key_path=args.controller_public_key,
    )
    exchange.write_ack(generation_id, ack)
    return True


def main() -> int:
    args = parse_args()
    if args.controller_private_key.is_symlink() or not args.controller_private_key.is_file():
        raise ValueError("controller private key must be a regular non-symlink file")
    if (args.controller_private_key.stat().st_mode & 0o777) != 0o600:
        raise ValueError("controller private key must have mode 0600")
    if args.controller_public_key.is_symlink() or not args.controller_public_key.is_file():
        raise ValueError("controller public key must be a regular non-symlink file")
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("trusted controller requires boto3; run with 'uv run --with boto3'") from error
    session = boto3.session.Session(region_name=args.region)
    client = session.client("s3")
    versioning = client.get_bucket_versioning(Bucket=args.bucket)
    if versioning.get("Status") != "Enabled":
        raise RuntimeError("checkpoint bucket versioning must be enabled before upload")
    exchange = SshExchange(args)
    idle_deadline = time.monotonic() + args.idle_timeout_seconds
    handled: set[str] = set()
    while time.monotonic() < idle_deadline:
        progressed = False
        for generation_id in exchange.list_requests():
            if generation_id not in handled:
                progressed |= process_request(exchange, client, args, generation_id)
                handled.add(generation_id)
        if progressed:
            idle_deadline = time.monotonic() + args.idle_timeout_seconds
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
