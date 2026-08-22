"""Validate an inert, hash-bound Step 5 canary launch packet.

This module never invokes a subprocess, cloud API, or workload.  It only checks
that a packet is internally consistent and reports whether every launch-time
substitution has been frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from intelligent_liars.step5_artifact_presigner import (
    validate_receipt as validate_artifact_put_receipt,
)
from intelligent_liars.step5_input_hydration import validate_url_manifest


PACKET_FORMAT = "tinylora_step5_canary_launch_packet_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE = re.compile(r"^[a-z0-9][a-z0-9./_-]*@sha256:[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
_S3_URI = re.compile(r"^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[^\s]+$")


class LaunchPacketError(ValueError):
    """The canary launch packet violates its fail-closed contract."""


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a packet while excluding its self-referential hash field."""
    payload = {key: item for key, item in value.items() if key != "packet_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _regular_file_sha256(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise LaunchPacketError(f"{label} must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise LaunchPacketError(
            f"{label} must be a regular non-symlink file"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LaunchPacketError(f"{label} must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _bound_s3_url(
    url: str,
    *,
    bucket: str,
    key: str,
    region: str,
    approval_time: datetime,
) -> None:
    parsed = urlsplit(url)
    expected_hosts = {
        f"{bucket}.s3.{region}.amazonaws.com",
        f"{bucket}.s3-{region}.amazonaws.com",
    }
    if region == "us-east-1":
        expected_hosts.add(f"{bucket}.s3.amazonaws.com")
    if (
        parsed.scheme != "https"
        or parsed.hostname not in expected_hosts
        or unquote(parsed.path).lstrip("/") != key
    ):
        raise LaunchPacketError("protected URL does not identify its frozen S3 object")
    query = parse_qs(parsed.query)
    try:
        if query["X-Amz-Algorithm"] != ["AWS4-HMAC-SHA256"]:
            raise ValueError("unsupported SigV4 algorithm")
        credential = query["X-Amz-Credential"][0].split("/")
        signed_headers = query["X-Amz-SignedHeaders"][0].split(";")
        signature = query["X-Amz-Signature"][0]
        signed = datetime.strptime(query["X-Amz-Date"][0], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        expires = int(query["X-Amz-Expires"][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise LaunchPacketError(
            "protected URL lacks a complete SigV4 binding"
        ) from error
    if (
        len(credential) != 5
        or re.fullmatch(r"[A-Z0-9]{16,128}", credential[0]) is None
        or credential[1] != signed.strftime("%Y%m%d")
        or credential[2] != region
        or credential[3:] != ["s3", "aws4_request"]
        or signed_headers != sorted(set(signed_headers))
        or "host" not in signed_headers
        or any(re.fullmatch(r"[a-z0-9-]+", header) is None for header in signed_headers)
        or re.fullmatch(r"[0-9a-fA-F]{64}", signature) is None
    ):
        raise LaunchPacketError("protected URL has an invalid SigV4 credential scope")
    if not 60 <= expires <= 604800 or not signed <= approval_time < signed + timedelta(
        seconds=expires
    ):
        raise LaunchPacketError("protected URL is not fresh at approval time")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchPacketError(f"{label} must be an object")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise LaunchPacketError(f"{label} must be a lowercase SHA-256")
    return value


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and _PLACEHOLDER.fullmatch(value) is not None


def _collect_placeholders(value: Any) -> set[str]:
    if isinstance(value, str):
        return {
            match.group(1) for match in re.finditer(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
        }
    if isinstance(value, Mapping):
        found: set[str] = set()
        for item in value.values():
            found.update(_collect_placeholders(item))
        return found
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        found = set()
        for item in value:
            found.update(_collect_placeholders(item))
        return found
    return set()


def _positive_finite(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise LaunchPacketError(f"{label} must be finite and positive")
    return float(value)


def _validate_s3(value: Any, *, label: str, allow_placeholder: bool = False) -> None:
    if allow_placeholder and _is_placeholder(value):
        return
    if not isinstance(value, str) or _S3_URI.fullmatch(value) is None:
        raise LaunchPacketError(f"{label} must be an exact s3:// URI")


def _s3_parts(value: Any, *, label: str) -> tuple[str, str]:
    _validate_s3(value, label=label)
    bucket, key = str(value)[5:].split("/", 1)
    return bucket, key


def _validate_input_url_receipt(
    packet: Mapping[str, Any],
    *,
    approval_time: datetime,
    controller: Mapping[str, Any],
) -> None:
    durability = _mapping(packet["durability"], label="durability")
    remote = _mapping(packet["remote_inputs"], label="remote_inputs")
    receipt_path = Path(str(controller["input_url_controller_receipt_path"]))
    receipt_bytes = _read_regular_bytes(
        receipt_path, label="input URL controller receipt"
    )
    expected_hash = _sha256(
        controller.get("input_url_controller_receipt_sha256"),
        label="input_url_controller_receipt_sha256",
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != expected_hash:
        raise LaunchPacketError("input URL controller receipt hash differs")
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchPacketError("input URL controller receipt is not JSON") from error
    receipt_fields = {
        "account_id",
        "content_sha256",
        "created_at",
        "expires_at",
        "expiry_seconds",
        "format",
        "host_gate",
        "manifest",
        "objects",
        "region",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != receipt_fields:
        raise LaunchPacketError("input URL controller receipt fields differ")
    content = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        hashlib.sha256(_canonical_bytes(content)).hexdigest()
        != receipt["content_sha256"]
    ):
        raise LaunchPacketError("input URL controller receipt commitment differs")
    if (
        receipt["format"] != "tinylora_step5_input_url_controller_receipt_v1"
        or receipt["account_id"] != durability["account_id"]
        or receipt["region"] != durability["region"]
    ):
        raise LaunchPacketError("input URL controller identity differs")
    try:
        created = datetime.fromisoformat(
            str(receipt["created_at"]).replace("Z", "+00:00")
        )
        expires = datetime.fromisoformat(
            str(receipt["expires_at"]).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise LaunchPacketError(
            "input URL controller timestamps are invalid"
        ) from error
    expiry_seconds = receipt["expiry_seconds"]
    if (
        created.tzinfo is None
        or expires.tzinfo is None
        or not isinstance(expiry_seconds, int)
        or not 60 <= expiry_seconds <= 604800
        or created + timedelta(seconds=expiry_seconds) != expires
        or not created <= approval_time <= created + timedelta(minutes=30)
        or approval_time >= expires
    ):
        raise LaunchPacketError("input URL controller receipt is not fresh at approval")

    manifest_descriptor = _mapping(
        receipt["manifest"], label="input URL manifest receipt"
    )
    manifest_bucket, manifest_key = _s3_parts(
        remote["input_url_manifest_s3_uri"], label="input_url_manifest_s3_uri"
    )
    if manifest_descriptor != {
        "bucket": manifest_bucket,
        "key": manifest_key,
        "sha256": manifest_descriptor.get("sha256"),
    }:
        raise LaunchPacketError("input URL manifest receipt binding differs")
    manifest_hash = _sha256(
        manifest_descriptor.get("sha256"), label="input URL manifest sha256"
    )
    manifest_path = Path(str(controller["input_url_manifest_output_path"]))
    manifest_bytes = _read_regular_bytes(manifest_path, label="input URL manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_hash:
        raise LaunchPacketError("input URL manifest hash differs from receipt")
    try:
        manifest = validate_url_manifest(json.loads(manifest_bytes))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise LaunchPacketError("input URL manifest is invalid") from error
    manifest_controller = _mapping(
        manifest["controller"], label="URL manifest controller"
    )
    for field, expected in (
        ("account_id", durability["account_id"]),
        ("bucket", manifest_bucket),
        ("region", durability["region"]),
        ("manifest_key", manifest_key),
        ("created_at", receipt["created_at"]),
        ("expires_at", receipt["expires_at"]),
        ("expiry_seconds", expiry_seconds),
    ):
        if manifest_controller.get(field) != expected:
            raise LaunchPacketError(f"input URL manifest controller {field} differs")

    objects_value = receipt["objects"]
    if not isinstance(objects_value, list) or not objects_value:
        raise LaunchPacketError("input URL controller objects are missing")
    objects: dict[str, Mapping[str, Any]] = {}
    for item_value in objects_value:
        item = _mapping(item_value, label="input URL controller object")
        key = item.get("key")
        if not isinstance(key, str) or key in objects:
            raise LaunchPacketError("input URL controller object keys differ")
        _sha256(item.get("sha256"), label="input URL controller object sha256")
        if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
            raise LaunchPacketError("input URL controller object size differs")
        objects[key] = item

    expected_hashes = {
        str(remote[name])
        for name in (
            "model_manifest_sha256",
            "model_completion_sha256",
            "frozen_inputs_tar_sha256",
            "frozen_inputs_completion_sha256",
            "pixmo_tar_sha256",
            "pixmo_manifest_sha256",
            "pixmo_completion_sha256",
        )
    }
    observed_hashes = {str(item["sha256"]) for item in objects.values()}
    if not expected_hashes <= observed_hashes:
        raise LaunchPacketError("input URL controller objects omit frozen hashes")

    model_bucket, model_prefix = _s3_parts(
        remote["model_s3_prefix"], label="model_s3_prefix"
    )
    frozen_bucket, frozen_archive_key = _s3_parts(
        remote["plan_s3_uri"], label="plan_s3_uri"
    )
    pixmo_bucket, pixmo_prefix = _s3_parts(
        remote["pixmo_s3_prefix"], label="pixmo_s3_prefix"
    )
    if {model_bucket, frozen_bucket, pixmo_bucket, manifest_bucket} != {
        manifest_bucket
    }:
        raise LaunchPacketError("input URL controller buckets differ")
    pixmo_archive_key = unquote(
        urlsplit(str(manifest["pixmo"]["archive_url"])).path
    ).lstrip("/")
    expected_key_hashes = {
        f"{model_prefix.rstrip('/')}/manifest.json": remote["model_manifest_sha256"],
        f"{model_prefix.rstrip('/')}/_COMPLETE.json": remote["model_completion_sha256"],
        frozen_archive_key: remote["frozen_inputs_tar_sha256"],
        f"{str(Path(frozen_archive_key).parent)}/_COMPLETE.json": remote[
            "frozen_inputs_completion_sha256"
        ],
        f"{pixmo_prefix.rstrip('/')}/manifest.json": remote["pixmo_manifest_sha256"],
        f"{pixmo_prefix.rstrip('/')}/_COMPLETE.json": remote["pixmo_completion_sha256"],
        pixmo_archive_key: remote["pixmo_tar_sha256"],
    }
    for key, expected in expected_key_hashes.items():
        if key not in objects or objects[key].get("sha256") != expected:
            raise LaunchPacketError(f"input URL controller hash differs for {key}")

    def manifest_urls(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [url for item in value.values() for url in manifest_urls(item)]
        return []

    for url in manifest_urls(
        {key: manifest[key] for key in ("model", "frozen_inputs", "pixmo")}
    ):
        parsed = urlsplit(url)
        key = unquote(parsed.path).lstrip("/")
        if key not in objects:
            raise LaunchPacketError(
                "input URL manifest references an unattested object"
            )
        _bound_s3_url(
            url,
            bucket=manifest_bucket,
            key=key,
            region=str(durability["region"]),
            approval_time=approval_time,
        )

    host_gate = _mapping(receipt["host_gate"], label="host gate receipt")
    host_key = host_gate.get("key")
    if not isinstance(host_key, str) or host_key not in objects:
        raise LaunchPacketError("host gate object is not attested")
    if host_gate.get("sha256") != objects[host_key].get("sha256") or host_gate.get(
        "bytes"
    ) != objects[host_key].get("bytes"):
        raise LaunchPacketError("host gate object binding differs")
    bootstrap_url = (
        _read_regular_bytes(
            Path(str(controller["input_url_manifest_url_file"])),
            label="input URL manifest bootstrap",
        )
        .decode()
        .strip()
    )
    host_gate_url = (
        _read_regular_bytes(
            Path(str(controller["host_gate_url_file"])), label="host gate URL"
        )
        .decode()
        .strip()
    )
    _bound_s3_url(
        bootstrap_url,
        bucket=manifest_bucket,
        key=manifest_key,
        region=str(durability["region"]),
        approval_time=approval_time,
    )
    _bound_s3_url(
        host_gate_url,
        bucket=manifest_bucket,
        key=host_key,
        region=str(durability["region"]),
        approval_time=approval_time,
    )


def _validate_local_contracts(value: Any, packet_dir: Path) -> None:
    contracts = _mapping(value, label="local_contracts")
    expected_formats = {
        "source_manifest": "tinylora_step5_source_manifest_v1",
        "expected_artifact_inventory": (
            "tinylora_step5_expected_artifact_inventory_v1"
        ),
    }
    if set(contracts) != set(expected_formats):
        raise LaunchPacketError("local_contracts inventory differs")
    for name, expected_format in expected_formats.items():
        descriptor = _mapping(contracts[name], label=name)
        if set(descriptor) != {"path", "sha256"}:
            raise LaunchPacketError(f"{name} descriptor fields differ")
        raw_path = descriptor["path"]
        if not isinstance(raw_path, str) or not raw_path:
            raise LaunchPacketError(f"{name} path is invalid")
        path = Path(raw_path)
        if not path.is_absolute():
            path = packet_dir / path
        actual = _regular_file_sha256(path.resolve(), label=name)
        expected = _sha256(descriptor["sha256"], label=f"{name} sha256")
        if actual != expected:
            raise LaunchPacketError(f"{name} hash does not match local bytes")
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise LaunchPacketError(f"{name} is not readable JSON") from error
        if not isinstance(payload, Mapping) or payload.get("format") != expected_format:
            raise LaunchPacketError(f"{name} has an unsupported format")


def _validate_complete_runtime(packet: Mapping[str, Any], packet_dir: Path) -> None:
    runtime = _mapping(packet["runtime"], label="runtime")
    image = runtime.get("image")
    if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
        raise LaunchPacketError("launch requires an immutable runtime image digest")
    image_digest = runtime.get("image_digest")
    if (
        not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
        or not image.endswith("@" + image_digest)
    ):
        raise LaunchPacketError(
            "runtime image and image_digest must identify the same bytes"
        )
    offer_id = runtime.get("offer_id")
    if not isinstance(offer_id, str) or not offer_id.isdigit():
        raise LaunchPacketError("launch requires one exact numeric offer_id")
    if not isinstance(runtime.get("gpu_name"), str) or not runtime["gpu_name"].strip():
        raise LaunchPacketError("launch requires the exact approved GPU name")
    if runtime.get("gpu_count") != 1:
        raise LaunchPacketError("the canary requires exactly one GPU")
    _positive_finite(runtime.get("gpu_vram_gib"), label="gpu_vram_gib")
    _positive_finite(runtime.get("disk_gib"), label="disk_gib")

    approval = _mapping(packet["approval"], label="approval")
    _positive_finite(
        approval.get("approved_hourly_price_usd"),
        label="approved_hourly_price_usd",
    )
    _positive_finite(
        approval.get("approved_max_cost_usd"), label="approved_max_cost_usd"
    )
    timestamp = approval.get("approved_at")
    if not isinstance(timestamp, str):
        raise LaunchPacketError("approved_at must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise LaunchPacketError("approved_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise LaunchPacketError("approved_at must include a timezone")

    commands = _mapping(packet["commands"], label="commands")
    argv = commands["lifecycle_argv"]
    assert isinstance(argv, list)

    def flag_value(flag: str) -> str:
        positions = [index for index, item in enumerate(argv) if item == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(argv):
            raise LaunchPacketError(f"lifecycle_argv requires exactly one {flag}")
        return str(argv[positions[0] + 1])

    expected_flags = {
        "--offer-id": str(runtime["offer_id"]),
        "--image": str(runtime["image"]),
        "--disk": str(runtime["disk_gib"]),
        "--approved-hourly-price": str(approval["approved_hourly_price_usd"]),
        "--approved-max-cost": str(approval["approved_max_cost_usd"]),
        "--controller-public-key-sha256": str(
            packet["controller_prerequisites"]["controller_public_key_sha256"]
        ),
        "--input-url-manifest-url-file": str(
            packet["controller_prerequisites"]["input_url_manifest_url_file"]
        ),
        "--artifact-put-url-file": str(
            packet["controller_prerequisites"]["artifact_put_url_file"]
        ),
        "--artifact-put-receipt": str(
            packet["controller_prerequisites"]["artifact_put_receipt_path"]
        ),
        "--artifact-put-receipt-sha256": str(
            packet["controller_prerequisites"]["artifact_put_receipt_sha256"]
        ),
        "--approved-at": str(approval["approved_at"]),
        "--host-gate-url-file": str(
            packet["controller_prerequisites"]["host_gate_url_file"]
        ),
        "--checkpoint-controller-script-sha256": str(
            packet["controller_contracts"]["checkpoint_controller"]["sha256"]
        ),
        "--checkpoint-controller-script": str(
            packet["controller_prerequisites"]["checkpoint_controller_script_path"]
        ),
        "--checkpoint-bucket": str(packet["durability"]["bucket"]),
        "--checkpoint-prefix": str(packet["durability"]["checkpoint_prefix"]).split(
            "/", 3
        )[-1],
        "--checkpoint-controller-private-key": str(
            packet["controller_prerequisites"]["controller_private_key_path"]
        ),
        "--expected-durable-uri": str(packet["durability"]["artifact_uri"]),
    }
    for flag, expected in expected_flags.items():
        if flag_value(flag) != expected:
            raise LaunchPacketError(f"{flag} differs from its frozen packet field")
    remote = commands["remote"]
    if not isinstance(remote, str) or str(runtime["image_digest"]) not in remote:
        raise LaunchPacketError(
            "remote command is not bound to the runtime image digest"
        )
    if flag_value("--remote-command") != remote:
        raise LaunchPacketError(
            "lifecycle remote command differs from the frozen template"
        )

    def command_flag(command_name: str, flag: str) -> str:
        command = commands.get(command_name)
        if not isinstance(command, str):
            raise LaunchPacketError(f"{command_name} must be a command string")
        tokens = shlex.split(command)
        positions = [index for index, item in enumerate(tokens) if item == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(tokens):
            raise LaunchPacketError(f"{command_name} requires exactly one {flag}")
        return tokens[positions[0] + 1]

    durability = _mapping(packet.get("durability"), label="durability")
    if durability.get("bucket_versioning_status") != "Enabled":
        raise LaunchPacketError("launch requires verified S3 bucket versioning Enabled")
    expected_receipt_hash = _sha256(
        durability.get("bucket_versioning_receipt_sha256"),
        label="bucket_versioning_receipt_sha256",
    )
    receipt_path = Path(str(durability.get("bucket_versioning_receipt_path", "")))
    if _regular_file_sha256(receipt_path, label="bucket versioning receipt") != (
        expected_receipt_hash
    ):
        raise LaunchPacketError("bucket versioning receipt hash differs")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchPacketError(
            "bucket versioning receipt is not readable JSON"
        ) from error
    expected_receipt = {
        "format": "tinylora_step5_bucket_versioning_receipt_v1",
        "bucket": durability.get("bucket"),
        "region": durability.get("region"),
        "account_id": durability.get("account_id"),
        "status": "Enabled",
    }
    if (
        not isinstance(expected_receipt["account_id"], str)
        or re.fullmatch(r"[0-9]{12}", expected_receipt["account_id"]) is None
    ):
        raise LaunchPacketError("durability account_id must be an AWS account ID")
    if (
        not isinstance(expected_receipt["region"], str)
        or re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-[0-9]", expected_receipt["region"])
        is None
    ):
        raise LaunchPacketError("durability region must be an AWS region")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        *expected_receipt,
        "checked_at",
    }:
        raise LaunchPacketError("bucket versioning receipt fields differ")
    for field, expected in expected_receipt.items():
        if receipt.get(field) != expected:
            raise LaunchPacketError(f"bucket versioning receipt {field} differs")
    checked_at = receipt.get("checked_at")
    if not isinstance(checked_at, str):
        raise LaunchPacketError("bucket versioning checked_at must be an ISO timestamp")
    try:
        checked = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise LaunchPacketError(
            "bucket versioning checked_at must be an ISO timestamp"
        ) from error
    if checked.tzinfo is None or not checked <= parsed <= checked + timedelta(
        minutes=30
    ):
        raise LaunchPacketError(
            "bucket versioning receipt must precede approval by at most 30 minutes"
        )
    _validate_s3(durability.get("artifact_uri"), label="artifact_uri")
    _validate_s3(durability.get("checkpoint_prefix"), label="checkpoint_prefix")
    _sha256(
        durability.get("checkpoint_controller_key_id"),
        label="checkpoint_controller_key_id",
    )

    if command_flag("versioning_receipt", "--bucket") != durability["bucket"]:
        raise LaunchPacketError("versioning receipt command bucket differs")
    receipt_output = Path(command_flag("versioning_receipt", "--output"))
    if not receipt_output.is_absolute():
        receipt_output = Path.cwd() / receipt_output
    if receipt_output.resolve() != receipt_path.resolve():
        raise LaunchPacketError("versioning receipt command output differs")
    artifact_bucket, artifact_key = str(durability["artifact_uri"])[5:].split("/", 1)
    if command_flag("artifact_upload_preparation", "--bucket") != artifact_bucket:
        raise LaunchPacketError("artifact preparation bucket differs")
    if command_flag("artifact_upload_preparation", "--key") != artifact_key:
        raise LaunchPacketError("artifact preparation key differs")
    for flag, expected in (
        ("--region", durability["region"]),
        ("--account-id", durability["account_id"]),
        ("--approved-at", approval["approved_at"]),
        ("--expires-in", "21600"),
    ):
        if command_flag("artifact_upload_preparation", flag) != expected:
            raise LaunchPacketError(f"artifact preparation {flag} differs")
    input_manifest_bucket, input_manifest_key = str(
        packet["remote_inputs"]["input_url_manifest_s3_uri"]
    )[5:].split("/", 1)
    input_command = "input_url_preparation"
    for flag, expected in (
        ("--account-id", durability["account_id"]),
        ("--region", durability["region"]),
        ("--manifest-bucket", input_manifest_bucket),
        ("--manifest-key", input_manifest_key),
        ("--expires-in", "21600"),
    ):
        if command_flag(input_command, flag) != expected:
            raise LaunchPacketError(f"input URL preparation {flag} differs")
    packet_input = Path(command_flag(input_command, "--packet"))
    if not packet_input.is_absolute():
        packet_input = packet_dir.parent / packet_input
    expected_packet = packet_dir / "tinylora_step5_canary_launch_packet_v1.json"
    if packet_input.resolve() != expected_packet.resolve():
        raise LaunchPacketError("input URL preparation packet path differs")

    controller = _mapping(
        packet.get("controller_prerequisites"), label="controller_prerequisites"
    )
    artifact_url_output = Path(
        command_flag("artifact_upload_preparation", "--url-file")
    )
    if not artifact_url_output.is_absolute():
        artifact_url_output = Path.cwd() / artifact_url_output
    if (
        artifact_url_output.resolve()
        != Path(str(controller["artifact_put_url_file"])).resolve()
    ):
        raise LaunchPacketError("artifact preparation URL file differs")
    artifact_receipt_output = Path(
        command_flag("artifact_upload_preparation", "--receipt")
    )
    if not artifact_receipt_output.is_absolute():
        artifact_receipt_output = Path.cwd() / artifact_receipt_output
    artifact_receipt_path = Path(str(controller["artifact_put_receipt_path"]))
    if artifact_receipt_output.resolve() != artifact_receipt_path.resolve():
        raise LaunchPacketError("artifact preparation receipt file differs")
    artifact_receipt_hash = _sha256(
        controller.get("artifact_put_receipt_sha256"),
        label="artifact_put_receipt_sha256",
    )
    artifact_receipt_bytes = _read_regular_bytes(
        artifact_receipt_path, label="artifact PUT receipt"
    )
    if hashlib.sha256(artifact_receipt_bytes).hexdigest() != artifact_receipt_hash:
        raise LaunchPacketError("artifact PUT receipt hash differs")
    try:
        artifact_receipt = json.loads(artifact_receipt_bytes)
        if not isinstance(artifact_receipt, Mapping):
            raise ValueError("receipt is not an object")
        verified_artifact_receipt = validate_artifact_put_receipt(
            artifact_receipt,
            url_bytes=_read_regular_bytes(
                Path(str(controller["artifact_put_url_file"])),
                label="artifact PUT URL",
            ),
            expected_receipt_sha256=artifact_receipt_hash,
            expected_durable_uri=str(durability["artifact_uri"]),
            expected_approved_at=str(approval["approved_at"]),
            now=parsed,
            max_approval_age_seconds=3600,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise LaunchPacketError("artifact PUT receipt is invalid") from error
    if (
        verified_artifact_receipt.get("account_id") != durability["account_id"]
        or verified_artifact_receipt.get("endpoint", {}).get("region")
        != durability["region"]
    ):
        raise LaunchPacketError("artifact PUT receipt account or region differs")
    for flag, controller_field in (
        ("--manifest-output", "input_url_manifest_output_path"),
        ("--url-file", "input_url_manifest_url_file"),
        ("--host-gate-url-file", "host_gate_url_file"),
        ("--receipt", "input_url_controller_receipt_path"),
    ):
        output = Path(command_flag(input_command, flag))
        if not output.is_absolute():
            output = packet_dir.parent / output
        if output.resolve() != Path(str(controller[controller_field])).resolve():
            raise LaunchPacketError(f"input URL preparation {flag} output differs")
    _validate_input_url_receipt(
        packet,
        approval_time=parsed,
        controller=controller,
    )
    public_path = Path(str(controller.get("controller_public_key_path", "")))
    private_path = Path(str(controller.get("controller_private_key_path", "")))
    expected_public_hash = _sha256(
        controller.get("controller_public_key_sha256"),
        label="controller_public_key_sha256",
    )
    if (
        _regular_file_sha256(public_path, label="controller public key")
        != expected_public_hash
    ):
        raise LaunchPacketError("controller public key hash differs")
    for path, label in (
        (private_path, "controller private key"),
        (
            Path(str(controller.get("input_url_controller_receipt_path", ""))),
            "input URL controller receipt",
        ),
        (
            Path(str(controller.get("input_url_manifest_output_path", ""))),
            "input URL manifest",
        ),
        (
            Path(str(controller.get("input_url_manifest_url_file", ""))),
            "input URL manifest secret",
        ),
        (
            Path(str(controller.get("artifact_put_url_file", ""))),
            "artifact PUT URL secret",
        ),
        (
            Path(str(controller.get("artifact_put_receipt_path", ""))),
            "artifact PUT receipt",
        ),
        (Path(str(controller.get("host_gate_url_file", ""))), "host gate URL secret"),
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.S_IMODE(path.stat().st_mode) != 0o600
        ):
            raise LaunchPacketError(f"{label} must be a 0600 regular file")


def _validate_runtime_publication(value: Any) -> None:
    runtime = _mapping(value, label="runtime")
    publication = _mapping(runtime.get("publication"), label="runtime publication")
    if set(publication) != {
        "amd64_manifest_digest",
        "index_digest",
        "raw_index_digest_verified",
        "sha256sums_verified",
        "source_commit",
        "workflow_run_id",
    }:
        raise LaunchPacketError("runtime publication evidence fields differ")
    for field in ("amd64_manifest_digest", "index_digest"):
        digest = publication.get(field)
        if (
            field == "index_digest"
            and _is_placeholder(digest)
            and digest == runtime.get("image_digest")
        ):
            continue
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise LaunchPacketError(f"runtime publication {field} is invalid")
    if publication["index_digest"] != runtime.get("image_digest"):
        raise LaunchPacketError("runtime publication index digest differs from image")
    if publication.get("raw_index_digest_verified") is not True:
        raise LaunchPacketError(
            "runtime raw index digest is not independently verified"
        )
    if publication.get("sha256sums_verified") is not True:
        raise LaunchPacketError("runtime publication artifacts are not hash verified")
    if (
        not isinstance(publication.get("source_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", publication["source_commit"]) is None
    ):
        raise LaunchPacketError("runtime publication source commit is invalid")
    if (
        not isinstance(publication.get("workflow_run_id"), str)
        or not publication["workflow_run_id"].isdigit()
    ):
        raise LaunchPacketError("runtime publication workflow run ID is invalid")


def validate_launch_packet(
    value: Mapping[str, Any], *, packet_dir: Path
) -> dict[str, Any]:
    """Validate one packet and report readiness without executing anything."""
    if value.get("format") != PACKET_FORMAT:
        raise LaunchPacketError("unsupported launch packet format")
    expected_hash = _sha256(value.get("packet_sha256"), label="packet_sha256")
    actual_hash = canonical_sha256(value)
    if expected_hash != actual_hash:
        raise LaunchPacketError("packet_sha256 does not match packet contents")

    execution = _mapping(value.get("execution"), label="execution")
    if execution != {"enabled": False, "execute_flag_present": False}:
        raise LaunchPacketError("launch packet must keep execute disabled")

    identity = _mapping(value.get("identity"), label="identity")
    for field in (
        "plan_sha256",
        "probe_qualification_file_sha256",
        "probe_qualification_receipt_sha256",
        "model_content_sha256",
        "pixmo_content_sha256",
    ):
        _sha256(identity.get(field), label=field)
    revision = identity.get("model_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise LaunchPacketError("model_revision must be an exact 40-hex revision")
    _validate_runtime_publication(value.get("runtime"))

    controller_contracts = _mapping(
        value.get("controller_contracts"), label="controller_contracts"
    )
    if set(controller_contracts) != {
        "artifact_upload_preparation",
        "artifact_presigner",
        "checkpoint_controller",
        "input_url_controller",
        "input_url_preparation",
        "lifecycle_wrapper",
        "versioning_receipt",
    }:
        raise LaunchPacketError("controller_contracts inventory differs")
    for name, descriptor_value in controller_contracts.items():
        descriptor = _mapping(descriptor_value, label=name)
        if set(descriptor) != {"path", "sha256"}:
            raise LaunchPacketError(f"{name} descriptor fields differ")
        path = Path(str(descriptor["path"]))
        if not path.is_absolute():
            path = packet_dir / path
        actual = _regular_file_sha256(path.resolve(), label=name)
        expected = _sha256(descriptor["sha256"], label=f"{name} sha256")
        if actual != expected:
            raise LaunchPacketError(f"{name} hash does not match local bytes")

    _validate_local_contracts(value.get("local_contracts"), packet_dir)
    remote_inputs = _mapping(value.get("remote_inputs"), label="remote_inputs")
    for name, item in remote_inputs.items():
        if name.endswith("_sha256"):
            _sha256(item, label=name)
    _validate_s3(remote_inputs.get("model_s3_prefix"), label="model_s3_prefix")
    _validate_s3(
        remote_inputs.get("input_url_manifest_s3_uri"),
        label="input_url_manifest_s3_uri",
    )
    _validate_s3(remote_inputs.get("pixmo_s3_prefix"), label="pixmo_s3_prefix")
    _validate_s3(
        remote_inputs.get("plan_s3_uri"),
        label="plan_s3_uri",
        allow_placeholder=True,
    )
    _validate_s3(
        remote_inputs.get("probe_s3_uri"),
        label="probe_s3_uri",
        allow_placeholder=True,
    )

    commands = _mapping(value.get("commands"), label="commands")
    expected_commands = {
        "artifact_upload_preparation",
        "input_url_preparation",
        "versioning_receipt",
        "remote",
        "host_qualification",
        "diagnostic",
        "software_recovery",
        "lifecycle_argv",
    }
    if set(commands) != expected_commands:
        raise LaunchPacketError("commands inventory differs")
    lifecycle = commands["lifecycle_argv"]
    if not isinstance(lifecycle, list) or not all(
        isinstance(item, str) and item for item in lifecycle
    ):
        raise LaunchPacketError("lifecycle_argv must be a nonempty string list")
    if len(lifecycle) < 2:
        raise LaunchPacketError("lifecycle_argv must name its controller and wrapper")
    wrapper_descriptor = _mapping(
        controller_contracts["lifecycle_wrapper"], label="lifecycle_wrapper"
    )
    wrapper_path = Path(str(wrapper_descriptor["path"]))
    if not wrapper_path.is_absolute():
        wrapper_path = packet_dir / wrapper_path
    argv_wrapper = Path(lifecycle[1])
    if not argv_wrapper.is_absolute():
        argv_wrapper = packet_dir.parent / argv_wrapper
    if argv_wrapper.resolve() != wrapper_path.resolve():
        raise LaunchPacketError(
            "lifecycle_argv wrapper differs from controller contract"
        )
    all_command_text = "\n".join(
        [
            *(str(commands[name]) for name in expected_commands - {"lifecycle_argv"}),
            *lifecycle,
        ]
    )
    if re.search(r"(?:^|\s)--execute(?:\s|$)", all_command_text):
        raise LaunchPacketError("launch packet must contain zero --execute flags")
    secret_patterns = (
        r"\bAWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)\s*=",
        r"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY)\s*=",
        r"\bVAST(?:AI)?_(?:API_)?KEY\s*=",
        r"--(?:api-key|token|password)(?:=|\s+)",
        r"Authorization\s*:\s*Bearer\s+",
    )
    if any(
        re.search(pattern, all_command_text, re.IGNORECASE)
        for pattern in secret_patterns
    ):
        raise LaunchPacketError("launch packet must not embed credentials")

    declared = value.get("remaining_substitutions")
    if not isinstance(declared, list) or declared != sorted(set(declared)):
        raise LaunchPacketError("remaining_substitutions must be sorted and unique")
    observed = sorted(_collect_placeholders(value))
    if declared != observed:
        raise LaunchPacketError(
            "remaining_substitutions differ from packet placeholders"
        )
    launch_ready = not observed
    if launch_ready:
        _validate_complete_runtime(value, packet_dir)
    return {
        "format": "tinylora_step5_canary_launch_validation_v1",
        "valid": True,
        "launch_ready": launch_ready,
        "packet_sha256": actual_hash,
        "remaining_substitutions": observed,
        "executed": False,
    }
