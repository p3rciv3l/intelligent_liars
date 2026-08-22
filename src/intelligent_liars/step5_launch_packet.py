"""Validate an inert, hash-bound Step 5 canary launch packet.

This module never invokes a subprocess, cloud API, or workload.  It only checks
that a packet is internally consistent and reports whether every launch-time
substitution has been frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


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
            match.group(1)
            for match in re.finditer(r"\$\{([A-Z][A-Z0-9_]*)\}", value)
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


def _validate_complete_runtime(packet: Mapping[str, Any]) -> None:
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
        raise LaunchPacketError("runtime image and image_digest must identify the same bytes")
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
        "--expected-durable-uri": str(packet["durability"]["artifact_uri"]),
    }
    for flag, expected in expected_flags.items():
        if flag_value(flag) != expected:
            raise LaunchPacketError(f"{flag} differs from its frozen packet field")
    remote = commands["remote"]
    if not isinstance(remote, str) or str(runtime["image_digest"]) not in remote:
        raise LaunchPacketError("remote command is not bound to the runtime image digest")
    if flag_value("--remote-command") != remote:
        raise LaunchPacketError("lifecycle remote command differs from the frozen template")

    durability = _mapping(packet.get("durability"), label="durability")
    if durability.get("bucket_versioning_status") != "Enabled":
        raise LaunchPacketError("launch requires verified S3 bucket versioning Enabled")
    _sha256(
        durability.get("bucket_versioning_receipt_sha256"),
        label="bucket_versioning_receipt_sha256",
    )
    _validate_s3(durability.get("artifact_uri"), label="artifact_uri")
    _validate_s3(durability.get("checkpoint_prefix"), label="checkpoint_prefix")
    _sha256(durability.get("checkpoint_controller_key_id"), label="checkpoint_controller_key_id")

    controller = _mapping(packet.get("controller_prerequisites"), label="controller_prerequisites")
    public_path = Path(str(controller.get("controller_public_key_path", "")))
    private_path = Path(str(controller.get("controller_private_key_path", "")))
    expected_public_hash = _sha256(
        controller.get("controller_public_key_sha256"),
        label="controller_public_key_sha256",
    )
    if _regular_file_sha256(public_path, label="controller public key") != expected_public_hash:
        raise LaunchPacketError("controller public key hash differs")
    for path, label in (
        (private_path, "controller private key"),
        (Path(str(controller.get("input_url_manifest_url_file", ""))), "input URL manifest secret"),
        (Path(str(controller.get("artifact_put_url_file", ""))), "artifact PUT URL secret"),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise LaunchPacketError(f"{label} must be a 0600 regular file")


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

    controller_contracts = _mapping(
        value.get("controller_contracts"), label="controller_contracts"
    )
    if set(controller_contracts) != {
        "artifact_upload_preparation",
        "checkpoint_controller",
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
        "checkpoint_controller",
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
    all_command_text = "\n".join(
        [*(str(commands[name]) for name in expected_commands - {"lifecycle_argv"}), *lifecycle]
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
        raise LaunchPacketError("remaining_substitutions differ from packet placeholders")
    launch_ready = not observed
    if launch_ready:
        _validate_complete_runtime(value)
    return {
        "format": "tinylora_step5_canary_launch_validation_v1",
        "valid": True,
        "launch_ready": launch_ready,
        "packet_sha256": actual_hash,
        "remaining_substitutions": observed,
        "executed": False,
    }
