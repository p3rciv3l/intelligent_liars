from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.step5_launch_packet import (
    LaunchPacketError,
    canonical_sha256,
    validate_launch_packet,
)


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source.json"
    inventory = tmp_path / "inventory.json"
    source_sha = _write_json(
        source,
        {
            "format": "tinylora_step5_source_manifest_v1",
            "files": [{"path": "pyproject.toml", "sha256": "1" * 64}],
        },
    )
    inventory_sha = _write_json(
        inventory,
        {
            "format": "tinylora_step5_expected_artifact_inventory_v1",
            "files": ["canary_bundle.tar", "canary_summary.json"],
        },
    )
    packet: dict[str, object] = {
        "format": "tinylora_step5_canary_launch_packet_v1",
        "execution": {"enabled": False, "execute_flag_present": False},
        "identity": {
            "plan_sha256": "2" * 64,
            "probe_qualification_sha256": "3" * 64,
            "model_revision": "4" * 40,
            "model_content_sha256": "5" * 64,
            "pixmo_content_sha256": "6" * 64,
        },
        "local_contracts": {
            "source_manifest": {"path": str(source), "sha256": source_sha},
            "expected_artifact_inventory": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
        },
        "remote_inputs": {
            "model_s3_prefix": "s3://bucket/model-cache/revision/content",
            "pixmo_s3_prefix": "s3://bucket/assets/pixmo/content",
            "plan_s3_uri": "${PLAN_S3_URI}",
            "probe_s3_uri": "${PROBE_S3_URI}",
        },
        "runtime": {
            "image": "${RUNTIME_IMAGE}",
            "image_digest": "${RUNTIME_IMAGE_DIGEST}",
            "offer_id": "${OFFER_ID}",
            "gpu_name": "${GPU_NAME}",
            "gpu_count": 1,
            "gpu_vram_gib": "${GPU_VRAM_GIB}",
            "disk_gib": 160,
        },
        "approval": {
            "approved_hourly_price_usd": "${APPROVED_HOURLY_PRICE_USD}",
            "approved_max_cost_usd": "${APPROVED_MAX_COST_USD}",
            "approved_at": "${APPROVED_AT}",
        },
        "commands": {
            "remote": "python scripts/run_tinylora_step5_screen.py --mode prerequisites --runtime-image-digest ${RUNTIME_IMAGE_DIGEST}",
            "host_qualification": "python scripts/qualify_vast_step5_host.py --download-url-env STEP5_HOST_GATE_URL",
            "diagnostic": "python scripts/diagnose_tinylora_step5_canary.py",
            "software_recovery": "python scripts/validate_tinylora_step5_launch_packet.py --packet configs/packet.json --allow-incomplete",
            "lifecycle_argv": [
                "python",
                "scripts/run_vast_step5_instance.py",
                "--offer-id",
                "${OFFER_ID}",
                "--image",
                "${RUNTIME_IMAGE}",
                "--disk",
                "160",
                "--approved-hourly-price",
                "${APPROVED_HOURLY_PRICE_USD}",
                "--approved-max-cost",
                "${APPROVED_MAX_COST_USD}",
                "--remote-command",
                "python scripts/run_tinylora_step5_screen.py --mode prerequisites --runtime-image-digest ${RUNTIME_IMAGE_DIGEST}",
            ],
        },
        "remaining_substitutions": [
            "APPROVED_AT",
            "APPROVED_HOURLY_PRICE_USD",
            "APPROVED_MAX_COST_USD",
            "GPU_NAME",
            "GPU_VRAM_GIB",
            "OFFER_ID",
            "PLAN_S3_URI",
            "PROBE_S3_URI",
            "RUNTIME_IMAGE",
            "RUNTIME_IMAGE_DIGEST",
        ],
    }
    packet["packet_sha256"] = canonical_sha256(packet)
    return packet


def test_incomplete_packet_is_valid_but_not_launch_ready(tmp_path: Path):
    result = validate_launch_packet(_packet(tmp_path), packet_dir=tmp_path)
    assert result["valid"] is True
    assert result["launch_ready"] is False
    assert "RUNTIME_IMAGE" in result["remaining_substitutions"]


def test_packet_rejects_execute_flag_and_enabled_execution(tmp_path: Path):
    for mutation in ("flag", "enabled"):
        packet = _packet(tmp_path)
        if mutation == "flag":
            packet["commands"]["lifecycle_argv"].append("--execute")  # type: ignore[index,union-attr]
        else:
            packet["execution"]["enabled"] = True  # type: ignore[index]
        packet["packet_sha256"] = canonical_sha256(packet)
        with pytest.raises(LaunchPacketError, match="execute"):
            validate_launch_packet(packet, packet_dir=tmp_path)


def test_packet_rejects_changed_local_contract(tmp_path: Path):
    packet = _packet(tmp_path)
    Path(packet["local_contracts"]["source_manifest"]["path"]).write_text("changed\n")  # type: ignore[index]
    with pytest.raises(LaunchPacketError, match="source_manifest hash"):
        validate_launch_packet(packet, packet_dir=tmp_path)


def test_complete_packet_requires_digest_and_exact_cost_approval(tmp_path: Path):
    packet = _packet(tmp_path)
    replacements = {
        "${RUNTIME_IMAGE}": "ghcr.io/example/step5@sha256:" + "a" * 64,
        "${RUNTIME_IMAGE_DIGEST}": "sha256:" + "a" * 64,
        "${OFFER_ID}": "12345678",
        "${GPU_NAME}": "RTX 3090",
        "${GPU_VRAM_GIB}": 24,
        "${APPROVED_HOURLY_PRICE_USD}": 0.25,
        "${APPROVED_MAX_COST_USD}": 1.50,
        "${APPROVED_AT}": "2026-08-22T20:00:00Z",
        "${PLAN_S3_URI}": "s3://bucket/step5/plan.tar",
        "${PROBE_S3_URI}": "s3://bucket/step5/probe.tar",
    }

    def replace(value: object) -> object:
        if isinstance(value, str):
            if value in replacements:
                return replacements[value]
            replaced = value
            for marker, replacement in replacements.items():
                replaced = replaced.replace(marker, str(replacement))
            return replaced
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    complete = replace(copy.deepcopy(packet))
    assert isinstance(complete, dict)
    complete["commands"]["lifecycle_argv"] = [  # type: ignore[index]
        str(item) for item in complete["commands"]["lifecycle_argv"]  # type: ignore[index]
    ]
    complete["remaining_substitutions"] = []
    complete["packet_sha256"] = canonical_sha256(complete)
    result = validate_launch_packet(complete, packet_dir=tmp_path)
    assert result["launch_ready"] is True

    complete["runtime"]["image"] = "ghcr.io/example/step5:latest"  # type: ignore[index]
    complete["packet_sha256"] = canonical_sha256(complete)
    with pytest.raises(LaunchPacketError, match="immutable runtime image"):
        validate_launch_packet(complete, packet_dir=tmp_path)


def test_complete_packet_rejects_command_field_drift(tmp_path: Path):
    packet = _packet(tmp_path)
    packet["runtime"].update(  # type: ignore[union-attr]
        {
            "image": "ghcr.io/example/step5@sha256:" + "a" * 64,
            "image_digest": "sha256:" + "a" * 64,
            "offer_id": "123",
            "gpu_name": "RTX 3090",
            "gpu_vram_gib": 24,
        }
    )
    packet["approval"] = {
        "approved_hourly_price_usd": 0.25,
        "approved_max_cost_usd": 1.5,
        "approved_at": "2026-08-22T20:00:00Z",
    }
    packet["remote_inputs"]["plan_s3_uri"] = "s3://bucket/plan"  # type: ignore[index]
    packet["remote_inputs"]["probe_s3_uri"] = "s3://bucket/probe"  # type: ignore[index]
    packet["commands"]["remote"] = "run --runtime sha256:" + "a" * 64  # type: ignore[index]
    packet["commands"]["lifecycle_argv"] = [  # type: ignore[index]
        "python",
        "wrapper.py",
        "--offer-id",
        "999",
        "--image",
        "ghcr.io/example/step5@sha256:" + "a" * 64,
        "--disk",
        "160",
        "--approved-hourly-price",
        "0.25",
        "--approved-max-cost",
        "1.5",
        "--remote-command",
        packet["commands"]["remote"],  # type: ignore[index]
    ]
    packet["remaining_substitutions"] = []
    packet["packet_sha256"] = canonical_sha256(packet)
    with pytest.raises(LaunchPacketError, match="offer-id"):
        validate_launch_packet(packet, packet_dir=tmp_path)


def test_packet_hash_detects_tampering(tmp_path: Path):
    packet = _packet(tmp_path)
    packet["runtime"]["disk_gib"] = 200  # type: ignore[index]
    with pytest.raises(LaunchPacketError, match="packet_sha256"):
        validate_launch_packet(packet, packet_dir=tmp_path)


def test_packet_rejects_embedded_credentials(tmp_path: Path):
    packet = _packet(tmp_path)
    packet["commands"]["diagnostic"] = "AWS_SESSION_TOKEN=not-allowed diagnose"  # type: ignore[index]
    packet["packet_sha256"] = canonical_sha256(packet)
    with pytest.raises(LaunchPacketError, match="credentials"):
        validate_launch_packet(packet, packet_dir=tmp_path)


def test_cli_is_inert_and_fails_closed_until_complete(tmp_path: Path):
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(_packet(tmp_path)))
    command = [
        sys.executable,
        "scripts/validate_tinylora_step5_launch_packet.py",
        "--packet",
        str(packet_path),
    ]
    blocked = subprocess.run(command, text=True, capture_output=True, check=False)
    assert blocked.returncode == 2
    assert '"executed": false' in blocked.stdout
    allowed = subprocess.run(
        [*command, "--allow-incomplete"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert allowed.returncode == 0
    assert '"launch_ready": false' in allowed.stdout
