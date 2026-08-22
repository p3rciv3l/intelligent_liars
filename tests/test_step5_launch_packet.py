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
    public_key = tmp_path / "public.pem"
    private_key = tmp_path / "private.pem"
    input_secret = tmp_path / "input.url"
    artifact_secret = tmp_path / "artifact.url"
    host_gate_secret = tmp_path / "host-gate.url"
    versioning_receipt = tmp_path / "versioning.json"
    for path, content in (
        (public_key, "public\n"),
        (private_key, "private\n"),
        (input_secret, "https://example.test/input?signed=yes\n"),
        (artifact_secret, "https://example.test/output?signed=yes\n"),
        (host_gate_secret, "https://example.test/gate?signed=yes\n"),
    ):
        path.write_text(content)
        path.chmod(0o600)
    packet: dict[str, object] = {
        "format": "tinylora_step5_canary_launch_packet_v1",
        "execution": {"enabled": False, "execute_flag_present": False},
        "identity": {
            "plan_sha256": "2" * 64,
            "probe_qualification_file_sha256": "3" * 64,
            "probe_qualification_receipt_sha256": "7" * 64,
            "model_revision": "4" * 40,
            "model_content_sha256": "5" * 64,
            "pixmo_content_sha256": "6" * 64,
        },
        "controller_contracts": {
            "artifact_upload_preparation": {
                "path": str(source),
                "sha256": source_sha,
            },
            "checkpoint_controller": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
            "input_url_controller": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
            "input_url_preparation": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
            "lifecycle_wrapper": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
            "versioning_receipt": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
        },
        "local_contracts": {
            "source_manifest": {"path": str(source), "sha256": source_sha},
            "expected_artifact_inventory": {
                "path": str(inventory),
                "sha256": inventory_sha,
            },
        },
        "remote_inputs": {
            "frozen_inputs_completion_sha256": "d" * 64,
            "frozen_inputs_tar_sha256": "c" * 64,
            "input_url_manifest_s3_uri": "s3://bucket/controller/input.json",
            "model_completion_sha256": "b" * 64,
            "model_manifest_sha256": "a" * 64,
            "model_s3_prefix": "s3://bucket/model-cache/revision/content",
            "pixmo_completion_sha256": "9" * 64,
            "pixmo_manifest_sha256": "f" * 64,
            "pixmo_s3_prefix": "s3://bucket/assets/pixmo/content",
            "pixmo_tar_sha256": "e" * 64,
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
            "artifact_upload_preparation": (
                "prepare --bucket bucket --key artifacts/canary.tar "
                f"--url-file {artifact_secret}"
            ),
            "versioning_receipt": (
                f"capture --bucket bucket --output {versioning_receipt}"
            ),
            "remote": "python scripts/run_tinylora_step5_screen.py --mode prerequisites --runtime-image-digest ${RUNTIME_IMAGE_DIGEST}",
            "host_qualification": "python scripts/qualify_vast_step5_host.py --download-url-env STEP5_HOST_GATE_URL",
            "input_url_preparation": (
                f"prepare --packet {tmp_path / 'tinylora_step5_canary_launch_packet_v1.json'} "
                "--account-id ${AWS_ACCOUNT_ID} --region ${AWS_REGION} "
                "--manifest-bucket bucket --manifest-key controller/input.json "
                f"--manifest-output {tmp_path / 'input.json'} "
                f"--url-file {input_secret} --host-gate-url-file {host_gate_secret} "
                f"--receipt {tmp_path / 'input-receipt.json'}"
            ),
            "diagnostic": "python scripts/diagnose_tinylora_step5_canary.py",
            "software_recovery": "python scripts/validate_tinylora_step5_launch_packet.py --packet configs/packet.json --allow-incomplete",
            "lifecycle_argv": [
                "python",
                str(inventory),
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
                "--controller-public-key-sha256",
                hashlib.sha256(public_key.read_bytes()).hexdigest(),
                "--input-url-manifest-url-file",
                str(input_secret),
                "--artifact-put-url-file",
                str(artifact_secret),
                "--expected-durable-uri",
                "s3://bucket/artifacts/canary.tar",
                "--host-gate-url-file",
                str(host_gate_secret),
                "--checkpoint-controller-script",
                str(inventory),
                "--checkpoint-controller-script-sha256",
                inventory_sha,
                "--checkpoint-bucket",
                "bucket",
                "--checkpoint-prefix",
                "checkpoints/canary",
                "--checkpoint-controller-private-key",
                str(private_key),
                "--remote-command",
                "python scripts/run_tinylora_step5_screen.py --mode prerequisites --runtime-image-digest ${RUNTIME_IMAGE_DIGEST}",
            ],
        },
        "durability": {
            "account_id": "${AWS_ACCOUNT_ID}",
            "artifact_uri": "s3://bucket/artifacts/canary.tar",
            "bucket": "bucket",
            "bucket_versioning_status": "Enabled",
            "bucket_versioning_receipt_path": str(versioning_receipt),
            "bucket_versioning_receipt_sha256": "${BUCKET_VERSIONING_RECEIPT_SHA256}",
            "checkpoint_controller_key_id": "8" * 64,
            "checkpoint_prefix": "s3://bucket/checkpoints/canary",
            "region": "${AWS_REGION}",
        },
        "controller_prerequisites": {
            "controller_public_key_path": str(public_key),
            "checkpoint_controller_script_path": str(inventory),
            "controller_public_key_sha256": hashlib.sha256(public_key.read_bytes()).hexdigest(),
            "controller_private_key_path": str(private_key),
            "input_url_manifest_url_file": str(input_secret),
            "artifact_put_url_file": str(artifact_secret),
            "host_gate_url_file": str(host_gate_secret),
            "input_url_controller_receipt_path": str(tmp_path / "input-receipt.json"),
            "input_url_controller_receipt_sha256": "${INPUT_URL_CONTROLLER_RECEIPT_SHA256}",
            "input_url_manifest_output_path": str(tmp_path / "input.json"),
        },
        "remaining_substitutions": [
            "APPROVED_AT",
            "APPROVED_HOURLY_PRICE_USD",
            "APPROVED_MAX_COST_USD",
            "AWS_ACCOUNT_ID",
            "AWS_REGION",
            "BUCKET_VERSIONING_RECEIPT_SHA256",
            "GPU_NAME",
            "GPU_VRAM_GIB",
            "INPUT_URL_CONTROLLER_RECEIPT_SHA256",
            "OFFER_ID",
            "PLAN_S3_URI",
            "PROBE_S3_URI",
            "RUNTIME_IMAGE",
            "RUNTIME_IMAGE_DIGEST",
        ],
    }
    packet["packet_sha256"] = canonical_sha256(packet)
    return packet


def _write_input_controller_outputs(packet: dict[str, object]) -> str:
    controller = packet["controller_prerequisites"]  # type: ignore[assignment]
    remote = packet["remote_inputs"]  # type: ignore[assignment]
    date_query = (
        "X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=ASIATESTACCESSKEY1%2F20260822%2Fus-west-2%2Fs3%2Faws4_request"
        "&X-Amz-Date=20260822T195000Z&X-Amz-Expires=21600"
        "&X-Amz-SignedHeaders=host&X-Amz-Signature=" + "a" * 64
    )

    def url(key: str) -> str:
        return f"https://bucket.s3.us-west-2.amazonaws.com/{key}?{date_query}"

    keys = {
        "model_manifest": "model-cache/revision/content/manifest.json",
        "model_completion": "model-cache/revision/content/_COMPLETE.json",
        "model_file": "model-cache/revision/content/files/shard.bin",
        "frozen_archive": "step5/plan.tar",
        "frozen_completion": "step5/_COMPLETE.json",
        "pixmo_archive": "assets/pixmo/content/archive.tar",
        "pixmo_manifest": "assets/pixmo/content/manifest.json",
        "pixmo_completion": "assets/pixmo/content/_COMPLETE.json",
    }
    hashes = {
        keys["model_manifest"]: remote["model_manifest_sha256"],
        keys["model_completion"]: remote["model_completion_sha256"],
        keys["model_file"]: "8" * 64,
        keys["frozen_archive"]: remote["frozen_inputs_tar_sha256"],
        keys["frozen_completion"]: remote["frozen_inputs_completion_sha256"],
        keys["pixmo_archive"]: remote["pixmo_tar_sha256"],
        keys["pixmo_manifest"]: remote["pixmo_manifest_sha256"],
        keys["pixmo_completion"]: remote["pixmo_completion_sha256"],
    }
    manifest = {
        "controller": {
            "account_id": "123456789012",
            "bucket": "bucket",
            "created_at": "2026-08-22T19:50:00Z",
            "expires_at": "2026-08-23T01:50:00Z",
            "expiry_seconds": 21600,
            "manifest_key": "controller/input.json",
            "region": "us-west-2",
        },
        "format": "tinylora_step5_input_url_manifest_v2",
        "model": {
            "completion_url": url(keys["model_completion"]),
            "manifest_url": url(keys["model_manifest"]),
            "file_urls": {"shard.bin": url(keys["model_file"])},
        },
        "frozen_inputs": {
            "completion_url": url(keys["frozen_completion"]),
            "archive_url": url(keys["frozen_archive"]),
        },
        "pixmo": {
            "completion_url": url(keys["pixmo_completion"]),
            "manifest_url": url(keys["pixmo_manifest"]),
            "archive_url": url(keys["pixmo_archive"]),
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest_path = Path(controller["input_url_manifest_output_path"])
    manifest_path.write_bytes(manifest_bytes)
    host_key = keys["model_file"]
    receipt = {
        "account_id": "123456789012",
        "created_at": "2026-08-22T19:50:00Z",
        "expires_at": "2026-08-23T01:50:00Z",
        "expiry_seconds": 21600,
        "format": "tinylora_step5_input_url_controller_receipt_v1",
        "host_gate": {"bytes": 1, "key": host_key, "sha256": hashes[host_key]},
        "manifest": {
            "bucket": "bucket",
            "key": "controller/input.json",
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "objects": [
            {"bytes": 1, "key": key, "sha256": digest, "verification": "head_sha256"}
            for key, digest in sorted(hashes.items())
        ],
        "region": "us-west-2",
    }
    receipt["content_sha256"] = hashlib.sha256(
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    receipt_path = Path(controller["input_url_controller_receipt_path"])
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    Path(controller["input_url_manifest_url_file"]).write_text(
        url("controller/input.json") + "\n"
    )
    Path(controller["host_gate_url_file"]).write_text(url(host_key) + "\n")
    for path in (
        receipt_path,
        manifest_path,
        Path(controller["input_url_manifest_url_file"]),
        Path(controller["host_gate_url_file"]),
    ):
        path.chmod(0o600)
    return hashlib.sha256(receipt_path.read_bytes()).hexdigest()


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
    with pytest.raises(LaunchPacketError, match="hash does not match"):
        validate_launch_packet(packet, packet_dir=tmp_path)


def test_complete_packet_requires_digest_and_exact_cost_approval(tmp_path: Path):
    packet = _packet(tmp_path)
    input_receipt_sha = _write_input_controller_outputs(packet)
    receipt_path = Path(packet["durability"]["bucket_versioning_receipt_path"])  # type: ignore[index]
    receipt_sha = _write_json(
        receipt_path,
        {
            "account_id": "123456789012",
            "bucket": "bucket",
            "checked_at": "2026-08-22T19:50:00Z",
            "format": "tinylora_step5_bucket_versioning_receipt_v1",
            "region": "us-west-2",
            "status": "Enabled",
        },
    )
    replacements = {
        "${RUNTIME_IMAGE}": "ghcr.io/example/step5@sha256:" + "a" * 64,
        "${RUNTIME_IMAGE_DIGEST}": "sha256:" + "a" * 64,
        "${OFFER_ID}": "12345678",
        "${GPU_NAME}": "RTX 3090",
        "${GPU_VRAM_GIB}": 24,
        "${APPROVED_HOURLY_PRICE_USD}": 0.25,
        "${APPROVED_MAX_COST_USD}": 1.50,
        "${APPROVED_AT}": "2026-08-22T20:00:00Z",
        "${AWS_ACCOUNT_ID}": "123456789012",
        "${AWS_REGION}": "us-west-2",
        "${BUCKET_VERSIONING_RECEIPT_SHA256}": receipt_sha,
        "${INPUT_URL_CONTROLLER_RECEIPT_SHA256}": input_receipt_sha,
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

    manifest_path = Path(
        complete["controller_prerequisites"]["input_url_manifest_output_path"]  # type: ignore[index]
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_text("{}\n")
    with pytest.raises(LaunchPacketError, match="manifest hash"):
        validate_launch_packet(complete, packet_dir=tmp_path)
    manifest_path.write_bytes(manifest_bytes)

    host_url_path = Path(
        complete["controller_prerequisites"]["host_gate_url_file"]  # type: ignore[index]
    )
    host_url = host_url_path.read_text()
    host_url_path.write_text(
        host_url.replace("model-cache/revision/content/files/shard.bin", "wrong.bin")
    )
    with pytest.raises(LaunchPacketError, match="frozen S3 object"):
        validate_launch_packet(complete, packet_dir=tmp_path)
    host_url_path.write_text(host_url)

    unsigned_host_url = host_url.split("?", 1)[0] + (
        "?X-Amz-Date=20260822T195000Z&X-Amz-Expires=21600\n"
    )
    host_url_path.write_text(unsigned_host_url)
    with pytest.raises(LaunchPacketError, match="complete SigV4"):
        validate_launch_packet(complete, packet_dir=tmp_path)
    host_url_path.write_text(host_url)

    refreshed_bucket_sha = _write_json(
        receipt_path,
        {
            "account_id": "123456789012",
            "bucket": "bucket",
            "checked_at": "2026-08-22T20:00:00Z",
            "format": "tinylora_step5_bucket_versioning_receipt_v1",
            "region": "us-west-2",
            "status": "Enabled",
        },
    )
    complete["durability"][  # type: ignore[index]
        "bucket_versioning_receipt_sha256"
    ] = refreshed_bucket_sha
    complete["approval"]["approved_at"] = "2026-08-22T20:25:00Z"  # type: ignore[index]
    complete["packet_sha256"] = canonical_sha256(complete)
    with pytest.raises(LaunchPacketError, match="not fresh"):
        validate_launch_packet(complete, packet_dir=tmp_path)
    complete["approval"]["approved_at"] = "2026-08-22T20:00:00Z"  # type: ignore[index]

    complete["runtime"]["image"] = "ghcr.io/example/step5:latest"  # type: ignore[index]
    complete["packet_sha256"] = canonical_sha256(complete)
    with pytest.raises(LaunchPacketError, match="immutable runtime image"):
        validate_launch_packet(complete, packet_dir=tmp_path)


def test_complete_packet_rejects_command_field_drift(tmp_path: Path):
    packet = _packet(tmp_path)
    input_receipt_sha = _write_input_controller_outputs(packet)
    receipt_path = Path(packet["durability"]["bucket_versioning_receipt_path"])  # type: ignore[index]
    receipt_sha = _write_json(
        receipt_path,
        {
            "account_id": "123456789012",
            "bucket": "bucket",
            "checked_at": "2026-08-22T19:50:00Z",
            "format": "tinylora_step5_bucket_versioning_receipt_v1",
            "region": "us-west-2",
            "status": "Enabled",
        },
    )
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
    packet["durability"].update(  # type: ignore[union-attr]
        {
            "account_id": "123456789012",
            "bucket_versioning_receipt_sha256": receipt_sha,
            "region": "us-west-2",
        }
    )
    packet["controller_prerequisites"][  # type: ignore[index]
        "input_url_controller_receipt_sha256"
    ] = input_receipt_sha
    packet["remote_inputs"]["plan_s3_uri"] = "s3://bucket/plan"  # type: ignore[index]
    packet["remote_inputs"]["probe_s3_uri"] = "s3://bucket/probe"  # type: ignore[index]
    packet["commands"]["remote"] = "run --runtime sha256:" + "a" * 64  # type: ignore[index]
    packet["commands"]["input_url_preparation"] = packet["commands"][  # type: ignore[index]
        "input_url_preparation"
    ].replace("${AWS_ACCOUNT_ID}", "123456789012").replace(  # type: ignore[union-attr]
        "${AWS_REGION}", "us-west-2"
    )
    packet["commands"]["lifecycle_argv"] = [  # type: ignore[index]
        "python",
        packet["controller_contracts"]["lifecycle_wrapper"]["path"],  # type: ignore[index]
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
        "--controller-public-key-sha256",
        packet["controller_prerequisites"]["controller_public_key_sha256"],  # type: ignore[index]
        "--input-url-manifest-url-file",
        packet["controller_prerequisites"]["input_url_manifest_url_file"],  # type: ignore[index]
        "--artifact-put-url-file",
        packet["controller_prerequisites"]["artifact_put_url_file"],  # type: ignore[index]
        "--expected-durable-uri",
        packet["durability"]["artifact_uri"],  # type: ignore[index]
        "--host-gate-url-file",
        packet["controller_prerequisites"]["host_gate_url_file"],  # type: ignore[index]
        "--checkpoint-controller-script",
        packet["controller_prerequisites"]["checkpoint_controller_script_path"],  # type: ignore[index]
        "--checkpoint-controller-script-sha256",
        packet["controller_contracts"]["checkpoint_controller"]["sha256"],  # type: ignore[index]
        "--checkpoint-bucket",
        packet["durability"]["bucket"],  # type: ignore[index]
        "--checkpoint-prefix",
        "checkpoints/canary",
        "--checkpoint-controller-private-key",
        packet["controller_prerequisites"]["controller_private_key_path"],  # type: ignore[index]
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


def test_complete_packet_rejects_fabricated_versioning_receipt(tmp_path: Path):
    packet = _packet(tmp_path)
    input_receipt_sha = _write_input_controller_outputs(packet)
    receipt_path = Path(packet["durability"]["bucket_versioning_receipt_path"])  # type: ignore[index]
    receipt_sha = _write_json(
        receipt_path,
        {
            "account_id": "123456789012",
            "bucket": "different-bucket",
            "checked_at": "2026-08-22T19:50:00Z",
            "format": "tinylora_step5_bucket_versioning_receipt_v1",
            "region": "us-west-2",
            "status": "Enabled",
        },
    )
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
    packet["durability"].update(  # type: ignore[union-attr]
        {
            "account_id": "123456789012",
            "bucket_versioning_receipt_sha256": receipt_sha,
            "region": "us-west-2",
        }
    )
    packet["controller_prerequisites"][  # type: ignore[index]
        "input_url_controller_receipt_sha256"
    ] = input_receipt_sha
    packet["commands"]["remote"] = "run sha256:" + "a" * 64  # type: ignore[index]
    packet["commands"]["input_url_preparation"] = packet["commands"][  # type: ignore[index]
        "input_url_preparation"
    ].replace("${AWS_ACCOUNT_ID}", "123456789012").replace(  # type: ignore[union-attr]
        "${AWS_REGION}", "us-west-2"
    )
    lifecycle = packet["commands"]["lifecycle_argv"]  # type: ignore[index]
    for marker, replacement in (
        ("${OFFER_ID}", "123"),
        ("${RUNTIME_IMAGE}", "ghcr.io/example/step5@sha256:" + "a" * 64),
        ("${RUNTIME_IMAGE_DIGEST}", "sha256:" + "a" * 64),
        ("${APPROVED_HOURLY_PRICE_USD}", "0.25"),
        ("${APPROVED_MAX_COST_USD}", "1.5"),
    ):
        lifecycle = [str(item).replace(marker, replacement) for item in lifecycle]
    remote_index = lifecycle.index("--remote-command") + 1
    lifecycle[remote_index] = packet["commands"]["remote"]  # type: ignore[index]
    packet["commands"]["lifecycle_argv"] = lifecycle  # type: ignore[index]
    packet["remaining_substitutions"] = []
    packet["packet_sha256"] = canonical_sha256(packet)
    with pytest.raises(LaunchPacketError, match="receipt"):
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
