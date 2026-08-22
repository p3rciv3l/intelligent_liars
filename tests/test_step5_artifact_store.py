from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.step5_artifact_store import (
    ArtifactContractError,
    build_artifact_manifest,
    build_final_receipt,
    build_head_receipt,
    build_roundtrip_receipt,
    sha256_file,
    validate_artifact_manifest,
    validate_final_receipt,
    validate_head_receipt,
    validate_roundtrip_receipt,
)


def _artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    for role in ("inputs", "logs", "checkpoints", "results"):
        (root / role).mkdir(parents=True, exist_ok=True)
    (root / "inputs" / "plan.json").write_text('{"frozen":true}\n')
    (root / "logs" / "worker.log").write_text("started\n")
    (root / "checkpoints" / "step-000010.pt").write_bytes(b"checkpoint")
    (root / "results" / "metrics.json").write_text('{"loss":1.5}\n')
    return root


def test_manifest_is_deterministic_and_layout_neutral(tmp_path: Path):
    root = _artifact_root(tmp_path)
    first = build_artifact_manifest(root, run_id="run-123")
    second = build_artifact_manifest(root, run_id="run-123")

    assert first == second
    assert validate_artifact_manifest(first) == first
    assert [item["logical_path"] for item in first["artifacts"]] == [
        "checkpoints/step-000010.pt",
        "inputs/plan.json",
        "logs/worker.log",
        "results/metrics.json",
    ]
    encoded = json.dumps(first, sort_keys=True)
    assert "s3://" not in encoded
    assert str(root) not in encoded
    assert len(first["artifact_set_id"]) == 64


def test_manifest_rejects_missing_role_and_unsafe_logical_path(tmp_path: Path):
    root = _artifact_root(tmp_path)
    (root / "logs" / "worker.log").unlink()
    with pytest.raises(ArtifactContractError, match="logs"):
        build_artifact_manifest(root, run_id="run-123")

    manifest = build_artifact_manifest(_artifact_root(tmp_path), run_id="run-123")
    manifest["artifacts"][0]["logical_path"] = "../escaped.pt"
    with pytest.raises(ArtifactContractError, match="logical_path"):
        validate_artifact_manifest(manifest)


def test_manifest_rejects_symlinks(tmp_path: Path):
    root = _artifact_root(tmp_path)
    (root / "results" / "linked.json").symlink_to(root / "results" / "metrics.json")
    with pytest.raises(ArtifactContractError, match="symlink"):
        build_artifact_manifest(root, run_id="run-123")


def test_head_receipt_records_success_and_detects_mismatch(tmp_path: Path):
    manifest = build_artifact_manifest(_artifact_root(tmp_path), run_id="run-123")
    artifact = manifest["artifacts"][0]
    receipt = build_head_receipt(
        manifest,
        logical_path=artifact["logical_path"],
        object_ref="provider-object-opaque-17",
        observed_size_bytes=artifact["size_bytes"],
        observed_sha256=artifact["sha256"],
        etag="opaque-etag",
        observed_at="2026-08-22T12:00:00Z",
    )
    assert receipt["verified"] is True
    assert validate_head_receipt(receipt, manifest) == receipt

    mismatch = build_head_receipt(
        manifest,
        logical_path=artifact["logical_path"],
        object_ref="provider-object-opaque-17",
        observed_size_bytes=artifact["size_bytes"] + 1,
        observed_sha256=artifact["sha256"],
        observed_at="2026-08-22T12:00:00Z",
    )
    assert mismatch["verified"] is False
    assert mismatch["mismatches"] == ["size_bytes"]
    assert validate_head_receipt(mismatch, manifest) == mismatch


def test_roundtrip_receipt_hashes_downloaded_bytes(tmp_path: Path):
    root = _artifact_root(tmp_path)
    manifest = build_artifact_manifest(root, run_id="run-123")
    artifact = next(
        item for item in manifest["artifacts"] if item["logical_path"].startswith("results/")
    )
    receipt = build_roundtrip_receipt(
        manifest,
        logical_path=artifact["logical_path"],
        object_ref="opaque-result-object",
        downloaded_path=root / artifact["logical_path"],
        observed_at="2026-08-22T12:01:00Z",
    )
    assert receipt["verified"] is True
    assert receipt["downloaded"]["sha256"] == sha256_file(
        root / artifact["logical_path"]
    )
    assert validate_roundtrip_receipt(receipt, manifest) == receipt


def test_final_receipt_requires_head_and_roundtrip_for_every_artifact(tmp_path: Path):
    root = _artifact_root(tmp_path)
    manifest = build_artifact_manifest(root, run_id="run-123")
    heads = []
    roundtrips = []
    for artifact in manifest["artifacts"]:
        object_ref = f"opaque/{artifact['logical_path']}"
        heads.append(
            build_head_receipt(
                manifest,
                logical_path=artifact["logical_path"],
                object_ref=object_ref,
                observed_size_bytes=artifact["size_bytes"],
                observed_sha256=artifact["sha256"],
                observed_at="2026-08-22T12:00:00Z",
            )
        )
        roundtrips.append(
            build_roundtrip_receipt(
                manifest,
                logical_path=artifact["logical_path"],
                object_ref=object_ref,
                downloaded_path=root / artifact["logical_path"],
                observed_at="2026-08-22T12:01:00Z",
            )
        )

    receipt = build_final_receipt(
        manifest,
        head_receipts=heads,
        roundtrip_receipts=roundtrips,
        completed_at="2026-08-22T12:02:00Z",
    )
    assert receipt["verified"] is True
    assert validate_final_receipt(receipt, manifest) == receipt

    with pytest.raises(ArtifactContractError, match="roundtrip receipt inventory"):
        build_final_receipt(
            manifest,
            head_receipts=heads,
            roundtrip_receipts=roundtrips[:-1],
            completed_at="2026-08-22T12:02:00Z",
        )


def test_final_receipt_refuses_failed_artifact_verification(tmp_path: Path):
    root = _artifact_root(tmp_path)
    manifest = build_artifact_manifest(root, run_id="run-123")
    heads = []
    roundtrips = []
    for index, artifact in enumerate(manifest["artifacts"]):
        heads.append(
            build_head_receipt(
                manifest,
                logical_path=artifact["logical_path"],
                object_ref=f"opaque-{index}",
                observed_size_bytes=artifact["size_bytes"] + (1 if index == 0 else 0),
                observed_sha256=artifact["sha256"],
                observed_at="2026-08-22T12:00:00Z",
            )
        )
        roundtrips.append(
            build_roundtrip_receipt(
                manifest,
                logical_path=artifact["logical_path"],
                object_ref=f"opaque-{index}",
                downloaded_path=root / artifact["logical_path"],
                observed_at="2026-08-22T12:01:00Z",
            )
        )
    with pytest.raises(ArtifactContractError, match="did not verify"):
        build_final_receipt(
            manifest,
            head_receipts=heads,
            roundtrip_receipts=roundtrips,
            completed_at="2026-08-22T12:02:00Z",
        )
