from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path

import boto3
import pytest
from botocore.config import Config

from intelligent_liars.step5_artifact_store import (
    LIFECYCLE_ARTIFACT_MANIFEST_FORMAT,
    build_lifecycle_artifact_manifest,
)
from intelligent_liars.step5_artifact_presigner import (
    build_receipt,
    canonical_bytes,
    sha256_bytes,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_vast_step5_instance.py"
SPEC = importlib.util.spec_from_file_location("run_vast_step5_instance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _instance(identifier: int, label: str, state: str = "running") -> dict[str, object]:
    return {"id": identifier, "label": label, "actual_status": state}


def test_labels_are_unique_and_project_scoped():
    first = MODULE.unique_label("run 5", "dim/64", 1)
    second = MODULE.unique_label("run 5", "dim/64", 1)
    assert first.startswith("codex-vast-tinylora-step5-run-5-dim-64-a1-")
    assert first != second
    assert MODULE.resume_label_matches(first, "run 5", "dim/64", 1) is True
    assert MODULE.resume_label_matches(first, "run 5", "another-arm", 1) is False


def test_worker_cap_counts_running_and_stopped_workers():
    workers = [
        _instance(index, f"codex-vast-tinylora-step5-run-arm-a{index}", state)
        for index, state in enumerate(("running", "stopped", "running"), start=1)
    ]
    with pytest.raises(RuntimeError, match="3/3"):
        MODULE.enforce_worker_cap(workers, max_workers=3, resume_instance_id=None)


def test_worker_cap_refuses_configuration_above_hard_cap():
    with pytest.raises(ValueError, match="between 1 and 3"):
        MODULE.enforce_worker_cap([], max_workers=4, resume_instance_id=None)


def test_existing_labeled_stopped_worker_can_be_resumed_at_cap():
    workers = [
        _instance(index, f"codex-vast-tinylora-step5-run-arm-a{index}", "stopped")
        for index in range(1, 4)
    ]
    MODULE.enforce_worker_cap(workers, max_workers=3, resume_instance_id="2")


def test_resume_refuses_already_over_cap_inventory():
    workers = [
        _instance(index, f"codex-vast-tinylora-step5-run-arm-a{index}", "stopped")
        for index in range(1, 5)
    ]
    with pytest.raises(RuntimeError, match="already exceeds"):
        MODULE.enforce_worker_cap(workers, max_workers=3, resume_instance_id="2")


def test_unlabeled_instance_cannot_be_claimed_as_resume_worker():
    with pytest.raises(ValueError, match="not a labeled Step 5"):
        MODULE.enforce_worker_cap(
            [_instance(7, "somebody-elses-worker", "stopped")],
            max_workers=3,
            resume_instance_id="7",
        )


def test_created_instance_can_be_recovered_only_by_exact_unique_label():
    inventory = [
        _instance(7, "codex-vast-tinylora-step5-run-arm-a1-unique"),
        _instance(8, "codex-vast-tinylora-step5-run-arm-a1-other"),
    ]
    assert MODULE.recover_created_instance_id(
        inventory,
        "codex-vast-tinylora-step5-run-arm-a1-unique",
    ) == "7"
    with pytest.raises(RuntimeError, match="uniquely recover"):
        MODULE.recover_created_instance_id(inventory, "missing")


def test_created_instance_id_poll_handles_eventual_inventory(
    monkeypatch: pytest.MonkeyPatch,
):
    inventories = iter(
        [
            [],
            [],
            [_instance(77, "codex-vast-tinylora-step5-run-arm-a1-unique")],
        ]
    )
    monkeypatch.setattr(MODULE, "show_instances", lambda _vastai: next(inventories))
    assert MODULE.poll_created_instance_id(
        "vastai",
        "codex-vast-tinylora-step5-run-arm-a1-unique",
        wait_seconds=2,
        poll_seconds=0,
    ) == "77"


def test_host_qualification_rejects_below_measured_minimum():
    result = MODULE.parse_host_qualification('{"download_mbps": 79.9}', 80.0)
    assert result["qualified"] is False
    assert result["minimum_download_mbps"] == 80.0


def test_host_qualification_accepts_nested_qualifier_contract():
    result = MODULE.parse_host_qualification(
        '{"decision": {"median_effective_mbps": 125.5, "accepted": true}}',
        100.0,
    )
    assert result["qualified"] is True
    assert result["download_mbps"] == 125.5


def test_cleanup_stops_until_both_durable_and_fetched_artifacts_verify():
    assert MODULE.cleanup_action(
        durable_verified=False,
        fetched_verified=True,
        workload_started=True,
    ) == "stop"
    assert MODULE.cleanup_action(
        durable_verified=True,
        fetched_verified=False,
        workload_started=True,
    ) == "stop"


def test_cleanup_destroys_after_both_verifications():
    assert MODULE.cleanup_action(
        durable_verified=True,
        fetched_verified=True,
        workload_started=True,
    ) == "destroy"


def test_cleanup_destroys_if_workload_never_started():
    assert MODULE.cleanup_action(
        durable_verified=False,
        fetched_verified=False,
        workload_started=False,
    ) == "destroy"


def test_preexisting_worker_is_stopped_even_before_workload_starts():
    assert MODULE.cleanup_action(
        durable_verified=False,
        fetched_verified=False,
        workload_started=False,
        preexisting_worker=True,
    ) == "stop"


def test_only_host_loss_allows_replacement():
    assert MODULE.replacement_allowed("host_loss") is True
    assert MODULE.replacement_allowed("software_failure") is False
    assert MODULE.replacement_allowed("unknown") is False


def test_remote_host_loss_claim_needs_controller_corroboration():
    diagnosis = {"classification": "host_loss"}
    running = _instance(1, "x", "running")
    offline = _instance(1, "x", "offline")
    assert MODULE.controller_corroborated_classification(diagnosis, running) == "unknown"
    assert MODULE.controller_corroborated_classification(diagnosis, offline) == "host_loss"


def test_unreachable_ssh_is_not_alone_enough_to_call_host_loss():
    assert MODULE.classify_unreachable(_instance(1, "x", "running")) == "unknown"
    assert MODULE.classify_unreachable(_instance(1, "x", "offline")) == "host_loss"


def test_vast_exited_state_counts_as_stopped_not_host_loss():
    instance = _instance(1, "x", "exited")
    assert MODULE.state_matches(MODULE.instance_state(instance), "stopped") is True
    assert MODULE.classify_unreachable(instance) == "unknown"


def test_diagnostic_classification_is_fail_closed():
    with pytest.raises(ValueError, match="Unsupported failure"):
        MODULE.classify_diagnostic('{"classification": "replace_it"}')


def test_artifact_manifest_verifies_hashes_and_sizes(tmp_path: Path):
    artifact = tmp_path / "checkpoints" / "step_25.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"durable checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = build_lifecycle_artifact_manifest(
        run_id="run-123",
        durable_uri="s3://bucket/run.tar",
        durable_bytes=len(b"durable checkpoint"),
        durable_sha256=digest,
        files=[
            {
                "path": "checkpoints/step_25.pt",
                "bytes": artifact.stat().st_size,
                "sha256": digest,
            }
        ],
    )
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    verified = MODULE.verify_artifact_manifest(
        tmp_path,
        "artifact_manifest.json",
        frozenset({"checkpoints/step_25.pt"}),
        expected_run_id="run-123",
    )
    assert verified["files"]["checkpoints/step_25.pt"]["sha256"] == digest


def test_artifact_manifest_rejects_path_traversal(tmp_path: Path):
    manifest = {
        "format": LIFECYCLE_ARTIFACT_MANIFEST_FORMAT,
        "run_id": "run-123",
        "artifact_set_id": "0" * 64,
        "files": [{"path": "../escape", "sha256": "0" * 64, "bytes": 1}],
        "durable_object": {
            "uri": "s3://bucket/run.tar",
            "bytes": 1,
            "sha256": "0" * 64,
        },
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe artifact path"):
        MODULE.verify_artifact_manifest(
            tmp_path,
            "artifact_manifest.json",
            frozenset({"../escape"}),
        )


def test_artifact_manifest_explicitly_rejects_ambiguous_legacy_v1(tmp_path: Path):
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "format": "tinylora_step5_artifact_manifest_v1",
                "files": [],
                "durable_object": {},
            }
        )
    )
    with pytest.raises(ValueError, match="legacy ambiguous v1"):
        MODULE.verify_artifact_manifest(
            tmp_path,
            "artifact_manifest.json",
            frozenset({"result.json"}),
        )


def test_artifact_manifest_rejects_hash_mismatch(tmp_path: Path):
    (tmp_path / "result.json").write_text("{}")
    manifest = build_lifecycle_artifact_manifest(
        run_id="run-123",
        durable_uri="s3://bucket/run.tar",
        durable_bytes=2,
        durable_sha256="0" * 64,
        files=[{"path": "result.json", "sha256": "0" * 64, "bytes": 2}],
    )
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.verify_artifact_manifest(
            tmp_path,
            "artifact_manifest.json",
            frozenset({"result.json"}),
        )


def test_artifact_manifest_rejects_unexpected_fetched_file(tmp_path: Path):
    (tmp_path / "result.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    manifest = build_lifecycle_artifact_manifest(
        run_id="run-123",
        durable_uri="s3://bucket/run.tar",
        durable_bytes=2,
        durable_sha256=digest,
        files=[{"path": "result.json", "sha256": digest, "bytes": 2}],
    )
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "unlisted.log").write_text("surprise")
    with pytest.raises(ValueError, match="inventory mismatch"):
        MODULE.verify_artifact_manifest(
            tmp_path,
            "artifact_manifest.json",
            frozenset({"result.json"}),
        )


def test_artifact_manifest_requires_controller_frozen_durable_uri(tmp_path: Path):
    (tmp_path / "result.json").write_text("{}")
    digest = hashlib.sha256(b"{}").hexdigest()
    manifest = build_lifecycle_artifact_manifest(
        run_id="run-123",
        durable_uri="s3://bucket/wrong.tar",
        durable_bytes=2,
        durable_sha256=digest,
        files=[{"path": "result.json", "sha256": digest, "bytes": 2}],
    )
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="controller-frozen"):
        MODULE.verify_artifact_manifest(
            tmp_path,
            "artifact_manifest.json",
            frozenset({"result.json"}),
            expected_run_id="run-123",
            expected_durable_uri="s3://bucket/intended.tar",
        )


def test_durable_archive_must_contain_exact_manifest_bytes(tmp_path: Path):
    archive_path = tmp_path / "durable.tar"
    member = tmp_path / "result.json"
    member.write_text("{}")
    with tarfile.open(archive_path, "w") as archive:
        archive.add(member, arcname="result.json")
    digest = hashlib.sha256(b"{}").hexdigest()
    MODULE.verify_durable_archive(
        archive_path,
        {"result.json": {"bytes": 2, "sha256": digest}},
    )
    with pytest.raises(ValueError, match="inventory"):
        MODULE.verify_durable_archive(
            archive_path,
            {"other.json": {"bytes": 2, "sha256": digest}},
        )


def test_controller_heads_and_roundtrips_exact_s3_version(monkeypatch: pytest.MonkeyPatch):
    body = b"durable object"
    digest = hashlib.sha256(body).hexdigest()
    checksum = base64.b64encode(bytes.fromhex(digest)).decode()

    def fake_run(command, **_kwargs):
        if "head-object" in command:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "VersionId": "version-7",
                        "ContentLength": len(body),
                        "ChecksumSHA256": checksum,
                    }
                )
            )
        output_path = Path(command[-3])
        output_path.write_bytes(body)
        return SimpleNamespace(stdout="{}")

    monkeypatch.setattr(MODULE, "run", fake_run)
    result = MODULE.controller_verify_s3_object(
        "aws",
        {"uri": "s3://bucket/key", "bytes": len(body), "sha256": digest},
    )
    assert result["verified"] is True
    assert result["version_id"] == "version-7"


def test_controller_rejects_s3_head_without_version(monkeypatch: pytest.MonkeyPatch):
    body = b"durable object"
    digest = hashlib.sha256(body).hexdigest()

    def fake_run(_command, **_kwargs):
        return SimpleNamespace(
            stdout=json.dumps(
                {"ContentLength": len(body), "ChecksumSHA256": "unused"}
            )
        )

    monkeypatch.setattr(MODULE, "run", fake_run)
    with pytest.raises(ValueError, match="version id"):
        MODULE.controller_verify_s3_object(
            "aws",
            {"uri": "s3://bucket/key", "bytes": len(body), "sha256": digest},
        )


def test_execute_requires_immutable_image_and_specific_cost_approval():
    args = argparse.Namespace(
        execute=True,
        confirmed_cost_approval=True,
        approved_hourly_price=0.2,
        approved_max_cost=2.0,
        offer_id="123",
        resume_instance_id=None,
        image="repo/image:latest",
        max_software_resumes=1,
        minimum_download_mbps=100.0,
        max_workers=3,
    )
    with pytest.raises(ValueError, match="immutable image digest"):
        MODULE.require_execute_contract(args)


def test_dry_run_does_not_need_cost_approval():
    args = argparse.Namespace(execute=False, max_workers=3)
    MODULE.require_execute_contract(args)


def test_dry_run_also_refuses_more_than_three_workers():
    args = argparse.Namespace(execute=False, max_workers=4)
    with pytest.raises(ValueError, match="between 1 and 3"):
        MODULE.require_execute_contract(args)


def test_execute_contract_requires_both_new_provisioned_files(tmp_path: Path):
    args = argparse.Namespace(
        execute=True,
        max_workers=3,
        confirmed_cost_approval=True,
        approved_hourly_price=0.2,
        approved_max_cost=2.0,
        offer_id="123",
        resume_instance_id=None,
        image="repo/image@sha256:" + "a" * 64,
        max_software_resumes=1,
        minimum_download_mbps=100.0,
        wait_seconds=10,
        state_wait_seconds=10,
        disk=100,
        artifact_put_url_file=tmp_path / "artifact-url",
        artifact_put_receipt=tmp_path / "artifact-receipt",
        artifact_put_receipt_sha256="a" * 64,
        approved_at="2026-08-22T00:00:00Z",
        artifact_put_max_approval_age_seconds=3600,
        expected_durable_uri="s3://bucket/key",
        input_url_manifest_url_file=None,
        controller_public_key=None,
        controller_public_key_sha256=None,
    )
    with pytest.raises(ValueError, match="hydration/host-gate URLs"):
        MODULE.require_execute_contract(args)


def test_artifact_put_contract_is_receipt_bound_and_mode_protected(tmp_path: Path):
    bucket = "step5-artifacts-example"
    key = "immutable/run-7/artifacts.tar"
    region = "us-west-2"
    session = boto3.session.Session(
        aws_access_key_id="AKIA" + "A" * 16,
        aws_secret_access_key="b" * 40,
        region_name=region,
    )
    client = session.client(
        "s3",
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )
    url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "IfNoneMatch": "*"},
        ExpiresIn=3600,
        HttpMethod="PUT",
    )
    signed_at = datetime.strptime(
        url.split("X-Amz-Date=", 1)[1].split("&", 1)[0], "%Y%m%dT%H%M%SZ"
    ).replace(tzinfo=timezone.utc)
    approved_at = (signed_at - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    receipt = build_receipt(
        url=url,
        bucket=bucket,
        key=key,
        region=region,
        account_id="123456789012",
        approved_at=approved_at,
        generated_at=signed_at,
        expiry_seconds=3600,
    )
    url_path = tmp_path / "artifact-url"
    receipt_path = tmp_path / "artifact-receipt.json"
    url_path.write_bytes((url + "\n").encode())
    receipt_bytes = canonical_bytes(receipt)
    receipt_path.write_bytes(receipt_bytes)
    url_path.chmod(0o600)
    receipt_path.chmod(0o600)
    args = argparse.Namespace(
        artifact_put_url_file=url_path,
        artifact_put_receipt=receipt_path,
        artifact_put_receipt_sha256=sha256_bytes(receipt_bytes),
        approved_at=approved_at,
        expected_durable_uri=f"s3://{bucket}/{key}",
        artifact_put_max_approval_age_seconds=3600,
    )
    verified, stable_url = MODULE.validate_artifact_put_contract(args)
    assert verified["method"] == "PUT"
    assert stable_url == (url + "\n").encode()
    with pytest.raises(ValueError, match="stale"):
        MODULE.revalidate_artifact_put_freshness(
            args,
            receipt,
            stable_url,
            current=datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
            + timedelta(hours=2),
        )

    receipt_path.chmod(0o644)
    with pytest.raises(ValueError, match="receipt file must have mode 0600"):
        MODULE.validate_artifact_put_contract(args)


def test_artifact_put_contract_rejects_frozen_receipt_hash_drift(tmp_path: Path):
    url_path = tmp_path / "artifact-url"
    receipt_path = tmp_path / "artifact-receipt.json"
    url_path.write_text("https://example.test/private\n")
    receipt_path.write_text("{}\n")
    url_path.chmod(0o600)
    receipt_path.chmod(0o600)
    args = argparse.Namespace(
        artifact_put_url_file=url_path,
        artifact_put_receipt=receipt_path,
        artifact_put_receipt_sha256="a" * 64,
        approved_at="2026-08-22T00:00:00Z",
        expected_durable_uri="s3://bucket/key",
        artifact_put_max_approval_age_seconds=3600,
    )
    with pytest.raises(ValueError, match="frozen SHA-256"):
        MODULE.validate_artifact_put_contract(args)


@pytest.mark.parametrize("recovery_path", ["qualification", "workload"])
def test_same_instance_recovery_refuses_stale_artifact_authorization(
    recovery_path: str, monkeypatch: pytest.MonkeyPatch
):
    starts: list[list[str]] = []

    def stale(*_args, **_kwargs):
        raise ValueError("artifact approval is stale")

    def unexpected_run(command, **_kwargs):
        starts.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE, "revalidate_artifact_put_freshness", stale)
    monkeypatch.setattr(MODULE, "run", unexpected_run)
    with pytest.raises(ValueError, match="stale"):
        MODULE.restart_stopped_instance(
            argparse.Namespace(vastai="vastai", recovery_path=recovery_path),
            target_id="worker-7",
            artifact_put_receipt={},
            artifact_put_url_bytes=b"https://expired.example\n",
            cost_deadline=time.monotonic() + 60,
        )
    assert starts == []


def test_both_internal_restart_paths_use_freshness_guard():
    source = inspect.getsource(MODULE.main)
    assert source.count("restart_stopped_instance(") == 2


def test_checkpoint_controller_command_keeps_private_key_local(tmp_path: Path):
    args = argparse.Namespace(
        checkpoint_controller_uv="uv",
        checkpoint_controller_script=tmp_path / "controller.py",
        checkpoint_remote_exchange_dir="/workspace/checkpoint-bridge",
        checkpoint_bucket="bucket-name",
        checkpoint_prefix="step5/checkpoints/run",
        checkpoint_controller_private_key=tmp_path / "private.pem",
        controller_public_key=tmp_path / "public.pem",
    )
    command = MODULE.checkpoint_controller_command(
        args,
        host="worker.example",
        port="2222",
        known_hosts=tmp_path / "known-hosts",
        ready_file=tmp_path / "ready.json",
        idle_timeout_seconds=900,
    )
    rendered = json.dumps(command)
    assert "private.pem" in rendered
    assert "worker.example" in rendered
    assert all("BEGIN PRIVATE KEY" not in value for value in command)


def test_remote_workload_fails_when_checkpoint_controller_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Worker:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def kill(self):
            self.returncode = -9

    class Sidecar:
        def poll(self):
            return 7

    worker = Worker()
    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *_args, **_kwargs: worker)
    result = MODULE.remote_run_with_checkpoint_controller(
        "host",
        "22",
        tmp_path / "known-hosts",
        "run worker",
        sidecar={"process": Sidecar()},
        timeout=10,
    )
    assert result.returncode == 125
    assert "controller exited" in result.stderr


def test_checkpoint_controller_must_publish_bound_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Process:
        pid = 42
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            del timeout
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    controller_script = tmp_path / "controller.py"
    controller_script.write_text("#!/usr/bin/env python3\n")
    args = argparse.Namespace(
        checkpoint_controller_ready_seconds=1,
        checkpoint_bucket="bucket-name",
        checkpoint_prefix="step5/checkpoints/run",
        checkpoint_remote_exchange_dir="/workspace/checkpoint-bridge",
        checkpoint_controller_script=controller_script,
        checkpoint_controller_script_sha256=hashlib.sha256(
            controller_script.read_bytes()
        ).hexdigest(),
    )

    def fake_command(_args, **kwargs):
        kwargs["ready_file"].write_text(
            json.dumps(
                {
                    "format": "tinylora_step5_checkpoint_controller_ready_v1",
                    "pid": 42,
                    "bucket": "bucket-name",
                    "prefix": "step5/checkpoints/run",
                    "remote_exchange_dir": "/workspace/checkpoint-bridge",
                    "ssh_host": "host",
                    "ssh_port": "22",
                    "versioning": "Enabled",
                    "ready_at": "2026-08-22T12:00:00Z",
                }
            )
        )
        kwargs["ready_file"].chmod(0o600)
        return ["controller"]

    monkeypatch.setattr(MODULE, "checkpoint_controller_command", fake_command)
    monkeypatch.setattr(
        MODULE,
        "checkpoint_controller_key_bytes",
        lambda _args: (b"private", b"public"),
    )
    monkeypatch.setattr(MODULE.subprocess, "Popen", lambda *_args, **_kwargs: process)
    state = MODULE.start_checkpoint_controller(
        args,
        host="host",
        port="22",
        known_hosts=tmp_path / "known-hosts",
        log_path=tmp_path / "controller.log",
        idle_timeout_seconds=900,
    )
    assert state["ready"]["pid"] == 42
    status = MODULE.stop_checkpoint_controller(state)
    assert status["unexpected_exit"] is False
    assert status["returncode"] == -15


def test_host_gate_secret_uses_fixed_nonoverridable_path():
    assert MODULE.HOST_GATE_URL_REMOTE_PATH == "/run/secrets/step5-host-gate-url"
    assert MODULE.INPUT_URL_MANIFEST_REMOTE_PATH != MODULE.HOST_GATE_URL_REMOTE_PATH


def test_host_gate_and_hydration_urls_must_be_distinct(tmp_path: Path):
    hydration = tmp_path / "hydration-url"
    host_gate = tmp_path / "host-gate-url"
    hydration.write_text("https://example.test/hydration\n")
    host_gate.write_text("https://example.test/hydration")
    hydration.chmod(0o600)
    host_gate.chmod(0o600)
    args = argparse.Namespace(
        input_url_manifest_url_file=hydration,
        host_gate_url_file=host_gate,
    )
    with pytest.raises(ValueError, match="distinct URLs"):
        MODULE.validate_distinct_url_sources(args)
    host_gate.write_text("https://example.test/host-gate\n")
    MODULE.validate_distinct_url_sources(args)


def test_private_url_sync_never_places_secret_in_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    secret_value = "https://example.test/object?X-Amz-Signature=private"
    source = tmp_path / "url"
    source.write_text(secret_value)
    source.chmod(0o600)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command)
        or SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    MODULE.sync_private_url_file(
        source,
        "/run/secrets/step5-input-url-manifest-url",
        "host",
        "22",
        tmp_path / "known-hosts",
        10,
    )
    assert secret_value not in json.dumps(commands)
    assert any("mktemp" in part for command in commands for part in command)


def test_public_key_sync_is_hash_bound_and_uses_fixed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    public = tmp_path / "public.pem"
    private = tmp_path / "private.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    digest = hashlib.sha256(public.read_bytes()).hexdigest()
    commands: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "openssl":
            return real_run(command, **kwargs)
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        fake_run,
    )
    MODULE.sync_hash_bound_public_key(
        public,
        digest,
        "host",
        "22",
        tmp_path / "known-hosts",
        10,
    )
    rendered = json.dumps(commands)
    assert MODULE.CONTROLLER_PUBLIC_KEY_REMOTE_PATH in rendered
    assert digest in rendered
    public.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        MODULE.sync_hash_bound_public_key(
            public,
            digest,
            "host",
            "22",
            tmp_path / "known-hosts",
            10,
        )
    private_digest = hashlib.sha256(private.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="public PEM"):
        MODULE.sync_hash_bound_public_key(
            private,
            private_digest,
            "host",
            "22",
            tmp_path / "known-hosts",
            10,
        )


def test_provisioning_reader_rejects_final_and_parent_symlinks(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    secret = real / "url"
    secret.write_text("https://example.test/private")
    secret.chmod(0o600)
    final_link = tmp_path / "final-link"
    final_link.symlink_to(secret)
    with pytest.raises(ValueError, match="symlinks"):
        MODULE.read_file_without_symlinks(final_link)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        MODULE.read_file_without_symlinks(parent_link / "url")


def _write_source_manifest(repository: Path, relative: str) -> Path:
    path = repository / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = repository / "step5_sources.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_source_manifest_v1",
                "files": [{"path": relative, "sha256": digest}],
            }
        )
    )
    return manifest_path


def test_source_allowlist_verifies_and_bundle_contains_only_allowlisted_files(
    tmp_path: Path,
):
    (tmp_path / "safe.py").write_text("print('safe')\n")
    (tmp_path / ".env").write_text("SECRET=do-not-copy\n")
    manifest_path = _write_source_manifest(tmp_path, "safe.py")
    verified = MODULE.verify_source_manifest(tmp_path, manifest_path)
    bundle = tmp_path / "bundle.tar.gz"
    receipt = MODULE.build_source_bundle(tmp_path, verified, bundle)
    with tarfile.open(bundle, "r:gz") as archive:
        assert set(archive.getnames()) == {"safe.py", "SOURCE_BUNDLE_MANIFEST.json"}
    assert receipt["files"] == 1


def test_source_allowlist_rejects_secret_like_path(tmp_path: Path):
    (tmp_path / ".env").write_text("SECRET=nope\n")
    manifest_path = _write_source_manifest(tmp_path, ".env")
    with pytest.raises(ValueError, match="Secret-like"):
        MODULE.verify_source_manifest(tmp_path, manifest_path)


def test_source_allowlist_rejects_symlink(tmp_path: Path):
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n")
    (tmp_path / "linked.py").symlink_to(outside)
    manifest_path = _write_source_manifest(tmp_path, "linked.py")
    with pytest.raises(ValueError, match="symlink"):
        MODULE.verify_source_manifest(tmp_path, manifest_path)


def test_source_allowlist_rejects_sealed_audit(tmp_path: Path):
    audit = tmp_path / "corpora" / "tinylora" / "pilot_v1" / "audit.jsonl"
    audit.parent.mkdir(parents=True)
    audit.write_text('{"sealed": true}\n')
    manifest_path = _write_source_manifest(
        tmp_path,
        "corpora/tinylora/pilot_v1/audit.jsonl",
    )
    with pytest.raises(ValueError, match="Sealed audit"):
        MODULE.verify_source_manifest(tmp_path, manifest_path)


def test_source_allowlist_rejects_sealed_audit_content_under_renamed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "renamed.jsonl").write_text('{"sealed": true}\n')
    digest = hashlib.sha256((tmp_path / "renamed.jsonl").read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "SEALED_AUDIT_SHA256", digest)
    manifest_path = _write_source_manifest(tmp_path, "renamed.jsonl")
    with pytest.raises(ValueError, match="content hash"):
        MODULE.verify_source_manifest(tmp_path, manifest_path)


def test_commands_reject_long_lived_credentials_and_presigned_url():
    with pytest.raises(ValueError, match="long-lived credentials"):
        MODULE.reject_credential_bearing_commands(
            ["AWS_SECRET_ACCESS_KEY=not-allowed python upload.py"]
        )
    with pytest.raises(ValueError, match="long-lived credentials"):
        MODULE.reject_credential_bearing_commands(
            ["curl 'https://example.test/object?X-Amz-Expires=900&X-Amz-Signature=abc'"]
        )


def test_commands_reject_bearer_headers_and_source_rejects_auth_files(tmp_path: Path):
    with pytest.raises(ValueError, match="long-lived credentials"):
        MODULE.reject_credential_bearing_commands(
            ["curl -H 'Authorization: Bearer abc' https://example.test"]
        )
    (tmp_path / ".netrc").write_text("machine example.test\n")
    manifest_path = _write_source_manifest(tmp_path, ".netrc")
    with pytest.raises(ValueError, match="Secret-like"):
        MODULE.verify_source_manifest(tmp_path, manifest_path)


def test_remote_artifact_directory_is_restricted_to_workspace():
    assert MODULE.validate_remote_artifact_dir("/workspace/results/arm") == (
        "/workspace/results/arm"
    )
    for unsafe in (
        "results/arm",
        "/workspace/../root",
        "/tmp/results",
        "/workspace/results;touch-pwned",
    ):
        with pytest.raises(ValueError, match="under /workspace"):
            MODULE.validate_remote_artifact_dir(unsafe)


def test_expected_artifact_inventory_is_controller_frozen(tmp_path: Path):
    inventory = tmp_path / "expected.json"
    inventory.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_expected_artifact_inventory_v1",
                "files": ["result.json", "checkpoints/step_25.pt"],
            }
        )
    )
    verified = MODULE.verify_expected_artifact_inventory(inventory)
    assert verified["files"] == frozenset(
        {"result.json", "checkpoints/step_25.pt"}
    )


def test_ssh_uses_only_isolated_pinned_known_hosts(tmp_path: Path):
    command = MODULE.ssh_prefix("host", "22", tmp_path / "known_hosts")
    joined = " ".join(str(value) for value in command)
    assert "StrictHostKeyChecking=yes" in joined
    assert f"UserKnownHostsFile={tmp_path / 'known_hosts'}" in joined
    assert "StrictHostKeyChecking=no" not in joined


def test_execute_contract_rejects_nan_cost_and_speed():
    args = argparse.Namespace(
        execute=True,
        max_workers=3,
        confirmed_cost_approval=True,
        approved_hourly_price=math.nan,
        approved_max_cost=2.0,
        offer_id="123",
        resume_instance_id=None,
        image="repo/image@sha256:" + "a" * 64,
        max_software_resumes=1,
        minimum_download_mbps=math.nan,
        wait_seconds=10,
        state_wait_seconds=10,
        disk=100,
    )
    with pytest.raises(ValueError, match="finite and positive"):
        MODULE.require_execute_contract(args)


def test_offer_price_cannot_exceed_specific_approval():
    MODULE.enforce_price(0.20, 0.20)
    with pytest.raises(RuntimeError, match="above approved"):
        MODULE.enforce_price(0.21, 0.20)


def test_stop_for_recovery_always_issues_provider_stop(
    monkeypatch: pytest.MonkeyPatch,
):
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(MODULE, "run", fake_run)
    monkeypatch.setattr(
        MODULE,
        "wait_for_state",
        lambda *_args: (True, "stopped", []),
    )
    result = MODULE.stop_for_recovery("vastai", "77", 1)
    assert commands == [["vastai", "stop", "instance", "77", "--retry", "3", "--raw"]]
    assert result["verified"] is True


def test_main_dry_run_performs_no_local_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    (tmp_path / "safe.py").write_text("print('safe')\n")
    source_manifest = _write_source_manifest(tmp_path, "safe.py")
    expected_inventory = tmp_path / "expected.json"
    expected_inventory.write_text(
        json.dumps(
            {
                "format": "tinylora_step5_expected_artifact_inventory_v1",
                "files": ["result.json"],
            }
        )
    )
    metadata = tmp_path / "metadata" / "dry-run.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--vastai",
            "/fake/vastai",
            "--offer-id",
            "123",
            "--run-id",
            "run-5",
            "--candidate",
            "arm-a",
            "--repo",
            str(tmp_path),
            "--source-manifest",
            str(source_manifest),
            "--expected-artifact-inventory",
            str(expected_inventory),
            "--fetch-dir",
            str(tmp_path / "fetch"),
            "--metadata",
            str(metadata),
            "--image",
            "repo/image:dry-run",
            "--remote-command",
            "python train.py",
            "--remote-artifact-dir",
            "/workspace/results/arm-a",
            "--host-qualification-command",
            "python qualify.py",
            "--diagnostic-command",
            "python diagnose.py",
        ],
    )
    assert MODULE.main() == 0
    assert not metadata.exists()
    assert "controller S3 HEAD" in capsys.readouterr().out
