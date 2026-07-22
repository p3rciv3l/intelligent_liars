from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from intelligent_liars.osworld_runtime import (
    AWSLifecycleWatchdog,
    AWS_MINIMUM_RUNTIME_ACTIONS,
    ArtifactManifest,
    BillableVolume,
    InfrastructureTarget,
    ManagedResource,
    VastLaunchPolicy,
    VastLifecycleWatchdog,
    VAST_EXCLUDED_KEY_SCOPES,
    VAST_REQUIRED_KEY_SCOPES,
    VAST_SKILL_SOURCE,
    apply_watchdog_plan,
    locate_vastai_path,
    offline_preflight,
    run_qwen_multimodal_smoke_test,
)
from intelligent_liars.run_control import BudgetDecision, LedgerValidationError


def test_preflight_reports_exact_env_names_and_never_values(tmp_path):
    vastai = tmp_path / "vastai"
    vastai.write_text("#!/bin/sh\n")
    vastai.chmod(0o755)
    environment = {
        "VASTAI_PATH": str(vastai),
        "VAST_API_KEY": "<present>",
        "AWS_ACCESS_KEY_ID": "<present>",
        "AWS_SECRET_ACCESS_KEY": "<present>",
        "AWS_REGION": "<present>",
        "QWEN_ENDPOINT_URL": "<present>",
        "QWEN_ENDPOINT_API_KEY": "<present>",
        "OSWORLD_CLIENT_PASSWORD": "<present>",
        "OSWORLD_ARTIFACT_DESTINATION": "<present>",
    }

    report = offline_preflight(
        environment,
        InfrastructureTarget(),
        VastLaunchPolicy("offer-123", Decimal("0.60"), True),
    ).as_dict()

    assert report["ready"] is True
    assert report["credential_values_included"] is False
    assert report["vastai_cli_available"] is True
    assert "<present>" not in json.dumps(report)
    assert {item["name"] for item in report["environment"]} == {
        "VAST_API_KEY",
        "VASTAI_PATH",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "QWEN_ENDPOINT_URL",
        "QWEN_ENDPOINT_API_KEY",
        "OSWORLD_CLIENT_PASSWORD",
        "OSWORLD_ARTIFACT_DESTINATION",
    }
    statuses = {item["name"]: item for item in report["environment"]}
    assert statuses["AWS_ACCESS_KEY_ID"]["required"] is False
    assert statuses["AWS_SECRET_ACCESS_KEY"]["required"] is False
    assert report["aws_auth_mode"] == "controller-iam-role"
    assert report["authorization"]["vast_required_scopes"] == {
        "misc": "offer search",
        "instance_read": "inventory, status, logs, and volumes",
        "instance_write": "create, manage, and destroy",
    }
    assert report["authorization"]["vast_excluded_scopes"] == [
        "billing_write",
        "user_write",
        "machine_*",
        "payment access",
    ]
    assert report["source"] == {
        "repository": "p3rciv3l/avi-skills",
        "commit": "f453c250b3c5a684f42f6cf6abb97463dddda14c",
        "path": ".agents/skills/vast-gpu-experiments/",
        "scripts": (
            "vast_find_offer.py",
            "vast_run_workload.py",
            "vast_cleanup.py",
        ),
    }
    assert VAST_SKILL_SOURCE == report["source"]


@pytest.mark.parametrize(
    "missing_name",
    (
        "OSWORLD_CLIENT_PASSWORD",
        "QWEN_ENDPOINT_URL",
        "QWEN_ENDPOINT_API_KEY",
    ),
)
def test_preflight_fails_when_password_or_qwen_endpoint_gate_is_missing(
    tmp_path,
    missing_name,
):
    vastai = tmp_path / "vastai"
    vastai.touch(mode=0o755)
    environment = {
        "VASTAI_PATH": str(vastai),
        "VAST_API_KEY": "<present>",
        "AWS_REGION": "<present>",
        "QWEN_ENDPOINT_URL": "<present>",
        "QWEN_ENDPOINT_API_KEY": "<present>",
        "OSWORLD_CLIENT_PASSWORD": "<present>",
        "OSWORLD_ARTIFACT_DESTINATION": "<present>",
    }
    del environment[missing_name]

    report = offline_preflight(
        environment,
        InfrastructureTarget(),
        VastLaunchPolicy("offer-123", Decimal("0.60"), True),
    )
    statuses = {item.name: item for item in report.environment}

    assert report.ready is False
    assert statuses[missing_name].required is True
    assert statuses[missing_name].present is False


def test_bootstrap_aws_keys_are_required_only_in_explicit_bootstrap_mode(tmp_path):
    vastai = tmp_path / "vastai"
    vastai.touch(mode=0o755)
    environment = {
        "VASTAI_PATH": str(vastai),
        "VAST_API_KEY": "<present>",
        "AWS_REGION": "<present>",
        "QWEN_ENDPOINT_URL": "<present>",
        "QWEN_ENDPOINT_API_KEY": "<present>",
        "OSWORLD_CLIENT_PASSWORD": "<present>",
        "OSWORLD_ARTIFACT_DESTINATION": "<present>",
    }
    report = offline_preflight(
        environment,
        InfrastructureTarget(),
        VastLaunchPolicy("offer", Decimal("0.60"), True),
        aws_auth_mode="bootstrap-access-keys",
    )
    statuses = {item.name: item for item in report.environment}

    assert report.ready is False
    assert statuses["AWS_ACCESS_KEY_ID"].required is True
    assert statuses["AWS_SECRET_ACCESS_KEY"].required is True
    assert statuses["AWS_SESSION_TOKEN"].required is False


def test_authorization_metadata_is_least_privilege_and_has_no_claimed_integrations():
    assert set(VAST_REQUIRED_KEY_SCOPES) == {"misc", "instance_read", "instance_write"}
    assert VAST_EXCLUDED_KEY_SCOPES == (
        "billing_write",
        "user_write",
        "machine_*",
        "payment access",
    )
    flattened = {
        action
        for actions in AWS_MINIMUM_RUNTIME_ACTIONS.values()
        for action in actions
    }
    assert "sts:GetCallerIdentity" in flattened
    assert "iam:PassRole" in flattened
    assert "ce:GetCostAndUsage" in flattened
    assert not any("gmail" in action.lower() or "bitwarden" in action.lower() for action in flattened)


def test_linux_vastai_path_is_located_and_nested_kvm_is_rejected(tmp_path):
    executable = tmp_path / "vastai"
    executable.touch(mode=0o755)
    assert locate_vastai_path({"VASTAI_PATH": str(executable)}) == executable.resolve()

    with pytest.raises(ValueError, match="nested KVM"):
        InfrastructureTarget(nested_kvm=True).validate()
    with pytest.raises(ValueError, match="exactly one GPU"):
        VastLaunchPolicy("offer", Decimal("0.60"), True, gpu_count=2)


def test_qwen_smoke_contract_uses_auth_and_openai_multimodal_shape():
    captured = {}

    class FakeHTTPClient:
        def create_chat_completion(self, request, *, bearer_token):
            captured["request"] = request
            captured["authenticated"] = bool(bearer_token)
            return {
                "id": "response-1",
                "model": request.model,
                "choices": [{"message": {"content": "desktop"}}],
            }

    result = run_qwen_multimodal_smoke_test(
        FakeHTTPClient(),
        endpoint_url="https://qwen.invalid/v1",
        bearer_token="<fake>",
        model="Qwen/Qwen3-VL-8B-Thinking",
        image_png=b"\x89PNG\r\n",
    )

    assert captured["authenticated"] is True
    assert captured["request"].endpoint_url.endswith("/v1/chat/completions")
    assert captured["request"].messages[0]["content"][1]["type"] == "image_url"
    assert result.assistant_content_present is True


def _verified_manifest(tmp_path, resource_ids):
    tmp_path.mkdir(parents=True, exist_ok=True)
    local = tmp_path / "bundle.tar"
    local.write_bytes(b"durable artifact")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()

    class FakeStore:
        def sha256(self, remote_uri):
            assert remote_uri.startswith("fake://")
            return digest

    manifest = ArtifactManifest(tmp_path / "artifacts.jsonl", "run-1")
    for resource_id in resource_ids:
        manifest.verify_and_append(
            artifact_id=f"artifact-{resource_id}",
            resource_id=resource_id,
            local_path=local,
            remote_uri=f"fake://{resource_id}",
            store=FakeStore(),
        )
    return manifest


def test_artifact_manifest_requires_matching_remote_hash_and_records_termination(tmp_path):
    local = tmp_path / "artifact"
    local.write_bytes(b"local")

    class WrongStore:
        def sha256(self, remote_uri):
            del remote_uri
            return "0" * 64

    manifest = ArtifactManifest(tmp_path / "manifest.jsonl", "run")
    with pytest.raises(LedgerValidationError, match="differ"):
        manifest.verify_and_append(
            artifact_id="artifact",
            resource_id="worker",
            local_path=local,
            remote_uri="fake://artifact",
            store=WrongStore(),
        )

    verified = _verified_manifest(tmp_path / "verified", ["worker"])
    verified.record_termination("aws", "worker")
    assert verified.termination_ids() == {"worker"}


def test_watchdog_stops_leases_but_allows_export_until_checksum_is_verified(tmp_path):
    artifacts = ArtifactManifest(tmp_path / "artifacts.jsonl", "run")
    resource = ManagedResource("aws", "worker-1", "run-label", 0, True, False)

    plan = AWSLifecycleWatchdog("run-label").reconcile(
        resources=[resource],
        volumes=[],
        desired_workers=1,
        endpoint_healthy=False,
        budget_decision=BudgetDecision.CONTINUE,
        artifacts=artifacts,
    )

    assert plan.accept_new_leases is False
    assert plan.allow_artifact_export is True
    assert plan.terminate_resource_ids == ()

    budget_plan = AWSLifecycleWatchdog("run-label").reconcile(
        resources=[resource],
        volumes=[],
        desired_workers=1,
        endpoint_healthy=True,
        budget_decision=BudgetDecision.HARD_STOP,
        artifacts=artifacts,
    )
    assert budget_plan.accept_new_leases is False
    assert budget_plan.allow_artifact_export is True


def test_tail_shrink_to_zero_terminates_workers_and_removes_billable_volumes(tmp_path):
    artifacts = _verified_manifest(tmp_path, ["worker-1", "worker-2"])
    resources = [
        ManagedResource("aws", "worker-1", "run-label", 0, False, True),
        ManagedResource("aws", "worker-2", "run-label", 1, False, True, state="stopped"),
    ]
    volumes = [
        BillableVolume("volume-1", "worker-1", "worker-1"),
        BillableVolume("volume-2", "worker-2", None),
    ]
    plan = AWSLifecycleWatchdog("run-label").reconcile(
        resources=resources,
        volumes=volumes,
        desired_workers=0,
        endpoint_healthy=True,
        budget_decision=BudgetDecision.CONTINUE,
        artifacts=artifacts,
    )

    assert plan.accept_new_leases is False
    assert plan.terminate_resource_ids == ("worker-1", "worker-2")
    assert plan.delete_volume_ids == ("volume-1", "volume-2")
    assert plan.awaiting_verification_ids == ()

    class FakeActions:
        calls = []

        def terminate_resource(self, provider, resource_id):
            self.calls.append(("terminate", provider, resource_id))

        def delete_volume(self, provider, volume_id):
            self.calls.append(("delete-volume", provider, volume_id))

    actions = FakeActions()
    apply_watchdog_plan(plan, actions, provider="aws")
    assert actions.calls == []
    apply_watchdog_plan(plan, actions, provider="aws", dry_run=False)
    assert not any(call[0] == "stop" for call in actions.calls)
    assert len(actions.calls) == 4


def test_vast_watchdog_cleans_stale_labeled_instance_only_after_verification(tmp_path):
    unverified = ArtifactManifest(tmp_path / "empty.jsonl", "run")
    stale = ManagedResource(
        "vast",
        "vast-1",
        "run-label",
        0,
        False,
        False,
        stale=True,
    )
    watchdog = VastLifecycleWatchdog("run-label")

    blocked = watchdog.reconcile(
        resources=[stale],
        volumes=[],
        desired_workers=1,
        endpoint_healthy=True,
        budget_decision=BudgetDecision.CONTINUE,
        artifacts=unverified,
    )
    assert blocked.terminate_resource_ids == ()
    assert blocked.awaiting_verification_ids == ("vast-1",)

    verified = _verified_manifest(tmp_path / "verified", ["vast-1"])
    allowed = watchdog.reconcile(
        resources=[stale],
        volumes=[],
        desired_workers=1,
        endpoint_healthy=True,
        budget_decision=BudgetDecision.CONTINUE,
        artifacts=verified,
    )
    assert allowed.terminate_resource_ids == ("vast-1",)


def test_documented_dry_run_ceilings_are_checked_configuration():
    path = (
        Path(__file__).parents[1]
        / "docs/evaluation/qwen3_vl_osworld_dry_run_ceilings.json"
    )
    ceilings = json.loads(path.read_text())
    assert ceilings["small"] == {"15": "2.76", "50": "8.11", "100": "15.77"}
    assert ceilings["full"] == {"15": "21.00", "50": "61.84", "100": "120.18"}
