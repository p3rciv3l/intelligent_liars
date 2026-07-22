from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from intelligent_liars.osworld_cloud import (
    ARTIFACT_BUCKET,
    AWS_REGION,
    AWS_STEP_CAP_HARD_STOP_USD,
    DURABLE_CHECKPOINT_STATE_KINDS,
    FULL_BUDGET,
    OSWORLD_REQUIRED_SERVICE_PORTS,
    PILOT_BUDGET,
    VAST_EVALUATION_HARD_STOP_USD,
    ApprovalError,
    ArtifactExport,
    BootstrapHandle,
    BootstrapValidationError,
    CloudOrchestrator,
    CloudRunSpec,
    DurableCheckpoint,
    Instance,
    RunTier,
    SecurityGroupRule,
    TaggedResource,
    VastInstance,
    VastOffer,
    create_approval_artifact,
    create_dry_run_plan,
    render_plan_json,
    verify_approval,
)
from intelligent_liars.run_control import (
    AWS_STEP_CAP_HARD_STOP_USD as CENTRAL_AWS_STEP_CAP_HARD_STOP_USD,
    VAST_EVALUATION_HARD_STOP_USD as CENTRAL_VAST_EVALUATION_HARD_STOP_USD,
    BudgetDecision,
    CostLedger,
    CostSample,
)


PINNED_IMAGE = (
    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:" + "a" * 64
)
PINNED_OSWORLD = "b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf"
PARETO_EVIDENCE_SHA256 = "9" * 64
PARETO_POINT_SHA256 = "8" * 64


def make_vast_offer(**overrides) -> VastOffer:
    values = {
        "offer_id": "12345",
        "gpu_model": "L40S",
        "hourly_rate_usd": Decimal("0.59"),
        "verified": True,
        "projected_completed_run_cost_usd": Decimal("9.50"),
        "projected_wall_clock_minutes": 900,
        "pareto_evidence_sha256": PARETO_EVIDENCE_SHA256,
        "selected_pareto_point_sha256": PARETO_POINT_SHA256,
        "frontier_min_completed_run_cost_usd": Decimal("9.50"),
        "frontier_best_wall_clock_minutes_at_min_cost": 900,
    }
    values.update(overrides)
    return VastOffer(**values)


def make_spec(
    *,
    tier: RunTier = RunTier.PILOT,
    client_count: int | None = None,
    projected: str = "4.00",
    maximum: str = "5",
    projected_vast: str = "9.50",
    maximum_vast: str = "20",
    step_cap: int = 50,
) -> CloudRunSpec:
    pilot = tier is RunTier.PILOT
    return CloudRunSpec(
        run_id="run-test-001",
        tier=tier,
        controller_cidr="198.51.100.7/32",
        vpc_id="vpc-test",
        subnet_id="subnet-test",
        client_count=client_count if client_count is not None else (2 if pilot else 8),
        controller_hourly_rate_usd=Decimal("0.052" if pilot else "0.094"),
        vast_offer=make_vast_offer(
            projected_completed_run_cost_usd=Decimal(projected_vast),
            frontier_min_completed_run_cost_usd=Decimal(projected_vast),
        ),
        vast_image=PINNED_IMAGE,
        controller_ttl_minutes=210 if pilot else 1800,
        client_ttl_minutes=180,
        vast_ttl_minutes=180 if pilot else 1800,
        projected_aws_spend_usd=Decimal(projected),
        maximum_aws_spend_usd=Decimal(maximum),
        projected_vast_spend_usd=Decimal(projected_vast),
        maximum_vast_spend_usd=Decimal(maximum_vast),
        osworld_commit=PINNED_OSWORLD,
        step_cap=step_cap,
    )


def make_approval(plan):
    approved = datetime(2026, 7, 22, 12, tzinfo=UTC)
    return create_approval_artifact(
        plan,
        approved_at=approved,
        expires_at=approved + timedelta(hours=1),
    )


def make_full_checkpoint(
    run_id: str = "run-test-001",
    sequence: int = 1,
) -> DurableCheckpoint:
    return DurableCheckpoint(
        run_id=run_id,
        sequence=sequence,
        artifact_prefix=(
            f"s3://{ARTIFACT_BUCKET}/runs/{run_id}/tasks/fake/attempt-0001/"
            f"checkpoints/{sequence:08d}/"
        ),
        artifact_checksums={"checkpoint.json": "a" * 64},
        state_kinds=DURABLE_CHECKPOINT_STATE_KINDS,
        resumable_manifest_sha256="b" * 64,
        remote_verified=True,
    )


class FakeAWS:
    def __init__(self) -> None:
        self.calls = []
        self.cost = Decimal("0")
        self.resources = ()

    def ensure_bucket(self, policy):
        self.calls.append(("bucket", policy))

    def create_security_group(self, *, name, vpc_id, tags):
        resource_id = "sg-host" if name.endswith("-host") else "sg-clients"
        self.calls.append(("create-sg", resource_id, name, vpc_id, dict(tags)))
        return resource_id

    def authorize_ingress(self, security_group_id, rule):
        self.calls.append(("ingress", security_group_id, rule))

    def import_key_pair(self, name, public_key, tags):
        self.calls.append(("import-key", name, public_key, dict(tags)))

    def delete_key_pair(self, name):
        self.calls.append(("delete-key", name))

    def launch_controller(self, request):
        self.calls.append(("launch-controller", request))
        return Instance("i-controller", "10.0.0.10", "198.51.100.8", ("vol-host",))

    def launch_osworld_clients(self, request):
        self.calls.append(("launch-clients", request))
        return tuple(
            Instance(f"i-client-{index}", f"10.0.1.{index + 1}", None, (f"vol-{index}",))
            for index in range(request.count)
        )

    def schedule_termination(self, resource_id, *, ttl_minutes, run_id):
        self.calls.append(("ttl", resource_id, ttl_minutes, run_id))

    def issue_sts_credentials(self, *, run_id, ttl_minutes):
        self.calls.append(("sts", run_id, ttl_minutes))
        return object()

    def stop_new_leases(self, controller_id):
        self.calls.append(("stop-leases", controller_id))

    def terminate_instance(self, resource_id):
        self.calls.append(("terminate", resource_id))

    def delete_volume(self, volume_id):
        self.calls.append(("delete-volume", volume_id))

    def delete_security_group(self, security_group_id):
        self.calls.append(("delete-sg", security_group_id))

    def sample_costs(self, run_id):
        return (
            CostSample("aws", run_id, self.cost, "2026-07-22T12:00:00+00:00"),
        )

    def list_tagged_resources(self):
        return self.resources


class FakeVast:
    def __init__(self) -> None:
        self.calls = []
        self.cost = Decimal("0")
        self.resources = ()

    def launch(self, **kwargs):
        self.calls.append(("launch", kwargs))
        return VastInstance("vast-1", "vast-user@203.0.113.9", kwargs["qwen_port"])

    def schedule_termination(self, resource_id, *, ttl_minutes, run_id):
        self.calls.append(("ttl", resource_id, ttl_minutes, run_id))

    def destroy(self, resource_id):
        self.calls.append(("destroy", resource_id))

    def sample_costs(self, run_id):
        return (
            CostSample("vast", run_id, self.cost, "2026-07-22T12:00:00+00:00"),
        )

    def list_tagged_resources(self):
        return self.resources


class FakeS3:
    def __init__(self, events=None) -> None:
        self.hashes = {}
        self.events = events if events is not None else []
        self.verified_runs = set()

    def put_file_if_absent(self, local_path, remote_uri):
        self.events.append(("upload", remote_uri))
        if remote_uri in self.hashes:
            raise RuntimeError("append-only collision")
        self.hashes[remote_uri] = hashlib.sha256(local_path.read_bytes()).hexdigest()

    def remote_sha256(self, remote_uri):
        self.events.append(("remote-sha256", remote_uri))
        return self.hashes[remote_uri]

    def run_export_verified(self, run_id):
        return run_id in self.verified_runs


class FakeSSH:
    def __init__(self) -> None:
        self.calls = []

    def generate_key(self, private_key_path):
        private_key_path.write_text("fake-private-key")
        private_key_path.chmod(0o644)
        self.calls.append(("generate-key", private_key_path))
        return b"ssh-ed25519 fake-public-key"

    def install_ephemeral_sts_credentials(
        self, controller, credentials, *, expires_in_minutes
    ):
        self.calls.append(("sts", controller.resource_id, bool(credentials), expires_in_minutes))

    def establish_qwen_local_forward(self, controller, vast, **kwargs):
        self.calls.append(("forward", controller.resource_id, vast.resource_id, kwargs))

    def configure_osworld(self, controller, **kwargs):
        self.calls.append(("configure", controller.resource_id, kwargs))


def make_orchestrator(tmp_path, *, events=None):
    aws = FakeAWS()
    vast = FakeVast()
    s3 = FakeS3(events)
    ssh = FakeSSH()
    return CloudOrchestrator(
        aws,
        vast,
        s3,
        ssh,
        key_directory=tmp_path / "keys",
    ), aws, vast, s3, ssh


def bootstrap(tmp_path, spec=None):
    orchestrator, aws, vast, s3, ssh = make_orchestrator(tmp_path)
    plan = create_dry_run_plan(spec or make_spec())
    approval = make_approval(plan)
    handle = orchestrator.bootstrap(
        plan,
        approval,
        now=datetime(2026, 7, 22, 12, 30, tzinfo=UTC),
    )
    return orchestrator, handle, aws, vast, s3, ssh


def test_dry_run_is_zero_side_effect_and_contains_no_secret_values(tmp_path):
    orchestrator, aws, vast, s3, ssh = make_orchestrator(tmp_path)
    plan = create_dry_run_plan(make_spec())
    rendered = render_plan_json(plan)

    assert aws.calls == vast.calls == ssh.calls == []
    assert s3.events == []
    assert str(tmp_path) not in rendered
    assert "credential_values_included" in rendered
    assert "AWS_SECRET_ACCESS_KEY=" not in rendered
    assert "VAST_API_KEY=" not in rendered
    assert "QWEN_ENDPOINT_API_KEY=" not in rendered
    assert f"s3://{ARTIFACT_BUCKET}/runs/run-test-001/" in rendered
    assert "vastai create instance 12345" in rendered
    assert "${HOST_SECURITY_GROUP_ID}" in rendered
    del orchestrator


def test_dry_run_uses_atomic_append_only_s3_upload_with_remote_checksum():
    plan = create_dry_run_plan(make_spec())
    put_index = next(
        index
        for index, command in enumerate(plan.commands)
        if command.startswith("aws s3api put-object")
    )
    put_command = plan.commands[put_index]
    head_command = plan.commands[put_index + 1]

    assert "--if-none-match '*'" in put_command
    assert "--checksum-algorithm SHA256" in put_command
    assert "--checksum-sha256 ${ARTIFACT_SHA256_BASE64}" in put_command
    assert "--body ${ARTIFACT_PATH}" in put_command
    assert head_command.startswith("aws s3api head-object")
    assert "--checksum-mode ENABLED" in head_command
    assert "--no-clobber" not in "\n".join(plan.commands)
    assert "aws s3 cp" not in "\n".join(plan.commands)


def test_security_groups_are_run_separated_and_never_public():
    plan = create_dry_run_plan(make_spec())

    assert plan.host_security_group_name != plan.client_security_group_name
    assert plan.spec.run_id in plan.host_security_group_name
    assert plan.host_rules == (
        SecurityGroupRule("tcp", 22, 22, source_cidr="198.51.100.7/32"),
    )
    assert plan.client_rules == tuple(
        SecurityGroupRule(
            "tcp",
            port,
            port,
            source_security_group_id="${HOST_SECURITY_GROUP_ID}",
        )
        for port in OSWORLD_REQUIRED_SERVICE_PORTS
    )
    assert OSWORLD_REQUIRED_SERVICE_PORTS == (
        5000,
        9222,
        8006,
        8080,
        5910,
    )
    assert all(rule.from_port == rule.to_port for rule in plan.client_rules)
    assert all(rule.source_security_group_id for rule in plan.client_rules)
    assert not any(rule.source_cidr for rule in plan.client_rules)
    with pytest.raises(BootstrapValidationError, match="prohibited"):
        SecurityGroupRule("tcp", 5900, 5900, source_cidr="0.0.0.0/0")


@pytest.mark.parametrize(
    "ports",
    [
        (5000, 9222, 8006, 8080),
        (5000, 9222, 8006, 8080, 5910, 5900),
        (9222, 5000, 8006, 8080, 5910),
    ],
)
def test_spec_fails_closed_when_service_ports_are_not_exact(ports):
    with pytest.raises(
        BootstrapValidationError,
        match="service ports must exactly equal",
    ):
        replace(make_spec(), service_ports=ports)


def test_spec_enforces_tiers_vast_pinning_ttls_and_promotion_boundary():
    pilot = make_spec()
    full = make_spec(
        tier=RunTier.FULL,
        projected="49.99",
        maximum="120",
        step_cap=100,
    )

    assert pilot.policy.controller_type == "t3.medium"
    assert pilot.policy.maximum_clients == 2
    assert full.policy.controller_type == "t3.large"
    assert full.policy.maximum_clients == 8
    assert pilot.region == AWS_REGION
    assert pilot.vast_disk_gb == 120
    with pytest.raises(BootstrapValidationError, match="between 1 and 2"):
        replace(pilot, client_count=3)
    with pytest.raises(BootstrapValidationError, match="exactly three hours"):
        replace(pilot, vast_ttl_minutes=181)
    with pytest.raises(BootstrapValidationError, match=r"strictly below \$60"):
        make_spec(
            tier=RunTier.FULL,
            projected="50.50",
            maximum="120",
            step_cap=100,
        )
    with pytest.raises(BootstrapValidationError, match="100-step ceiling"):
        make_spec(
            tier=RunTier.FULL,
            projected="49",
            maximum="120.01",
            step_cap=100,
        )
    with pytest.raises(BootstrapValidationError, match="pinned by digest"):
        replace(pilot, vast_image="pytorch/pytorch:latest-cuda")


def test_cloud_spec_has_no_duplicate_vast_maximum_declaration():
    source = (
        Path(__file__).parents[1]
        / "src/intelligent_liars/osworld_cloud.py"
    ).read_text()
    assert source.count("    maximum_vast_spend_usd: Decimal\n") == 1
    assert AWS_STEP_CAP_HARD_STOP_USD is CENTRAL_AWS_STEP_CAP_HARD_STOP_USD
    assert VAST_EVALUATION_HARD_STOP_USD is CENTRAL_VAST_EVALUATION_HARD_STOP_USD


def test_approval_is_content_addressed_and_fails_closed_on_every_mismatch():
    plan = create_dry_run_plan(make_spec())
    approval = make_approval(plan)
    now = datetime(2026, 7, 22, 12, 30, tzinfo=UTC)

    verify_approval(plan, approval, now=now)
    with pytest.raises(ApprovalError, match="required"):
        verify_approval(plan, None, now=now)
    with pytest.raises(ApprovalError, match="content hash"):
        verify_approval(plan, replace(approval, approval_id="0" * 64), now=now)
    changed = create_dry_run_plan(
        replace(
            make_spec(),
            vast_offer=make_vast_offer(
                offer_id="other-offer",
                hourly_rate_usd=Decimal("0.60"),
            ),
        )
    )
    with pytest.raises(ApprovalError, match="exact launch plan"):
        verify_approval(changed, approval, now=now)
    with pytest.raises(ApprovalError, match="expired"):
        verify_approval(
            plan,
            approval,
            now=datetime(2026, 7, 22, 13, tzinfo=UTC),
        )


def test_vast_offer_requires_measured_pareto_optimum_and_binds_approval():
    with pytest.raises(BootstrapValidationError, match="necessity justification"):
        make_vast_offer(
            offer_id="multi-gpu-offer",
            hourly_rate_usd=Decimal("1.20"),
            gpu_count=2,
            higher_hourly_spend=True,
        )
    with pytest.raises(BootstrapValidationError, match="necessity justification"):
        make_vast_offer(
            projected_completed_run_cost_usd=Decimal("10"),
            projected_wall_clock_minutes=850,
        )

    base_plan = create_dry_run_plan(make_spec())
    approved = make_approval(base_plan)
    justified_offer = make_vast_offer(
        offer_id="multi-gpu-offer",
        hourly_rate_usd=Decimal("1.20"),
        gpu_count=2,
        projected_completed_run_cost_usd=Decimal("8.75"),
        projected_wall_clock_minutes=420,
        frontier_min_completed_run_cost_usd=Decimal("8.50"),
        frontier_best_wall_clock_minutes_at_min_cost=400,
        higher_hourly_spend=True,
        necessity_justification="single GPU misses the approved completion deadline",
    )
    changed_plan = create_dry_run_plan(
        replace(
            make_spec(),
            vast_offer=justified_offer,
            projected_vast_spend_usd=Decimal("8.75"),
        )
    )

    assert base_plan.approval_payload["vast"]["selection_objective"] == (
        "projected_completed_run_cost_then_wall_clock"
    )
    assert base_plan.approval_payload["vast"]["pareto_evidence_sha256"] == (
        PARETO_EVIDENCE_SHA256
    )
    assert base_plan.approval_payload["vast"]["selected_pareto_point_sha256"] == (
        PARETO_POINT_SHA256
    )
    assert (
        base_plan.approval_payload["vast"][
            "frontier_min_completed_run_cost_usd"
        ]
        == "9.50"
    )
    assert (
        base_plan.approval_payload["vast"][
            "frontier_best_wall_clock_minutes_at_min_cost"
        ]
        == 900
    )
    assert changed_plan.approval_payload["vast"]["gpu_count"] == 2
    assert (
        changed_plan.approval_payload["vast"]["necessity_justification"]
        == "single GPU misses the approved completion deadline"
    )
    with pytest.raises(ApprovalError, match="exact launch plan"):
        verify_approval(
            changed_plan,
            approved,
            now=datetime(2026, 7, 22, 12, 30, tzinfo=UTC),
        )
    changed_approval = make_approval(changed_plan)
    rejustified_plan = create_dry_run_plan(
        replace(
            make_spec(),
            vast_offer=replace(
                justified_offer,
                necessity_justification="lower interruption risk",
            ),
            projected_vast_spend_usd=Decimal("8.75"),
        )
    )
    with pytest.raises(ApprovalError, match="exact launch plan"):
        verify_approval(
            rejustified_plan,
            changed_approval,
            now=datetime(2026, 7, 22, 12, 30, tzinfo=UTC),
        )


def test_approval_binds_separate_provider_ceiling_schema():
    plan = create_dry_run_plan(make_spec())
    approval = make_approval(plan)
    assert plan.approval_payload["provider_budgets"] == {
        "aws": {
            "projected_spend_usd": "4.00",
            "maximum_spend_usd": "5",
            "provider_hard_ceiling_usd": "75",
        },
        "vast": {
            "projected_spend_usd": "9.50",
            "maximum_spend_usd": "20",
            "provider_hard_ceiling_usd": "20",
        },
    }
    changed = create_dry_run_plan(
        replace(make_spec(), maximum_aws_spend_usd=Decimal("6"))
    )
    with pytest.raises(ApprovalError, match="exact launch plan"):
        verify_approval(
            changed,
            approval,
            now=datetime(2026, 7, 22, 12, 30, tzinfo=UTC),
        )


def test_bootstrap_uses_private_clients_loopback_forward_sts_and_independent_ttls(
    tmp_path,
):
    orchestrator, handle, aws, vast, _s3, ssh = bootstrap(tmp_path)

    bucket = next(call[1] for call in aws.calls if call[0] == "bucket")
    assert bucket.region == "us-east-1"
    assert bucket.encryption == "AES256"
    assert bucket.block_public_access is True
    assert bucket.abort_incomplete_multipart_days > 0
    controller_request = next(call[1] for call in aws.calls if call[0] == "launch-controller")
    client_request = next(call[1] for call in aws.calls if call[0] == "launch-clients")
    assert controller_request.public_address is True
    assert controller_request.instance_type == "t3.medium"
    assert client_request.instance_type == "t3.xlarge"
    assert client_request.private_addresses is True
    assert client_request.vpc_id == controller_request.vpc_id
    assert all(client.public_address is None for client in handle.clients)
    ttl_calls = [call for call in aws.calls if call[0] == "ttl"]
    assert {call[1] for call in ttl_calls} == {
        "i-controller",
        "i-client-0",
        "i-client-1",
    }
    assert [call for call in vast.calls if call[0] == "ttl"] == [
        ("ttl", "vast-1", 180, "run-test-001")
    ]
    forward = next(call for call in ssh.calls if call[0] == "forward")
    assert forward[3]["local_bind_host"] == "127.0.0.1"
    assert forward[3]["remote_bind_host"] == "127.0.0.1"
    configure = next(call for call in ssh.calls if call[0] == "configure")
    assert configure[2]["use_private_client_addresses"] is True
    assert configure[2]["qwen_endpoint"].startswith("http://127.0.0.1:")
    assert any(call[0] == "sts" for call in aws.calls)
    assert any(call[0] == "delete-key" for call in aws.calls)
    assert not list((tmp_path / "keys").glob("*.pem"))
    del orchestrator


def test_key_pair_and_partial_resources_are_deleted_when_bootstrap_fails(tmp_path):
    orchestrator, aws, vast, _s3, _ssh = make_orchestrator(tmp_path)
    original_schedule_termination = aws.schedule_termination

    def fail_after_clients_launch(resource_id, *, ttl_minutes, run_id):
        original_schedule_termination(
            resource_id,
            ttl_minutes=ttl_minutes,
            run_id=run_id,
        )
        if resource_id == "i-client-1":
            raise RuntimeError("fake client TTL failure")

    aws.schedule_termination = fail_after_clients_launch
    plan = create_dry_run_plan(make_spec())
    with pytest.raises(RuntimeError, match="fake client TTL failure"):
        orchestrator.bootstrap(
            plan,
            make_approval(plan),
            now=datetime(2026, 7, 22, 12, 30, tzinfo=UTC),
        )

    instance_volumes = {
        "i-client-0": "vol-0",
        "i-client-1": "vol-1",
        "i-controller": "vol-host",
    }
    for resource_id, volume_id in instance_volumes.items():
        termination_index = aws.calls.index(("terminate", resource_id))
        volume_deletion_index = aws.calls.index(("delete-volume", volume_id))
        assert termination_index < volume_deletion_index
    assert ("destroy", "vast-1") in vast.calls
    assert any(call[0] == "delete-key" for call in aws.calls)
    assert not list((tmp_path / "keys").glob("*.pem"))


def test_artifacts_are_uploaded_and_verified_before_any_termination(tmp_path):
    events = []
    orchestrator, aws, vast, s3, ssh = make_orchestrator(tmp_path, events=events)
    original_terminate = aws.terminate_instance
    original_destroy = vast.destroy
    aws.terminate_instance = lambda resource_id: (
        events.append(("terminate", resource_id)),
        original_terminate(resource_id),
    )[-1]
    vast.destroy = lambda resource_id: (
        events.append(("destroy", resource_id)),
        original_destroy(resource_id),
    )[-1]
    plan = create_dry_run_plan(make_spec())
    handle = orchestrator.bootstrap(
        plan,
        make_approval(plan),
        now=datetime(2026, 7, 22, 12, 30, tzinfo=UTC),
    )
    artifact = tmp_path / "run.tar"
    artifact.write_bytes(b"complete durable export")
    orchestrator.register_checkpoint(handle, make_full_checkpoint())

    with pytest.raises(RuntimeError, match="verified before teardown"):
        orchestrator.teardown(handle)
    orchestrator.verify_exports(
        handle,
        [ArtifactExport(artifact, "exports/run.tar", tuple(handle.compute_resource_ids))],
    )
    orchestrator.teardown(handle)

    verification_index = max(
        index for index, event in enumerate(events) if event[0] == "remote-sha256"
    )
    termination_index = min(
        index for index, event in enumerate(events) if event[0] in {"terminate", "destroy"}
    )
    assert verification_index < termination_index
    assert handle.terminated_resource_ids == handle.compute_resource_ids
    assert s3.hashes
    assert any(call[0] == "delete-volume" for call in aws.calls)
    assert not any(call[0] == "stop-instance" for call in aws.calls)
    del ssh


def test_spend_sampling_uses_cumulative_ledger_and_exact_boundaries(tmp_path):
    orchestrator, handle, aws, vast, _s3, _ssh = bootstrap(tmp_path)
    ledger = CostLedger(tmp_path / "costs.jsonl")

    aws.cost = Decimal("4.50")
    vast.cost = Decimal("0.59")
    assert (
        orchestrator.sample_spend(handle, ledger)
        is BudgetDecision.STOP_NEW_LEASES
    )
    aws.cost = Decimal("75")
    assert orchestrator.sample_spend(handle, ledger) is BudgetDecision.HARD_STOP
    assert ledger.totals_by_resource()[("aws", "run-test-001")] == Decimal("75")
    assert PILOT_BUDGET.decide(Decimal("4.499")) is BudgetDecision.CONTINUE
    assert PILOT_BUDGET.decide(Decimal("4.50")) is BudgetDecision.STOP_NEW_LEASES
    assert PILOT_BUDGET.decide(Decimal("75")) is BudgetDecision.HARD_STOP
    assert FULL_BUDGET.decide(Decimal("64.99")) is BudgetDecision.CONTINUE
    assert FULL_BUDGET.decide(Decimal("65")) is BudgetDecision.STOP_NEW_LEASES
    assert FULL_BUDGET.decide(Decimal("120")) is BudgetDecision.HARD_STOP
    assert (
        orchestrator.sample_spend(
            handle,
            ledger,
            other_model_run_cost_usd=Decimal("65"),
        )
        is BudgetDecision.HARD_STOP
    )


def test_vast_evaluation_cap_is_cumulative_across_tranches_and_resources(tmp_path):
    spec = make_spec(
        tier=RunTier.FULL,
        projected="20",
        maximum="70",
        client_count=8,
    )
    orchestrator, handle, aws, vast, _s3, _ssh = bootstrap(tmp_path, spec)
    ledger = CostLedger(tmp_path / "cumulative-vast-costs.jsonl")
    ledger.append(
        CostSample("vast", "tranche-one", Decimal("14"), "2026-07-22T10:00:00Z")
    )
    ledger.append(
        CostSample(
            "vast",
            "tranche-two",
            Decimal("5.999"),
            "2026-07-22T11:00:00Z",
        )
    )
    aws.cost = Decimal("0")
    vast.cost = Decimal("0")

    assert VAST_EVALUATION_HARD_STOP_USD == Decimal("20")
    assert orchestrator.sample_spend(handle, ledger) is BudgetDecision.CONTINUE
    assert sum(
        cost
        for (provider, _resource), cost in ledger.totals_by_resource().items()
        if provider == "vast"
    ) == Decimal("19.999")

    vast.cost = Decimal("0.001")
    assert orchestrator.sample_spend(handle, ledger) is BudgetDecision.HARD_STOP
    assert sum(
        cost
        for (provider, _resource), cost in ledger.totals_by_resource().items()
        if provider == "vast"
    ) == Decimal("20.000")


@pytest.mark.parametrize(
    ("step_cap", "below", "at"),
    (
        (50, Decimal("74.999"), Decimal("75")),
        (100, Decimal("119.999"), Decimal("120")),
    ),
)
def test_aws_provider_caps_are_cumulative_and_step_specific(
    tmp_path,
    step_cap,
    below,
    at,
):
    spec = make_spec(
        tier=RunTier.FULL,
        projected="49" if step_cap == 100 else "20",
        maximum=str(AWS_STEP_CAP_HARD_STOP_USD[step_cap]),
        projected_vast="0",
        client_count=8,
        step_cap=step_cap,
    )
    orchestrator, handle, aws, vast, _s3, _ssh = bootstrap(tmp_path, spec)
    ledger = CostLedger(tmp_path / f"aws-{step_cap}.jsonl")
    ledger.append(
        CostSample("aws", "controller", Decimal("10"), "2026-07-22T10:00:00Z")
    )
    ledger.append(
        CostSample(
            "aws",
            "clients",
            below - Decimal("10"),
            "2026-07-22T10:00:00Z",
        )
    )
    aws.cost = Decimal("0")
    vast.cost = Decimal("0")

    assert orchestrator.sample_spend(handle, ledger) is BudgetDecision.STOP_NEW_LEASES
    ledger.append(
        CostSample(
            "aws",
            "clients",
            at - Decimal("10"),
            "2026-07-22T11:00:00Z",
        )
    )
    assert orchestrator.sample_spend(handle, ledger) is BudgetDecision.HARD_STOP


def test_hard_stop_stops_leases_bounds_drain_then_terminates(tmp_path):
    orchestrator, handle, aws, vast, _s3, _ssh = bootstrap(tmp_path)
    artifact = tmp_path / "drain.tar"
    artifact.write_bytes(b"drained")
    observed = []

    def drain(seconds):
        observed.append(seconds)
        return [
            ArtifactExport(
                artifact,
                "hard-stop/drain.tar",
                tuple(handle.compute_resource_ids),
            )
        ]

    orchestrator.hard_stop(
        handle,
        artifact_drain_seconds=300,
        drain=drain,
        checkpoint=make_full_checkpoint(),
    )

    assert observed == [300]
    assert ("stop-leases", "i-controller") in aws.calls
    assert ("destroy", "vast-1") in vast.calls
    with pytest.raises(BootstrapValidationError, match="positive bound"):
        orchestrator.hard_stop(
            handle,
            artifact_drain_seconds=0,
            drain=drain,
            checkpoint=make_full_checkpoint(),
        )


def test_tail_modulus_shrinks_immediately_only_after_checksum_verification(tmp_path):
    spec = make_spec(
        tier=RunTier.FULL,
        projected="20",
        maximum="70",
        client_count=8,
    )
    orchestrator, handle, aws, _vast, _s3, _ssh = bootstrap(tmp_path, spec)

    handle.verified_resource_ids.update(
        client.resource_id for client in handle.clients[1:]
    )
    orchestrator.register_checkpoint(handle, make_full_checkpoint())
    assert orchestrator.shrink_tail(handle, remaining_tasks=1) == 1
    assert len(handle.clients) == 1
    assert len([call for call in aws.calls if call[0] == "terminate"]) == 7

    fresh = BootstrapHandle(
        spec=spec,
        controller=handle.controller,
        clients=[Instance("tail-unverified", "10.0.2.1")],
        vast=handle.vast,
        host_security_group_id="sg-host",
        client_security_group_id="sg-client",
        latest_checkpoint=make_full_checkpoint(),
    )
    with pytest.raises(RuntimeError, match="checksum-verified"):
        orchestrator.shrink_tail(fresh, remaining_tasks=0)


def test_reconciliation_cleans_only_tagged_safe_or_expired_orphans(tmp_path):
    orchestrator, _handle, aws, vast, s3, _ssh = bootstrap(tmp_path)
    aws.resources = (
        TaggedResource(
            "aws",
            "orphan-safe",
            "client",
            "run-test-001",
            "running",
            False,
            ("orphan-volume",),
            True,
            True,
            True,
        ),
        TaggedResource(
            "aws",
            "orphan-wait",
            "client",
            "run-test-001",
            "running",
            False,
        ),
        TaggedResource(
            "aws",
            "other-run",
            "client",
            "other",
            "running",
            True,
            artifact_verified=True,
            resumable_manifest_verified=True,
            complete_checkpoint_verified=True,
        ),
    )
    vast.resources = (
        TaggedResource(
            "vast",
            "orphan-expired",
            "gpu",
            "run-test-001",
            "running",
            True,
            artifact_verified=True,
            resumable_manifest_verified=True,
            complete_checkpoint_verified=True,
        ),
        TaggedResource(
            "vast",
            "orphan-expired-unsynced",
            "gpu",
            "run-test-001",
            "running",
            True,
        ),
    )

    assert orchestrator.reconcile_orphans(run_id="run-test-001") == (
        "orphan-safe",
        "orphan-expired",
    )
    assert ("terminate", "orphan-safe") in aws.calls
    assert ("delete-volume", "orphan-volume") in aws.calls
    assert ("terminate", "orphan-wait") not in aws.calls
    assert ("terminate", "other-run") not in aws.calls
    assert ("destroy", "orphan-expired") in vast.calls
    assert ("destroy", "orphan-expired-unsynced") not in vast.calls

    s3.verified_runs.add("run-test-001")
    assert orchestrator.reconcile_orphans(run_id="run-test-001") == (
        "orphan-safe",
        "orphan-wait",
        "orphan-expired",
        "orphan-expired-unsynced",
    )
