from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from intelligent_liars.run_control import (
    BudgetDecision,
    BudgetPolicy,
    CostLedger,
    CostSample,
    EvaluationBudgetPolicy,
    _decimal,
    canonical_json,
    desired_worker_count,
    file_sha256,
    stable_sha256,
)


AWS_REGION = "us-east-1"
ARTIFACT_BUCKET = "intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21"
OSWORLD_CLIENT_TYPE = "t3.xlarge"
OSWORLD_CLIENT_RATE_USD = Decimal("0.2295")
OSWORLD_REQUIRED_SERVICE_PORTS = (5000, 9222, 8006, 8080, 5910)
PILOT_BUDGET = BudgetPolicy(Decimal("4.50"), Decimal("5"))
FULL_BUDGET = BudgetPolicy(Decimal("65"), Decimal("70"))
VAST_DISK_GB = 120
PILOT_VAST_TTL_MINUTES = 180


class BootstrapValidationError(ValueError):
    pass


class ApprovalError(PermissionError):
    pass


class RunTier(StrEnum):
    PILOT = "pilot"
    FULL = "full"


@dataclass(frozen=True)
class TierPolicy:
    controller_type: str
    maximum_clients: int
    budget: BudgetPolicy


TIER_POLICIES = {
    RunTier.PILOT: TierPolicy("t3.medium", 2, PILOT_BUDGET),
    RunTier.FULL: TierPolicy("t3.large", 8, FULL_BUDGET),
}


@dataclass(frozen=True)
class SecurityGroupRule:
    protocol: str
    from_port: int
    to_port: int
    source_cidr: str | None = None
    source_security_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.protocol != "tcp" or not (0 < self.from_port <= self.to_port <= 65535):
            raise BootstrapValidationError("security-group rules must use valid TCP ports")
        if bool(self.source_cidr) == bool(self.source_security_group_id):
            raise BootstrapValidationError("a rule requires exactly one source")
        if self.source_cidr:
            network = ipaddress.ip_network(self.source_cidr, strict=True)
            if network.prefixlen == 0:
                raise BootstrapValidationError("0.0.0.0/0 and ::/0 are prohibited")


@dataclass(frozen=True)
class BucketPolicy:
    name: str = ARTIFACT_BUCKET
    region: str = AWS_REGION
    encryption: str = "AES256"
    block_public_access: bool = True
    abort_incomplete_multipart_days: int = 7
    append_only_run_prefixes: bool = True


@dataclass(frozen=True)
class VastOffer:
    offer_id: str
    gpu_model: str
    hourly_rate_usd: Decimal
    verified: bool
    gpu_count: int = 1

    def __post_init__(self) -> None:
        if not self.offer_id or not self.gpu_model:
            raise BootstrapValidationError("a concrete Vast offer ID and GPU model are required")
        object.__setattr__(
            self,
            "hourly_rate_usd",
            _decimal(self.hourly_rate_usd, "vast hourly_rate_usd"),
        )
        if not self.verified:
            raise BootstrapValidationError("the selected Vast offer must be verified")
        if self.gpu_count != 1:
            raise BootstrapValidationError("exactly one Vast GPU is required")


@dataclass(frozen=True)
class CloudRunSpec:
    run_id: str
    tier: RunTier
    controller_cidr: str
    vpc_id: str
    subnet_id: str
    client_count: int
    controller_hourly_rate_usd: Decimal
    vast_offer: VastOffer
    vast_image: str
    controller_ttl_minutes: int
    client_ttl_minutes: int
    vast_ttl_minutes: int
    projected_total_usd: Decimal
    maximum_spend_usd: Decimal
    osworld_commit: str
    step_cap: int
    service_ports: tuple[int, ...] = OSWORLD_REQUIRED_SERVICE_PORTS
    qwen_port: int = 8000
    artifact_bucket: str = ARTIFACT_BUCKET
    region: str = AWS_REGION
    worker_dtype: str = "bfloat16"
    workers_per_gpu: int = 1
    vast_disk_gb: int = VAST_DISK_GB
    destroy_after_verified_export: bool = True

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "tier", RunTier(self.tier))
        except ValueError as exc:
            raise BootstrapValidationError("tier must be pilot or full") from exc
        for name in (
            "controller_hourly_rate_usd",
            "projected_total_usd",
            "maximum_spend_usd",
        ):
            object.__setattr__(self, name, _decimal(getattr(self, name), name))
        self.validate()

    @property
    def policy(self) -> TierPolicy:
        return TIER_POLICIES[self.tier]

    @property
    def artifact_prefix(self) -> str:
        return f"s3://{self.artifact_bucket}/runs/{self.run_id}/"

    @property
    def tags(self) -> dict[str, str]:
        return {"run_id": self.run_id, "managed_by": "intelligent-liars-osworld"}

    def validate(self) -> None:
        if not self.run_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in self.run_id):
            raise BootstrapValidationError("run_id must be lowercase and tag-safe")
        if self.region != AWS_REGION or self.artifact_bucket != ARTIFACT_BUCKET:
            raise BootstrapValidationError("the artifact bucket and AWS region are fixed")
        network = ipaddress.ip_network(self.controller_cidr, strict=True)
        if network.prefixlen == 0:
            raise BootstrapValidationError("controller_cidr must be explicitly restricted")
        if not self.vpc_id or not self.subnet_id:
            raise BootstrapValidationError("a concrete VPC and subnet are required")
        if not 1 <= self.client_count <= self.policy.maximum_clients:
            raise BootstrapValidationError(
                f"{self.tier.value} client_count must be between 1 and {self.policy.maximum_clients}"
            )
        if self.service_ports != OSWORLD_REQUIRED_SERVICE_PORTS:
            raise BootstrapValidationError(
                "OSWorld service ports must exactly equal "
                f"{OSWORLD_REQUIRED_SERVICE_PORTS}"
            )
        if (
            self.worker_dtype != "bfloat16"
            or self.workers_per_gpu != 1
            or self.vast_disk_gb != VAST_DISK_GB
        ):
            raise BootstrapValidationError("Vast requires one BF16 worker per GPU and 120GB disk")
        if "@sha256:" not in self.vast_image or not all(
            token in self.vast_image.lower() for token in ("cuda", "pytorch")
        ):
            raise BootstrapValidationError("Vast CUDA/PyTorch image must be pinned by digest")
        if any(
            ttl <= 0
            for ttl in (
                self.controller_ttl_minutes,
                self.client_ttl_minutes,
                self.vast_ttl_minutes,
            )
        ):
            raise BootstrapValidationError("all resources require independent positive TTLs")
        if self.tier is RunTier.PILOT and self.vast_ttl_minutes != PILOT_VAST_TTL_MINUTES:
            raise BootstrapValidationError("pilot Vast TTL must be exactly three hours")
        if not self.destroy_after_verified_export:
            raise BootstrapValidationError("destroy-after-verified-export must remain enabled")
        if len(self.osworld_commit) != 40 or any(c not in "0123456789abcdef" for c in self.osworld_commit):
            raise BootstrapValidationError("OSWorld must be pinned to a lowercase 40-character commit")
        if self.maximum_spend_usd > self.policy.budget.hard_stop_usd:
            raise BootstrapValidationError("maximum spend exceeds the tier hard stop")
        if self.projected_total_usd > self.maximum_spend_usd:
            raise BootstrapValidationError("projected spend exceeds maximum spend")
        if self.step_cap <= 0:
            raise BootstrapValidationError("step_cap must be positive")
        if (
            self.step_cap == 100
            and not EvaluationBudgetPolicy().may_promote_to_100_steps(
                self.projected_total_usd
            )
        ):
            raise BootstrapValidationError(
                "100-step projected total must be strictly below $60"
            )


@dataclass(frozen=True)
class DryRunPlan:
    schema_version: int
    spec: CloudRunSpec
    host_security_group_name: str
    client_security_group_name: str
    host_rules: tuple[SecurityGroupRule, ...]
    client_rules: tuple[SecurityGroupRule, ...]
    teardown_plan: tuple[str, ...]
    commands: tuple[str, ...]

    @property
    def approval_payload(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "schema_version": self.schema_version,
            "run_id": spec.run_id,
            "vast": {
                "offer_id": spec.vast_offer.offer_id,
                "gpu_model": spec.vast_offer.gpu_model,
                "gpu_count": spec.vast_offer.gpu_count,
                "exact_hourly_rate_usd": str(spec.vast_offer.hourly_rate_usd),
                "image": spec.vast_image,
                "disk_gb": spec.vast_disk_gb,
                "worker_dtype": spec.worker_dtype,
                "workers_per_gpu": spec.workers_per_gpu,
            },
            "aws": {
                "region": spec.region,
                "controller_type": spec.policy.controller_type,
                "controller_count": 1,
                "controller_hourly_rate_usd": str(spec.controller_hourly_rate_usd),
                "client_type": OSWORLD_CLIENT_TYPE,
                "client_count": spec.client_count,
                "client_hourly_rate_usd": str(OSWORLD_CLIENT_RATE_USD),
                "vpc_id": spec.vpc_id,
                "subnet_id": spec.subnet_id,
            },
            "ttls_minutes": {
                "controller": spec.controller_ttl_minutes,
                "clients": spec.client_ttl_minutes,
                "vast": spec.vast_ttl_minutes,
            },
            "network": {
                "controller_cidr": spec.controller_cidr,
                "host_ingress": ["tcp/22 from controller_cidr"],
                "client_service_ports": list(spec.service_ports),
                "client_ingress_source": "host-security-group",
                "client_addresses": "private",
                "qwen_local_bind": f"127.0.0.1:{spec.qwen_port}",
                "qwen_remote_bind": f"127.0.0.1:{spec.qwen_port}",
            },
            "artifact_uri": spec.artifact_prefix,
            "artifact_bucket_policy": {
                "encryption": "AES256",
                "block_public_access": True,
                "abort_incomplete_multipart_days": 7,
                "append_only_run_prefixes": True,
            },
            "projected_spend_usd": str(spec.projected_total_usd),
            "maximum_spend_usd": str(spec.maximum_spend_usd),
            "teardown_plan": list(self.teardown_plan),
            "osworld_commit": spec.osworld_commit,
            "step_cap": spec.step_cap,
        }

    @property
    def content_hash(self) -> str:
        return stable_sha256(self.approval_payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": True,
            "approval_hash": self.content_hash,
            "approval_payload": self.approval_payload,
            "security_groups": {
                "host": {
                    "name": self.host_security_group_name,
                    "rules": [_rule_dict(rule) for rule in self.host_rules],
                },
                "client": {
                    "name": self.client_security_group_name,
                    "rules": [_rule_dict(rule) for rule in self.client_rules],
                },
            },
            "commands": list(self.commands),
            "credential_values_included": False,
        }


def _rule_dict(rule: SecurityGroupRule) -> dict[str, Any]:
    return {
        "protocol": rule.protocol,
        "from_port": rule.from_port,
        "to_port": rule.to_port,
        "source_cidr": rule.source_cidr,
        "source_security_group_id": rule.source_security_group_id,
    }


def create_dry_run_plan(spec: CloudRunSpec) -> DryRunPlan:
    host_name = f"osworld-{spec.run_id}-host"
    client_name = f"osworld-{spec.run_id}-clients"
    host_rules = (SecurityGroupRule("tcp", 22, 22, source_cidr=spec.controller_cidr),)
    client_rules = tuple(
        SecurityGroupRule(
            "tcp",
            port,
            port,
            source_security_group_id="${HOST_SECURITY_GROUP_ID}",
        )
        for port in spec.service_ports
    )
    teardown = (
        "stop new leases and allow only bounded artifact drain",
        "upload append-only artifacts and verify remote SHA-256",
        "terminate clients and delete unattached billable volumes",
        "terminate controller and Vast GPU",
        "delete run security groups and verify no tagged resources remain",
    )
    commands = (
        f"aws s3api head-bucket --bucket {ARTIFACT_BUCKET}",
        (
            "aws ec2 create-security-group "
            f"--group-name {host_name} --vpc-id {spec.vpc_id} "
            f"--description osworld-host-{spec.run_id}"
        ),
        (
            "aws ec2 authorize-security-group-ingress "
            "--group-id ${HOST_SECURITY_GROUP_ID} --protocol tcp --port 22 "
            f"--cidr {spec.controller_cidr}"
        ),
        (
            "aws ec2 create-security-group "
            f"--group-name {client_name} --vpc-id {spec.vpc_id} "
            f"--description osworld-clients-{spec.run_id}"
        ),
        *(
            "aws ec2 authorize-security-group-ingress "
            "--group-id ${CLIENT_SECURITY_GROUP_ID} --protocol tcp "
            f"--port {port} --source-group ${{HOST_SECURITY_GROUP_ID}}"
            for port in spec.service_ports
        ),
        (
            "vastai create instance "
            f"{spec.vast_offer.offer_id} --image {spec.vast_image} "
            f"--disk {spec.vast_disk_gb} --label osworld-{spec.run_id}"
        ),
        (
            "ssh -N -L "
            f"127.0.0.1:{spec.qwen_port}:127.0.0.1:{spec.qwen_port} "
            "${VAST_SSH_TARGET}"
        ),
        (
            "python -m osworld.run --provider aws "
            f"--region {spec.region} --vpc-id {spec.vpc_id} --subnet-id {spec.subnet_id} "
            "--client-security-group-id ${CLIENT_SECURITY_GROUP_ID} "
            "--use-private-client-addresses"
        ),
        f"aws s3 cp ${{ARTIFACT_PATH}} {spec.artifact_prefix} --no-clobber",
        f"aws s3api head-object --bucket {ARTIFACT_BUCKET} --key runs/{spec.run_id}/${{ARTIFACT_KEY}}",
    )
    return DryRunPlan(
        schema_version=1,
        spec=spec,
        host_security_group_name=host_name,
        client_security_group_name=client_name,
        host_rules=host_rules,
        client_rules=client_rules,
        teardown_plan=teardown,
        commands=commands,
    )


@dataclass(frozen=True)
class ApprovalArtifact:
    approval_id: str
    plan_hash: str
    run_id: str
    approved_at: str
    expires_at: str

    def as_dict(self) -> dict[str, str]:
        return {
            "approval_id": self.approval_id,
            "plan_hash": self.plan_hash,
            "run_id": self.run_id,
            "approved_at": self.approved_at,
            "expires_at": self.expires_at,
        }


def create_approval_artifact(
    plan: DryRunPlan,
    *,
    approved_at: datetime,
    expires_at: datetime,
) -> ApprovalArtifact:
    if approved_at.tzinfo is None or expires_at.tzinfo is None or expires_at <= approved_at:
        raise ApprovalError("approval timestamps must be aware and increasing")
    body = {
        "plan_hash": plan.content_hash,
        "run_id": plan.spec.run_id,
        "approved_at": approved_at.astimezone(UTC).isoformat(),
        "expires_at": expires_at.astimezone(UTC).isoformat(),
    }
    return ApprovalArtifact(approval_id=stable_sha256(body), **body)


def verify_approval(
    plan: DryRunPlan,
    approval: ApprovalArtifact | None,
    *,
    now: datetime | None = None,
) -> None:
    if approval is None:
        raise ApprovalError("a content-addressed approval artifact is required")
    body = {
        "plan_hash": approval.plan_hash,
        "run_id": approval.run_id,
        "approved_at": approval.approved_at,
        "expires_at": approval.expires_at,
    }
    if approval.approval_id != stable_sha256(body):
        raise ApprovalError("approval artifact content hash mismatch")
    if approval.plan_hash != plan.content_hash or approval.run_id != plan.spec.run_id:
        raise ApprovalError("approval does not bind the exact launch plan")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ApprovalError("approval verification time must be timezone-aware")
    if current < datetime.fromisoformat(approval.approved_at):
        raise ApprovalError("approval artifact is not active yet")
    if current >= datetime.fromisoformat(approval.expires_at):
        raise ApprovalError("approval artifact has expired")


@dataclass(frozen=True)
class ControllerRequest:
    run_id: str
    instance_type: str
    region: str
    vpc_id: str
    subnet_id: str
    security_group_id: str
    key_pair_name: str
    public_address: bool
    ttl_minutes: int
    tags: Mapping[str, str]


@dataclass(frozen=True)
class ClientRequest:
    run_id: str
    count: int
    instance_type: str
    region: str
    vpc_id: str
    subnet_id: str
    security_group_id: str
    private_addresses: bool
    ttl_minutes: int
    osworld_commit: str
    tags: Mapping[str, str]


@dataclass(frozen=True)
class Instance:
    resource_id: str
    private_address: str
    public_address: str | None = None
    volume_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VastInstance:
    resource_id: str
    ssh_target: str
    qwen_port: int


@dataclass(frozen=True)
class TaggedResource:
    provider: str
    resource_id: str
    kind: str
    run_id: str
    state: str
    ttl_expired: bool
    volume_ids: tuple[str, ...] = ()
    artifact_verified: bool = False


class AWSControl(Protocol):
    def ensure_bucket(self, policy: BucketPolicy) -> None: ...

    def create_security_group(
        self, *, name: str, vpc_id: str, tags: Mapping[str, str]
    ) -> str: ...

    def authorize_ingress(self, security_group_id: str, rule: SecurityGroupRule) -> None: ...

    def import_key_pair(self, name: str, public_key: bytes, tags: Mapping[str, str]) -> None: ...

    def delete_key_pair(self, name: str) -> None: ...

    def launch_controller(self, request: ControllerRequest) -> Instance: ...

    def launch_osworld_clients(self, request: ClientRequest) -> tuple[Instance, ...]: ...

    def schedule_termination(
        self, resource_id: str, *, ttl_minutes: int, run_id: str
    ) -> None: ...

    def issue_sts_credentials(self, *, run_id: str, ttl_minutes: int) -> object: ...

    def stop_new_leases(self, controller_id: str) -> None: ...

    def terminate_instance(self, resource_id: str) -> None: ...

    def delete_volume(self, volume_id: str) -> None: ...

    def delete_security_group(self, security_group_id: str) -> None: ...

    def sample_costs(self, run_id: str) -> tuple[CostSample, ...]: ...

    def list_tagged_resources(self) -> tuple[TaggedResource, ...]: ...


class VastControl(Protocol):
    def launch(
        self,
        *,
        offer: VastOffer,
        image: str,
        disk_gb: int,
        run_id: str,
        qwen_port: int,
    ) -> VastInstance: ...

    def schedule_termination(
        self, resource_id: str, *, ttl_minutes: int, run_id: str
    ) -> None: ...

    def destroy(self, resource_id: str) -> None: ...

    def sample_costs(self, run_id: str) -> tuple[CostSample, ...]: ...

    def list_tagged_resources(self) -> tuple[TaggedResource, ...]: ...


class S3Control(Protocol):
    def put_file_if_absent(self, local_path: Path, remote_uri: str) -> None: ...

    def remote_sha256(self, remote_uri: str) -> str: ...

    def run_export_verified(self, run_id: str) -> bool: ...


class SSHControl(Protocol):
    def generate_key(self, private_key_path: Path) -> bytes: ...

    def install_ephemeral_sts_credentials(
        self, controller: Instance, credentials: object, *, expires_in_minutes: int
    ) -> None: ...

    def establish_qwen_local_forward(
        self,
        controller: Instance,
        vast: VastInstance,
        *,
        local_bind_host: str,
        local_port: int,
        remote_bind_host: str,
        remote_port: int,
    ) -> None: ...

    def configure_osworld(
        self,
        controller: Instance,
        *,
        commit: str,
        client_security_group_id: str,
        use_private_client_addresses: bool,
        qwen_endpoint: str,
        artifact_prefix: str,
    ) -> None: ...


@dataclass
class BootstrapHandle:
    spec: CloudRunSpec
    controller: Instance
    clients: list[Instance]
    vast: VastInstance
    host_security_group_id: str
    client_security_group_id: str
    verified_resource_ids: set[str] = field(default_factory=set)
    terminated_resource_ids: set[str] = field(default_factory=set)

    @property
    def compute_resource_ids(self) -> set[str]:
        return {
            self.controller.resource_id,
            self.vast.resource_id,
            *(client.resource_id for client in self.clients),
        }


@dataclass(frozen=True)
class ArtifactExport:
    local_path: Path
    relative_key: str
    covered_resource_ids: tuple[str, ...]


class CloudOrchestrator:
    def __init__(
        self,
        aws: AWSControl,
        vast: VastControl,
        s3: S3Control,
        ssh: SSHControl,
        *,
        key_directory: Path,
    ) -> None:
        self.aws = aws
        self.vast = vast
        self.s3 = s3
        self.ssh = ssh
        self.key_directory = key_directory

    def bootstrap(
        self,
        plan: DryRunPlan,
        approval: ApprovalArtifact | None,
        *,
        now: datetime | None = None,
    ) -> BootstrapHandle:
        verify_approval(plan, approval, now=now)
        spec = plan.spec
        self.aws.ensure_bucket(BucketPolicy())
        host_sg: str | None = None
        client_sg: str | None = None
        controller: Instance | None = None
        vast_instance: VastInstance | None = None
        clients: tuple[Instance, ...] = ()
        key_name = f"osworld-{spec.run_id}-ephemeral"
        key_path = self.key_directory / f"{key_name}.pem"
        key_imported = False
        try:
            host_sg = self.aws.create_security_group(
                name=plan.host_security_group_name,
                vpc_id=spec.vpc_id,
                tags=spec.tags,
            )
            client_sg = self.aws.create_security_group(
                name=plan.client_security_group_name,
                vpc_id=spec.vpc_id,
                tags=spec.tags,
            )
            self.aws.authorize_ingress(host_sg, plan.host_rules[0])
            for port in spec.service_ports:
                self.aws.authorize_ingress(
                    client_sg,
                    SecurityGroupRule(
                        "tcp",
                        port,
                        port,
                        source_security_group_id=host_sg,
                    ),
                )
            self.key_directory.mkdir(parents=True, exist_ok=True)
            public_key = self.ssh.generate_key(key_path)
            os.chmod(key_path, 0o600)
            if key_path.stat().st_mode & 0o777 != 0o600:
                raise BootstrapValidationError("ephemeral private key mode must be 0600")
            self.aws.import_key_pair(key_name, public_key, spec.tags)
            key_imported = True
            controller = self.aws.launch_controller(
                ControllerRequest(
                    run_id=spec.run_id,
                    instance_type=spec.policy.controller_type,
                    region=spec.region,
                    vpc_id=spec.vpc_id,
                    subnet_id=spec.subnet_id,
                    security_group_id=host_sg,
                    key_pair_name=key_name,
                    public_address=True,
                    ttl_minutes=spec.controller_ttl_minutes,
                    tags=spec.tags,
                )
            )
            self.aws.schedule_termination(
                controller.resource_id,
                ttl_minutes=spec.controller_ttl_minutes,
                run_id=spec.run_id,
            )
            vast_instance = self.vast.launch(
                offer=spec.vast_offer,
                image=spec.vast_image,
                disk_gb=spec.vast_disk_gb,
                run_id=spec.run_id,
                qwen_port=spec.qwen_port,
            )
            self.vast.schedule_termination(
                vast_instance.resource_id,
                ttl_minutes=spec.vast_ttl_minutes,
                run_id=spec.run_id,
            )
            credentials = self.aws.issue_sts_credentials(
                run_id=spec.run_id,
                ttl_minutes=min(spec.controller_ttl_minutes, 60),
            )
            self.ssh.install_ephemeral_sts_credentials(
                controller,
                credentials,
                expires_in_minutes=min(spec.controller_ttl_minutes, 60),
            )
            self.ssh.establish_qwen_local_forward(
                controller,
                vast_instance,
                local_bind_host="127.0.0.1",
                local_port=spec.qwen_port,
                remote_bind_host="127.0.0.1",
                remote_port=vast_instance.qwen_port,
            )
            self.ssh.configure_osworld(
                controller,
                commit=spec.osworld_commit,
                client_security_group_id=client_sg,
                use_private_client_addresses=True,
                qwen_endpoint=f"http://127.0.0.1:{spec.qwen_port}/v1",
                artifact_prefix=spec.artifact_prefix,
            )
            clients = self.aws.launch_osworld_clients(
                ClientRequest(
                    run_id=spec.run_id,
                    count=spec.client_count,
                    instance_type=OSWORLD_CLIENT_TYPE,
                    region=spec.region,
                    vpc_id=spec.vpc_id,
                    subnet_id=spec.subnet_id,
                    security_group_id=client_sg,
                    private_addresses=True,
                    ttl_minutes=spec.client_ttl_minutes,
                    osworld_commit=spec.osworld_commit,
                    tags=spec.tags,
                )
            )
            if len(clients) != spec.client_count or any(
                not client.private_address for client in clients
            ):
                raise RuntimeError("official OSWorld provider returned an invalid client set")
            for client in clients:
                self.aws.schedule_termination(
                    client.resource_id,
                    ttl_minutes=spec.client_ttl_minutes,
                    run_id=spec.run_id,
                )
            return BootstrapHandle(
                spec=spec,
                controller=controller,
                clients=list(clients),
                vast=vast_instance,
                host_security_group_id=host_sg,
                client_security_group_id=client_sg,
            )
        except Exception:
            for client in clients:
                self.aws.terminate_instance(client.resource_id)
            if controller is not None:
                self.aws.terminate_instance(controller.resource_id)
            if vast_instance is not None:
                self.vast.destroy(vast_instance.resource_id)
            if client_sg is not None:
                self.aws.delete_security_group(client_sg)
            if host_sg is not None:
                self.aws.delete_security_group(host_sg)
            raise
        finally:
            key_path.unlink(missing_ok=True)
            if key_imported:
                self.aws.delete_key_pair(key_name)

    def verify_exports(
        self,
        handle: BootstrapHandle,
        exports: Sequence[ArtifactExport],
    ) -> None:
        covered: set[str] = set()
        for export in exports:
            if export.relative_key.startswith("/") or ".." in Path(export.relative_key).parts:
                raise BootstrapValidationError("artifact keys must remain in the run prefix")
            remote_uri = f"{handle.spec.artifact_prefix}{export.relative_key}"
            self.s3.put_file_if_absent(export.local_path, remote_uri)
            local_hash = file_sha256(export.local_path)
            if self.s3.remote_sha256(remote_uri).lower() != local_hash:
                raise RuntimeError("remote artifact SHA-256 verification failed")
            covered.update(export.covered_resource_ids)
        unknown = covered - handle.compute_resource_ids
        if unknown:
            raise BootstrapValidationError("artifact export covers unknown resources")
        handle.verified_resource_ids.update(covered)

    def shrink_tail(
        self,
        handle: BootstrapHandle,
        *,
        remaining_tasks: int,
    ) -> int:
        target = desired_worker_count(remaining_tasks, handle.spec.client_count)
        surplus = handle.clients[target:]
        for client in surplus:
            if client.resource_id not in handle.verified_resource_ids:
                raise RuntimeError("tail worker artifact is not checksum-verified")
        for client in surplus:
            self.aws.terminate_instance(client.resource_id)
            for volume_id in client.volume_ids:
                self.aws.delete_volume(volume_id)
            handle.terminated_resource_ids.add(client.resource_id)
        del handle.clients[target:]
        return target

    def teardown(self, handle: BootstrapHandle) -> None:
        remaining = handle.compute_resource_ids - handle.terminated_resource_ids
        if not remaining <= handle.verified_resource_ids:
            raise RuntimeError("all compute artifacts must be verified before teardown")
        for client in tuple(handle.clients):
            if client.resource_id not in handle.terminated_resource_ids:
                self.aws.terminate_instance(client.resource_id)
                for volume_id in client.volume_ids:
                    self.aws.delete_volume(volume_id)
                handle.terminated_resource_ids.add(client.resource_id)
        self.aws.terminate_instance(handle.controller.resource_id)
        for volume_id in handle.controller.volume_ids:
            self.aws.delete_volume(volume_id)
        handle.terminated_resource_ids.add(handle.controller.resource_id)
        self.vast.destroy(handle.vast.resource_id)
        handle.terminated_resource_ids.add(handle.vast.resource_id)
        self.aws.delete_security_group(handle.client_security_group_id)
        self.aws.delete_security_group(handle.host_security_group_id)

    def sample_spend(
        self,
        handle: BootstrapHandle,
        ledger: CostLedger,
        *,
        authorized_active_cost_usd: Decimal = Decimal("0"),
        other_model_run_cost_usd: Decimal = Decimal("0"),
    ) -> BudgetDecision:
        for sample in (*self.aws.sample_costs(handle.spec.run_id), *self.vast.sample_costs(handle.spec.run_id)):
            ledger.append(sample)
        decision = handle.spec.policy.budget.decide(
            ledger.total(),
            authorized_active_cost_usd,
        )
        combined = EvaluationBudgetPolicy().combined_decision(
            ledger.total(),
            other_model_run_cost_usd,
        )
        if combined is BudgetDecision.HARD_STOP:
            return combined
        return decision

    def hard_stop(
        self,
        handle: BootstrapHandle,
        *,
        artifact_drain_seconds: int,
        drain: Callable[[int], Sequence[ArtifactExport]],
    ) -> None:
        if artifact_drain_seconds <= 0:
            raise BootstrapValidationError("artifact drain must have a positive bound")
        self.aws.stop_new_leases(handle.controller.resource_id)
        exports = drain(artifact_drain_seconds)
        self.verify_exports(handle, exports)
        self.teardown(handle)

    def reconcile_orphans(self, *, run_id: str) -> tuple[str, ...]:
        if not run_id:
            raise BootstrapValidationError("run_id is required for reconciliation")
        resources = (
            *self.aws.list_tagged_resources(),
            *self.vast.list_tagged_resources(),
        )
        cleaned = []
        for resource in resources:
            if resource.run_id != run_id or resource.state == "terminated":
                continue
            safe = resource.artifact_verified or self.s3.run_export_verified(run_id)
            if not safe and not resource.ttl_expired:
                continue
            if resource.provider == "aws":
                self.aws.terminate_instance(resource.resource_id)
                for volume_id in resource.volume_ids:
                    self.aws.delete_volume(volume_id)
            elif resource.provider == "vast":
                self.vast.destroy(resource.resource_id)
            else:
                continue
            cleaned.append(resource.resource_id)
        return tuple(cleaned)


def render_plan_json(plan: DryRunPlan) -> str:
    rendered = plan.as_dict()
    encoded = canonical_json(rendered)
    forbidden = (
        "AWS_SECRET_ACCESS_KEY=",
        "AWS_SESSION_TOKEN=",
        "VAST_API_KEY=",
        "QWEN_ENDPOINT_API_KEY=",
        "BEGIN PRIVATE KEY",
    )
    if any(value in encoded for value in forbidden):
        raise BootstrapValidationError("dry-run plan contains secret material")
    return json.dumps(rendered, indent=2, sort_keys=True) + "\n"
