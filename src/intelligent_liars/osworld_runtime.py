from __future__ import annotations

import base64
import json
import os
import shutil
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from intelligent_liars.run_control import (
    BudgetDecision,
    LedgerValidationError,
    _append_json_line,
    _decimal,
    file_sha256,
    now_iso,
)


VAST_SKILL_SOURCE = {
    "repository": "p3rciv3l/avi-skills",
    "commit": "f453c250b3c5a684f42f6cf6abb97463dddda14c",
    "path": ".agents/skills/vast-gpu-experiments/",
    "scripts": (
        "vast_find_offer.py",
        "vast_run_workload.py",
        "vast_cleanup.py",
    ),
}

VAST_REQUIRED_KEY_SCOPES = {
    "misc": "offer search",
    "instance_read": "inventory, status, logs, and volumes",
    "instance_write": "create, manage, and destroy",
}
VAST_EXCLUDED_KEY_SCOPES = (
    "billing_write",
    "user_write",
    "machine_*",
    "payment access",
)
AWS_MINIMUM_RUNTIME_ACTIONS = {
    "identity": ("sts:GetCallerIdentity",),
    "ec2": (
        "ec2:DescribeImages",
        "ec2:DescribeInstances",
        "ec2:DescribeTags",
        "ec2:DescribeVolumes",
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:CreateTags",
        "ec2:DeleteVolume",
    ),
    "scheduler": (
        "scheduler:CreateSchedule",
        "scheduler:GetSchedule",
        "scheduler:DeleteSchedule",
        "iam:PassRole",
    ),
    "artifacts": (
        "s3:ListBucket",
        "s3:PutObject",
        "s3:GetObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts",
    ),
    "optional_cost_read": ("ce:GetCostAndUsage",),
}

REQUIRED_ENV_NAMES = (
    "VAST_API_KEY",
    "AWS_REGION",
    "QWEN_ENDPOINT_URL",
    "QWEN_ENDPOINT_API_KEY",
    "OSWORLD_CLIENT_PASSWORD",
    "OSWORLD_ARTIFACT_DESTINATION",
)
AWS_BOOTSTRAP_ENV_NAMES = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
OPTIONAL_ENV_NAMES = ("AWS_SESSION_TOKEN", "VASTAI_PATH")


@dataclass(frozen=True)
class EnvironmentStatus:
    name: str
    required: bool
    present: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "required": self.required,
            "present": self.present,
        }


def environment_statuses(
    environment: Mapping[str, str],
    *,
    aws_auth_mode: str = "controller-iam-role",
) -> tuple[EnvironmentStatus, ...]:
    if aws_auth_mode not in {"controller-iam-role", "bootstrap-access-keys"}:
        raise ValueError("aws_auth_mode must be controller-iam-role or bootstrap-access-keys")
    return tuple(
        EnvironmentStatus(name, True, bool(environment.get(name)))
        for name in REQUIRED_ENV_NAMES
    ) + tuple(
        EnvironmentStatus(
            name,
            aws_auth_mode == "bootstrap-access-keys",
            bool(environment.get(name)),
        )
        for name in AWS_BOOTSTRAP_ENV_NAMES
    ) + tuple(
        EnvironmentStatus(name, False, bool(environment.get(name)))
        for name in OPTIONAL_ENV_NAMES
    )


def locate_vastai_path(environment: Mapping[str, str]) -> Path | None:
    configured = environment.get("VASTAI_PATH")
    candidate = Path(configured).expanduser() if configured else None
    if candidate is None:
        discovered = shutil.which("vastai")
        candidate = Path(discovered) if discovered else None
    if (
        candidate is None
        or not candidate.is_file()
        or not os.access(candidate, os.X_OK)
    ):
        return None
    return candidate.resolve()


@dataclass(frozen=True)
class InfrastructureTarget:
    desktop_provider: str = "aws"
    execution_mode: str = "official-osworld-host-client"
    nested_kvm: bool = False

    def validate(self) -> None:
        if self.desktop_provider != "aws":
            raise ValueError("desktop_provider must be official AWS OSWorld")
        if self.execution_mode != "official-osworld-host-client":
            raise ValueError("execution_mode must be official-osworld-host-client")
        if self.nested_kvm:
            raise ValueError("nested KVM on Capy is prohibited")


@dataclass(frozen=True)
class VastLaunchPolicy:
    offer_id: str
    estimated_hourly_rate_usd: Decimal
    price_approved: bool
    gpu_count: int = 1

    def __post_init__(self) -> None:
        if not self.offer_id:
            raise ValueError("a concrete Vast offer_id is required")
        object.__setattr__(
            self,
            "estimated_hourly_rate_usd",
            _decimal(self.estimated_hourly_rate_usd, "estimated_hourly_rate_usd"),
        )
        if self.gpu_count != 1:
            raise ValueError("Vast launch policy requires exactly one GPU by default")


@dataclass(frozen=True)
class OfflinePreflight:
    environment: tuple[EnvironmentStatus, ...]
    vastai_cli_available: bool
    target_valid: bool
    concrete_vast_offer: bool
    vast_price_approved: bool
    aws_auth_mode: str
    dry_run: bool = True

    @property
    def ready(self) -> bool:
        return (
            all(item.present for item in self.environment if item.required)
            and self.vastai_cli_available
            and self.target_valid
            and self.concrete_vast_offer
            and self.vast_price_approved
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "ready": self.ready,
            "credential_values_included": False,
            "environment": [item.as_dict() for item in self.environment],
            "vastai_cli_available": self.vastai_cli_available,
            "target_valid": self.target_valid,
            "concrete_vast_offer": self.concrete_vast_offer,
            "vast_price_approved": self.vast_price_approved,
            "aws_auth_mode": self.aws_auth_mode,
            "authorization": {
                "vast_required_scopes": VAST_REQUIRED_KEY_SCOPES,
                "vast_excluded_scopes": list(VAST_EXCLUDED_KEY_SCOPES),
                "aws_minimum_runtime_actions": {
                    group: list(actions)
                    for group, actions in AWS_MINIMUM_RUNTIME_ACTIONS.items()
                },
                "aws_scope_requirement": (
                    "run VPC/resources, artifact prefix, and termination role only"
                ),
            },
            "source": VAST_SKILL_SOURCE,
        }


def offline_preflight(
    environment: Mapping[str, str],
    target: InfrastructureTarget,
    vast_policy: VastLaunchPolicy,
    *,
    aws_auth_mode: str = "controller-iam-role",
) -> OfflinePreflight:
    target.validate()
    return OfflinePreflight(
        environment=environment_statuses(environment, aws_auth_mode=aws_auth_mode),
        vastai_cli_available=locate_vastai_path(environment) is not None,
        target_valid=True,
        concrete_vast_offer=bool(vast_policy.offer_id),
        vast_price_approved=vast_policy.price_approved,
        aws_auth_mode=aws_auth_mode,
    )


@dataclass(frozen=True)
class QwenMultimodalRequest:
    endpoint_url: str
    model: str
    messages: tuple[Mapping[str, Any], ...]
    max_tokens: int


class QwenHTTPClient(Protocol):
    def create_chat_completion(
        self,
        request: QwenMultimodalRequest,
        *,
        bearer_token: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class QwenSmokeResult:
    response_id: str
    model: str
    assistant_content_present: bool


def run_qwen_multimodal_smoke_test(
    client: QwenHTTPClient,
    *,
    endpoint_url: str,
    bearer_token: str,
    model: str,
    image_png: bytes,
) -> QwenSmokeResult:
    if not endpoint_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise ValueError("Qwen endpoint must use HTTPS except for loopback tests")
    if not bearer_token:
        raise ValueError("Qwen endpoint authentication is required")
    if not image_png:
        raise ValueError("a PNG image is required")
    image_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
    request = QwenMultimodalRequest(
        endpoint_url=endpoint_url.rstrip("/") + "/chat/completions",
        model=model,
        messages=(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this screenshot briefly."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ),
        max_tokens=32,
    )
    response = client.create_chat_completion(request, bearer_token=bearer_token)
    try:
        response_id = response["id"]
        response_model = response["model"]
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("invalid OpenAI-compatible multimodal response") from exc
    if not isinstance(response_id, str) or not isinstance(response_model, str):
        raise ValueError("invalid OpenAI-compatible response identity")
    return QwenSmokeResult(response_id, response_model, bool(content))


class ArtifactStore(Protocol):
    def sha256(self, remote_uri: str) -> str: ...


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    run_id: str
    resource_id: str
    local_path: str
    remote_uri: str
    sha256: str
    verified_at: str

    def as_dict(self) -> dict[str, str]:
        return {
            "event": "artifact_verified",
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "resource_id": self.resource_id,
            "local_path": self.local_path,
            "remote_uri": self.remote_uri,
            "sha256": self.sha256,
            "verified_at": self.verified_at,
        }


class ArtifactManifest:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id

    def verify_and_append(
        self,
        *,
        artifact_id: str,
        resource_id: str,
        local_path: Path,
        remote_uri: str,
        store: ArtifactStore,
    ) -> VerifiedArtifact:
        local_hash = file_sha256(local_path)
        remote_hash = store.sha256(remote_uri)
        if local_hash.lower() != remote_hash.lower():
            raise LedgerValidationError("local and remote artifact SHA-256 hashes differ")
        artifact = VerifiedArtifact(
            artifact_id=artifact_id,
            run_id=self.run_id,
            resource_id=resource_id,
            local_path=str(local_path),
            remote_uri=remote_uri,
            sha256=local_hash,
            verified_at=now_iso(),
        )
        if artifact_id in self.verified_artifact_ids():
            raise LedgerValidationError("artifact_id already exists")
        _append_json_line(self.path, artifact.as_dict())
        return artifact

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), start=1):
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise LedgerValidationError(
                    f"invalid artifact manifest event at line {line_number}"
                ) from exc
            events.append(event)
        return events

    def verified_artifact_ids(self) -> set[str]:
        return {
            event["artifact_id"]
            for event in self.events()
            if event.get("event") == "artifact_verified"
        }

    def verified_resource_ids(self) -> set[str]:
        return {
            event["resource_id"]
            for event in self.events()
            if event.get("event") == "artifact_verified"
        }

    def record_termination(self, provider: str, resource_id: str) -> None:
        if resource_id not in self.verified_resource_ids():
            raise LedgerValidationError("resource artifacts are not durably verified")
        if resource_id in self.termination_ids():
            raise LedgerValidationError("resource termination already recorded")
        _append_json_line(
            self.path,
            {
                "event": "resource_terminated",
                "provider": provider,
                "resource_id": resource_id,
                "terminated_at": now_iso(),
            },
        )

    def termination_ids(self) -> set[str]:
        return {
            event["resource_id"]
            for event in self.events()
            if event.get("event") == "resource_terminated"
        }


class WatchdogState(StrEnum):
    ACCEPTING = "accepting"
    DRAINING = "draining"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class ManagedResource:
    provider: str
    resource_id: str
    label: str
    ordinal: int
    active_lease: bool
    completed: bool
    state: str = "running"
    stale: bool = False


@dataclass(frozen=True)
class BillableVolume:
    volume_id: str
    source_resource_id: str
    attached_to: str | None


@dataclass(frozen=True)
class WatchdogPlan:
    state: WatchdogState
    accept_new_leases: bool
    allow_artifact_export: bool
    terminate_resource_ids: tuple[str, ...]
    delete_volume_ids: tuple[str, ...]
    awaiting_verification_ids: tuple[str, ...]


class LifecycleActions(Protocol):
    def terminate_resource(self, provider: str, resource_id: str) -> None: ...

    def delete_volume(self, provider: str, volume_id: str) -> None: ...


class LifecycleWatchdog:
    provider: str

    def __init__(self, managed_label: str) -> None:
        self.managed_label = managed_label
        self.state = WatchdogState.ACCEPTING

    def reconcile(
        self,
        *,
        resources: Sequence[ManagedResource],
        volumes: Sequence[BillableVolume],
        desired_workers: int,
        endpoint_healthy: bool,
        budget_decision: BudgetDecision,
        artifacts: ArtifactManifest,
    ) -> WatchdogPlan:
        if desired_workers < 0:
            raise ValueError("desired_workers must be non-negative")
        managed = sorted(
            (
                resource
                for resource in resources
                if resource.provider == self.provider
                and resource.label == self.managed_label
                and resource.state != "terminated"
            ),
            key=lambda resource: resource.ordinal,
        )
        healthy = endpoint_healthy and budget_decision is BudgetDecision.CONTINUE
        accepting = healthy and desired_workers > 0
        draining = not healthy or desired_workers == 0
        verified = artifacts.verified_resource_ids()
        terminate = []
        awaiting = []
        for index, resource in enumerate(managed):
            surplus = index >= desired_workers
            candidate = not resource.active_lease and (
                resource.completed
                or surplus
                or draining
                or resource.state == "stopped"
                or resource.stale
            )
            if not candidate:
                continue
            if resource.resource_id in verified:
                terminate.append(resource.resource_id)
            else:
                awaiting.append(resource.resource_id)

        delete_volumes = tuple(
            volume.volume_id
            for volume in volumes
            if volume.source_resource_id in terminate
            or (
                volume.attached_to is None
                and volume.source_resource_id in verified
            )
        )
        active_or_waiting = any(resource.active_lease for resource in managed) or bool(awaiting)
        if not managed and not volumes and draining:
            state = WatchdogState.TERMINATED
        elif draining and active_or_waiting:
            state = WatchdogState.DRAINING
        elif terminate or delete_volumes:
            state = WatchdogState.TERMINATING
        else:
            state = WatchdogState.ACCEPTING if accepting else WatchdogState.DRAINING
        self.state = state
        return WatchdogPlan(
            state=state,
            accept_new_leases=accepting,
            allow_artifact_export=active_or_waiting,
            terminate_resource_ids=tuple(terminate),
            delete_volume_ids=delete_volumes,
            awaiting_verification_ids=tuple(awaiting),
        )


class AWSLifecycleWatchdog(LifecycleWatchdog):
    provider = "aws"


class VastLifecycleWatchdog(LifecycleWatchdog):
    provider = "vast"


def apply_watchdog_plan(
    plan: WatchdogPlan,
    actions: LifecycleActions,
    *,
    provider: str,
    dry_run: bool = True,
) -> None:
    if dry_run:
        return
    for resource_id in plan.terminate_resource_ids:
        actions.terminate_resource(provider, resource_id)
    for volume_id in plan.delete_volume_ids:
        actions.delete_volume(provider, volume_id)
