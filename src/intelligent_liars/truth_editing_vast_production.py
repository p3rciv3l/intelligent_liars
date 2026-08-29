"""Bounded Vast lifecycle for a truth-editing production study phase.

This module deepens the prerequisite lifecycle rather than creating another cloud
controller: it adds pinned model-cache hydration, periodic durable checkpoint
publication, a compact output fetch, and a post-destroy zero-lineage proof.  Offer
IDs are inputs that are revalidated at execution time, never configuration authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from intelligent_liars.truth_editing_vast_prerequisites import (
    EphemeralWorkloadSecret,
    FROZEN_RUNTIME_ID,
    JobConfig,
    Offer,
    RunCommand,
    VastPrerequisiteError,
    build_bundle,
    canonical_bytes,
    execute_lifecycle,
    lifecycle_plan,
    sha256,
)
from intelligent_liars.truth_editing_final_checkpoint_publication import (
    FinalCheckpointPublicationError,
    validate_compact_vast_output_contract,
)


FORMAT = "truth_editing_vast_production_job_v1"
ADAPTIVE_PRODUCTION_JOB_FORMAT = "truth_editing_vast_adaptive_production_job_v1"
PLAN_FORMAT = "truth_editing_vast_production_dry_run_v1"
RECEIPT_FORMAT = "truth_editing_vast_production_lifecycle_v1"
PHASES = frozenset(
    {"discovery", "expanded", "finalist", "adaptive", "timed_canary"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProductionVastError(RuntimeError):
    """The production job violated a frozen identity or lifecycle boundary."""


@dataclass(frozen=True)
class AdaptiveProductionJobConfig(JobConfig):
    """The long-running main-study lifecycle contract.

    This distinct version keeps the prerequisite, timed-canary, and legacy
    production parser limits unchanged while permitting the explicitly reserved
    24-hour, 45 USD infrastructure envelope for the adaptive main run.
    """

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdaptiveProductionJobConfig":
        return cls._from_mapping(  # type: ignore[return-value]
            raw, expected_format=ADAPTIVE_PRODUCTION_JOB_FORMAT
        )

    def validate(self) -> None:
        self._validate_frozen_invariants()
        if self.runtime_id != FROZEN_RUNTIME_ID:
            raise VastPrerequisiteError(
                "adaptive production requires the current off-host checkpoint runtime"
            )
        if self.maximum_cost_usd > 45.0:
            raise VastPrerequisiteError(
                "adaptive production infrastructure cost exceeds $45"
            )
        if self.maximum_elapsed_seconds > 24 * 3600:
            raise VastPrerequisiteError(
                "adaptive production elapsed cap exceeds 24 hours"
            )


@dataclass(frozen=True)
class ProductionVastConfig:
    phase: str
    base_job: JobConfig
    model_cache_directory: str
    expected_model_sha256: str
    expected_snapshot_manifest_sha256: str
    hydrate_command: tuple[str, ...]
    verify_command: tuple[str, ...]
    production_config_path: str
    production_config_sha256: str
    workload_command: tuple[str, ...]
    checkpoint_directory: str
    checkpoint_interval_seconds: int
    checkpoint_stream_command: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProductionVastConfig":
        if set(raw) != {"format", "phase", "base_job", "model_cache", "study", "checkpoints"} or raw.get("format") != FORMAT:
            raise ProductionVastError("production job fields or format changed")
        phase = _text(raw["phase"], "phase")
        if phase not in PHASES:
            raise ProductionVastError(
                "phase must be discovery, expanded, finalist, adaptive, or timed_canary"
            )
        cache = _exact(
            raw["model_cache"],
            {
                "remote_directory",
                "expected_model_sha256",
                "expected_snapshot_manifest_sha256",
                "hydrate_command",
                "verify_command",
            },
            "model_cache",
        )
        study = _exact(
            raw["study"],
            {
                "production_config_path",
                "production_config_sha256",
                "workload_command",
            },
            "study",
        )
        checkpoints = _exact(raw["checkpoints"], {"remote_directory", "interval_seconds", "stream_command"}, "checkpoints")
        base_raw = _mapping(raw["base_job"], "base_job")
        base_format = base_raw.get("format")
        if phase == "adaptive" and base_format != ADAPTIVE_PRODUCTION_JOB_FORMAT:
            raise ProductionVastError(
                "adaptive production requires its versioned adaptive base job"
            )
        if phase != "adaptive" and base_format == ADAPTIVE_PRODUCTION_JOB_FORMAT:
            role = "timed canary" if phase == "timed_canary" else "legacy production"
            raise ProductionVastError(
                f"{role} cannot use the adaptive production base job"
            )
        try:
            base = (
                AdaptiveProductionJobConfig.from_mapping(base_raw)
                if phase == "adaptive"
                else JobConfig.from_mapping(base_raw)
            )
        except Exception as error:
            raise ProductionVastError(f"base_job is invalid: {error}") from error
        config = cls(
            phase=phase,
            base_job=base,
            model_cache_directory=_absolute(cache["remote_directory"], "model_cache.remote_directory"),
            expected_model_sha256=_digest(cache["expected_model_sha256"], "model_cache.expected_model_sha256"),
            expected_snapshot_manifest_sha256=_digest(
                cache["expected_snapshot_manifest_sha256"],
                "model_cache.expected_snapshot_manifest_sha256",
            ),
            hydrate_command=_command(cache["hydrate_command"], "model_cache.hydrate_command"),
            verify_command=_command(cache["verify_command"], "model_cache.verify_command"),
            production_config_path=_relative(study["production_config_path"], "study.production_config_path"),
            production_config_sha256=_digest(
                study["production_config_sha256"],
                "study.production_config_sha256",
            ),
            workload_command=_command(study["workload_command"], "study.workload_command"),
            checkpoint_directory=_absolute(checkpoints["remote_directory"], "checkpoints.remote_directory"),
            checkpoint_interval_seconds=_integer(checkpoints["interval_seconds"], "checkpoints.interval_seconds", 30),
            checkpoint_stream_command=_command(checkpoints["stream_command"], "checkpoints.stream_command"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.production_config_path not in self.base_job.bundle_paths:
            raise ProductionVastError("production config must be present in the bundle allowlist")
        if self.checkpoint_directory == self.base_job.remote_output_dir:
            raise ProductionVastError("checkpoint directory must be below, not equal to, output directory")
        output_prefix = self.base_job.remote_output_dir.rstrip("/") + "/"
        if not self.checkpoint_directory.startswith(output_prefix):
            raise ProductionVastError("checkpoint directory must be inside the compact output directory")
        if self.model_cache_directory in {self.base_job.remote_workdir, self.base_job.remote_output_dir}:
            raise ProductionVastError("model cache must have a dedicated remote directory")
        if self.phase == "timed_canary":
            if not any(
                "run_truth_editing_timed_canary.py" in part
                for part in self.workload_command
            ):
                raise ProductionVastError(
                    "timed canary must select the canonical timed-canary entrypoint"
                )
            required_canary_outputs = {
                "timed-canary-v6-adaptive-r10/observation.json",
                "timed-canary-v6-adaptive-r10/receipt.json",
                "timed-canary-v6-adaptive-r10/monitoring/wandb-run.json",
                "timed-canary-v6-adaptive-r10/monitoring/wandb-events.jsonl",
                "timed-canary-v6-adaptive-r10/providers/production-judge-budget/manifest.json",
                "model/cache-hydration-receipt.json",
                "model/model-verification-receipt.json",
            }
            if not required_canary_outputs.issubset(self.base_job.expected_outputs):
                raise ProductionVastError(
                    "timed canary outputs must include observation, receipt, monitoring, and model receipts"
                )
        elif self.phase == "adaptive":
            if (
                not any(
                    part.endswith("run_truth_editing_cuda_fleet_controller.py")
                    for part in self.workload_command
                )
                or "--phase" in self.workload_command
            ):
                raise ProductionVastError(
                    "adaptive workload must use the phase-free CUDA fleet controller"
                )
            for flag in (
                "--fleet-config",
                "--capacity-policy",
                "--capacity-receipt",
                "--output-root",
                "--checkpoint-publication-root",
                "--model-registry-config",
                "--offhost-key-prefix",
                "--final-model-slug",
            ):
                if flag not in self.workload_command:
                    raise ProductionVastError(
                        f"adaptive workload must bind {flag}"
                    )
            required_adaptive_outputs = {
                "checkpoints/adaptive-latest.json",
                "finalization/final-model-publication-receipt.json",
            }
            if not required_adaptive_outputs.issubset(
                self.base_job.expected_outputs
            ):
                raise ProductionVastError(
                    "adaptive outputs must include adaptive-latest.json and the "
                    "verified final model publication receipt"
                )
            if (
                "--adaptive" not in self.checkpoint_stream_command
                or "--study-config-sha256" not in self.checkpoint_stream_command
                or "--phase" in self.checkpoint_stream_command
            ):
                raise ProductionVastError(
                    "adaptive checkpoint stream must bind adaptive study identity"
                )
        elif not any(
            left == "--phase" and right == self.phase
            for left, right in zip(self.workload_command, self.workload_command[1:])
        ):
            raise ProductionVastError(
                "workload command must bind the selected phase as --phase <phase>"
            )
        if self.base_job.maximum_upload_gib > 1.0:
            raise ProductionVastError("compact output fetch must be capped at 1 GiB")
        try:
            validate_compact_vast_output_contract(
                expected_outputs=self.base_job.expected_outputs,
                maximum_upload_gib=self.base_job.maximum_upload_gib,
            )
        except FinalCheckpointPublicationError as error:
            raise ProductionVastError(str(error)) from error
        if self.phase != "timed_canary" and not any(
            path.startswith("checkpoints/")
            for path in self.base_job.expected_outputs
        ):
            raise ProductionVastError("compact outputs must include a checkpoint receipt")

    def validate_repository(
        self,
        repo: Path,
        *,
        production_config_opener: Callable[[Path], Any] | None = None,
    ) -> None:
        """Strict-open and hash-check the exact production config selected by the job."""

        config_path = repo.resolve() / self.production_config_path
        if config_path.is_symlink() or not config_path.is_file():
            raise ProductionVastError("production config is missing or unsafe")
        actual_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if actual_sha256 != self.production_config_sha256:
            raise ProductionVastError("production config SHA-256 differs from the job binding")
        if production_config_opener is None:
            from .truth_editing_production import ProductionRunConfig

            production_config_opener = ProductionRunConfig.open
        try:
            production = production_config_opener(config_path)
        except Exception as error:
            raise ProductionVastError(
                f"production config strict-open failed: {type(error).__name__}: {error}"
            ) from error
        if (
            production.verified_model_sha256 != self.expected_model_sha256
            or production.verified_snapshot_manifest_sha256
            != self.expected_snapshot_manifest_sha256
        ):
            raise ProductionVastError(
                "production config model identity differs from the Vast job"
            )


def build_production_bundle(
    repo: Path,
    config: ProductionVastConfig,
    output: Path,
    *,
    maximum_bytes: int = 1024**3,
    production_config_opener: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Create the deterministic, allowlisted production source bundle."""

    config.validate_repository(
        repo, production_config_opener=production_config_opener
    )
    return dict(build_bundle(repo, config.base_job, output, maximum_bytes=maximum_bytes))


def production_lifecycle_plan(
    *,
    vastai: str,
    config: ProductionVastConfig,
    offer: Offer,
    bundle: Path,
    fetch_dir: Path,
    ssh_identity: Path | None = None,
) -> dict[str, Any]:
    """Build a non-mutating plan; the supplied ephemeral offer is revalidated later."""

    host_hourly_usd = offer.requested_hourly_price_usd(config.base_job.disk_gib)
    compound = _remote_production_command(config, host_hourly_usd=host_hourly_usd)
    base = replace(
        config.base_job,
        workload_command=("bash", "-lc", compound),
    )
    base_plan = lifecycle_plan(
        vastai=vastai,
        config=base,
        offer=offer,
        bundle=bundle,
        fetch_dir=fetch_dir,
        ssh_identity=ssh_identity,
    )
    unsigned = {
        "format": PLAN_FORMAT,
        "phase": config.phase,
        "label": base_plan["label"],
        "offer_id_candidate": offer.offer_id,
        "model": {
            "repository": base.model_id,
            "revision": base.model_revision,
            "cache_directory": config.model_cache_directory,
            "expected_model_sha256": config.expected_model_sha256,
            "expected_snapshot_manifest_sha256": config.expected_snapshot_manifest_sha256,
        },
        "production_config_path": config.production_config_path,
        "production_config_sha256": config.production_config_sha256,
        "checkpoint": {
            "remote_directory": config.checkpoint_directory,
            "interval_seconds": config.checkpoint_interval_seconds,
        },
        "base_lifecycle": base_plan,
    }
    return {**unsigned, "self_sha256": sha256(unsigned)}


BaseExecute = Callable[..., dict[str, Any]]


def execute_production_lifecycle(
    *,
    plan: Mapping[str, Any],
    config: ProductionVastConfig,
    metadata_path: Path,
    workload_secret: EphemeralWorkloadSecret,
    base_execute: BaseExecute = execute_lifecycle,
    run_command: RunCommand = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    minimum_aws_validity_seconds: int | None = None,
) -> dict[str, Any]:
    """Execute the base lifecycle and prove no instance from its label remains.

    The base controller performs exact offer/created-instance validation, bounded
    execution, verified compact fetch, and unconditional destruction.  Its receipt
    is retained beside this receipt even if the final lineage assertion fails.
    """

    if metadata_path.exists():
        raise ProductionVastError("production lifecycle metadata already exists")
    if plan.get("format") != PLAN_FORMAT or plan.get("self_sha256") != sha256(
        {key: value for key, value in plan.items() if key != "self_sha256"}
    ):
        raise ProductionVastError("production lifecycle plan identity is invalid")
    base_plan = _mapping(plan.get("base_lifecycle"), "base_lifecycle")
    identity_value = base_plan.get("ssh_identity")
    if not isinstance(identity_value, str) or not identity_value.strip():
        raise ProductionVastError(
            "production execution requires an explicit SSH identity"
        )
    identity_path = Path(identity_value)
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ProductionVastError(
            "production execution requires an explicit SSH identity regular file"
        )
    if config.phase == "adaptive":
        aws_credentials = workload_secret.aws_credentials
        if aws_credentials is None:
            raise ProductionVastError(
                "adaptive production execution requires AWS credentials"
            )
        try:
            full_lease_validity = config.base_job.maximum_elapsed_seconds + 300
            required_validity = (
                full_lease_validity
                if minimum_aws_validity_seconds is None
                else minimum_aws_validity_seconds
            )
            if not 600 <= required_validity <= full_lease_validity:
                raise ProductionVastError(
                    "adaptive AWS validity override is outside the safe range"
                )
            aws_credentials.assert_valid_for(required_validity)
        except VastPrerequisiteError as error:
            raise ProductionVastError(str(error)) from error
    base_metadata = metadata_path.with_name(metadata_path.name + ".base.json")
    base_receipt: dict[str, Any] | None = None
    try:
        base_receipt = base_execute(
            plan=base_plan,
            config=replace(
                config.base_job,
                workload_command=(
                    "bash",
                    "-lc",
                    _remote_production_command(
                        config,
                        host_hourly_usd=float(
                            base_plan["offer"]["hourly_price_usd"]
                        ),
                    ),
                ),
            ),
            metadata_path=base_metadata,
            workload_secret=workload_secret,
            run_command=run_command,
            sleeper=sleeper,
        )
    finally:
        # This assertion runs after success, workload failure, timeout, or interruption.
        _verify_zero_lineage_instances(
            run_command=run_command,
            vastai=str(base_plan["create_command"][0]),
            label=str(plan["label"]),
            sleeper=sleeper,
        )
    if base_receipt is None or base_receipt.get("destroyed") is not True:
        raise ProductionVastError("base lifecycle lacks successful destruction evidence")
    unsigned = {
        "format": RECEIPT_FORMAT,
        "phase": config.phase,
        "plan_sha256": plan["self_sha256"],
        "base_lifecycle_receipt_sha256": base_receipt.get("self_sha256"),
        "instance_id": base_receipt.get("instance_id"),
        "estimated_cost_usd": base_receipt.get("estimated_cost_usd"),
        "zero_lineage_instances_verified": True,
    }
    receipt = {**unsigned, "self_sha256": sha256(unsigned)}
    _atomic_json(metadata_path, receipt)
    return receipt


def _remote_production_command(
    config: ProductionVastConfig, *, host_hourly_usd: float
) -> str:
    hydrate = shlex.join(config.hydrate_command)
    verify = shlex.join(config.verify_command)
    workload = shlex.join(config.workload_command)
    stream = shlex.join(config.checkpoint_stream_command)
    checkpoint_dir = shlex.quote(config.checkpoint_directory)
    interval = config.checkpoint_interval_seconds
    read_only_cache = (
        f"chmod -R a-w {shlex.quote(config.model_cache_directory)}; "
        if any(
            part.endswith("run_truth_editing_cuda_fleet_controller.py")
            for part in config.workload_command
        )
        else ""
    )
    # The streamer terminates the worker on publication failure. The final
    # publication is mandatory, so the compact fetched archive contains a durable
    # terminal checkpoint/receipt even when the study itself exits nonzero.
    hourly_rate_environment = (
        f"export TRUTH_EDITING_HOST_HOURLY_USD={host_hourly_usd}; "
        "export TRUTH_EDITING_HOST_LEASE_STARTED_AT_UTC="
        '"$(date -u +%Y-%m-%dT%H:%M:%SZ)"; '
        if config.phase == "adaptive"
        else f"export TRUTH_EDITING_GPU_HOURLY_USD={host_hourly_usd}; "
    )
    checkpoint_setup = (
        ""
        if config.phase == "adaptive"
        else f"mkdir -p {checkpoint_dir}; "
    )
    aws_refresh_setup = (
        "python scripts/configure_truth_editing_aws_refresh.py initialize "
        "--root /workspace/.truth-editing-aws; "
        "unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN "
        "AWS_CREDENTIAL_EXPIRATION; "
        "export AWS_PROFILE=truth-editing-refresh; "
        "export AWS_CONFIG_FILE=/workspace/.truth-editing-aws/config; "
        "export AWS_SHARED_CREDENTIALS_FILE=/workspace/.truth-editing-aws/credentials; "
        if config.phase == "adaptive"
        else ""
    )
    return (
        "set -euo pipefail; "
        f"{hourly_rate_environment}"
        f"export TRUTH_EDITING_MODEL_ID={shlex.quote(config.base_job.model_id)}; "
        f"export TRUTH_EDITING_MODEL_REVISION={shlex.quote(config.base_job.model_revision)}; "
        f"export TRUTH_EDITING_EXPECTED_MODEL_SHA256={config.expected_model_sha256}; "
        "export TRUTH_EDITING_EXPECTED_SNAPSHOT_MANIFEST_SHA256="
        f"{config.expected_snapshot_manifest_sha256}; "
        f"{checkpoint_setup}"
        f"{aws_refresh_setup}"
        f"{hydrate}; {verify}; {read_only_cache}"
        f"({workload}) & worker_pid=$!; "
        "(set +e; while kill -0 \"$worker_pid\" 2>/dev/null; do "
        f"remaining={interval}; "
        "while [ \"$remaining\" -gt 0 ] && kill -0 \"$worker_pid\" 2>/dev/null; do "
        "sleep 1; remaining=$((remaining - 1)); done; "
        "kill -0 \"$worker_pid\" 2>/dev/null || break; "
        f"{stream} || {{ kill \"$worker_pid\" 2>/dev/null; exit 90; }}; done) & streamer_pid=$!; "
        "set +e; wait \"$worker_pid\"; worker_status=$?; "
        "wait \"$streamer_pid\"; streamer_status=$?; set -e; "
        f"{stream}; "
        "test \"$worker_status\" -eq 0; test \"$streamer_status\" -eq 0"
    )


def _verify_zero_lineage_instances(
    *,
    run_command: RunCommand,
    vastai: str,
    label: str,
    sleeper: Callable[[float], None],
) -> None:
    for attempt in range(5):
        result = run_command(
            [vastai, "show", "instances", "--raw"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            rows = json.loads(result.stdout)
            if not isinstance(rows, list):
                raise TypeError
            remaining = [row for row in rows if isinstance(row, Mapping) and row.get("label") == label]
        except (TypeError, json.JSONDecodeError) as error:
            raise ProductionVastError("Vast instance inventory is unreadable") from error
        if not remaining:
            return
        if attempt < 4:
            sleeper(2.0)
    identifiers = [str(row.get("id", "unknown")) for row in remaining]
    raise ProductionVastError(f"lineage instance remains after destruction: {identifiers}")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionVastError(f"{name} must be an object")
    return value


def _exact(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != fields:
        raise ProductionVastError(f"{name} fields changed")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProductionVastError(f"{name} must be nonempty trimmed text")
    return value


def _command(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProductionVastError(f"{name} must be a nonempty argv list")
    command = tuple(_text(part, f"{name} item") for part in value)
    if any("\x00" in part or "\n" in part for part in command):
        raise ProductionVastError(f"{name} contains unsafe characters")
    return command


def _absolute(value: Any, name: str) -> str:
    path = _text(value, name)
    if not path.startswith("/") or ".." in Path(path).parts:
        raise ProductionVastError(f"{name} must be a safe absolute path")
    return path.rstrip("/")


def _relative(value: Any, name: str) -> str:
    path = _text(value, name)
    parts = Path(path).parts
    if path.startswith("/") or ".." in parts or "\\" in path:
        raise ProductionVastError(f"{name} must be a safe relative path")
    return path


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductionVastError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value: Any, name: str) -> str:
    digest = _text(value, name)
    if _SHA256.fullmatch(digest) is None:
        raise ProductionVastError(f"{name} must be a lowercase SHA-256")
    return digest
