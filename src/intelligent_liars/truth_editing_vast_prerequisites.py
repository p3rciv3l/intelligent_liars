"""Fail-closed Vast lifecycle planning for truth-editing prerequisite inference.

The controller intentionally ships an explicit file allowlist instead of a repository
checkout.  It can execute a single bounded instance lifecycle, but callers must opt in;
normal use and all tests are dry-run only.
"""

from __future__ import annotations

import hashlib
import configparser
import gzip
import io
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any


FORMAT = "truth_editing_vast_prerequisite_job_v1"
RECEIPT_FORMAT = "truth_editing_vast_prerequisite_lifecycle_v1"
OUTPUT_ARCHIVE_FORMAT = "truth_editing_vast_output_archive_v1"
LABEL_PREFIX = "codex-vast-truth-prerequisites"
FROZEN_BASE_IMAGE = (
    "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@"
    "sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
)
FROZEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
FROZEN_MODEL_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
FROZEN_RUNTIME_ID = (
    "truth-editing-cu124-torch251-transformers4571-fa274post1-optuna490-wandb0290-boto34018-r3"
)
LEGACY_RUNTIME_IDS = frozenset(
    {
        "truth-editing-cu124-torch251-transformers4571-fa274post1-optuna490-wandb0290-r2"
    }
)
HARD_SPEND_CEILING_USD = 15.0
REMOTE_DIAGNOSTIC_TAIL_CHARS = 4000
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9._/+\-]+$")
_SECRET_PARTS = frozenset(
    {
        ".aws",
        ".env",
        ".git",
        ".netrc",
        ".secrets",
        ".ssh",
        "auth.json",
        "credentials",
        "id_ed25519",
        "id_rsa",
        "secrets",
        "token",
        "tokens",
    }
)


class VastPrerequisiteError(RuntimeError):
    """A prerequisite job violated its immutable or lifecycle contract."""


@dataclass(frozen=True, repr=False)
class AwsWorkloadCredentials:
    """Validated AWS credentials that exist only in controller memory."""

    source: str
    _access_key_id: str
    _secret_access_key: str
    _session_token: str
    expires_at: datetime | None

    def assert_valid_for(
        self, seconds: int, *, now: datetime | None = None
    ) -> None:
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
            raise VastPrerequisiteError("AWS credential validity window is invalid")
        if self.expires_at is None:
            return
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        if self.expires_at <= current + timedelta(seconds=seconds):
            raise VastPrerequisiteError(
                "AWS credentials expire before the paid workload can finish"
            )

    def stdin_values(self) -> tuple[str, str, str, str]:
        return (
            self._access_key_id,
            self._secret_access_key,
            self._session_token,
            self.expires_at.isoformat() if self.expires_at is not None else "",
        )


def resolve_aws_workload_credentials(
    *,
    environment: Mapping[str, str],
    minimum_validity_seconds: int,
    profile_name: str | None = None,
    credentials_file: Path | None = None,
    now: datetime | None = None,
) -> AwsWorkloadCredentials:
    """Resolve one explicit local credential source without network discovery.

    Supplying ``profile_name`` selects exactly one section of the shared credentials
    file. Otherwise only the three standard environment credential variables are
    considered. Metadata, SSO, credential-process, and implicit provider-chain
    discovery are intentionally outside this launch seam.
    """

    if profile_name is not None:
        if not profile_name or profile_name != profile_name.strip():
            raise VastPrerequisiteError("AWS credentials profile is malformed")
        path = credentials_file or Path.home() / ".aws" / "credentials"
        if path.is_symlink() or not path.is_file():
            raise VastPrerequisiteError("AWS credentials file is missing or unsafe")
        if path.stat().st_mode & 0o077:
            raise VastPrerequisiteError("AWS credentials file must be private")
        parser = configparser.RawConfigParser(interpolation=None)
        try:
            with path.open(encoding="utf-8") as stream:
                parser.read_file(stream)
        except (OSError, UnicodeError, configparser.Error) as error:
            raise VastPrerequisiteError("AWS credentials file is unreadable") from error
        if not parser.has_section(profile_name):
            raise VastPrerequisiteError("AWS credentials profile is missing")
        values = {
            "AWS_ACCESS_KEY_ID": parser.get(
                profile_name, "aws_access_key_id", fallback=""
            ),
            "AWS_SECRET_ACCESS_KEY": parser.get(
                profile_name, "aws_secret_access_key", fallback=""
            ),
            "AWS_SESSION_TOKEN": parser.get(
                profile_name, "aws_session_token", fallback=""
            ),
            "AWS_CREDENTIAL_EXPIRATION": parser.get(
                profile_name, "expiration", fallback=""
            ),
        }
        source = "shared_credentials_profile"
    else:
        if credentials_file is not None:
            raise VastPrerequisiteError(
                "AWS credentials file requires an explicit profile"
            )
        values = {name: environment.get(name, "") for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_CREDENTIAL_EXPIRATION",
        )}
        source = "environment"

    access = _credential_text(values["AWS_ACCESS_KEY_ID"], "access key", 16, 128)
    secret = _credential_text(
        values["AWS_SECRET_ACCESS_KEY"], "secret access key", 16, 256
    )
    token_raw = values["AWS_SESSION_TOKEN"]
    expiration_raw = values["AWS_CREDENTIAL_EXPIRATION"]
    token = (
        _credential_text(token_raw, "session token", 8, 16384)
        if token_raw
        else ""
    )
    if bool(token) != bool(expiration_raw):
        raise VastPrerequisiteError(
            "AWS credentials with a session token require a bounded expiration"
        )
    expiration: datetime | None = None
    if expiration_raw:
        try:
            expiration = datetime.fromisoformat(expiration_raw.replace("Z", "+00:00"))
        except ValueError as error:
            raise VastPrerequisiteError(
                "AWS credentials expiration is malformed"
            ) from error
        if expiration.tzinfo is None:
            raise VastPrerequisiteError("AWS credentials expiration must include timezone")
        expiration = expiration.astimezone(timezone.utc)
    resolved = AwsWorkloadCredentials(
        source=source,
        _access_key_id=access,
        _secret_access_key=secret,
        _session_token=token,
        expires_at=expiration,
    )
    resolved.assert_valid_for(minimum_validity_seconds, now=now)
    return resolved


def resolve_aws_cli_profile_credentials(
    *,
    profile_name: str,
    minimum_validity_seconds: int,
    aws_cli: str = "aws",
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now: datetime | None = None,
) -> AwsWorkloadCredentials:
    """Resolve an explicit AWS CLI profile, including cached SSO/role profiles.

    The CLI command contains only the profile name. Credential JSON is captured in
    memory and converted immediately into the same stdin-only envelope used by
    direct environment/shared-file sources.
    """

    if (
        not profile_name
        or profile_name != profile_name.strip()
        or any(character.isspace() or ord(character) < 32 for character in profile_name)
    ):
        raise VastPrerequisiteError("AWS credentials profile is malformed")
    if not aws_cli or aws_cli != aws_cli.strip() or "\x00" in aws_cli:
        raise VastPrerequisiteError("AWS CLI path is malformed")
    result = run_command(
        [
            aws_cli,
            "configure",
            "export-credentials",
            "--profile",
            profile_name,
            "--format",
            "process",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise VastPrerequisiteError("AWS credentials profile could not be resolved")
    try:
        raw = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise VastPrerequisiteError(
            "AWS credentials profile returned malformed data"
        ) from error
    if not isinstance(raw, Mapping):
        raise VastPrerequisiteError("AWS credentials profile returned malformed data")
    expected = {
        "Version",
        "AccessKeyId",
        "SecretAccessKey",
        "SessionToken",
        "Expiration",
    }
    if not set(raw).issubset(expected) or raw.get("Version") != 1:
        raise VastPrerequisiteError("AWS credentials profile returned malformed data")
    resolved = resolve_aws_workload_credentials(
        environment={
            "AWS_ACCESS_KEY_ID": str(raw.get("AccessKeyId", "")),
            "AWS_SECRET_ACCESS_KEY": str(raw.get("SecretAccessKey", "")),
            "AWS_SESSION_TOKEN": str(raw.get("SessionToken", "")),
            "AWS_CREDENTIAL_EXPIRATION": str(raw.get("Expiration", "")),
        },
        minimum_validity_seconds=minimum_validity_seconds,
        now=now,
    )
    return AwsWorkloadCredentials(
        source="aws_cli_profile",
        _access_key_id=resolved._access_key_id,
        _secret_access_key=resolved._secret_access_key,
        _session_token=resolved._session_token,
        expires_at=resolved.expires_at,
    )


def _credential_text(value: object, label: str, minimum: int, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise VastPrerequisiteError(f"AWS credentials {label} is missing or malformed")
    return value


@dataclass(frozen=True, repr=False, init=False)
class EphemeralWorkloadSecret:
    """An in-memory secret envelope delivered to a remote workload over SSH stdin.

    Secret values are deliberately absent from every serializable lifecycle
    object. Only allowlisted environment names may cross the command seam; the
    values never enter argv, the source bundle, a config, an event, or a receipt.
    """

    _entries: tuple[tuple[str, str], ...]
    aws_credentials: AwsWorkloadCredentials | None

    def __init__(
        self,
        environment_name: str,
        value: str,
        secondary_environment_name: str | None = None,
        secondary_value: str | None = None,
        *,
        aws_credentials: AwsWorkloadCredentials | None = None,
    ) -> None:
        entries = [(environment_name, value)]
        if secondary_environment_name is not None:
            entries.append((secondary_environment_name, secondary_value or ""))
        if aws_credentials is not None:
            access, secret, token, expiration = aws_credentials.stdin_values()
            entries.extend(
                (
                    ("AWS_ACCESS_KEY_ID", access),
                    ("AWS_SECRET_ACCESS_KEY", secret),
                    ("AWS_SESSION_TOKEN", token),
                    ("AWS_CREDENTIAL_EXPIRATION", expiration),
                )
            )
        object.__setattr__(self, "_entries", tuple(entries))
        object.__setattr__(self, "aws_credentials", aws_credentials)

    @classmethod
    def openrouter(cls, value: str) -> "EphemeralWorkloadSecret":
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 4096
            or any(character in value for character in ("\x00", "\r", "\n"))
        ):
            raise VastPrerequisiteError("OPENROUTER_API_KEY is missing or malformed")
        return cls("OPENROUTER_API_KEY", value)

    @classmethod
    def production(
        cls,
        *,
        openrouter_value: str,
        wandb_value: str | None,
        aws_credentials: AwsWorkloadCredentials | None = None,
    ) -> "EphemeralWorkloadSecret":
        """Build the production stdin envelope with optional W&B monitoring.

        OpenRouter remains mandatory because it is part of evaluation.  W&B is
        operational telemetry only: an absent or syntactically unsafe value is
        converted to an empty second stdin line, which makes the remote wrapper
        explicitly unset ``WANDB_API_KEY`` and continue without monitoring.
        """

        required = cls.openrouter(openrouter_value)
        optional = (
            wandb_value
            if isinstance(wandb_value, str)
            and 8 <= len(wandb_value) <= 4096
            and wandb_value == wandb_value.strip()
            and not any(
                character.isspace() or ord(character) < 32
                for character in wandb_value
            )
            else ""
        )
        return cls(
            required.environment_name,
            required._entries[0][1],
            "WANDB_API_KEY",
            optional,
            aws_credentials=aws_credentials,
        )

    @property
    def environment_name(self) -> str:
        return self._entries[0][0]

    @property
    def environment_names(self) -> tuple[str, ...]:
        return tuple(name for name, _value in self._entries)

    def stdin_payload(self) -> str:
        return "".join(value + "\n" for _name, value in self._entries)

    def redact(self, value: str) -> str:
        redacted = value
        for _name, secret in self._entries:
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise VastPrerequisiteError("value is not canonical JSON") from error


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class JobConfig:
    image: str
    runtime_id: str
    model_id: str
    model_revision: str
    disk_gib: int
    minimum_gpu_vram_gib: float
    maximum_elapsed_seconds: int
    maximum_cost_usd: float
    maximum_download_gib: float
    maximum_upload_gib: float
    remote_workdir: str
    remote_output_dir: str
    bootstrap_command: tuple[str, ...]
    workload_command: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    bundle_paths: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "JobConfig":
        return cls._from_mapping(raw, expected_format=FORMAT)

    @classmethod
    def _from_mapping(
        cls, raw: Mapping[str, Any], *, expected_format: str
    ) -> "JobConfig":
        """Parse the shared lifecycle fields under one caller-owned format.

        Subclasses may expose a separate versioned contract and a stricter or
        broader resource policy without weakening the prerequisite parser.
        """

        fields = {
            "format",
            "image",
            "runtime_id",
            "model",
            "resources",
            "paths",
            "commands",
            "expected_outputs",
            "bundle_paths",
        }
        if set(raw) != fields or raw.get("format") != expected_format:
            raise VastPrerequisiteError("job config fields or format changed")
        model = _exact_object(raw["model"], {"repository", "revision"}, "model")
        resources = _exact_object(
            raw["resources"],
            {
                "disk_gib",
                "minimum_gpu_vram_gib",
                "maximum_elapsed_seconds",
                "maximum_cost_usd",
                "maximum_download_gib",
                "maximum_upload_gib",
            },
            "resources",
        )
        paths = _exact_object(
            raw["paths"], {"remote_workdir", "remote_output_dir"}, "paths"
        )
        commands = _exact_object(
            raw["commands"], {"bootstrap", "workload"}, "commands"
        )
        config = cls(
            image=_text(raw["image"], "image"),
            runtime_id=_text(raw["runtime_id"], "runtime_id"),
            model_id=_text(model["repository"], "model.repository"),
            model_revision=_text(model["revision"], "model.revision"),
            disk_gib=_integer(resources["disk_gib"], "disk_gib", 80),
            minimum_gpu_vram_gib=_number(
                resources["minimum_gpu_vram_gib"], "minimum_gpu_vram_gib", 23
            ),
            maximum_elapsed_seconds=_integer(
                resources["maximum_elapsed_seconds"],
                "maximum_elapsed_seconds",
                60,
            ),
            maximum_cost_usd=_number(
                resources["maximum_cost_usd"], "maximum_cost_usd", 0.01
            ),
            maximum_download_gib=_number(
                resources["maximum_download_gib"], "maximum_download_gib", 1
            ),
            maximum_upload_gib=_number(
                resources["maximum_upload_gib"], "maximum_upload_gib", 0
            ),
            remote_workdir=_absolute_remote_path(
                paths["remote_workdir"], "remote_workdir"
            ),
            remote_output_dir=_absolute_remote_path(
                paths["remote_output_dir"], "remote_output_dir"
            ),
            bootstrap_command=_command(commands["bootstrap"], "bootstrap"),
            workload_command=_command(commands["workload"], "workload"),
            expected_outputs=_path_list(raw["expected_outputs"], "expected_outputs"),
            bundle_paths=_path_list(raw["bundle_paths"], "bundle_paths"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self._validate_frozen_invariants()
        if self.maximum_cost_usd >= HARD_SPEND_CEILING_USD:
            raise VastPrerequisiteError("maximum cost must be strictly below $15")
        if self.maximum_elapsed_seconds > 18 * 3600:
            raise VastPrerequisiteError("elapsed cap exceeds the bounded prerequisite job")

    def _validate_frozen_invariants(self) -> None:
        """Validate invariants shared by every versioned Vast job contract."""

        if self.image != FROZEN_BASE_IMAGE:
            raise VastPrerequisiteError("Vast job must use the digest-pinned CUDA 12.4 image")
        if self.runtime_id not in {FROZEN_RUNTIME_ID, *LEGACY_RUNTIME_IDS}:
            raise VastPrerequisiteError("runtime identity differs from the frozen lock")
        if (self.model_id, self.model_revision) != (
            FROZEN_MODEL_ID,
            FROZEN_MODEL_REVISION,
        ):
            raise VastPrerequisiteError("model repository or revision changed")
        if self.remote_output_dir == self.remote_workdir:
            raise VastPrerequisiteError("outputs must be separate from the uploaded bundle")
        if len(set(self.bundle_paths)) != len(self.bundle_paths):
            raise VastPrerequisiteError("bundle paths must be unique")
        for path in self.bundle_paths:
            _validate_relative_path(path)
        for path in self.expected_outputs:
            _validate_relative_path(path)


@dataclass(frozen=True)
class Offer:
    offer_id: int
    gpu_name: str
    gpu_count: int
    gpu_ram_gib: float
    base_hourly_price_usd: float
    advertised_hourly_price_usd: float
    storage_cost_usd_per_gib_month: float
    internet_download_cost_usd_per_gib: float
    internet_upload_cost_usd_per_gib: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Offer":
        return cls._from_mapping(raw, expected_gpu_count=1)

    @classmethod
    def from_multi_gpu_mapping(
        cls, raw: Mapping[str, Any], *, expected_gpu_count: int = 8
    ) -> "Offer":
        """Parse one exact homogeneous multi-GPU offer for a persistent fleet host."""

        if expected_gpu_count != 8:
            raise VastPrerequisiteError("persistent fleet requires exactly eight GPUs")
        return cls._from_mapping(raw, expected_gpu_count=expected_gpu_count)

    @classmethod
    def _from_mapping(
        cls, raw: Mapping[str, Any], *, expected_gpu_count: int
    ) -> "Offer":
        advertised_hourly_price = _number(
            raw.get("dph_total"), "offer.dph_total", 0
        )
        has_base = raw.get("dph_base") is not None
        has_storage = raw.get("storage_cost") is not None
        if has_base != has_storage:
            raise VastPrerequisiteError(
                "offer must provide both dph_base and storage_cost"
            )
        # Historical production fixtures only recorded Vast's aggregate rate.
        # Preserve that conservative legacy interpretation while requiring new
        # disk-aware offer captures to carry both component prices.
        base_hourly_price = (
            _number(raw.get("dph_base"), "offer.dph_base", 0)
            if has_base
            else advertised_hourly_price
        )
        storage_cost = (
            _number(raw.get("storage_cost"), "offer.storage_cost", 0)
            if has_storage
            else 0.0
        )
        offer = cls(
            offer_id=_integer(raw.get("id"), "offer.id", 1),
            gpu_name=_text(raw.get("gpu_name"), "offer.gpu_name"),
            gpu_count=_integer(raw.get("num_gpus"), "offer.num_gpus", 1),
            gpu_ram_gib=_normalize_vram(raw.get("gpu_ram")),
            base_hourly_price_usd=base_hourly_price,
            advertised_hourly_price_usd=advertised_hourly_price,
            storage_cost_usd_per_gib_month=storage_cost,
            internet_download_cost_usd_per_gib=_number(
                raw.get("inet_down_cost"), "offer.inet_down_cost", 0
            ),
            internet_upload_cost_usd_per_gib=_number(
                raw.get("inet_up_cost"), "offer.inet_up_cost", 0
            ),
        )
        if offer.gpu_count != expected_gpu_count:
            raise VastPrerequisiteError(
                f"Vast offer requires exactly {expected_gpu_count} GPU(s)"
            )
        if offer.advertised_hourly_price_usd < offer.base_hourly_price_usd:
            raise VastPrerequisiteError("offer total hourly price is below its base price")
        return offer

    def disk_hourly_price_usd(self, disk_gib: int) -> float:
        return self.storage_cost_usd_per_gib_month * disk_gib / (30 * 24)

    def requested_hourly_price_usd(self, disk_gib: int) -> float:
        return self.base_hourly_price_usd + self.disk_hourly_price_usd(disk_gib)


def validate_offer(config: JobConfig, offer: Offer) -> float:
    if offer.gpu_ram_gib < config.minimum_gpu_vram_gib:
        raise VastPrerequisiteError("offer has insufficient GPU memory")
    projected = (
        offer.requested_hourly_price_usd(config.disk_gib)
        * config.maximum_elapsed_seconds
        / 3600
        + offer.internet_download_cost_usd_per_gib * config.maximum_download_gib
        + offer.internet_upload_cost_usd_per_gib * config.maximum_upload_gib
    )
    if not math.isfinite(projected) or projected > config.maximum_cost_usd:
        raise VastPrerequisiteError("offer can exceed the approved bounded cost")
    return projected


def build_bundle(
    repo: Path, config: JobConfig, output: Path, *, maximum_bytes: int = 128 * 1024**2
) -> dict[str, Any]:
    """Create a deterministic tar containing only regular, explicitly listed files."""

    root = repo.resolve()
    if output.exists():
        raise VastPrerequisiteError(f"refusing to overwrite bundle: {output}")
    inventory: list[dict[str, Any]] = []
    total = 0
    resolved_files: list[tuple[str, Path]] = []
    for relative in sorted(config.bundle_paths):
        _validate_relative_path(relative)
        candidate = root / relative
        try:
            status = candidate.lstat()
        except OSError as error:
            raise VastPrerequisiteError(f"bundle input is missing: {relative}") from error
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise VastPrerequisiteError(f"bundle input is not a regular file: {relative}")
        resolved = candidate.resolve()
        if root not in resolved.parents:
            raise VastPrerequisiteError(f"bundle input escapes repository: {relative}")
        total += status.st_size
        if total > maximum_bytes:
            raise VastPrerequisiteError("bundle exceeds narrow prerequisite size cap")
        entry = {
            "path": relative,
            "bytes": status.st_size,
            "sha256": file_sha256(candidate),
        }
        inventory.append(entry)
        resolved_files.append((relative, candidate))
    unsigned = {"format": "truth_editing_vast_bundle_v1", "files": inventory}
    manifest = {**unsigned, "self_sha256": sha256(unsigned)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as gzip_file:
            with tarfile.open(fileobj=gzip_file, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative, candidate in resolved_files:
                    info = archive.gettarinfo(str(candidate), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with candidate.open("rb") as stream:
                        archive.addfile(info, stream)
                payload = canonical_bytes(manifest) + b"\n"
                info = tarfile.TarInfo("bundle-manifest.json")
                info.size = len(payload)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, fileobj=io.BytesIO(payload))
    return {**manifest, "archive_sha256": file_sha256(output)}


def lifecycle_plan(
    *,
    vastai: str,
    config: JobConfig,
    offer: Offer,
    bundle: Path,
    fetch_dir: Path,
    ssh_identity: Path | None = None,
) -> dict[str, Any]:
    projected = validate_offer(config, offer)
    disk_hourly_price = offer.disk_hourly_price_usd(config.disk_gib)
    requested_hourly_price = offer.requested_hourly_price_usd(config.disk_gib)
    label = f"{LABEL_PREFIX}-{int(time.time())}-{offer.offer_id}"
    create = [
        vastai,
        "create",
        "instance",
        str(offer.offer_id),
        "--image",
        config.image,
        "--disk",
        str(config.disk_gib),
        "--label",
        label,
        "--ssh",
        "--direct",
        "--raw",
    ]
    remote_archive = "/workspace/truth-prerequisites.tar.gz"
    remote_output_archive = "/workspace/truth-prerequisite-outputs.tar.gz"
    setup = (
        f"mkdir -p {shlex.quote(config.remote_workdir)} {shlex.quote(config.remote_output_dir)} "
        f"&& tar -xzf {remote_archive} -C {shlex.quote(config.remote_workdir)} "
        f"&& cd {shlex.quote(config.remote_workdir)} "
        f"&& timeout --signal=TERM --kill-after=60 {config.maximum_elapsed_seconds}s "
        f"bash -lc {shlex.quote(shlex.join(config.bootstrap_command) + ' && ' + shlex.join(config.workload_command))}"
    )
    copy_identity = ["-i", str(ssh_identity)] if ssh_identity is not None else []
    return {
        "format": "truth_editing_vast_prerequisite_dry_run_v1",
        "offer": {
            "id": offer.offer_id,
            "gpu_name": offer.gpu_name,
            "gpu_count": offer.gpu_count,
            "gpu_ram_gib": offer.gpu_ram_gib,
            "base_hourly_price_usd": offer.base_hourly_price_usd,
            "advertised_hourly_price_usd": offer.advertised_hourly_price_usd,
            "storage_cost_usd_per_gib_month": offer.storage_cost_usd_per_gib_month,
            "disk_hourly_price_usd": disk_hourly_price,
            # Kept for receipt consumers; this is the requested-disk all-in rate.
            "hourly_price_usd": requested_hourly_price,
            "internet_download_cost_usd_per_gib": offer.internet_download_cost_usd_per_gib,
            "internet_upload_cost_usd_per_gib": offer.internet_upload_cost_usd_per_gib,
            "maximum_download_gib": config.maximum_download_gib,
            "maximum_upload_gib": config.maximum_upload_gib,
            "projected_max_cost_usd": projected,
        },
        "image": config.image,
        "disk_gib": config.disk_gib,
        "maximum_elapsed_seconds": config.maximum_elapsed_seconds,
        "maximum_cost_usd": config.maximum_cost_usd,
        "label": label,
        "create_command": create,
        "copy_to_command_template": [
            vastai,
            "copy",
            *copy_identity,
            f"local:{bundle}",
            f"C.<INSTANCE_ID>:{remote_archive}",
        ],
        "bundle": str(bundle),
        "remote_archive": remote_archive,
        "remote_output_archive": remote_output_archive,
        "ssh_identity": str(ssh_identity) if ssh_identity is not None else None,
        "remote_command": setup,
        "fetch_command_template": [
            vastai,
            "copy",
            *copy_identity,
            f"C.<INSTANCE_ID>:{remote_output_archive}",
            "local:<STAGED_ARCHIVE>",
        ],
        "fetch_dir": str(fetch_dir),
        "destroy_command_template": [vastai, "destroy", "instance", "<INSTANCE_ID>", "--raw"],
        "expected_outputs": list(config.expected_outputs),
    }


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def execute_lifecycle(
    *,
    plan: Mapping[str, Any],
    config: JobConfig,
    metadata_path: Path,
    run_command: RunCommand = subprocess.run,
    sleeper: Callable[[float], None] = time.sleep,
    workload_secret: EphemeralWorkloadSecret | None = None,
) -> dict[str, Any]:
    """Execute one lifecycle and always destroy a created instance.

    This entrypoint is intentionally separate from plan creation so unit tests can
    prove cleanup behavior without contacting Vast.
    """

    if metadata_path.exists():
        raise VastPrerequisiteError("lifecycle metadata path already exists")
    fetch_dir = Path(str(plan["fetch_dir"]))
    if fetch_dir.exists():
        raise VastPrerequisiteError("fetch directory already exists")
    instance_id: str | None = None
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline: float | None = None
    exit_code: int | None = None
    error_text: str | None = None
    artifact_archive: dict[str, Any] | None = None
    staging_root: Path | None = None
    try:
        actual_offer = _revalidate_offer(
            run_command,
            str(plan["create_command"][0]),
            config,
            Offer(
                offer_id=int(plan["offer"]["id"]),
                gpu_name=str(plan["offer"]["gpu_name"]),
                gpu_count=int(plan["offer"]["gpu_count"]),
                gpu_ram_gib=float(plan["offer"]["gpu_ram_gib"]),
                base_hourly_price_usd=float(plan["offer"]["base_hourly_price_usd"]),
                advertised_hourly_price_usd=float(
                    plan["offer"]["advertised_hourly_price_usd"]
                ),
                storage_cost_usd_per_gib_month=float(
                    plan["offer"]["storage_cost_usd_per_gib_month"]
                ),
                internet_download_cost_usd_per_gib=float(
                    plan["offer"]["internet_download_cost_usd_per_gib"]
                ),
                internet_upload_cost_usd_per_gib=float(
                    plan["offer"]["internet_upload_cost_usd_per_gib"]
                ),
            ),
        )
        events.append({"event": "offer_revalidated", "offer_id": actual_offer.offer_id})
        created = run_command(
            list(plan["create_command"]), check=True, capture_output=True, text=True
        )
        raw = json.loads(created.stdout)
        instance_id = str(raw.get("new_contract") or raw.get("id") or "")
        if not instance_id.isdigit():
            raise VastPrerequisiteError("Vast create did not return an instance ID")
        events.append({"event": "created", "instance_id": instance_id})
        deadline = time.monotonic() + config.maximum_elapsed_seconds
        identity = (
            Path(str(plan["ssh_identity"]))
            if plan.get("ssh_identity") is not None
            else None
        )
        ssh = _wait_until_ready(
            run_command,
            str(plan["create_command"][0]),
            instance_id,
            deadline,
            sleeper,
            identity=identity,
            approved_offer=actual_offer,
            config=config,
        )
        events.append({"event": "instance_ready", "instance_id": instance_id})
        if identity is not None:
            bundle_path = Path(str(plan["bundle"]))
            try:
                bundle_size_bytes = bundle_path.stat().st_size
                bundle_sha256 = file_sha256(bundle_path)
            except OSError as error:
                raise VastPrerequisiteError(
                    f"local upload bundle is unreadable: {type(error).__name__}: {error}"
                ) from None
            run_command(
                _scp_command(
                    ssh,
                    source=str(bundle_path),
                    destination=str(plan["remote_archive"]),
                    identity=identity,
                    recursive=False,
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=_remaining(deadline),
            )
            verified_upload = run_command(
                _ssh_command(
                    ssh,
                    _remote_input_bundle_verification_command(
                        archive_path=str(plan["remote_archive"]),
                        expected_size_bytes=bundle_size_bytes,
                        expected_sha256=bundle_sha256,
                    ),
                    identity=identity,
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=_remaining(deadline),
            )
            if verified_upload.returncode != 0:
                stdout_tail, _ = _bounded_remote_diagnostic(
                    getattr(verified_upload, "stdout", ""), workload_secret
                )
                stderr_tail, _ = _bounded_remote_diagnostic(
                    getattr(verified_upload, "stderr", ""), workload_secret
                )
                diagnostic = "\n".join(
                    part for part in (stdout_tail, stderr_tail) if part.strip()
                )
                raise VastPrerequisiteError(
                    "remote bundle verification failed "
                    f"with exit {verified_upload.returncode}; output tail: {diagnostic}"
                )
        else:
            _run_template_with_retry(
                run_command,
                plan["copy_to_command_template"],
                instance_id,
                deadline=deadline,
                sleeper=sleeper,
            )
        upload_event: dict[str, Any] = {"event": "bundle_uploaded"}
        if identity is not None:
            upload_event.update(
                {
                    "size_bytes": bundle_size_bytes,
                    "archive_sha256": bundle_sha256,
                    "remote_identity_verified": True,
                }
            )
        events.append(upload_event)
        remote_command = str(plan["remote_command"])
        workload_input: str | None = None
        if workload_secret is not None:
            remote_command = _secret_stdin_wrapper(
                remote_command, workload_secret.environment_names
            )
            workload_input = workload_secret.stdin_payload()
        ssh_command = _ssh_command(ssh, remote_command, identity=identity)
        try:
            result = run_command(
                ssh_command,
                check=False,
                capture_output=True,
                text=True,
                input=workload_input,
                timeout=_remaining(deadline),
            )
        except (OSError, subprocess.SubprocessError) as error:
            detail = f"{type(error).__name__}: {error}"
            if workload_secret is not None:
                detail = workload_secret.redact(detail)
            raise VastPrerequisiteError(
                f"remote workload invocation failed: {detail}"
            ) from None
        exit_code = result.returncode
        stdout_tail, stdout_truncated = _bounded_remote_diagnostic(
            getattr(result, "stdout", ""), workload_secret
        )
        stderr_tail, stderr_truncated = _bounded_remote_diagnostic(
            getattr(result, "stderr", ""), workload_secret
        )
        events.append(
            {
                "event": "workload_finished",
                "exit_code": exit_code,
                "stdout_tail": stdout_tail,
                "stdout_truncated": stdout_truncated,
                "stderr_tail": stderr_tail,
                "stderr_truncated": stderr_truncated,
            }
        )
        if exit_code != 0:
            diagnostic = "\n".join(
                part for part in (stdout_tail, stderr_tail) if part.strip()
            )
            raise VastPrerequisiteError(
                f"remote workload exited {exit_code}; output tail: {diagnostic}"
            )
        packed = run_command(
            _ssh_command(
                ssh,
                _remote_pack_command(
                    output_dir=config.remote_output_dir,
                    archive_path=str(plan["remote_output_archive"]),
                    expected_outputs=config.expected_outputs,
                ),
                identity=identity,
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=_remaining(deadline),
        )
        remote_identity = _parse_output_archive_identity(packed.stdout, config)
        events.append(
            {
                "event": "artifact_archive_packed",
                "archive_sha256": remote_identity["archive_sha256"],
                "size_bytes": remote_identity["size_bytes"],
            }
        )
        fetch_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{fetch_dir.name}.fetch-", dir=fetch_dir.parent)
        )
        staged_archive = staging_root / "outputs.tar.gz"
        if identity is not None:
            run_command(
                _scp_command(
                    ssh,
                    source=str(plan["remote_output_archive"]),
                    destination=str(staged_archive),
                    identity=identity,
                    recursive=False,
                    source_is_remote=True,
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=_remaining(deadline),
            )
        else:
            fetch_template = [
                str(part).replace("<STAGED_ARCHIVE>", str(staged_archive))
                for part in plan["fetch_command_template"]
            ]
            _run_template_with_retry(
                run_command,
                fetch_template,
                instance_id,
                deadline=deadline,
                sleeper=sleeper,
            )
        _verify_downloaded_archive(staged_archive, remote_identity, config)
        staged_outputs = staging_root / "published"
        _extract_output_archive(
            staged_archive,
            staged_outputs,
            maximum_extracted_bytes=config.disk_gib * 1024**3,
        )
        expected_outputs = config.expected_outputs
        _verify_fetched_outputs(staged_outputs, expected_outputs)
        staged_outputs.replace(fetch_dir)
        artifact_archive = {
            **remote_identity,
            "expected_outputs": list(expected_outputs),
            "published_directory": str(fetch_dir),
        }
        events.append({"event": "artifacts_fetched"})
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        if workload_secret is not None:
            error_text = workload_secret.redact(error_text)
        raise
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        destroyed = False
        destroy_error: str | None = None
        if instance_id is not None:
            try:
                _run_template(
                    run_command,
                    plan["destroy_command_template"],
                    instance_id,
                    timeout=60,
                )
                destroyed = True
                events.append({"event": "destroyed"})
            except BaseException as error:  # cleanup evidence must survive interruption
                destroy_error = f"{type(error).__name__}: {error}"
        elapsed = time.monotonic() - started
        unsigned = {
            "format": RECEIPT_FORMAT,
            "offer": plan["offer"],
            "image": plan["image"],
            "label": plan["label"],
            "instance_id": instance_id,
            "events": events,
            "elapsed_seconds": elapsed,
            "estimated_cost_usd": (
                elapsed * float(plan["offer"]["hourly_price_usd"]) / 3600
            ),
            "maximum_network_cost_usd": (
                float(plan["offer"]["internet_download_cost_usd_per_gib"])
                * float(plan["offer"]["maximum_download_gib"])
                + float(plan["offer"]["internet_upload_cost_usd_per_gib"])
                * float(plan["offer"]["maximum_upload_gib"])
            ),
            "projected_all_in_max_cost_usd": plan["offer"]["projected_max_cost_usd"],
            "exit_code": exit_code,
            "artifact_archive": artifact_archive,
            "error": error_text,
            "destroyed": destroyed,
            "destroy_error": destroy_error,
        }
        receipt = {**unsigned, "self_sha256": sha256(unsigned)}
        _atomic_private_json(metadata_path, receipt)
        if instance_id is not None and not destroyed:
            raise VastPrerequisiteError(
                f"instance cleanup failed; manual destroy required for {instance_id}: {destroy_error}"
            )
    return receipt


def _secret_stdin_wrapper(
    remote_command: str, environment_names: Sequence[str]
) -> str:
    names = tuple(environment_names)
    allowed = {
        ("OPENROUTER_API_KEY",),
        ("OPENROUTER_API_KEY", "WANDB_API_KEY"),
        (
            "OPENROUTER_API_KEY",
            "WANDB_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_CREDENTIAL_EXPIRATION",
        ),
    }
    if names not in allowed:
        raise VastPrerequisiteError("workload secret environment name is not allowlisted")
    required_names = {"OPENROUTER_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}
    statements: list[str] = []
    for name in names:
        if name in required_names:
            statements.append(
                f"IFS= read -r {name} && test -n \"${name}\" && export {name}"
            )
        else:
            statements.append(
                f"IFS= read -r {name} || {name}=; "
                f'if test -n "${name}"; then export {name}; else unset {name}; fi'
            )
    prefix = " && ".join(f"{{ {statement}; }}" for statement in statements) + " && "
    # ``remote_command`` is a compound lifecycle program. Applying ``exec``
    # directly would execute only its first simple command and leave the rest
    # as arguments. One quoted Bash scope preserves the whole program and its
    # terminal exit status while keeping secret values exclusively on stdin.
    execute = f"exec bash -lc {shlex.quote(remote_command)}"
    return prefix + execute


def _bounded_remote_diagnostic(
    value: object, workload_secret: EphemeralWorkloadSecret | None
) -> tuple[str, bool]:
    """Return a receipt-safe tail, redacting complete secrets before truncation."""

    text = value if isinstance(value, str) else ""
    if workload_secret is not None:
        text = workload_secret.redact(text)
    truncated = len(text) > REMOTE_DIAGNOSTIC_TAIL_CHARS
    return text[-REMOTE_DIAGNOSTIC_TAIL_CHARS:], truncated


def _run_template(
    run_command: RunCommand,
    template: Sequence[str],
    instance_id: str,
    *,
    timeout: float,
) -> None:
    command = [str(part).replace("<INSTANCE_ID>", instance_id) for part in template]
    run_command(command, check=True, capture_output=True, text=True, timeout=timeout)


def _run_template_with_retry(
    run_command: RunCommand,
    template: Sequence[str],
    instance_id: str,
    *,
    deadline: float,
    sleeper: Callable[[float], None],
) -> None:
    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            _run_template(
                run_command,
                template,
                instance_id,
                timeout=_remaining(deadline),
            )
            return
        except (subprocess.SubprocessError, OSError) as error:
            last_error = error
            if attempt < 2:
                sleeper(min(5.0, _remaining(deadline)))
    assert last_error is not None
    raise VastPrerequisiteError("Vast copy failed after bounded retries") from last_error


def _revalidate_offer(
    run_command: RunCommand,
    vastai: str,
    config: JobConfig,
    approved: Offer,
) -> Offer:
    matches: list[Mapping[str, Any]] = []
    last_inventory_error: BaseException | None = None
    # Vast's inventory endpoint is eventually consistent and its documented
    # ``id=...`` filter intermittently returns an empty list.  Re-query the
    # narrow eligible inventory a bounded number of times, but still require
    # one exact approved ID below before any instance is created.
    for _ in range(5):
        try:
            result = run_command(
                [
                    vastai,
                    "search",
                    "offers",
                    (
                        f"gpu_name={approved.gpu_name.replace(' ', '_')} "
                        "rentable=true verified=true rented=false "
                        f"num_gpus={approved.gpu_count} direct_port_count>=1"
                    ),
                    "--raw",
                    "--limit",
                    "200",
                    "--order",
                    "dph_total",
                    "--storage",
                    str(config.disk_gib),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            rows = json.loads(result.stdout)
            matches = [
                row
                for row in rows
                if int(row.get("id", -1)) == approved.offer_id
            ]
            last_inventory_error = None
        except (
            subprocess.SubprocessError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            last_inventory_error = error
            continue
        if matches:
            break
    if last_inventory_error is not None and not matches:
        raise VastPrerequisiteError(
            "current Vast offer inventory failed after bounded retries"
        ) from last_inventory_error
    if not matches:
        # The exact create call below is still pinned to ``approved.offer_id``.
        # Vast's search inventory can omit a live ask across repeated reads, so
        # the created instance is authoritatively checked before any upload.
        return approved
    if len(matches) != 1:
        raise VastPrerequisiteError("approved Vast offer is not uniquely identifiable")
    current = (
        Offer.from_mapping(matches[0])
        if approved.gpu_count == 1
        else Offer.from_multi_gpu_mapping(
            matches[0], expected_gpu_count=approved.gpu_count
        )
    )
    validate_offer(config, current)
    if not math.isclose(
        current.advertised_hourly_price_usd,
        current.requested_hourly_price_usd(config.disk_gib),
        abs_tol=1e-9,
    ):
        raise VastPrerequisiteError("current Vast offer total price excludes requested disk")
    if (
        current.gpu_name != approved.gpu_name
        or current.gpu_count != approved.gpu_count
        or current.gpu_ram_gib != approved.gpu_ram_gib
        or current.base_hourly_price_usd != approved.base_hourly_price_usd
        or current.storage_cost_usd_per_gib_month
        != approved.storage_cost_usd_per_gib_month
        or current.internet_download_cost_usd_per_gib
        != approved.internet_download_cost_usd_per_gib
        or current.internet_upload_cost_usd_per_gib
        != approved.internet_upload_cost_usd_per_gib
    ):
        raise VastPrerequisiteError("approved Vast offer identity or price changed")
    return current


def _wait_until_ready(
    run_command: RunCommand,
    vastai: str,
    instance_id: str,
    deadline: float,
    sleeper: Callable[[float], None],
    *,
    identity: Path | None = None,
    approved_offer: Offer | None = None,
    config: JobConfig | None = None,
) -> str:
    last_state = "unknown"
    while True:
        result = run_command(
            [vastai, "show", "instance", instance_id, "--raw"],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(30.0, _remaining(deadline)),
        )
        try:
            raw = json.loads(result.stdout)
            if isinstance(raw, list):
                if len(raw) != 1:
                    raise ValueError
                raw = raw[0]
            if approved_offer is not None and config is not None:
                _validate_created_instance(raw, approved_offer, config)
            last_state = str(raw.get("actual_status") or raw.get("status") or "unknown").lower()
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise VastPrerequisiteError("Vast instance readiness response is unreadable") from error
        if last_state in {"dead", "error", "failed", "offline"}:
            raise VastPrerequisiteError(f"Vast instance entered terminal state {last_state}")
        if last_state == "running":
            try:
                ssh = run_command(
                    [vastai, "ssh-url", instance_id],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=min(30.0, _remaining(deadline)),
                ).stdout.strip()
                run_command(
                    _ssh_command(ssh, "true", identity=identity),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=min(30.0, _remaining(deadline)),
                )
                return ssh
            except (subprocess.SubprocessError, OSError, VastPrerequisiteError):
                pass
        sleeper(min(5.0, _remaining(deadline)))


def _validate_created_instance(
    raw: Mapping[str, Any], approved: Offer, config: JobConfig
) -> None:
    gpu_name = _text(raw.get("gpu_name"), "instance.gpu_name")
    gpu_count = _integer(raw.get("num_gpus"), "instance.num_gpus", 1)
    gpu_ram_gib = _normalize_vram(raw.get("gpu_ram"))
    base_hourly_price = _number(raw.get("dph_base"), "instance.dph_base", 0)
    total_hourly_price = _number(raw.get("dph_total"), "instance.dph_total", 0)
    nested_pricing = raw.get("instance")
    nested_disk_hour = (
        nested_pricing.get("diskHour")
        if isinstance(nested_pricing, Mapping)
        else None
    )
    top_level_disk_hour = raw.get("diskHour")
    if (
        nested_disk_hour is not None
        and top_level_disk_hour is not None
        and nested_disk_hour != top_level_disk_hour
    ):
        raise VastPrerequisiteError("created instance disk price fields disagree")
    disk_hourly_price = _number(
        nested_disk_hour if nested_disk_hour is not None else top_level_disk_hour,
        "instance.diskHour",
        0,
    )
    download_cost = _number(
        raw.get("inet_down_cost"), "instance.inet_down_cost", 0
    )
    upload_cost = _number(raw.get("inet_up_cost"), "instance.inet_up_cost", 0)
    expected_disk_hourly = approved.disk_hourly_price_usd(config.disk_gib)
    expected_total_hourly = approved.requested_hourly_price_usd(config.disk_gib)
    if (
        gpu_name != approved.gpu_name
        or gpu_count != approved.gpu_count
        or gpu_ram_gib != approved.gpu_ram_gib
        or base_hourly_price != approved.base_hourly_price_usd
        or not math.isclose(disk_hourly_price, expected_disk_hourly, abs_tol=1e-9)
        or not math.isclose(total_hourly_price, expected_total_hourly, abs_tol=1e-9)
        or download_cost != approved.internet_download_cost_usd_per_gib
        or upload_cost != approved.internet_upload_cost_usd_per_gib
    ):
        raise VastPrerequisiteError("created instance identity or price changed")


def _remaining(deadline: float | None) -> float:
    if deadline is None:
        raise VastPrerequisiteError("instance deadline was not established")
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Vast prerequisite hard elapsed limit reached")
    return value


def _remote_pack_command(
    *, output_dir: str, archive_path: str, expected_outputs: tuple[str, ...]
) -> str:
    """Return one fail-closed command that packs and identifies remote outputs."""

    checks = " ".join(
        "if ! test -f "
        + shlex.quote(str(PurePosixPath(output_dir) / path))
        + "; then printf 'missing expected output: %s\\n' "
        + shlex.quote(path)
        + " >&2; exit 73; fi;"
        for path in expected_outputs
    )
    prefix = f"{checks} " if checks else ""
    return (
        "set -euo pipefail; "
        f"{prefix}rm -f {shlex.quote(archive_path)}; "
        f"tar -C {shlex.quote(output_dir)} -czf {shlex.quote(archive_path)} .; "
        f"archive_sha256=$(sha256sum {shlex.quote(archive_path)} | cut -d' ' -f1); "
        f"size_bytes=$(stat -c %s {shlex.quote(archive_path)}); "
        "printf '{\"format\":\""
        + OUTPUT_ARCHIVE_FORMAT
        + "\",\"archive_sha256\":\"%s\",\"size_bytes\":%s}\\n' "
        '"$archive_sha256" "$size_bytes"'
    )


def _parse_output_archive_identity(stdout: str, config: JobConfig) -> dict[str, Any]:
    try:
        raw = json.loads(stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise VastPrerequisiteError("remote output archive identity is unreadable") from error
    if not isinstance(raw, Mapping) or set(raw) != {
        "format",
        "archive_sha256",
        "size_bytes",
    }:
        raise VastPrerequisiteError("remote output archive identity fields changed")
    digest = raw["archive_sha256"]
    size = raw["size_bytes"]
    if raw["format"] != OUTPUT_ARCHIVE_FORMAT or not isinstance(digest, str) or not _SHA.fullmatch(digest):
        raise VastPrerequisiteError("remote output archive identity is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise VastPrerequisiteError("remote output archive size is invalid")
    maximum_bytes = math.floor(config.maximum_upload_gib * 1024**3)
    if size > maximum_bytes:
        raise VastPrerequisiteError("remote output archive exceeds maximum upload bound")
    return {
        "format": OUTPUT_ARCHIVE_FORMAT,
        "archive_sha256": digest,
        "size_bytes": size,
    }


def _verify_downloaded_archive(
    archive_path: Path, identity: Mapping[str, Any], config: JobConfig
) -> None:
    try:
        size = archive_path.stat().st_size
    except OSError as error:
        raise VastPrerequisiteError("output archive transfer did not produce a file") from error
    maximum_bytes = math.floor(config.maximum_upload_gib * 1024**3)
    if size > maximum_bytes:
        raise VastPrerequisiteError("downloaded output archive exceeds maximum upload bound")
    if size != identity["size_bytes"] or file_sha256(archive_path) != identity["archive_sha256"]:
        raise VastPrerequisiteError("downloaded output archive identity mismatch")


def _extract_output_archive(
    archive_path: Path, destination: Path, *, maximum_extracted_bytes: int
) -> None:
    """Safely extract regular files and directories into a new staged directory."""

    destination.mkdir(mode=0o700)
    seen: set[str] = set()
    total = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 100_000:
                raise VastPrerequisiteError("output archive has too many members")
            for member in members:
                normalized = member.name.removeprefix("./").rstrip("/")
                if not normalized:
                    continue
                path = PurePosixPath(normalized)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in normalized
                    or normalized in seen
                ):
                    raise VastPrerequisiteError("output archive contains an unsafe member")
                if not (member.isfile() or member.isdir()):
                    raise VastPrerequisiteError("output archive contains a non-regular member")
                seen.add(normalized)
                target = destination.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                if member.size < 0:
                    raise VastPrerequisiteError("output archive member has an invalid size")
                total += member.size
                if total > maximum_extracted_bytes:
                    raise VastPrerequisiteError("output archive exceeds the extraction byte bound")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source = archive.extractfile(member)
                if source is None:
                    raise VastPrerequisiteError("output archive member is unreadable")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if target.stat().st_size != member.size:
                    raise VastPrerequisiteError("output archive member size changed during extraction")
    except (OSError, tarfile.TarError) as error:
        raise VastPrerequisiteError("output archive is unreadable") from error


def _verify_fetched_outputs(fetch_dir: Path, expected: tuple[str, ...]) -> None:
    missing = [path for path in expected if not (fetch_dir / path).is_file()]
    if missing:
        raise VastPrerequisiteError(f"fetched artifacts are incomplete: {missing}")


def _ssh_command(
    url: str, remote_command: str, *, identity: Path | None = None
) -> list[str]:
    match = re.fullmatch(r"ssh://([^:@/]+)@([^:/]+):(\d+)", url)
    if match is None:
        raise VastPrerequisiteError("Vast ssh-url has an unsupported format")
    user, host, port = match.groups()
    identity_args = ["-i", str(identity)] if identity is not None else []
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        *identity_args,
        "-p",
        port,
        f"{user}@{host}",
        remote_command,
    ]


def _scp_command(
    url: str,
    *,
    source: str,
    destination: str,
    identity: Path,
    recursive: bool,
    source_is_remote: bool = False,
) -> list[str]:
    match = re.fullmatch(r"ssh://([^:@/]+)@([^:/]+):(\d+)", url)
    if match is None:
        raise VastPrerequisiteError("Vast ssh-url has an unsupported format")
    user, host, port = match.groups()
    remote = f"{user}@{host}:"
    resolved_source = f"{remote}{source}" if source_is_remote else source
    resolved_destination = destination if source_is_remote else f"{remote}{destination}"
    recursive_args = ["-r"] if recursive else []
    return [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        "-i",
        str(identity),
        "-P",
        port,
        *recursive_args,
        resolved_source,
        resolved_destination,
    ]


def _remote_input_bundle_verification_command(
    *, archive_path: str, expected_size_bytes: int, expected_sha256: str
) -> str:
    """Build a fail-closed remote identity check for a just-uploaded bundle."""

    if expected_size_bytes < 0:
        raise VastPrerequisiteError("expected upload bundle size is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise VastPrerequisiteError("expected upload bundle SHA-256 is invalid")
    archive = shlex.quote(archive_path)
    return (
        "set -eu; "
        f"test -f {archive} && test ! -L {archive} || "
        "{ echo 'uploaded bundle is missing or not a regular file' >&2; exit 74; }; "
        f"observed_bytes=$(wc -c < {archive}) || "
        "{ echo 'uploaded bundle byte length is unreadable' >&2; exit 75; }; "
        "observed_bytes=$(printf '%s' \"$observed_bytes\" | tr -d '[:space:]'); "
        f"test \"$observed_bytes\" = {shlex.quote(str(expected_size_bytes))} || "
        "{ echo 'uploaded bundle byte length differs' >&2; exit 75; }; "
        f"observed_sha256=$(sha256sum -- {archive} | awk '{{print $1}}') || "
        "{ echo 'uploaded bundle SHA-256 is unreadable' >&2; exit 76; }; "
        f"test \"$observed_sha256\" = {shlex.quote(expected_sha256)} || "
        "{ echo 'uploaded bundle SHA-256 differs' >&2; exit 76; }"
    )


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _exact_object(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise VastPrerequisiteError(f"{name} fields changed")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VastPrerequisiteError(f"{name} must be nonempty trimmed text")
    return value


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise VastPrerequisiteError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VastPrerequisiteError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise VastPrerequisiteError(f"{name} must be finite and >= {minimum}")
    return result


def _command(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise VastPrerequisiteError(f"{name} must be a nonempty argv list")
    output = tuple(_text(part, f"{name} item") for part in value)
    if any("\x00" in part or "\n" in part for part in output):
        raise VastPrerequisiteError(f"{name} contains unsafe characters")
    return output


def _path_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise VastPrerequisiteError(f"{name} must be a nonempty path list")
    return tuple(_text(item, f"{name} item") for item in value)


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not _SAFE_PATH.fullmatch(value):
        raise VastPrerequisiteError(f"unsafe bundle path: {value}")
    if any(part.lower() in _SECRET_PARTS for part in path.parts):
        raise VastPrerequisiteError(f"secret-like bundle path is forbidden: {value}")


def _absolute_remote_path(value: Any, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts or not _SAFE_PATH.fullmatch(text):
        raise VastPrerequisiteError(f"{name} is not a safe absolute path")
    return text


def _normalize_vram(value: Any) -> float:
    raw = _number(value, "offer.gpu_ram", 1)
    return raw / 1024 if raw > 1024 else raw
MINIMUM_REFRESHABLE_AWS_VALIDITY_SECONDS = 300


def adaptive_aws_required_validity_seconds(
    requested_seconds: int | None,
    full_lease_validity_seconds: int,
) -> int:
    """Resolve the admission window for credentials refreshed every four minutes."""

    required = (
        full_lease_validity_seconds
        if requested_seconds is None
        else requested_seconds
    )
    if not MINIMUM_REFRESHABLE_AWS_VALIDITY_SECONDS <= required <= full_lease_validity_seconds:
        raise ValueError(
            "AWS minimum validity must be between 300 seconds and the adaptive "
            "lease plus cleanup window"
        )
    return required
