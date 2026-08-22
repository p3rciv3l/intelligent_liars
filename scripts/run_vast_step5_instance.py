#!/usr/bin/env python3
"""Run one fail-closed Vast Step 5 worker, or print its lifecycle dry run.

This wrapper never replaces a machine.  It can mark replacement as permissible
only after a host-loss classification; software failures remain attached to the
same stopped instance and may be resumed in place with an explicit recovery
command.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
import re
import secrets
import shlex
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from intelligent_liars.step5_artifact_store import (
    ArtifactContractError,
    LIFECYCLE_ARTIFACT_MANIFEST_FORMAT,
    validate_lifecycle_artifact_manifest,
)
from intelligent_liars.step5_artifact_presigner import parse_utc, validate_receipt


PROJECT_LABEL_PREFIX = "codex-vast-tinylora-step5"
CONTROLLER_PUBLIC_KEY_REMOTE_PATH = "/workspace/inputs/controller-public.pem"
INPUT_URL_MANIFEST_REMOTE_PATH = "/run/secrets/step5-input-url-manifest-url"
HOST_GATE_URL_REMOTE_PATH = "/run/secrets/step5-host-gate-url"
HARD_MAX_WORKERS = 3
ARTIFACT_APPROVAL_MAX_AGE_SECONDS = 3600
ARTIFACT_MINIMUM_REMAINING_SECONDS = 900
SEALED_AUDIT_SHA256 = "40e2756176387514fc265bdf6225e73b356d7fe57b641405e6c4d2ecbe498e91"
HOST_LOSS_STATES = frozenset({"dead", "error", "failed", "offline", "unavailable"})
STOPPED_STATES = frozenset({"exited", "stopped"})
FAILURE_CLASSES = frozenset({"software_failure", "host_loss", "unknown"})
SECRET_PATH_PARTS = frozenset(
    {
        ".aws",
        ".env",
        ".git",
        ".netrc",
        ".secrets",
        ".ssh",
        "auth.json",
        "config.local",
        "credentials",
        "id_ed25519",
        "id_rsa",
        "hf_token.txt",
        "secrets",
        "token",
        "tokens",
    }
)
SECRET_COMMAND_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)\s*="),
    re.compile(r"(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY)\s*="),
    re.compile(r"--(?:api-key|token|password)(?:=|\s+)"),
    re.compile(r"Authorization\s*:\s*Bearer\s+", re.IGNORECASE),
    re.compile(r"[?&]X-Amz-(?:Signature|Credential)=", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastai", required=True)
    parser.add_argument("--offer-id")
    parser.add_argument("--resume-instance-id")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--disk", type=int, default=160)
    parser.add_argument("--cache-volume-id")
    parser.add_argument("--cache-mount", default="/workspace/cache")
    parser.add_argument("--remote-command", required=True)
    parser.add_argument("--remote-artifact-dir", required=True)
    parser.add_argument(
        "--artifact-manifest-name",
        default="artifact_manifest.json",
    )
    parser.add_argument("--expected-artifact-inventory", type=Path, required=True)
    parser.add_argument("--host-qualification-command", required=True)
    parser.add_argument("--minimum-download-mbps", type=float, default=100.0)
    parser.add_argument("--diagnostic-command", required=True)
    parser.add_argument("--software-recovery-command")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--artifact-put-url-file", type=Path)
    parser.add_argument("--artifact-put-receipt", type=Path)
    parser.add_argument("--artifact-put-receipt-sha256")
    parser.add_argument("--approved-at")
    parser.add_argument(
        "--remote-artifact-put-url-file",
        default="/run/secrets/step5-artifact-put-url",
    )
    parser.add_argument("--expected-durable-uri")
    parser.add_argument("--input-url-manifest-url-file", type=Path)
    parser.add_argument("--host-gate-url-file", type=Path)
    parser.add_argument("--controller-public-key", type=Path)
    parser.add_argument("--controller-public-key-sha256")
    parser.add_argument("--checkpoint-controller-script", type=Path)
    parser.add_argument("--checkpoint-controller-script-sha256")
    parser.add_argument("--checkpoint-controller-uv", default="uv")
    parser.add_argument("--checkpoint-bucket")
    parser.add_argument("--checkpoint-prefix")
    parser.add_argument("--checkpoint-controller-private-key", type=Path)
    parser.add_argument(
        "--checkpoint-remote-exchange-dir",
        default="/workspace/checkpoint-bridge",
    )
    parser.add_argument("--checkpoint-controller-ready-seconds", type=int, default=60)
    parser.add_argument("--max-software-resumes", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=HARD_MAX_WORKERS)
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--state-wait-seconds", type=int, default=120)
    parser.add_argument(
        "--controller-lock",
        type=Path,
        default=Path("/tmp/intelligent-liars-step5-vast.lock"),
    )
    parser.add_argument("--approved-hourly-price", type=float)
    parser.add_argument("--approved-max-cost", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmed-cost-approval", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not cleaned:
        raise ValueError(f"Label component has no safe characters: {value!r}")
    return cleaned[:32]


def unique_label(run_id: str, candidate: str, attempt: int) -> str:
    if attempt < 1:
        raise ValueError("attempt must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    nonce = secrets.token_hex(3)
    return (
        f"{PROJECT_LABEL_PREFIX}-{safe_component(run_id)}-"
        f"{safe_component(candidate)}-a{attempt}-{stamp}-{nonce}"
    )


def resume_label_matches(label: str, run_id: str, candidate: str, attempt: int) -> bool:
    binding = f"-{safe_component(run_id)}-{safe_component(candidate)}-a{attempt}-"
    return label.startswith(f"{PROJECT_LABEL_PREFIX}-") and binding in label


def parse_json_payload(text: str) -> Any:
    """Parse a JSON response, accepting a final JSON line after log output."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Expected JSON output, received an empty response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        for line in reversed(stripped.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON payload found in output: {stripped[:500]}")


def parse_instance_id(text: str) -> str:
    payload = parse_json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("Vast create response is not a JSON object")
    for key in ("new_contract", "instance_id", "id"):
        if key in payload:
            return str(payload[key])
    raise ValueError(f"Vast create response has no instance id: {text[:500]}")


def parse_ssh_url(text: str) -> tuple[str, str]:
    match = re.search(r"ssh://(?:[^@]+@)?([^:/\s]+):(\d+)", text)
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"ssh\s+-p\s+(\d+)\s+(?:[^@]+@)?([^\s]+)", text)
    if match:
        return match.group(2), match.group(1)
    raise ValueError(f"Cannot parse Vast SSH URL: {text[:500]}")


def ssh_options(known_hosts: Path) -> list[str]:
    return [
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
    ]


def ssh_prefix(host: str, port: str, known_hosts: Path) -> list[str]:
    return [
        "ssh",
        *ssh_options(known_hosts),
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-p",
        port,
        f"root@{host}",
    ]


def capture_known_host(host: str, port: str, path: Path) -> dict[str, Any]:
    captured = run(["ssh-keyscan", "-p", port, host], check=False)
    lines = [line for line in captured.stdout.splitlines() if line and not line.startswith("#")]
    if captured.returncode != 0 or not lines:
        raise RuntimeError("Could not capture the worker SSH host key")
    content = "\n".join(lines) + "\n"
    path.write_text(content)
    path.chmod(0o600)
    return {"sha256": hashlib.sha256(content.encode()).hexdigest(), "keys": len(lines)}


def instance_label(instance: dict[str, Any]) -> str:
    return str(instance.get("label") or instance.get("name") or "")


def instance_id(instance: dict[str, Any]) -> str:
    return str(instance.get("id") or instance.get("contract_id") or "")


def project_workers(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in instances
        if instance_label(item).startswith(f"{PROJECT_LABEL_PREFIX}-")
    ]


def enforce_worker_cap(
    instances: list[dict[str, Any]],
    *,
    max_workers: int,
    resume_instance_id: str | None,
) -> None:
    if not 1 <= max_workers <= HARD_MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {HARD_MAX_WORKERS}")
    workers = project_workers(instances)
    if len(workers) > max_workers:
        raise RuntimeError(
            f"Project inventory already exceeds the worker cap: {len(workers)}/{max_workers}"
        )
    if resume_instance_id:
        if not any(instance_id(item) == str(resume_instance_id) for item in workers):
            raise ValueError("resume instance is not a labeled Step 5 project worker")
        return
    if len(workers) >= max_workers:
        ids = ", ".join(instance_id(item) for item in workers)
        raise RuntimeError(
            f"Refusing a new worker: {len(workers)}/{max_workers} Step 5 workers "
            f"already exist (including stopped workers): {ids}"
        )


@contextlib.contextmanager
def controller_lock(path: Path):
    """Serialize inventory-and-create decisions made by this controller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def recover_created_instance_id(
    instances: list[dict[str, Any]],
    expected_label: str,
) -> str:
    matches = [item for item in instances if instance_label(item) == expected_label]
    if len(matches) != 1 or not instance_id(matches[0]):
        raise RuntimeError(
            "Could not uniquely recover a created instance from its exact unique label"
        )
    return instance_id(matches[0])


def poll_created_instance_id(
    vastai: str,
    expected_label: str,
    wait_seconds: int,
    poll_seconds: float = 2.0,
) -> str:
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        matches = [
            item
            for item in show_instances(vastai)
            if instance_label(item) == expected_label
        ]
        if len(matches) == 1 and instance_id(matches[0]):
            return instance_id(matches[0])
        if len(matches) > 1:
            raise RuntimeError("Unique create label unexpectedly matched multiple workers")
        time.sleep(poll_seconds)
    raise RuntimeError(
        f"Created instance id remains ambiguous for exact label {expected_label}"
    )


def show_instances(vastai: str) -> list[dict[str, Any]]:
    result = run([vastai, "show", "instances", "--raw"])
    payload = parse_json_payload(result.stdout)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Vast instance inventory is not a list of objects")
    return payload


def hourly_price(record: dict[str, Any]) -> float:
    for key in ("dph_total", "dph_base", "cost_per_hour", "hourly_price"):
        if record.get(key) is not None:
            value = float(record[key])
            if math.isfinite(value) and value > 0:
                return value
    raise ValueError("Vast record lacks a finite positive hourly price")


def query_offer(vastai: str, offer_id: str) -> dict[str, Any]:
    result = run(
        [vastai, "search", "offers", f"id={offer_id}", "--raw", "--limit", "5"]
    )
    payload = parse_json_payload(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("Vast offer response is not a list")
    matches = [item for item in payload if str(item.get("id")) == str(offer_id)]
    if len(matches) != 1:
        raise ValueError("Approved Vast offer is unavailable or not unique")
    return matches[0]


def enforce_price(actual: float, approved_hourly_price: float) -> None:
    if actual > approved_hourly_price + 1e-9:
        raise RuntimeError(
            f"Offer costs ${actual:.6f}/h, above approved ${approved_hourly_price:.6f}/h"
        )


def remaining_cost_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Approved maximum-cost runtime has been exhausted")
    return remaining


def find_instance(instances: list[dict[str, Any]], target_id: str) -> dict[str, Any] | None:
    return next((item for item in instances if instance_id(item) == str(target_id)), None)


def instance_state(instance: dict[str, Any] | None) -> str:
    if instance is None:
        return "absent"
    for key in ("actual_status", "cur_state", "status", "intended_status"):
        value = instance.get(key)
        if value:
            return str(value).lower()
    return "unknown"


def state_matches(observed: str, expected: str) -> bool:
    if expected == "stopped":
        return observed in STOPPED_STATES
    return observed == expected


def wait_for_state(
    vastai: str,
    target_id: str,
    expected: str,
    wait_seconds: int,
) -> tuple[bool, str, list[dict[str, Any]]]:
    deadline = time.time() + wait_seconds
    last_inventory: list[dict[str, Any]] = []
    last_state = "unknown"
    while time.time() < deadline:
        last_inventory = show_instances(vastai)
        last_state = instance_state(find_instance(last_inventory, target_id))
        if state_matches(last_state, expected):
            return True, last_state, last_inventory
        time.sleep(5)
    return False, last_state, last_inventory


def wait_for_real_ssh(
    vastai: str,
    target_id: str,
    wait_seconds: int,
    known_hosts: Path,
) -> tuple[str, str, dict[str, Any]]:
    deadline = time.time() + wait_seconds
    last = ""
    while time.time() < deadline:
        url = run([vastai, "ssh-url", target_id], check=False)
        last = (url.stdout or "") + (url.stderr or "")
        if url.returncode == 0:
            try:
                host, port = parse_ssh_url(last)
            except ValueError:
                time.sleep(10)
                continue
            try:
                if known_hosts.stat().st_size == 0:
                    pin = capture_known_host(host, port, known_hosts)
                else:
                    pin = {
                        "sha256": hash_file(known_hosts),
                        "keys": len(known_hosts.read_text().splitlines()),
                    }
            except (OSError, RuntimeError):
                time.sleep(10)
                continue
            probe = run([*ssh_prefix(host, port, known_hosts), "true"], check=False)
            if probe.returncode == 0:
                return host, port, pin
        time.sleep(10)
    raise TimeoutError(f"SSH never became usable for {target_id}: {last[:500]}")


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_manifest_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe artifact path in manifest: {relative!r}")
    target = root.joinpath(*pure.parts)
    unresolved = root
    for part in pure.parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise ValueError(f"Path traverses a symlink: {relative}")
    try:
        target.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes its trusted root: {relative}") from error
    return target


def secret_like_path(relative: str) -> bool:
    parts = tuple(part.lower() for part in PurePosixPath(relative).parts)
    return any(
        part in SECRET_PATH_PARTS
        or part.endswith((".key", ".pem", ".p12", ".pfx"))
        or "secret" in part
        for part in parts
    )


def sealed_audit_path(relative: str) -> bool:
    pure = PurePosixPath(relative)
    return pure.name.lower() == "audit.jsonl" or (
        "pilot_v1" in tuple(part.lower() for part in pure.parts)
        and "audit" in pure.name.lower()
    )


def reject_credential_bearing_commands(commands: list[str | None]) -> None:
    for command in commands:
        if not command:
            continue
        if any(pattern.search(command) for pattern in SECRET_COMMAND_PATTERNS):
            raise ValueError(
                "Commands must not contain long-lived credentials; use a scoped, "
                "short-lived signed URL or workload identity"
            )


def verify_source_manifest(repository: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify a positive allowlist without following symlinks or secret-like paths."""
    repository = repository.resolve()
    if not repository.is_dir():
        raise ValueError("repo must be a directory when using a source allowlist")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("source manifest must be a regular non-symlink file")
    payload = parse_json_payload(manifest_path.read_text())
    if not isinstance(payload, dict) or payload.get("format") != (
        "tinylora_step5_source_manifest_v1"
    ):
        raise ValueError("Unsupported Step 5 source manifest")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Source manifest must contain a nonempty files list")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Source manifest file entry is not an object")
        relative = str(item.get("path", ""))
        expected_hash = str(item.get("sha256", ""))
        if relative in seen:
            raise ValueError(f"Duplicate source path: {relative}")
        seen.add(relative)
        if secret_like_path(relative):
            raise ValueError(f"Secret-like source path is forbidden: {relative}")
        if sealed_audit_path(relative):
            raise ValueError(f"Sealed audit source path is forbidden: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Invalid source SHA-256: {relative}")
        path = safe_manifest_path(repository, relative)
        unresolved = repository
        for part in PurePosixPath(relative).parts:
            unresolved = unresolved / part
            if unresolved.is_symlink():
                raise ValueError(f"Source path traverses a symlink: {relative}")
        try:
            path.resolve().relative_to(repository)
        except ValueError as error:
            raise ValueError(f"Source path escapes repository: {relative}") from error
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty allowlisted source: {relative}")
        actual_hash = hash_file(path)
        if actual_hash == SEALED_AUDIT_SHA256:
            raise ValueError(f"Sealed audit content hash is forbidden: {relative}")
        if actual_hash != expected_hash:
            raise ValueError(f"Source hash mismatch: {relative}")
        verified.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": actual_hash}
        )
    return {
        "format": "tinylora_step5_verified_source_v1",
        "source_manifest_sha256": hash_file(manifest_path),
        "files": verified,
    }


def build_source_bundle(
    repository: Path,
    verified: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Create a deterministic tarball containing exactly the verified allowlist."""
    repository = repository.resolve()
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for item in sorted(verified["files"], key=lambda value: value["path"]):
            path = safe_manifest_path(repository, item["path"])
            info = archive.gettarinfo(str(path), arcname=item["path"])
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
        receipt = json.dumps(verified, indent=2, sort_keys=True).encode() + b"\n"
        receipt_info = tarfile.TarInfo("SOURCE_BUNDLE_MANIFEST.json")
        receipt_info.size = len(receipt)
        receipt_info.mode = 0o644
        receipt_info.mtime = 0
        archive.addfile(receipt_info, io.BytesIO(receipt))
    verify_source_bundle(destination, verified)
    return {
        "sha256": hash_file(destination),
        "bytes": destination.stat().st_size,
        "files": len(verified["files"]),
    }


def verify_source_bundle(bundle: Path, verified: dict[str, Any]) -> None:
    expected = {str(item["path"]): item for item in verified["files"]}
    expected_names = set(expected) | {"SOURCE_BUNDLE_MANIFEST.json"}
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("Source bundle inventory differs from its verified allowlist")
        for member in members:
            if not member.isfile():
                raise ValueError(f"Source bundle contains a non-file member: {member.name}")
            if member.name == "SOURCE_BUNDLE_MANIFEST.json":
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read bundled source: {member.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"Bundled source hash mismatch: {member.name}")


def verify_artifact_manifest(
    artifact_root: Path,
    manifest_name: str,
    expected_files: frozenset[str],
    *,
    expected_run_id: str | None = None,
    expected_durable_uri: str | None = None,
) -> dict[str, Any]:
    manifest_path = safe_manifest_path(artifact_root, manifest_name)
    payload = parse_json_payload(manifest_path.read_text())
    if not isinstance(payload, dict) or payload.get("format") != (
        LIFECYCLE_ARTIFACT_MANIFEST_FORMAT
    ):
        raise ValueError(
            "Unsupported Step 5 lifecycle artifact manifest; legacy ambiguous v1 "
            "manifests are rejected"
        )
    try:
        payload = validate_lifecycle_artifact_manifest(payload)
    except ArtifactContractError as error:
        raise ValueError(str(error)) from error
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ValueError("Artifact manifest run_id differs from controller run_id")
    files = payload["files"]
    verified: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Artifact manifest file entry is not an object")
        relative = str(item["path"])
        if relative in seen:
            raise ValueError(f"Duplicate artifact path: {relative}")
        seen.add(relative)
        expected_hash = str(item["sha256"])
        expected_bytes = item["bytes"]
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"Invalid SHA-256 for artifact: {relative}")
        path = safe_manifest_path(artifact_root, relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Missing or non-regular artifact: {relative}")
        if path.stat().st_size != int(expected_bytes):
            raise ValueError(f"Artifact size mismatch: {relative}")
        actual_hash = hash_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Artifact hash mismatch: {relative}")
        verified[relative] = {"bytes": path.stat().st_size, "sha256": actual_hash}
    if set(verified) != set(expected_files):
        raise ValueError("Artifact manifest differs from controller-frozen inventory")
    actual_inventory: set[str] = set()
    for path in artifact_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Fetched artifact tree contains a symlink: {path}")
        if path.is_file():
            actual_inventory.add(path.relative_to(artifact_root).as_posix())
    expected_inventory = set(verified) | {PurePosixPath(manifest_name).as_posix()}
    if actual_inventory != expected_inventory:
        missing = sorted(expected_inventory - actual_inventory)
        extra = sorted(actual_inventory - expected_inventory)
        raise ValueError(f"Fetched artifact inventory mismatch; missing={missing}, extra={extra}")
    durable_object = payload["durable_object"]
    durable_uri = durable_object["uri"]
    durable_hash = durable_object["sha256"]
    durable_bytes = durable_object["bytes"]
    if expected_durable_uri is not None and durable_uri != expected_durable_uri:
        raise ValueError("Durable object URI differs from controller-frozen destination")
    return {
        "format": payload["format"],
        "run_id": payload["run_id"],
        "artifact_set_id": payload["artifact_set_id"],
        "manifest_sha256": hash_file(manifest_path),
        "files": verified,
        "durable_object": {
            "uri": durable_uri,
            "sha256": durable_hash,
            "bytes": durable_bytes,
        },
    }


def verify_expected_artifact_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected artifact inventory must be a regular file")
    payload = parse_json_payload(path.read_text())
    if not isinstance(payload, dict) or payload.get("format") != (
        "tinylora_step5_expected_artifact_inventory_v1"
    ):
        raise ValueError("Unsupported expected artifact inventory")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Expected artifact inventory must contain files")
    normalized: list[str] = []
    for value in files:
        relative = str(value)
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"Unsafe expected artifact path: {relative}")
        normalized.append(pure.as_posix())
    if len(normalized) != len(set(normalized)):
        raise ValueError("Expected artifact inventory contains duplicate paths")
    return {
        "sha256": hash_file(path),
        "files": frozenset(normalized),
    }


def validate_remote_artifact_dir(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or ".." in pure.parts
        or not pure.as_posix().startswith("/workspace/")
        or re.fullmatch(r"/workspace/[A-Za-z0-9._/-]+", pure.as_posix()) is None
    ):
        raise ValueError("remote-artifact-dir must be traversal-free under /workspace")
    return pure.as_posix()


def validate_remote_secret_path(value: str) -> str:
    pure = PurePosixPath(value)
    if (
        not pure.is_absolute()
        or ".." in pure.parts
        or not pure.as_posix().startswith("/run/secrets/")
        or re.fullmatch(r"/run/secrets/[A-Za-z0-9._-]+", pure.as_posix()) is None
    ):
        raise ValueError("remote secret path must be a direct child of /run/secrets")
    return pure.as_posix()


def parse_host_qualification(text: str, minimum_download_mbps: float) -> dict[str, Any]:
    payload = parse_json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("Host qualification output is not an object")
    decision = payload.get("decision", {})
    if not isinstance(decision, dict):
        raise ValueError("Host qualification decision is not an object")
    measured_value = payload.get("download_mbps", decision.get("median_effective_mbps"))
    if measured_value is None:
        raise ValueError("Host qualification lacks a measured download speed")
    measured = float(measured_value)
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("Host qualification download speed must be finite and positive")
    hook_accepted = payload.get("qualified", decision.get("accepted", True))
    if not isinstance(hook_accepted, bool):
        raise ValueError("Host qualification accepted decision must be boolean")
    qualified = hook_accepted and measured >= minimum_download_mbps
    return {
        **payload,
        "download_mbps": measured,
        "minimum_download_mbps": minimum_download_mbps,
        "qualified": qualified,
    }


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, urllib.parse.unquote(parsed.path.lstrip("/"))


def controller_verify_s3_object(
    aws_cli: str,
    descriptor: dict[str, Any],
    expected_files: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """HEAD and round-trip an exact S3 version from the trusted controller."""
    bucket, key = parse_s3_uri(str(descriptor["uri"]))
    head_result = run(
        [
            aws_cli,
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--checksum-mode",
            "ENABLED",
            "--output",
            "json",
        ]
    )
    head = parse_json_payload(head_result.stdout)
    if not isinstance(head, dict):
        raise ValueError("S3 HEAD response is not an object")
    version_id = str(head.get("VersionId", ""))
    if not version_id or version_id == "null":
        raise ValueError("S3 object lacks an immutable version id")
    if int(head.get("ContentLength", -1)) != int(descriptor["bytes"]):
        raise ValueError("S3 object size does not match the artifact manifest")
    checksum_b64 = str(head.get("ChecksumSHA256", ""))
    head_checksum: str | None = None
    if checksum_b64:
        try:
            head_checksum = base64.b64decode(checksum_b64, validate=True).hex()
        except (ValueError, TypeError) as error:
            raise ValueError("S3 ChecksumSHA256 is invalid base64") from error
        if head_checksum != descriptor["sha256"]:
            raise ValueError("S3 HEAD checksum does not match the artifact manifest")
    file_descriptor, download_name = tempfile.mkstemp(prefix="tinylora-step5-roundtrip-")
    os.close(file_descriptor)
    download_path = Path(download_name)
    try:
        run(
            [
                aws_cli,
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--version-id",
                version_id,
                "--checksum-mode",
                "ENABLED",
                str(download_path),
                "--output",
                "json",
            ]
        )
        roundtrip_hash = hash_file(download_path)
        roundtrip_bytes = download_path.stat().st_size
        if expected_files is not None:
            verify_durable_archive(download_path, expected_files)
    finally:
        download_path.unlink(missing_ok=True)
    if roundtrip_bytes != int(descriptor["bytes"]):
        raise ValueError("S3 round-trip size does not match the artifact manifest")
    if roundtrip_hash != descriptor["sha256"]:
        raise ValueError("S3 round-trip hash does not match the artifact manifest")
    return {
        "verified": True,
        "uri": descriptor["uri"],
        "version_id": version_id,
        "bytes": roundtrip_bytes,
        "sha256": roundtrip_hash,
        "head_checksum_sha256": head_checksum,
        "controller_roundtrip_checksum_sha256": roundtrip_hash,
    }


def verify_durable_archive(
    archive_path: Path,
    expected_files: dict[str, dict[str, Any]],
) -> None:
    """Prove the round-tripped tar contains exactly the manifest-declared bytes."""
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected_files):
            raise ValueError("Durable archive inventory differs from artifact manifest")
        for member in members:
            if not member.isfile():
                raise ValueError(f"Durable archive contains non-file member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read durable archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            expected = expected_files[member.name]
            if size != int(expected["bytes"]):
                raise ValueError(f"Durable archive member size mismatch: {member.name}")
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Durable archive member hash mismatch: {member.name}")


def classify_diagnostic(text: str) -> dict[str, Any]:
    payload = parse_json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("Diagnostic output is not an object")
    classification = str(payload.get("classification", "unknown"))
    if classification not in FAILURE_CLASSES:
        raise ValueError(f"Unsupported failure classification: {classification}")
    return {**payload, "classification": classification}


def classify_unreachable(instance: dict[str, Any] | None) -> str:
    return "host_loss" if instance_state(instance) in HOST_LOSS_STATES else "unknown"


def controller_corroborated_classification(
    diagnosis: dict[str, Any],
    instance: dict[str, Any] | None,
) -> str:
    classification = str(diagnosis["classification"])
    if classification == "host_loss" and classify_unreachable(instance) != "host_loss":
        return "unknown"
    return classification


def replacement_allowed(classification: str) -> bool:
    return classification == "host_loss"


def cleanup_action(
    *,
    durable_verified: bool,
    fetched_verified: bool,
    workload_started: bool,
    preexisting_worker: bool = False,
) -> str:
    if preexisting_worker and not (durable_verified and fetched_verified):
        return "stop"
    if not workload_started:
        return "destroy"
    return "destroy" if durable_verified and fetched_verified else "stop"


def require_execute_contract(args: argparse.Namespace) -> None:
    if not 1 <= args.max_workers <= HARD_MAX_WORKERS:
        raise ValueError(f"max-workers must be between 1 and {HARD_MAX_WORKERS}")
    if not args.execute:
        return
    if not args.confirmed_cost_approval:
        raise ValueError("Execution requires confirmed cost approval")
    if args.approved_hourly_price is None or args.approved_max_cost is None:
        raise ValueError("Execution requires approved hourly price and maximum cost")
    if not all(
        math.isfinite(value) and value > 0
        for value in (args.approved_hourly_price, args.approved_max_cost)
    ):
        raise ValueError("Approved hourly price and maximum cost must be finite and positive")
    if bool(args.offer_id) == bool(args.resume_instance_id):
        raise ValueError("Execution requires exactly one of offer-id or resume-instance-id")
    if "@sha256:" not in args.image:
        raise ValueError("Execution requires an immutable image digest")
    if args.max_software_resumes < 0:
        raise ValueError("max-software-resumes cannot be negative")
    if not math.isfinite(args.minimum_download_mbps) or args.minimum_download_mbps <= 0:
        raise ValueError("minimum-download-mbps must be finite and positive")
    if args.wait_seconds <= 0 or args.state_wait_seconds <= 0 or args.disk <= 0:
        raise ValueError("disk and wait durations must be positive")
    if (
        getattr(args, "artifact_put_url_file", None) is None
        or getattr(args, "artifact_put_receipt", None) is None
        or getattr(args, "artifact_put_receipt_sha256", None) is None
        or getattr(args, "approved_at", None) is None
        or getattr(args, "expected_durable_uri", None) is None
    ):
        raise ValueError(
            "Execution requires a protected, receipt-bound artifact PUT and frozen durable URI"
        )
    parse_s3_uri(args.expected_durable_uri)
    if (
        getattr(args, "input_url_manifest_url_file", None) is None
        or getattr(args, "host_gate_url_file", None) is None
        or getattr(args, "controller_public_key", None) is None
        or getattr(args, "controller_public_key_sha256", None) is None
    ):
        raise ValueError(
            "Execution requires protected hydration/host-gate URLs and hash-bound controller public key files"
        )
    if any(
        getattr(args, name, None) is None
        for name in (
            "checkpoint_controller_script",
            "checkpoint_controller_script_sha256",
            "checkpoint_bucket",
            "checkpoint_prefix",
            "checkpoint_controller_private_key",
        )
    ):
        raise ValueError("Execution requires a trusted checkpoint controller sidecar")
    if args.checkpoint_controller_ready_seconds <= 0:
        raise ValueError("checkpoint controller readiness timeout must be positive")


def validate_checkpoint_controller_contract(args: argparse.Namespace) -> None:
    script_content, _script_mode = read_file_without_symlinks(
        args.checkpoint_controller_script.absolute()
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_controller_script_sha256)
        is None
        or hashlib.sha256(script_content).hexdigest()
        != args.checkpoint_controller_script_sha256
    ):
        raise ValueError("checkpoint controller script differs from frozen SHA-256")
    checkpoint_controller_key_bytes(args)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{1,61}[A-Za-z0-9]", args.checkpoint_bucket) is None:
        raise ValueError("checkpoint bucket is invalid")
    prefix = PurePosixPath(args.checkpoint_prefix.strip("/"))
    if not prefix.parts or ".." in prefix.parts:
        raise ValueError("checkpoint prefix is invalid")
    validate_remote_artifact_dir(args.checkpoint_remote_exchange_dir)


def checkpoint_controller_key_bytes(args: argparse.Namespace) -> tuple[bytes, bytes]:
    private_content, private_mode = read_file_without_symlinks(
        args.checkpoint_controller_private_key.absolute()
    )
    if private_mode != 0o600:
        raise ValueError("checkpoint controller private key must have mode 0600")
    public_content, _public_mode = read_file_without_symlinks(
        args.controller_public_key.absolute()
    )
    private_public = subprocess.run(
        ["openssl", "pkey", "-pubout", "-outform", "DER"],
        input=private_content,
        check=False,
        capture_output=True,
    )
    public_der = subprocess.run(
        ["openssl", "pkey", "-pubin", "-outform", "DER"],
        input=public_content,
        check=False,
        capture_output=True,
    )
    if (
        private_public.returncode != 0
        or public_der.returncode != 0
        or private_public.stdout != public_der.stdout
    ):
        raise ValueError("checkpoint controller private/public key pair differs")
    return private_content, public_content


def create_command(args: argparse.Namespace, label: str) -> list[str]:
    if not args.offer_id:
        raise ValueError("offer-id is required to create an instance")
    command = [
        args.vastai,
        "create",
        "instance",
        args.offer_id,
        "--image",
        args.image,
        "--disk",
        str(args.disk),
        "--label",
        label,
        "--ssh",
        "--direct",
        "--cancel-unavail",
        "--raw",
    ]
    if args.cache_volume_id:
        command.extend(
            ["--link-volume", args.cache_volume_id, "--mount-path", args.cache_mount]
        )
    return command


def remote_run(
    host: str,
    port: str,
    known_hosts: Path,
    command: str,
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return run(
        [*ssh_prefix(host, port, known_hosts), command],
        check=False,
        timeout=timeout,
    )


def checkpoint_controller_command(
    args: argparse.Namespace,
    *,
    host: str,
    port: str,
    known_hosts: Path,
    ready_file: Path,
    idle_timeout_seconds: int,
    private_key: Path | None = None,
    public_key: Path | None = None,
    controller_script: Path | None = None,
) -> list[str]:
    return [
        args.checkpoint_controller_uv,
        "run",
        "--with",
        "boto3",
        "python",
        str(controller_script or args.checkpoint_controller_script.absolute()),
        "--ssh-host",
        host,
        "--ssh-port",
        port,
        "--known-hosts",
        str(known_hosts),
        "--remote-exchange-dir",
        args.checkpoint_remote_exchange_dir,
        "--bucket",
        args.checkpoint_bucket,
        "--prefix",
        args.checkpoint_prefix,
        "--controller-private-key",
        str(private_key or args.checkpoint_controller_private_key.absolute()),
        "--controller-public-key",
        str(public_key or args.controller_public_key.absolute()),
        "--url-expiry-seconds",
        "1200",
        "--idle-timeout-seconds",
        str(idle_timeout_seconds),
        "--ready-file",
        str(ready_file),
    ]


def start_checkpoint_controller(
    args: argparse.Namespace,
    *,
    host: str,
    port: str,
    known_hosts: Path,
    log_path: Path,
    idle_timeout_seconds: int,
) -> dict[str, Any]:
    snapshots: list[Path] = []
    ready_dir: Path | None = None
    log_handle: Any | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        private_content, public_content = checkpoint_controller_key_bytes(args)
        script_content, _script_mode = read_file_without_symlinks(
            args.checkpoint_controller_script.absolute()
        )
        if hashlib.sha256(script_content).hexdigest() != args.checkpoint_controller_script_sha256:
            raise ValueError("checkpoint controller script changed after preflight")
        for label, suffix, content in (
            ("private", ".pem", private_content),
            ("public", ".pem", public_content),
            ("script", ".py", script_content),
        ):
            descriptor, snapshot_name = tempfile.mkstemp(
                prefix=f"step5-checkpoint-controller-{label}-", suffix=suffix
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            snapshots.append(Path(snapshot_name))
        private_snapshot, public_snapshot, script_snapshot = snapshots
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ready_dir = Path(
            tempfile.mkdtemp(prefix="step5-checkpoint-controller-ready-")
        ).resolve()
        os.chmod(ready_dir, 0o700)
        ready_file = ready_dir / "ready.json"
        log_handle = log_path.open("ab")
        command = checkpoint_controller_command(
            args,
            host=host,
            port=port,
            known_hosts=known_hosts,
            ready_file=ready_file,
            idle_timeout_seconds=idle_timeout_seconds,
            private_key=private_snapshot,
            public_key=public_snapshot,
            controller_script=script_snapshot,
        )
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + args.checkpoint_controller_ready_seconds
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"checkpoint controller exited before readiness with {returncode}"
                )
            if ready_file.exists():
                ready_content, ready_mode = read_file_without_symlinks(ready_file)
                if ready_mode != 0o600:
                    raise RuntimeError("checkpoint controller readiness mode differs")
                payload = parse_json_payload(ready_content.decode("utf-8"))
                expected = {
                    "format": "tinylora_step5_checkpoint_controller_ready_v1",
                    "pid": process.pid,
                    "bucket": args.checkpoint_bucket,
                    "prefix": args.checkpoint_prefix,
                    "remote_exchange_dir": args.checkpoint_remote_exchange_dir,
                    "ssh_host": host,
                    "ssh_port": str(port),
                    "versioning": "Enabled",
                }
                if not isinstance(payload, dict) or any(
                    payload.get(key) != value for key, value in expected.items()
                ):
                    raise RuntimeError("checkpoint controller readiness binding differs")
                if not isinstance(payload.get("ready_at"), str):
                    raise RuntimeError("checkpoint controller readiness timestamp is missing")
                return {
                    "process": process,
                    "log_handle": log_handle,
                    "log_path": log_path,
                    "ready_file": ready_file,
                    "ready": payload,
                    "started_at": now(),
                    "command_sha256": hashlib.sha256(
                        json.dumps(command, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "snapshots": snapshots,
                    "ready_dir": ready_dir,
                }
            time.sleep(0.2)
        raise TimeoutError("checkpoint controller did not become ready")
    except BaseException:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()
        if ready_dir is not None:
            (ready_dir / "ready.json").unlink(missing_ok=True)
            ready_dir.rmdir()
        for snapshot in snapshots:
            snapshot.unlink(missing_ok=True)
        raise


def stop_checkpoint_controller(state: dict[str, Any]) -> dict[str, Any]:
    process: subprocess.Popen[bytes] = state["process"]
    unexpected_returncode = process.poll()
    if unexpected_returncode is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    returncode = process.returncode
    state["log_handle"].close()
    ready_file: Path = state["ready_file"]
    ready_file.unlink(missing_ok=True)
    state["ready_dir"].rmdir()
    for snapshot in state["snapshots"]:
        snapshot.unlink(missing_ok=True)
    log_path: Path = state["log_path"]
    return {
        "started_at": state["started_at"],
        "ended_at": now(),
        "ready": state["ready"],
        "command_sha256": state["command_sha256"],
        "unexpected_exit": unexpected_returncode is not None,
        "returncode": returncode,
        "log_path": str(log_path),
        "log_sha256": hash_file(log_path),
        "log_bytes": log_path.stat().st_size,
    }


def remote_run_with_checkpoint_controller(
    host: str,
    port: str,
    known_hosts: Path,
    command: str,
    *,
    sidecar: dict[str, Any],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    ssh_command = [*ssh_prefix(host, port, known_hosts), command]
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        worker = subprocess.Popen(ssh_command, stdout=stdout, stderr=stderr)
        sidecar_failed = False
        while worker.poll() is None:
            if sidecar["process"].poll() is not None:
                sidecar_failed = True
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
                break
            if time.monotonic() >= deadline:
                worker.kill()
                worker.wait(timeout=5)
                raise subprocess.TimeoutExpired(ssh_command, timeout)
            time.sleep(0.2)
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
    if sidecar_failed:
        error += "\ntrusted checkpoint controller exited during workload"
        returncode = 125
    else:
        returncode = worker.returncode
    return subprocess.CompletedProcess(ssh_command, returncode, output, error)


def stop_for_recovery(
    vastai: str,
    target_id: str,
    wait_seconds: int,
) -> dict[str, Any]:
    stopped = run(
        [vastai, "stop", "instance", target_id, "--retry", "3", "--raw"],
        check=False,
    )
    verified, state, inventory = wait_for_state(
        vastai,
        target_id,
        "stopped",
        wait_seconds,
    )
    return {
        "action": "stop",
        "command_succeeded": stopped.returncode == 0,
        "verified": verified,
        "observed_state": state,
        "output": (stopped.stdout or stopped.stderr or "")[:1000],
        "inventory_count": len(inventory),
    }


def destroy_and_verify(
    vastai: str,
    target_id: str,
    wait_seconds: int,
) -> dict[str, Any]:
    destroyed = run(
        [vastai, "destroy", "instance", target_id, "--retry", "3", "--raw"],
        check=False,
    )
    verified, state, inventory = wait_for_state(
        vastai,
        target_id,
        "absent",
        wait_seconds,
    )
    return {
        "action": "destroy",
        "command_succeeded": destroyed.returncode == 0,
        "verified": verified,
        "observed_state": state,
        "output": (destroyed.stdout or destroyed.stderr or "")[:1000],
        "inventory_count": len(inventory),
    }


def sync_source_bundle(
    bundle: Path,
    host: str,
    port: str,
    known_hosts: Path,
    timeout: float,
) -> None:
    run(
        [*ssh_prefix(host, port, known_hosts), "mkdir -p /workspace/workload"],
        capture=False,
        timeout=timeout,
    )
    run(
        [
            "scp",
            "-P",
            port,
            *ssh_options(known_hosts),
            str(bundle),
            f"root@{host}:/workspace/source_bundle.tar.gz",
        ],
        capture=False,
        timeout=timeout,
    )
    run(
        [
            *ssh_prefix(host, port, known_hosts),
            "tar -xzf /workspace/source_bundle.tar.gz -C /workspace/workload",
        ],
        capture=False,
        timeout=timeout,
    )


def read_file_without_symlinks(path: Path, *, max_bytes: int = 1024 * 1024) -> tuple[bytes, int]:
    """Read one file through no-follow directory descriptors and a stable fd."""
    absolute = path.absolute()
    parts = absolute.parts
    directory_fd = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
    file_fd: int | None = None
    try:
        for part in parts[1:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("provisioned controller path must be a regular file")
        if metadata.st_size > max_bytes:
            raise ValueError("provisioned controller file exceeds the size limit")
        with os.fdopen(file_fd, "rb", closefd=True) as stream:
            file_fd = None
            content = stream.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError("provisioned controller file exceeds the size limit")
        return content, stat.S_IMODE(metadata.st_mode)
    except OSError as error:
        raise ValueError("provisioned controller path may not traverse symlinks") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def remote_install_bytes(
    content: bytes,
    *,
    remote_path: str,
    directory: str,
    mode: str,
    expected_sha256: str,
    host: str,
    port: str,
    known_hosts: Path,
    timeout: float,
) -> None:
    """Atomically install stdin bytes without exposing them in argv or a remote race."""
    incoming_template = f"{directory}/.step5-provision.XXXXXX"
    command = "; ".join(
        [
            "set -eu",
            f"test -d {shlex.quote(str(PurePosixPath(directory).parent))}",
            f"test ! -L {shlex.quote(str(PurePosixPath(directory).parent))}",
            f"if [ -e {shlex.quote(directory)} ]; then test -d {shlex.quote(directory)} && test ! -L {shlex.quote(directory)}; else mkdir -m 700 {shlex.quote(directory)}; fi",
            f"incoming=$(mktemp {shlex.quote(incoming_template)})",
            'trap \'rm -f "$incoming"\' EXIT',
            'cat > "$incoming"',
            f"printf '%s  %s\\n' {shlex.quote(expected_sha256)} \"$incoming\" | sha256sum -c - >/dev/null",
            f'chmod {shlex.quote(mode)} "$incoming"',
            f'mv -f "$incoming" {shlex.quote(remote_path)}',
            "trap - EXIT",
        ]
    )
    subprocess.run(
        [*ssh_prefix(host, port, known_hosts), command],
        input=content,
        check=True,
        capture_output=True,
        timeout=timeout,
    )


def private_url_file_bytes(source: Path) -> bytes:
    content, mode = read_file_without_symlinks(source)
    if mode & 0o077:
        raise ValueError("private URL file must be mode 0600 or stricter")
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("private URL file must contain UTF-8") from error
    if len(lines) != 1 or not lines[0].startswith("https://"):
        raise ValueError("private URL file must contain an HTTPS URL")
    if content not in {lines[0].encode("utf-8"), (lines[0] + "\n").encode("utf-8")}:
        raise ValueError("private URL file may not contain surrounding whitespace")
    return content


def validate_artifact_put_contract(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bytes]:
    url_content, url_mode = read_file_without_symlinks(
        args.artifact_put_url_file.absolute()
    )
    if url_mode != 0o600:
        raise ValueError("artifact PUT URL file must have mode 0600")
    try:
        url_lines = url_content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("artifact PUT URL file must contain UTF-8") from error
    if (
        len(url_lines) != 1
        or not url_lines[0].startswith("https://")
        or url_content != (url_lines[0] + "\n").encode()
    ):
        raise ValueError("artifact PUT URL file must contain exactly one HTTPS URL line")
    receipt_content, receipt_mode = read_file_without_symlinks(
        args.artifact_put_receipt.absolute()
    )
    if receipt_mode != 0o600:
        raise ValueError("artifact PUT receipt file must have mode 0600")
    try:
        receipt = json.loads(receipt_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact PUT receipt must contain JSON") from error
    if not isinstance(receipt, dict):
        raise ValueError("artifact PUT receipt must contain an object")
    if hashlib.sha256(receipt_content).hexdigest() != args.artifact_put_receipt_sha256:
        raise ValueError("artifact PUT receipt file differs from frozen SHA-256")
    verified = revalidate_artifact_put_freshness(
        args,
        receipt,
        url_content,
        current=datetime.now(timezone.utc),
    )
    return verified, url_content


def revalidate_artifact_put_freshness(
    args: argparse.Namespace,
    receipt: dict[str, Any],
    url_content: bytes,
    *,
    current: datetime,
) -> dict[str, Any]:
    verified = validate_receipt(
        receipt,
        url_bytes=url_content,
        expected_receipt_sha256=args.artifact_put_receipt_sha256,
        expected_durable_uri=args.expected_durable_uri,
        expected_approved_at=args.approved_at,
        now=current,
        max_approval_age_seconds=ARTIFACT_APPROVAL_MAX_AGE_SECONDS,
    )
    remaining = (
        parse_utc(verified["expires_at"], label="expires_at") - current
    ).total_seconds()
    if remaining <= ARTIFACT_MINIMUM_REMAINING_SECONDS:
        raise ValueError("artifact PUT URL has less than 15 minutes remaining")
    return verified


def validate_distinct_url_sources(args: argparse.Namespace) -> None:
    hydration = private_url_file_bytes(
        args.input_url_manifest_url_file.absolute()
    ).decode("utf-8").removesuffix("\n")
    host_gate = private_url_file_bytes(args.host_gate_url_file.absolute()).decode(
        "utf-8"
    ).removesuffix("\n")
    if hydration == host_gate:
        raise ValueError("hydration and host-gate URL files must contain distinct URLs")


def sync_private_url_file(
    source: Path,
    remote_path: str,
    host: str,
    port: str,
    known_hosts: Path,
    timeout: float,
) -> None:
    content = private_url_file_bytes(source)
    sync_private_url_bytes(
        content,
        remote_path=remote_path,
        host=host,
        port=port,
        known_hosts=known_hosts,
        timeout=timeout,
    )


def sync_private_url_bytes(
    content: bytes,
    *,
    remote_path: str,
    host: str,
    port: str,
    known_hosts: Path,
    timeout: float,
) -> None:
    remote_install_bytes(
        content,
        remote_path=remote_path,
        directory="/run/secrets",
        mode="600",
        expected_sha256=hashlib.sha256(content).hexdigest(),
        host=host,
        port=port,
        known_hosts=known_hosts,
        timeout=timeout,
    )


def sync_hash_bound_public_key(
    source: Path,
    expected_sha256: str,
    host: str,
    port: str,
    known_hosts: Path,
    timeout: float,
) -> None:
    content, _mode = read_file_without_symlinks(source)
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("controller public key SHA-256 is invalid")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("controller public key SHA-256 differs from frozen identity")
    parsed = subprocess.run(
        ["openssl", "pkey", "-pubin", "-noout"],
        input=content,
        check=False,
        capture_output=True,
    )
    if parsed.returncode != 0:
        raise ValueError("controller public key must be a parseable public PEM")
    remote_install_bytes(
        content,
        remote_path=CONTROLLER_PUBLIC_KEY_REMOTE_PATH,
        directory="/workspace/inputs",
        mode="644",
        expected_sha256=expected_sha256,
        host=host,
        port=port,
        known_hosts=known_hosts,
        timeout=timeout,
    )


def restart_stopped_instance(
    args: argparse.Namespace,
    *,
    target_id: str,
    artifact_put_receipt: dict[str, Any],
    artifact_put_url_bytes: bytes,
    cost_deadline: float,
) -> subprocess.CompletedProcess[str]:
    """Restart the same worker only while its upload authorization remains fresh."""
    revalidate_artifact_put_freshness(
        args,
        artifact_put_receipt,
        artifact_put_url_bytes,
        current=datetime.now(timezone.utc),
    )
    return run(
        [args.vastai, "start", "instance", target_id, "--retry", "3", "--raw"],
        check=False,
        timeout=remaining_cost_seconds(cost_deadline),
    )


def verify_remote_source_receipt(
    host: str,
    port: str,
    known_hosts: Path,
    expected: dict[str, Any],
    timeout: float,
) -> None:
    receipt = remote_run(
        host,
        port,
        known_hosts,
        "cat /workspace/workload/SOURCE_BUNDLE_MANIFEST.json",
        timeout=timeout,
    )
    if receipt.returncode != 0 or parse_json_payload(receipt.stdout) != expected:
        raise ValueError("Remote source receipt does not match this run binding and allowlist")


def fetch_artifacts(
    host: str,
    port: str,
    known_hosts: Path,
    remote_artifact_dir: str,
    fetch_dir: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    if fetch_dir.exists():
        raise FileExistsError(f"Refusing to mix artifacts with existing path: {fetch_dir}")
    fetch_dir.parent.mkdir(parents=True, exist_ok=True)
    return run(
        [
            "scp",
            "-r",
            "-P",
            port,
            *ssh_options(known_hosts),
            f"root@{host}:{remote_artifact_dir}",
            str(fetch_dir),
        ],
        check=False,
        capture=False,
        timeout=timeout,
    )


def print_dry_run(args: argparse.Namespace, label: str) -> None:
    print(f"label: {label}")
    if args.resume_instance_id:
        print(f"inventory: require labeled worker {args.resume_instance_id}; start it if stopped")
    else:
        print(f"inventory: refuse when {args.max_workers} Step 5 workers already exist")
        print(f"create: {shlex.join(create_command(args, label))}")
    print(f"qualify hook sha256: {command_sha256(args.host_qualification_command)}")
    print(f"reject: download speed below {args.minimum_download_mbps:g} Mbps")
    print(
        f"sync: hash-verified allowlist {args.source_manifest.resolve()} "
        "-> /workspace/workload"
    )
    print(f"run/resume command sha256: {command_sha256(args.remote_command)}")
    print(f"diagnostic hook sha256: {command_sha256(args.diagnostic_command)}")
    print("durable verification: controller S3 HEAD + exact-version round-trip SHA-256")
    print(f"fetch: {args.remote_artifact_dir} -> {args.fetch_dir.resolve()}")
    print("cleanup: destroy only after durable receipt and fetched hashes verify; else stop")
    print("replacement: never automatic; permissible only for classified host loss")


def command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode()).hexdigest()


def main() -> int:
    args = parse_args()
    require_execute_contract(args)
    artifact_put_receipt: dict[str, Any] | None = None
    artifact_put_url_bytes: bytes | None = None
    if args.execute:
        validate_checkpoint_controller_contract(args)
        validate_distinct_url_sources(args)
        artifact_put_receipt, artifact_put_url_bytes = validate_artifact_put_contract(args)
    reject_credential_bearing_commands(
        [
            args.remote_command,
            args.host_qualification_command,
            args.diagnostic_command,
            args.software_recovery_command,
        ]
    )
    verified_source = verify_source_manifest(
        args.repo.resolve(),
        args.source_manifest.resolve(),
    )
    source_receipt = {
        **verified_source,
        "binding": {
            "run_id": args.run_id,
            "candidate": args.candidate,
            "attempt": args.attempt,
        },
    }
    expected_inventory = verify_expected_artifact_inventory(
        args.expected_artifact_inventory.resolve()
    )
    args.remote_artifact_dir = validate_remote_artifact_dir(args.remote_artifact_dir)
    args.remote_artifact_put_url_file = validate_remote_secret_path(
        args.remote_artifact_put_url_file
    )
    label = unique_label(args.run_id, args.candidate, args.attempt)
    metadata: dict[str, Any] = {
        "format": "tinylora_step5_vast_lifecycle_v1",
        "dry_run": not args.execute,
        "run_id": args.run_id,
        "candidate": args.candidate,
        "attempt": args.attempt,
        "label": label,
        "offer_id": args.offer_id,
        "resume_instance_id": args.resume_instance_id,
        "image": args.image,
        "cache_volume_id": args.cache_volume_id,
        "minimum_download_mbps": args.minimum_download_mbps,
        "max_workers": args.max_workers,
        "source": source_receipt,
        "expected_artifact_inventory_sha256": expected_inventory["sha256"],
        "started_at": now(),
        "replacement_performed": False,
        "replacement_allowed": False,
    }
    if artifact_put_receipt is not None:
        metadata["artifact_put_authorization"] = {
            "receipt_sha256": args.artifact_put_receipt_sha256,
            "receipt_id": artifact_put_receipt["receipt_id"],
            "durable_uri": artifact_put_receipt["durable_uri"],
            "expires_at": artifact_put_receipt["expires_at"],
            "approved_at": artifact_put_receipt["approved_at"],
            "method": artifact_put_receipt["method"],
        }
    print_dry_run(args, label)
    if not args.execute:
        return 0

    target_id: str | None = None
    source_bundle: Path | None = None
    known_hosts: Path | None = None
    create_attempted = False
    workload_started = False
    durable_verified = False
    fetched_verified = False
    intentionally_stopped = False
    exit_code = 1
    host = ""
    port = ""
    try:
        file_descriptor, bundle_name = tempfile.mkstemp(
            prefix="tinylora-step5-source-",
            suffix=".tar.gz",
        )
        os.close(file_descriptor)
        source_bundle = Path(bundle_name)
        metadata["source_bundle"] = build_source_bundle(
            args.repo.resolve(),
            source_receipt,
            source_bundle,
        )
        known_hosts_descriptor, known_hosts_name = tempfile.mkstemp(
            prefix="tinylora-step5-known-hosts-"
        )
        os.close(known_hosts_descriptor)
        known_hosts = Path(known_hosts_name)
        with controller_lock(args.controller_lock):
            inventory = show_instances(args.vastai)
            enforce_worker_cap(
                inventory,
                max_workers=args.max_workers,
                resume_instance_id=args.resume_instance_id,
            )
            if args.resume_instance_id:
                target_id = str(args.resume_instance_id)
                current = find_instance(inventory, target_id)
                assert current is not None
                if not resume_label_matches(
                    instance_label(current),
                    args.run_id,
                    args.candidate,
                    args.attempt,
                ):
                    raise ValueError("Resume worker label does not match this run binding")
                actual_hourly_price = hourly_price(current)
                assert args.approved_hourly_price is not None
                enforce_price(actual_hourly_price, args.approved_hourly_price)
                cost_started_monotonic = time.monotonic()
                metadata["active_label"] = instance_label(current)
                metadata["resumed_existing_worker"] = True
            else:
                assert args.offer_id is not None
                offer = query_offer(args.vastai, args.offer_id)
                actual_hourly_price = hourly_price(offer)
                assert args.approved_hourly_price is not None
                enforce_price(actual_hourly_price, args.approved_hourly_price)
                assert artifact_put_receipt is not None
                assert artifact_put_url_bytes is not None
                revalidate_artifact_put_freshness(
                    args,
                    artifact_put_receipt,
                    artifact_put_url_bytes,
                    current=datetime.now(timezone.utc),
                )
                cost_started_monotonic = time.monotonic()
                command = create_command(args, label)
                create_attempted = True
                created = run(command)
                try:
                    target_id = parse_instance_id(created.stdout)
                except ValueError:
                    target_id = poll_created_instance_id(
                        args.vastai,
                        label,
                        args.state_wait_seconds,
                    )
                metadata["active_label"] = label
                metadata["resumed_existing_worker"] = False
        assert args.approved_hourly_price is not None
        assert args.approved_max_cost is not None
        maximum_runtime_seconds = args.approved_max_cost / actual_hourly_price * 3600.0
        cost_deadline = cost_started_monotonic + maximum_runtime_seconds
        assert artifact_put_receipt is not None
        artifact_remaining_seconds = (
            parse_utc(artifact_put_receipt["expires_at"], label="expires_at").timestamp()
            - time.time()
            - 300
        )
        if artifact_remaining_seconds <= 0:
            raise ValueError("artifact PUT URL expired before worker provisioning")
        cost_deadline = min(
            cost_deadline,
            time.monotonic() + artifact_remaining_seconds,
        )
        metadata["actual_hourly_price"] = actual_hourly_price
        metadata["approved_max_cost"] = args.approved_max_cost
        metadata["maximum_runtime_seconds"] = maximum_runtime_seconds
        if args.resume_instance_id:
            current = find_instance(show_instances(args.vastai), target_id)
            if state_matches(instance_state(current), "stopped"):
                assert artifact_put_receipt is not None
                assert artifact_put_url_bytes is not None
                revalidate_artifact_put_freshness(
                    args,
                    artifact_put_receipt,
                    artifact_put_url_bytes,
                    current=datetime.now(timezone.utc),
                )
                started = run(
                    [args.vastai, "start", "instance", target_id, "--retry", "3", "--raw"],
                    check=False,
                    timeout=remaining_cost_seconds(cost_deadline),
                )
                metadata["resume_start_exit_code"] = started.returncode
                if started.returncode != 0:
                    raise RuntimeError("Could not start the existing stopped worker")
        metadata["instance_id"] = target_id
        host, port, host_key = wait_for_real_ssh(
            args.vastai,
            target_id,
            min(args.wait_seconds, int(remaining_cost_seconds(cost_deadline))),
            known_hosts,
        )
        metadata.update(
            {"ssh_host": host, "ssh_port": port, "ssh_ready_at": now(), "host_key": host_key}
        )

        if not args.resume_instance_id:
            assert source_bundle is not None
            sync_source_bundle(
                source_bundle,
                host,
                port,
                known_hosts,
                remaining_cost_seconds(cost_deadline),
            )
        assert artifact_put_url_bytes is not None
        sync_private_url_bytes(
            artifact_put_url_bytes,
            remote_path=args.remote_artifact_put_url_file,
            host=host,
            port=port,
            known_hosts=known_hosts,
            timeout=remaining_cost_seconds(cost_deadline),
        )
        assert args.input_url_manifest_url_file is not None
        sync_private_url_file(
            args.input_url_manifest_url_file.absolute(),
            INPUT_URL_MANIFEST_REMOTE_PATH,
            host,
            port,
            known_hosts,
            remaining_cost_seconds(cost_deadline),
        )
        assert args.host_gate_url_file is not None
        sync_private_url_file(
            args.host_gate_url_file.absolute(),
            HOST_GATE_URL_REMOTE_PATH,
            host,
            port,
            known_hosts,
            remaining_cost_seconds(cost_deadline),
        )
        assert args.controller_public_key is not None
        assert args.controller_public_key_sha256 is not None
        sync_hash_bound_public_key(
            args.controller_public_key.absolute(),
            args.controller_public_key_sha256,
            host,
            port,
            known_hosts,
            remaining_cost_seconds(cost_deadline),
        )
        verify_remote_source_receipt(
            host,
            port,
            known_hosts,
            source_receipt,
            remaining_cost_seconds(cost_deadline),
        )

        qualification_resumes = 0
        while True:
            qualification = remote_run(
                host,
                port,
                known_hosts,
                args.host_qualification_command,
                timeout=remaining_cost_seconds(cost_deadline),
            )
            metadata.setdefault("host_qualification_attempts", []).append(
                {"at": now(), "exit_code": qualification.returncode}
            )
            if qualification.returncode == 0:
                break
            diagnostic = remote_run(
                host,
                port,
                known_hosts,
                f"cd /workspace/workload && {args.diagnostic_command}",
                timeout=remaining_cost_seconds(cost_deadline),
            )
            if diagnostic.returncode == 0:
                diagnosis = classify_diagnostic(diagnostic.stdout)
                current = find_instance(show_instances(args.vastai), target_id)
                classification = controller_corroborated_classification(
                    diagnosis,
                    current,
                )
                diagnosis["controller_classification"] = classification
                diagnosis["controller_instance_state"] = instance_state(current)
                metadata.setdefault("diagnostics", []).append(diagnosis)
            else:
                current = find_instance(show_instances(args.vastai), target_id)
                classification = classify_unreachable(current)
                metadata.setdefault("diagnostics", []).append(
                    {
                        "classification": classification,
                        "diagnostic_exit_code": diagnostic.returncode,
                        "instance_state": instance_state(current),
                    }
                )
            metadata["failure_classification"] = classification
            metadata["failure_subtype"] = "qualification_command_failure"
            metadata["replacement_allowed"] = replacement_allowed(classification)
            if classification != "software_failure":
                return 1
            stopped = stop_for_recovery(args.vastai, target_id, args.state_wait_seconds)
            metadata.setdefault("recovery_stops", []).append(stopped)
            intentionally_stopped = True
            if (
                not stopped["verified"]
                or not args.software_recovery_command
                or qualification_resumes >= args.max_software_resumes
            ):
                return 1
            assert artifact_put_receipt is not None
            assert artifact_put_url_bytes is not None
            restarted = restart_stopped_instance(
                args,
                target_id=target_id,
                artifact_put_receipt=artifact_put_receipt,
                artifact_put_url_bytes=artifact_put_url_bytes,
                cost_deadline=cost_deadline,
            )
            if restarted.returncode != 0:
                metadata["recovery_start_exit_code"] = restarted.returncode
                return 1
            host, port, _host_key = wait_for_real_ssh(
                args.vastai,
                target_id,
                min(args.wait_seconds, int(remaining_cost_seconds(cost_deadline))),
                known_hosts,
            )
            recovery = remote_run(
                host,
                port,
                known_hosts,
                f"cd /workspace/workload && {args.software_recovery_command}",
                timeout=remaining_cost_seconds(cost_deadline),
            )
            metadata.setdefault("software_recoveries", []).append(
                {"at": now(), "exit_code": recovery.returncode, "phase": "qualification"}
            )
            if recovery.returncode != 0:
                return 1
            qualification_resumes += 1
            intentionally_stopped = False
        metadata["host_qualification_exit_code"] = qualification.returncode
        host_result = parse_host_qualification(
            qualification.stdout,
            args.minimum_download_mbps,
        )
        metadata["host_qualification"] = host_result
        if not host_result["qualified"]:
            metadata["failure_classification"] = "host_loss"
            metadata["failure_subtype"] = "minimum_download_speed"
            metadata["replacement_allowed"] = True
            return 1

        metadata["workload_started_at"] = now()
        workload_started = True
        software_resumes = 0
        while True:
            controller_attempt = len(metadata.get("checkpoint_controller_attempts", [])) + 1
            controller_log = args.metadata.parent / (
                f"{safe_component(args.run_id)}-{safe_component(args.candidate)}-"
                f"checkpoint-controller-{controller_attempt}.log"
            )
            try:
                sidecar = start_checkpoint_controller(
                    args,
                    host=host,
                    port=port,
                    known_hosts=known_hosts,
                    log_path=controller_log,
                    idle_timeout_seconds=max(
                        300,
                        int(remaining_cost_seconds(cost_deadline)) + 60,
                    ),
                )
            except Exception as error:
                failed_status = {
                    "started_at": now(),
                    "ended_at": now(),
                    "ready": False,
                    "error": f"{type(error).__name__}: {error}",
                    "log_path": str(controller_log),
                    "log_bytes": controller_log.stat().st_size
                    if controller_log.exists()
                    else 0,
                    "log_sha256": hash_file(controller_log)
                    if controller_log.exists()
                    else None,
                }
                metadata.setdefault("checkpoint_controller_attempts", []).append(
                    failed_status
                )
                raise
            try:
                workload = remote_run_with_checkpoint_controller(
                    host,
                    port,
                    known_hosts,
                    f"cd /workspace/workload && {args.remote_command}",
                    sidecar=sidecar,
                    timeout=remaining_cost_seconds(cost_deadline),
                )
            finally:
                controller_status = stop_checkpoint_controller(sidecar)
                metadata.setdefault("checkpoint_controller_attempts", []).append(
                    controller_status
                )
            if controller_status["unexpected_exit"] and workload.returncode == 0:
                workload = subprocess.CompletedProcess(
                    workload.args,
                    125,
                    workload.stdout,
                    workload.stderr
                    + "\ntrusted checkpoint controller exited before workload completion",
                )
            exit_code = workload.returncode
            metadata.setdefault("workload_attempts", []).append(
                {
                    "at": now(),
                    "exit_code": exit_code,
                    "checkpoint_controller_ready": True,
                    "checkpoint_controller_unexpected_exit": controller_status[
                        "unexpected_exit"
                    ],
                }
            )
            if exit_code == 0:
                break

            diagnostic = remote_run(
                host,
                port,
                known_hosts,
                f"cd /workspace/workload && {args.diagnostic_command}",
                timeout=remaining_cost_seconds(cost_deadline),
            )
            if diagnostic.returncode == 0:
                diagnosis = classify_diagnostic(diagnostic.stdout)
                current = find_instance(show_instances(args.vastai), target_id)
                classification = controller_corroborated_classification(
                    diagnosis,
                    current,
                )
                diagnosis["controller_classification"] = classification
                diagnosis["controller_instance_state"] = instance_state(current)
                metadata.setdefault("diagnostics", []).append(diagnosis)
            else:
                current = find_instance(show_instances(args.vastai), target_id)
                classification = classify_unreachable(current)
                metadata.setdefault("diagnostics", []).append(
                    {
                        "classification": classification,
                        "diagnostic_exit_code": diagnostic.returncode,
                        "instance_state": instance_state(current),
                    }
                )
            metadata["failure_classification"] = classification
            metadata["replacement_allowed"] = replacement_allowed(classification)
            if classification != "software_failure":
                return exit_code or 1

            stopped = stop_for_recovery(args.vastai, target_id, args.state_wait_seconds)
            metadata.setdefault("recovery_stops", []).append(stopped)
            intentionally_stopped = True
            if (
                not stopped["verified"]
                or not args.software_recovery_command
                or software_resumes >= args.max_software_resumes
            ):
                return exit_code or 1

            assert artifact_put_receipt is not None
            assert artifact_put_url_bytes is not None
            restarted = restart_stopped_instance(
                args,
                target_id=target_id,
                artifact_put_receipt=artifact_put_receipt,
                artifact_put_url_bytes=artifact_put_url_bytes,
                cost_deadline=cost_deadline,
            )
            if restarted.returncode != 0:
                metadata["recovery_start_exit_code"] = restarted.returncode
                return exit_code or 1
            host, port, _host_key = wait_for_real_ssh(
                args.vastai,
                target_id,
                min(args.wait_seconds, int(remaining_cost_seconds(cost_deadline))),
                known_hosts,
            )
            recovery = remote_run(
                host,
                port,
                known_hosts,
                f"cd /workspace/workload && {args.software_recovery_command}",
                timeout=remaining_cost_seconds(cost_deadline),
            )
            metadata.setdefault("software_recoveries", []).append(
                {"at": now(), "exit_code": recovery.returncode}
            )
            if recovery.returncode != 0:
                intentionally_stopped = False
                return recovery.returncode or 1
            software_resumes += 1
            intentionally_stopped = False

        fetched = fetch_artifacts(
            host,
            port,
            known_hosts,
            args.remote_artifact_dir,
            args.fetch_dir.resolve(),
            remaining_cost_seconds(cost_deadline),
        )
        metadata["fetch_exit_code"] = fetched.returncode
        if fetched.returncode == 0:
            try:
                verified = verify_artifact_manifest(
                    args.fetch_dir.resolve(),
                    args.artifact_manifest_name,
                    expected_inventory["files"],
                    expected_run_id=args.run_id,
                    expected_durable_uri=args.expected_durable_uri,
                )
                metadata["fetched_artifacts"] = verified
                fetched_verified = True
            except (OSError, ValueError) as error:
                metadata["artifact_verification_error"] = str(error)
        if fetched_verified:
            preverification_stop = stop_for_recovery(
                args.vastai,
                target_id,
                args.state_wait_seconds,
            )
            metadata["preverification_stop"] = preverification_stop
            intentionally_stopped = True
            if not preverification_stop["verified"]:
                return 1
            try:
                durable_result = controller_verify_s3_object(
                    args.aws_cli,
                    metadata["fetched_artifacts"]["durable_object"],
                    metadata["fetched_artifacts"]["files"],
                )
                metadata["durable_verification"] = durable_result
                durable_verified = durable_result["verified"]
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                metadata["durable_verification_error"] = str(error)
        metadata["fetched_verified"] = fetched_verified
        metadata["durable_verified"] = durable_verified
        return 0 if durable_verified and fetched_verified else 1
    except KeyboardInterrupt:
        metadata["interrupted"] = True
        return 130
    except Exception as error:
        metadata["wrapper_error"] = f"{type(error).__name__}: {error}"
        return 1
    finally:
        metadata["ended_at"] = now()
        teardown_error: str | None = None
        if target_id is None and create_attempted:
            try:
                with controller_lock(args.controller_lock):
                    target_id = poll_created_instance_id(
                        args.vastai,
                        label,
                        args.state_wait_seconds,
                    )
                metadata["instance_id_recovered_during_teardown"] = target_id
            except Exception as error:
                metadata["orphan_risk"] = {
                    "exact_label": label,
                    "requires_operator_inventory_and_stop": True,
                    "error": f"{type(error).__name__}: {error}",
                }
                teardown_error = (
                    f"instance creation is ambiguous for exact label {label}; "
                    "operator inventory and stop are mandatory"
                )
        if target_id is not None:
            action = cleanup_action(
                durable_verified=durable_verified,
                fetched_verified=fetched_verified,
                workload_started=workload_started,
                preexisting_worker=bool(args.resume_instance_id) or intentionally_stopped,
            )
            try:
                if action == "destroy":
                    teardown = destroy_and_verify(
                        args.vastai,
                        target_id,
                        args.state_wait_seconds,
                    )
                else:
                    teardown = stop_for_recovery(
                        args.vastai,
                        target_id,
                        args.state_wait_seconds,
                    )
            except Exception as error:
                teardown = {
                    "action": action,
                    "verified": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            metadata["teardown"] = teardown
            metadata["recovery_required"] = action == "stop"
            if action == "stop":
                metadata["recovery_instance_id"] = target_id
            if not teardown["verified"]:
                teardown_error = (
                    f"{action} of instance {target_id} was not verified; inspect metadata"
                )
        if source_bundle is not None:
            source_bundle.unlink(missing_ok=True)
        if known_hosts is not None:
            known_hosts.unlink(missing_ok=True)
        write_metadata(args.metadata, metadata)
        if teardown_error:
            raise RuntimeError(teardown_error)


if __name__ == "__main__":
    raise SystemExit(main())
