#!/usr/bin/env python3
"""Run one Vast pilot worker with real SSH readiness and fail-safe teardown."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


WORKLOAD_MANIFEST_FORMAT = "tinylora_workload_archive_v1"
ARTIFACT_INVENTORY_FORMAT = "tinylora_pilot_artifact_inventory_v1"
SENSITIVE_COMPONENTS = {".git", ".secrets"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastai", required=True)
    parser.add_argument("--offer-id", required=True)
    parser.add_argument("--rank", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--workload-archive", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument("--fetch-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--disk", type=int, default=80)
    parser.add_argument("--remote-command", required=True)
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmed-cost-approval", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def parse_instance_id(text: str) -> str:
    payload = json.loads(text)
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


def ssh_prefix(host: str, port: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "-p",
        port,
        f"root@{host}",
    ]


def wait_for_real_ssh(vastai: str, instance_id: str, wait_seconds: int) -> tuple[str, str]:
    deadline = time.time() + wait_seconds
    last = ""
    while time.time() < deadline:
        url = run([vastai, "ssh-url", instance_id], check=False)
        last = (url.stdout or "") + (url.stderr or "")
        if url.returncode == 0:
            try:
                host, port = parse_ssh_url(last)
            except ValueError:
                time.sleep(10)
                continue
            probe = run([*ssh_prefix(host, port), "true"], check=False)
            if probe.returncode == 0:
                return host, port
        time.sleep(10)
    raise TimeoutError(f"SSH never became usable for {instance_id}: {last[:500]}")


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_member_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    for part in path.parts:
        if part in SENSITIVE_COMPONENTS or part == ".env" or part.startswith(".env."):
            raise ValueError(f"sensitive path is forbidden in workload archive: {name!r}")
    return path


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.exists() or component.is_symlink():
            if component.is_symlink():
                raise ValueError(f"{label} contains a symlinked path component: {component}")


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    _reject_symlink_chain(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file: {path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _read_regular_json_snapshot(
    path: Path, *, label: str
) -> tuple[dict[str, Any], str]:
    raw = _read_regular_bytes(path, label=label)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {label}: root must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _read_regular_json(path: Path, *, label: str) -> dict[str, Any]:
    payload, _ = _read_regular_json_snapshot(path, label=label)
    return payload


def _validate_hash_map(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must contain a non-empty files object")
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        if not isinstance(raw_name, str):
            raise ValueError(f"{label} has a non-string path")
        name = _validate_archive_member_path(raw_name).as_posix()
        if name != raw_name:
            raise ValueError(f"{label} path is not canonical: {raw_name!r}")
        if not isinstance(raw_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_digest):
            raise ValueError(f"{label} has invalid SHA-256 for {raw_name!r}")
        result[name] = raw_digest
    return result


def _validate_path_list(value: Any, *, label: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain a non-empty files list")
    result: set[str] = set()
    for raw_name in value:
        if not isinstance(raw_name, str):
            raise ValueError(f"{label} has a non-string path")
        name = _validate_archive_member_path(raw_name).as_posix()
        if name != raw_name:
            raise ValueError(f"{label} path is not canonical: {raw_name!r}")
        if name in result:
            raise ValueError(f"{label} contains a duplicate path: {name!r}")
        result.add(name)
    return result


def validate_workload_archive(archive: Path, manifest: Path) -> dict[str, Any]:
    """Validate an immutable, explicitly allowlisted workload before any rental."""
    _reject_symlink_chain(archive, label="workload archive")
    if not archive.is_file() or not archive.name.endswith(".tar.gz"):
        raise ValueError(f"workload must be a regular .tar.gz archive, not a directory: {archive}")
    payload = _read_regular_json(manifest, label="workload manifest")
    if payload.get("format") != WORKLOAD_MANIFEST_FORMAT:
        raise ValueError(f"unsupported workload manifest format: {payload.get('format')!r}")
    expected_archive_hash = payload.get("archive_sha256")
    if not isinstance(expected_archive_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_archive_hash
    ):
        raise ValueError("workload manifest has invalid archive_sha256")
    actual_archive_hash = sha256_file(archive)
    if actual_archive_hash != expected_archive_hash:
        raise ValueError("workload archive SHA-256 does not match its frozen manifest")
    expected_files = _validate_hash_map(payload.get("files"), label="workload manifest")
    actual_files: dict[str, str] = {}
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            seen: set[str] = set()
            for member in bundle.getmembers():
                name = _validate_archive_member_path(member.name).as_posix()
                if name in seen:
                    raise ValueError(f"duplicate workload archive member: {name}")
                seen.add(name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"workload archive may contain only regular files and directories: {name}"
                    )
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read workload archive member: {name}")
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                actual_files[name] = digest.hexdigest()
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"invalid workload archive: {error}") from error
    if actual_files.keys() != expected_files.keys():
        missing = sorted(expected_files.keys() - actual_files.keys())
        extra = sorted(actual_files.keys() - expected_files.keys())
        raise ValueError(f"workload archive allowlist mismatch: missing={missing}, extra={extra}")
    mismatched = sorted(
        name for name, digest in actual_files.items() if digest != expected_files[name]
    )
    if mismatched:
        raise ValueError(f"workload archive member SHA-256 mismatch: {mismatched}")
    return {
        "format": WORKLOAD_MANIFEST_FORMAT,
        "archive_sha256": actual_archive_hash,
        "files": actual_files,
    }


def load_artifact_inventory_snapshot(path: Path, rank: int) -> tuple[set[str], str]:
    payload, snapshot_sha256 = _read_regular_json_snapshot(path, label="artifact inventory")
    if payload.get("format") != ARTIFACT_INVENTORY_FORMAT:
        raise ValueError(f"unsupported artifact inventory format: {payload.get('format')!r}")
    files = _validate_path_list(payload.get("files"), label="artifact inventory")
    prefix = f"rank_{rank}/"
    if any(not name.startswith(prefix) for name in files):
        raise ValueError(f"artifact inventory may contain only {prefix!r} paths")
    return files, snapshot_sha256


def load_artifact_inventory(path: Path, rank: int) -> set[str]:
    files, _ = load_artifact_inventory_snapshot(path, rank)
    return files


def verified_artifact_hashes(
    fetch_dir: Path, rank: int, expected_inventory: set[str]
) -> dict[str, str]:
    """Verify the fetched tree exactly matches a controller-frozen inventory."""
    result_dir = fetch_dir / f"rank_{rank}"
    required_names = (
        f"rank_{rank}/result.json",
        f"rank_{rank}/pilot_state.pt",
        f"rank_{rank}/tinylora_rank{rank}_basis.pt",
    )
    if not set(required_names).issubset(expected_inventory):
        raise ValueError("artifact inventory omits the required pilot artifact set")
    if result_dir.is_symlink():
        raise ValueError(f"fetched artifact directory must not be a symlink: {result_dir}")
    actual_paths: dict[str, Path] = {}
    if result_dir.is_dir():
        for root, directories, files in os.walk(result_dir, followlinks=False):
            root_path = Path(root)
            for directory in directories:
                path = root_path / directory
                if path.is_symlink():
                    raise ValueError(f"fetched artifact directory contains a symlink: {path}")
            for filename in files:
                path = root_path / filename
                if path.is_symlink():
                    raise ValueError(f"fetched artifact file is a symlink: {path}")
                if not path.is_file():
                    raise ValueError(f"fetched artifact is not a regular file: {path}")
                actual_paths[path.relative_to(fetch_dir).as_posix()] = path
    missing = [name for name in required_names if name not in actual_paths or actual_paths[name].stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete fetched artifact set: {missing}")
    if actual_paths.keys() != expected_inventory:
        missing_inventory = sorted(expected_inventory - actual_paths.keys())
        extra = sorted(actual_paths.keys() - expected_inventory)
        raise ValueError(
            f"fetched artifact inventory mismatch: missing={missing_inventory}, extra={extra}"
        )
    hashes = {name: sha256_file(path) for name, path in actual_paths.items()}
    return hashes


def cleanup_action(*, workload_started: bool, artifacts_verified: bool) -> str:
    """Preserve a worker whenever it may contain the only artifact copy."""
    if workload_started and not artifacts_verified:
        return "stop"
    return "destroy"


def require_empty_artifact_destination(fetch_dir: Path, rank: int) -> None:
    _reject_symlink_chain(fetch_dir, label="artifact destination")
    result_dir = fetch_dir / f"rank_{rank}"
    if result_dir.exists():
        raise FileExistsError(
            f"Refusing to mix fetched artifacts with an existing result: {result_dir}"
        )


def main() -> int:
    args = parse_args()
    workload = validate_workload_archive(args.workload_archive, args.workload_manifest)
    artifact_inventory, artifact_inventory_sha256 = load_artifact_inventory_snapshot(
        args.artifact_inventory, args.rank
    )
    label = f"codex-vast-tinylora-rank-{args.rank}-retry"
    create = [
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
        "--raw",
    ]
    print(f"create: {shlex.join(create)}", flush=True)
    print(
        f"sync verified archive: {args.workload_archive.resolve()} -> /workspace/workload.tar.gz",
        flush=True,
    )
    print(f"run: {args.remote_command}", flush=True)
    print(f"fetch: /workspace/workload/results/rank_{args.rank}", flush=True)
    print("cleanup: destroy only after fetched artifacts verify; otherwise stop for recovery", flush=True)
    if not args.execute:
        return 0
    if not args.confirmed_cost_approval:
        raise ValueError("Execution requires confirmed cost approval")
    require_empty_artifact_destination(args.fetch_dir, args.rank)
    metadata: dict[str, Any] = {
        "format": "tinylora_vast_worker_lifecycle_v1",
        "rank": args.rank,
        "offer_id": args.offer_id,
        "image": args.image,
        "disk_gb": args.disk,
        "started_at": now(),
        "workload_archive_sha256": workload["archive_sha256"],
        "workload_file_sha256": workload["files"],
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "expected_artifact_inventory": sorted(artifact_inventory),
    }
    instance_id: str | None = None
    exit_code = 1
    workload_started = False
    artifacts_verified = False
    try:
        created = run(create)
        instance_id = parse_instance_id(created.stdout)
        metadata["instance_id"] = instance_id
        print(f"instance: {instance_id}", flush=True)
        host, port = wait_for_real_ssh(args.vastai, instance_id, args.wait_seconds)
        metadata.update({"ssh_host": host, "ssh_port": port, "ssh_ready_at": now()})
        print(f"ssh ready: {host}:{port}", flush=True)
        run([*ssh_prefix(host, port), "mkdir -p /workspace/workload"], capture=False)
        archive = args.workload_archive.resolve()
        if sha256_file(archive) != workload["archive_sha256"]:
            raise ValueError("workload archive changed after controller validation")
        scp = [
            "scp",
            "-P",
            port,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            str(archive),
            f"root@{host}:/workspace/workload.tar.gz",
        ]
        run(scp, capture=False)
        remote_hash = run(
            [*ssh_prefix(host, port), "sha256sum /workspace/workload.tar.gz"],
        ).stdout.split()[0]
        if remote_hash != workload["archive_sha256"]:
            raise ValueError("uploaded workload archive failed remote SHA-256 verification")
        run(
            [
                *ssh_prefix(host, port),
                "tar --no-same-owner --no-same-permissions -xzf "
                "/workspace/workload.tar.gz -C /workspace/workload",
            ],
            capture=False,
        )
        workload_started = True
        metadata["workload_started_at"] = now()
        remote = run(
            [*ssh_prefix(host, port), f"cd /workspace/workload && {args.remote_command}"],
            check=False,
            capture=False,
        )
        exit_code = remote.returncode
        metadata["workload_exit_code"] = exit_code
        args.fetch_dir.mkdir(parents=True, exist_ok=True)
        fetch = [
            "scp",
            "-r",
            "-P",
            port,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"root@{host}:/workspace/workload/results/rank_{args.rank}",
            str(args.fetch_dir.resolve()),
        ]
        fetched = run(fetch, check=False, capture=False)
        metadata["fetch_exit_code"] = fetched.returncode
        if fetched.returncode == 0:
            try:
                metadata["artifact_sha256"] = verified_artifact_hashes(
                    args.fetch_dir.resolve(), args.rank, artifact_inventory
                )
                artifacts_verified = True
            except (FileNotFoundError, OSError) as error:
                metadata["artifact_verification_error"] = str(error)
        metadata["artifacts_verified"] = artifacts_verified
        if not artifacts_verified:
            return 1
        return exit_code
    finally:
        metadata["ended_at"] = now()
        if instance_id is not None:
            action = cleanup_action(
                workload_started=workload_started,
                artifacts_verified=artifacts_verified,
            )
            cleaned = run(
                [args.vastai, action, "instance", instance_id, "--retry", "3", "--raw"],
                check=False,
            )
            metadata["cleanup_action"] = action
            metadata["cleanup_succeeded"] = cleaned.returncode == 0
            metadata["cleanup_output"] = (cleaned.stdout or cleaned.stderr or "")[:1000]
            metadata["recovery_required"] = action == "stop"
            print(
                f"{action} {instance_id}: {metadata['cleanup_succeeded']}",
                flush=True,
            )
        write_metadata(args.metadata, metadata)


if __name__ == "__main__":
    raise SystemExit(main())
