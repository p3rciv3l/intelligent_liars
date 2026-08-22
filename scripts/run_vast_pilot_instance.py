#!/usr/bin/env python3
"""Run one Vast pilot worker with real SSH readiness and fail-safe teardown."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastai", required=True)
    parser.add_argument("--offer-id", required=True)
    parser.add_argument("--rank", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--repo", type=Path, required=True)
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


def verified_artifact_hashes(fetch_dir: Path, rank: int) -> dict[str, str]:
    """Return hashes only when the complete pilot artifact set is present."""
    result_dir = fetch_dir / f"rank_{rank}"
    required = (
        result_dir / "result.json",
        result_dir / "pilot_state.pt",
        result_dir / f"tinylora_rank{rank}_basis.pt",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete fetched artifact set: {missing}")
    return {
        str(path.relative_to(fetch_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in required
    }


def cleanup_action(*, workload_started: bool, artifacts_verified: bool) -> str:
    """Preserve a worker whenever it may contain the only artifact copy."""
    if workload_started and not artifacts_verified:
        return "stop"
    return "destroy"


def require_empty_artifact_destination(fetch_dir: Path, rank: int) -> None:
    result_dir = fetch_dir / f"rank_{rank}"
    if result_dir.exists():
        raise FileExistsError(
            f"Refusing to mix fetched artifacts with an existing result: {result_dir}"
        )


def main() -> int:
    args = parse_args()
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
    print(f"sync: {args.repo.resolve()} -> /workspace/workload", flush=True)
    print(f"run: {args.remote_command}", flush=True)
    print(f"fetch: /workspace/workload/results/rank_{args.rank}", flush=True)
    print("cleanup: destroy only after fetched artifacts verify; otherwise stop for recovery", flush=True)
    if not args.execute:
        return 0
    if not args.confirmed_cost_approval:
        raise ValueError("Execution requires confirmed cost approval")
    require_empty_artifact_destination(args.fetch_dir.resolve(), args.rank)
    metadata: dict[str, Any] = {
        "format": "tinylora_vast_worker_lifecycle_v1",
        "rank": args.rank,
        "offer_id": args.offer_id,
        "image": args.image,
        "disk_gb": args.disk,
        "started_at": now(),
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
        repo = args.repo.resolve()
        remote_target = (
            "/workspace/workload.tar.gz" if repo.is_file() else "/workspace/workload"
        )
        local_source = str(repo) if repo.is_file() else f"{repo}/."
        scp = [
            "scp",
            "-r",
            "-P",
            port,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            local_source,
            f"root@{host}:{remote_target}",
        ]
        run(scp, capture=False)
        if repo.is_file():
            run(
                [
                    *ssh_prefix(host, port),
                    "tar -xzf /workspace/workload.tar.gz -C /workspace/workload",
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
                    args.fetch_dir.resolve(), args.rank
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
