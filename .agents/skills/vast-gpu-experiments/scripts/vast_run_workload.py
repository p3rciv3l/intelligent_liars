#!/usr/bin/env python3
"""Dry-run-first Vast.ai workload lifecycle wrapper."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_VASTAI = "/Users/student/Library/Python/3.10/bin/vastai"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastai-path", default=os.environ.get("VASTAI_PATH", DEFAULT_VASTAI))
    parser.add_argument("--offer-id", required=True, help="Vast offer ID from vast_find_offer.py.")
    parser.add_argument("--command", required=True, help="Workload command to run remotely.")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Local repo or workload directory to sync.")
    parser.add_argument("--remote-dir", default="/workspace/workload")
    parser.add_argument("--fetch-dir", type=Path, default=Path.cwd() / "vast-results")
    parser.add_argument("--artifact", action="append", default=[], help="Remote artifact path to fetch. Repeatable.")
    parser.add_argument("--setup-command", action="append", default=[], help="Setup command to run before workload. Repeatable.")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk", type=int, default=80)
    parser.add_argument("--label", default=None)
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--keep-instance", action="store_true")
    parser.add_argument("--confirmed-cost-approval", action="store_true", help="Required with --execute after the user confirms instance and pricing details.")
    parser.add_argument("--confirm-keep-instance", action="store_true", help="Required with --keep-instance and --execute because billing continues.")
    parser.add_argument("--dry-run", action="store_true", help="Print lifecycle only. This is the default unless --execute is set.")
    parser.add_argument("--execute", action="store_true", help="Actually rent, run, fetch, and clean up.")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=capture, check=check)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def parse_instance_id(stdout: str) -> str:
    try:
        data = json.loads(stdout)
        for key in ("new_contract", "instance_id", "id"):
            if key in data:
                return str(data[key])
    except json.JSONDecodeError:
        pass
    match = re.search(r"\b(?:new_contract|instance_id|id)\D+(\d+)\b", stdout)
    if match:
        return match.group(1)
    raise RuntimeError(f"Could not parse instance ID from create output: {stdout[:500]}")


def parse_ssh_target(text: str) -> tuple[str, str]:
    text = text.strip().strip('"')
    patterns = [
        r"ssh://(?:[^@]+@)?([^:/\s]+):(\d+)",
        r"ssh\s+(?:-[pP]\s+(\d+)\s+)?(?:[^@]+@)?([^@\s]+)",
        r"(?:[^@]+@)?([^:\s]+):(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 2 and groups[0] and groups[1]:
            if groups[0].isdigit():
                return groups[1], groups[0]
            return groups[0], groups[1]
    raise RuntimeError(f"Could not parse ssh-url output: {text}")


def wait_for_ssh(vastai: str, instance_id: str, wait_seconds: int, poll_seconds: int) -> tuple[str, str]:
    deadline = time.time() + wait_seconds
    last_output = ""
    while time.time() < deadline:
        proc = run([vastai, "ssh-url", instance_id], check=False)
        last_output = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and last_output.strip():
            try:
                return parse_ssh_target(last_output)
            except RuntimeError:
                pass
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for SSH URL. Last output: {last_output[:500]}")


def metadata_path(args: argparse.Namespace, label: str) -> Path:
    base = args.metadata_dir or (args.repo / ".codex" / "vast_runs")
    return base / f"{label}.json"


def build_remote_command(args: argparse.Namespace) -> str:
    parts = [f"cd {shlex.quote(args.remote_dir)}"]
    parts.extend(args.setup_command)
    parts.append(args.command)
    return " && ".join(f"({part})" if "&&" in part or ";" in part else part for part in parts)


def print_plan(args: argparse.Namespace, label: str, remote_command: str, metadata_file: Path) -> None:
    create_cmd = [
        args.vastai_path,
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
    print("Vast workload dry run")
    print(f"  create: {shell_join(create_cmd)}")
    print(f"  wait: {args.vastai_path} ssh-url INSTANCE_ID")
    print(f"  sync: {args.vastai_path} copy local:{args.repo.resolve()} C.INSTANCE_ID:{args.remote_dir}")
    print(f"  run: ssh -p PORT root@HOST {shlex.quote(remote_command)}")
    artifacts = args.artifact or [f"{args.remote_dir}/results"]
    for artifact in artifacts:
        print(f"  fetch: {args.vastai_path} copy C.INSTANCE_ID:{artifact} local:{args.fetch_dir.resolve()}")
    if args.keep_instance:
        print("  cleanup: skipped because --keep-instance is set")
    else:
        print(f"  cleanup: {args.vastai_path} destroy instance INSTANCE_ID --raw")
    print(f"  metadata: {metadata_file}")
    print("No rental occurs without --execute --confirmed-cost-approval.")


def main() -> int:
    args = parse_args()
    label = args.label or f"codex-vast-{timestamp()}"
    remote_command = build_remote_command(args)
    metadata_file = metadata_path(args, label)
    artifacts = args.artifact or [f"{args.remote_dir}/results"]

    if not args.execute:
        print_plan(args, label, remote_command, metadata_file)
        return 0
    if not args.confirmed_cost_approval:
        print(
            "Refusing to execute without --confirmed-cost-approval. "
            "Show the user the instance, hourly price, estimated total cost, workload, and cleanup plan first.",
            file=sys.stderr,
        )
        return 2
    if args.keep_instance and not args.confirm_keep_instance:
        print(
            "Refusing to keep a billable GPU instance without --confirm-keep-instance. "
            "Default behavior is to destroy the instance after fetching artifacts.",
            file=sys.stderr,
        )
        return 2

    metadata: dict[str, Any] = {
        "label": label,
        "offer_id": args.offer_id,
        "image": args.image,
        "disk_gb": args.disk,
        "repo": str(args.repo.resolve()),
        "remote_dir": args.remote_dir,
        "setup_commands": args.setup_command,
        "command": args.command,
        "artifacts": artifacts,
        "started_at": iso_now(),
    }
    instance_id = None
    exit_code = 1
    try:
        create_cmd = [
            args.vastai_path,
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
        print(f"Creating instance: {shell_join(create_cmd)}", flush=True)
        created = run(create_cmd)
        instance_id = parse_instance_id(created.stdout)
        metadata["instance_id"] = instance_id

        print(f"Waiting for SSH on instance {instance_id}", flush=True)
        host, port = wait_for_ssh(args.vastai_path, instance_id, args.wait_seconds, args.poll_seconds)
        metadata["ssh_host"] = host
        metadata["ssh_port"] = port

        sync_cmd = [args.vastai_path, "copy", f"local:{args.repo.resolve()}", f"C.{instance_id}:{args.remote_dir}"]
        print(f"Syncing workload: {shell_join(sync_cmd)}", flush=True)
        run(sync_cmd, capture=False)

        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ServerAliveInterval=30",
            "-p",
            port,
            f"root@{host}",
            remote_command,
        ]
        print(f"Running workload: {shell_join(ssh_cmd)}", flush=True)
        proc = subprocess.Popen(ssh_cmd)
        exit_code = proc.wait()
        metadata["exit_code"] = exit_code

        args.fetch_dir.mkdir(parents=True, exist_ok=True)
        fetched: list[str] = []
        for artifact in artifacts:
            fetch_cmd = [args.vastai_path, "copy", f"C.{instance_id}:{artifact}", f"local:{args.fetch_dir.resolve()}"]
            print(f"Fetching artifact: {shell_join(fetch_cmd)}", flush=True)
            fetch_proc = run(fetch_cmd, check=False, capture=False)
            if fetch_proc.returncode == 0:
                fetched.append(str(args.fetch_dir.resolve()))
        metadata["fetched_artifact_paths"] = fetched
        return exit_code
    finally:
        metadata["ended_at"] = iso_now()
        if instance_id and not args.keep_instance:
            destroy_cmd = [args.vastai_path, "destroy", "instance", instance_id, "--raw"]
            print(f"Destroying instance: {shell_join(destroy_cmd)}", flush=True)
            destroy_proc = run(destroy_cmd, check=False)
            metadata["destroyed"] = destroy_proc.returncode == 0
            metadata["destroy_output"] = (destroy_proc.stdout or destroy_proc.stderr or "")[:1000]
        elif instance_id:
            metadata["destroyed"] = False
            metadata["kept_instance"] = True
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"Wrote metadata: {metadata_file}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
