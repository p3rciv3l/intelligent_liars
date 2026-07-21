#!/usr/bin/env python3
"""Find and clean up stale Codex-labeled Vast.ai instances."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

DEFAULT_VASTAI = "/Users/student/Library/Python/3.10/bin/vastai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vastai-path", default=os.environ.get("VASTAI_PATH", DEFAULT_VASTAI))
    parser.add_argument("--label-prefix", default="codex-vast-")
    parser.add_argument("--older-than-hours", type=float, default=6.0)
    parser.add_argument("--raw-instances", help="Read instance JSON from file instead of calling Vast.")
    parser.add_argument("--include-unknown-age", action="store_true", help="Allow cleanup of matching labels with unknown age.")
    parser.add_argument("--dry-run", action="store_true", help="Report only. This is the default unless --execute is set.")
    parser.add_argument("--execute", action="store_true", help="Destroy stale matching instances.")
    return parser.parse_args()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=True)


def load_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.raw_instances:
        with open(args.raw_instances, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        try:
            proc = run([args.vastai_path, "show", "instances", "--raw"])
        except FileNotFoundError:
            raise SystemExit(f"Vast CLI not found: {args.vastai_path}")
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(exc.stderr or exc.stdout)
            raise SystemExit(exc.returncode)
        data = json.loads(proc.stdout)
    if isinstance(data, dict):
        for key in ("instances", "results"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise SystemExit("Expected a JSON list of instances.")
    return [item for item in data if isinstance(item, dict)]


def instance_id(instance: dict[str, Any]) -> str | None:
    for key in ("id", "contract_id", "instance_id"):
        if key in instance and instance[key] is not None:
            return str(instance[key])
    return None


def label_of(instance: dict[str, Any]) -> str:
    label = instance.get("label") or instance.get("name") or ""
    if isinstance(label, list):
        return " ".join(str(item) for item in label)
    return str(label)


def parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 10_000_000_000:
            raw = raw / 1000.0
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def created_at(instance: dict[str, Any]) -> datetime | None:
    for key in ("created_at", "create_time", "start_date", "start_time", "timestamp"):
        parsed = parse_time(instance.get(key))
        if parsed:
            return parsed
    return None


def main() -> int:
    args = parse_args()
    instances = load_instances(args)
    now = datetime.now(timezone.utc)
    stale: list[dict[str, Any]] = []
    for instance in instances:
        label = label_of(instance)
        if not label.startswith(args.label_prefix):
            continue
        created = created_at(instance)
        age_hours = None
        if created:
            age_hours = (now - created).total_seconds() / 3600.0
        is_stale = age_hours is not None and age_hours >= args.older_than_hours
        if age_hours is None and args.include_unknown_age:
            is_stale = True
        if is_stale:
            stale.append(
                {
                    "id": instance_id(instance),
                    "label": label,
                    "age_hours": None if age_hours is None else round(age_hours, 2),
                    "status": instance.get("actual_status") or instance.get("status"),
                    "gpu_name": instance.get("gpu_name"),
                }
            )

    print(json.dumps({"matched_stale": stale, "count": len(stale), "executing": bool(args.execute)}, indent=2))
    if not args.execute:
        print("Dry run only. Re-run with --execute to destroy these instances.")
        return 0

    failures = 0
    for item in stale:
        if not item["id"]:
            failures += 1
            continue
        proc = subprocess.run(
            [args.vastai_path, "destroy", "instance", str(item["id"]), "--raw"],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            failures += 1
            sys.stderr.write(proc.stderr or proc.stdout)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
