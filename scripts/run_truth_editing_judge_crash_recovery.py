#!/usr/bin/env python3
"""Parent-observed live judge SIGKILL, restart, and zero-call replay probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER = Path(__file__).with_name("run_truth_editing_live_judge_calibration.py")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _frontier(attempt_root: Path) -> tuple[list[str], list[str]]:
    completed = sorted(
        path.parent.parent.name for path in attempt_root.glob("*/*/completed.json")
    )
    pending = sorted(
        path.parent.parent.name
        for path in attempt_root.glob("*/*/pending.json")
        if not (path.parent / "completed.json").exists()
        and not (path.parent / "failed.json").exists()
    )
    return completed, pending


def _attempt_inventory(attempt_root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for request_dir in sorted(path for path in attempt_root.iterdir() if path.is_dir()):
        for attempt_dir in sorted(path for path in request_dir.iterdir() if path.is_dir()):
            states = {
                path.stem: _file_sha(path)
                for path in sorted(attempt_dir.glob("*.json"))
            }
            inventory.append(
                {
                    "request_sha256": request_dir.name,
                    "attempt": attempt_dir.name,
                    "states": states,
                }
            )
    return inventory


def _pending_authorized_exposure(attempt_root: Path, request_ids: list[str]) -> float:
    total = 0.0
    for request_id in request_ids:
        paths = list((attempt_root / request_id).glob("*/pending.json"))
        if len(paths) != 1:
            raise RuntimeError("pending crash request has an ambiguous attempt inventory")
        total += float(json.loads(paths[0].read_text())["authorized_usd"])
    return total


def _command(plan: Path, root: Path, output: Path, concurrency: int) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        str(plan),
        "--cache-dir",
        str(root / "cache"),
        "--attempt-dir",
        str(root / "attempts"),
        "--output",
        str(output),
        "--max-concurrency",
        str(concurrency),
        "--execute-live",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--kill-after-completed", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if not 1 <= args.max_concurrency <= 8 or args.kill_after_completed < 1:
        raise RuntimeError("crash probe bounds are invalid")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = args.plan.resolve(strict=True)
    attempts = root / "attempts"
    first_output = root / "interrupted-report.json"
    stderr_stream = root.joinpath("interrupted.stderr.log").open("x")
    child = subprocess.Popen(
        _command(plan, root, first_output, args.max_concurrency),
        stdout=subprocess.DEVNULL,
        stderr=stderr_stream,
    )
    try:
        deadline = time.monotonic() + args.timeout_seconds
        while True:
            completed, pending = _frontier(attempts)
            if len(completed) >= args.kill_after_completed and pending:
                child.send_signal(signal.SIGKILL)
                break
            if child.poll() is not None:
                raise RuntimeError("live child exited before the requested crash frontier")
            if time.monotonic() >= deadline:
                raise TimeoutError("live child did not reach the requested crash frontier")
            time.sleep(0.02)
        return_code = child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        stderr_stream.close()
    completed, pending = _frontier(attempts)
    if return_code != -signal.SIGKILL or not pending:
        raise RuntimeError("parent did not observe the required SIGKILL frontier")
    kill_event = {
        "format": "truth_editing_judge_parent_observed_kill_v1",
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "child_return_code": return_code,
        "signal": "SIGKILL",
        "max_concurrency": args.max_concurrency,
        "completed_request_sha256s": completed,
        "pending_request_sha256s": pending,
        "plan_file_sha256": _file_sha(plan),
    }
    kill_event["content_sha256"] = _sha(kill_event)
    _write_new(root / "kill-event.json", kill_event)

    restart_output = root / "restart-report.json"
    subprocess.run(
        _command(plan, root, restart_output, args.max_concurrency), check=True
    )
    completed_after_restart, pending_after_restart = _frontier(attempts)
    inventory_after_restart = _attempt_inventory(attempts)
    replay_output = root / "replay-report.json"
    subprocess.run(_command(plan, root, replay_output, args.max_concurrency), check=True)
    completed_after_replay, pending_after_replay = _frontier(attempts)
    inventory_after_replay = _attempt_inventory(attempts)
    if (
        completed_after_replay != completed_after_restart
        or pending_after_replay != pending_after_restart
        or inventory_after_replay != inventory_after_restart
        or restart_output.read_bytes() != replay_output.read_bytes()
    ):
        raise RuntimeError("complete replay changed calls, pending frontier, or report")
    report = json.loads(restart_output.read_text())
    receipt = {
        "format": "truth_editing_judge_parent_observed_crash_recovery_receipt_v1",
        "kill_event": kill_event,
        "completed_calls_after_restart": len(completed_after_restart),
        "pending_ambiguous_after_restart": len(pending_after_restart),
        "ambiguous_additional_authorized_exposure_usd": (
            _pending_authorized_exposure(attempts, pending_after_restart)
        ),
        "replay_added_completed_calls": 0,
        "attempt_inventory_after_restart_sha256": _sha(inventory_after_restart),
        "attempt_inventory_after_replay_sha256": _sha(inventory_after_replay),
        "replay_report_byte_identical": True,
        "restart_report_content_sha256": report["content_sha256"],
        "restart_report_file_sha256": _file_sha(restart_output),
        "terminal_operation_count": len(report["completed_operation_ids"])
        + len(report["failed_operation_ids"]),
        "confirmed_completed_spend_usd": report["actual_spend_usd"],
    }
    receipt["content_sha256"] = _sha(receipt)
    _write_new(args.receipt_output.resolve(), receipt)


if __name__ == "__main__":
    main()
