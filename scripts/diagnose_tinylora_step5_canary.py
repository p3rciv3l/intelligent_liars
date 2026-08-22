#!/usr/bin/env python3
"""Emit conservative same-machine diagnostics for a Step 5 canary failure."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("/workspace/runs/step5-canary")
    )
    return parser.parse_args()


def _probe(command: list[str]) -> dict[str, object]:
    if shutil.which(command[0]) is None:
        return {"available": False}
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired as error:
        def tail(value: str | bytes | None) -> str:
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            return (value or "")[-2000:]

        return {
            "available": True,
            "timed_out": True,
            "timeout_seconds": 20,
            "stdout_tail": tail(error.stdout),
            "stderr_tail": tail(error.stderr),
        }
    return {
        "available": True,
        "timed_out": False,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    latest = run_root / "checkpoint_store" / "latest.json"
    payload = {
        "format": "tinylora_step5_canary_diagnostic_v1",
        # If this command can run over SSH, the lifecycle controller has not
        # independently corroborated host loss. Treat it as software first.
        "classification": "software_failure",
        "diagnosed_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "run_root_exists": run_root.is_dir(),
        "latest_checkpoint_pointer_exists": latest.is_file(),
        "disk": _probe(["df", "-h", "/workspace"]),
        "gpu": _probe(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"]),
        "python": _probe(["python", "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"]),
        "offline_model_cache": os.environ.get("HF_HUB_OFFLINE") == "1",
        "replacement_requested": False,
        "recovery": "resume the same labeled instance and exact checkpoint identity",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
