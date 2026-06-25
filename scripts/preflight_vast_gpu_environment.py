#!/usr/bin/env python3
"""Preflight a Vast GPU environment before launching expensive Qwen work."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


BYTES_PER_GB = 1024**3


def disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return usage.free / BYTES_PER_GB


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def nvidia_smi_query() -> tuple[list[dict[str, Any]], str | None]:
    if not command_exists("nvidia-smi"):
        return [], "nvidia-smi not found"
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        return [], completed.stderr.strip() or "nvidia-smi failed"

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            return [], f"could not parse nvidia-smi line: {line!r}"
        index, name, memory_total, memory_free, utilization = parts
        try:
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mib": int(memory_total),
                    "memory_free_mib": int(memory_free),
                    "utilization_gpu_percent": int(utilization),
                }
            )
        except ValueError as exc:
            return [], f"could not parse nvidia-smi numeric values from {line!r}: {exc}"
    return gpus, None


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    dev_shm = args.dev_shm.resolve()
    workspace = args.workspace.resolve()
    hf_home = Path(os.environ.get("HF_HOME", "")).expanduser()
    tmpdir = Path(os.environ.get("TMPDIR", "")).expanduser()
    errors: list[str] = []
    warnings: list[str] = []

    dev_shm_free = disk_free_gb(dev_shm)
    workspace_free = disk_free_gb(workspace)
    if dev_shm_free is None:
        errors.append(f"dev_shm path is not available: {dev_shm}")
    elif dev_shm_free < args.min_dev_shm_free_gb:
        errors.append(
            f"{dev_shm} has {dev_shm_free:.1f} GiB free; "
            f"need at least {args.min_dev_shm_free_gb:.1f} GiB"
        )

    if workspace_free is None:
        warnings.append(f"workspace path is not available: {workspace}")
    elif workspace_free < args.min_workspace_free_gb:
        warnings.append(
            f"{workspace} has {workspace_free:.1f} GiB free; "
            f"recommended at least {args.min_workspace_free_gb:.1f} GiB"
        )

    if not project_root.exists():
        errors.append(f"project root does not exist: {project_root}")
    elif not (project_root / "pyproject.toml").exists():
        errors.append(f"project root is missing pyproject.toml: {project_root}")

    if args.require_env_file and not (project_root / ".env").exists():
        errors.append(f"required .env file is missing under project root: {project_root / '.env'}")

    if not hf_home:
        errors.append("HF_HOME is not set")
    elif not path_is_under(hf_home, dev_shm):
        errors.append(f"HF_HOME must be under {dev_shm}; got {hf_home}")

    if not tmpdir:
        errors.append("TMPDIR is not set")
    elif not path_is_under(tmpdir, dev_shm):
        errors.append(f"TMPDIR must be under {dev_shm}; got {tmpdir}")

    if os.environ.get("HF_HUB_DISABLE_XET") != "1":
        warnings.append("HF_HUB_DISABLE_XET is not set to 1")

    gpus, gpu_error = nvidia_smi_query()
    if gpu_error is not None:
        errors.append(gpu_error)
    if len(gpus) < args.expected_gpus:
        errors.append(f"found {len(gpus)} GPUs; expected at least {args.expected_gpus}")
    low_vram = [
        gpu
        for gpu in gpus
        if gpu["memory_total_mib"] < args.min_gpu_memory_mib
    ]
    if low_vram:
        errors.append(
            "GPU memory below requirement: "
            + ", ".join(
                f"gpu{gpu['index']}={gpu['memory_total_mib']}MiB"
                for gpu in low_vram
            )
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "paths": {
            "project_root": str(project_root),
            "dev_shm": str(dev_shm),
            "workspace": str(workspace),
            "hf_home": str(hf_home) if hf_home else "",
            "tmpdir": str(tmpdir) if tmpdir else "",
        },
        "disk_free_gb": {
            "dev_shm": None if dev_shm_free is None else round(dev_shm_free, 3),
            "workspace": None if workspace_free is None else round(workspace_free, 3),
        },
        "env": {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "TMPDIR": os.environ.get("TMPDIR", ""),
            "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
        "gpus": gpus,
        "requirements": {
            "expected_gpus": args.expected_gpus,
            "min_gpu_memory_mib": args.min_gpu_memory_mib,
            "min_dev_shm_free_gb": args.min_dev_shm_free_gb,
            "min_workspace_free_gb": args.min_workspace_free_gb,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check remote Vast GPU paths, environment, disk, and GPUs before launch."
    )
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/intelligent_liars"))
    parser.add_argument("--dev-shm", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--min-gpu-memory-mib", type=int, default=80_000)
    parser.add_argument("--min-dev-shm-free-gb", type=float, default=120.0)
    parser.add_argument("--min-workspace-free-gb", type=float, default=5.0)
    parser.add_argument("--require-env-file", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = preflight(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
