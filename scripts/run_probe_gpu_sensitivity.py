#!/usr/bin/env python3
"""Run a non-canonical PyTorch MPS/CPU probe sensitivity comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from intelligent_liars.probe_gpu_sensitivity import run_probe_gpu_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a sensitivity-only PyTorch probe from pooled features. "
            "This does not replace canonical sklearn/liblinear evidence."
        )
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="Pooled feature cache HDF5."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New result JSON path."
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Cached decoder layer."
    )
    parser.add_argument(
        "--c", type=float, default=1.0, help="Inverse L2 regularization strength."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Split, sample, and fit seed."
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Cached task to include. Repeatable; defaults to all cached tasks.",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Fit device. CPU is used only when explicitly selected.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum LBFGS iterations."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-7,
        help="LBFGS gradient and parameter-change tolerance.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_probe_gpu_sensitivity(
        cache_path=args.cache,
        output_path=args.output,
        layer=args.layer,
        regularization_c=args.c,
        random_seed=args.seed,
        tasks=args.tasks,
        device=args.device,
        max_steps=args.max_steps,
        tolerance=args.tolerance,
    )
    print(
        "Wrote sensitivity-only probe "
        f"device={summary.device} layer={summary.layer} "
        f"tasks={list(summary.tasks)} evaluations={summary.evaluations} "
        f"path={summary.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
