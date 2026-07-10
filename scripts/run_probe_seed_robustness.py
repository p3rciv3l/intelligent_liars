#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.probe_robustness_runner import (  # noqa: E402
    ProbeRobustnessExperimentSpec,
    run_probe_seed_robustness,
)


def _parse_excluded_job(value: str) -> tuple[str, int]:
    candidate_id, separator, seed_text = value.rpartition(":")
    if not separator or not candidate_id:
        raise argparse.ArgumentTypeError(
            "excluded job must use <candidate_id>:<seed>"
        )
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "excluded job seed must be an integer"
        ) from exc
    return candidate_id, seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned no-REPE probe seed-robustness manifest."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve canonical cache and result paths.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Run only this candidate ID. Repeat to select multiple candidates.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Run only this seed. Repeat to select multiple seeds.",
    )
    parser.add_argument(
        "--exclude-job",
        action="append",
        type=_parse_excluded_job,
        default=[],
        help=(
            "Skip <candidate_id>:<seed> without changing manifest identity. "
            "Repeat to exclude multiple active jobs."
        ),
    )
    parser.add_argument(
        "--max-parallel",
        "--max-workers",
        dest="max_parallel",
        type=int,
        default=None,
        help="Bounded process-pool size. Defaults to the experiment spec value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate reuse and print pending commands without executing them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = ProbeRobustnessExperimentSpec.default(args.project_root)
    effective_max_parallel = (
        spec.default_max_parallel
        if args.max_parallel is None
        else args.max_parallel
    )
    summary = run_probe_seed_robustness(
        spec,
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
        candidate_ids=tuple(args.candidates) if args.candidates else None,
        seeds=tuple(args.seeds) if args.seeds else None,
        excluded_jobs=tuple(args.exclude_job),
    )
    for planned in summary.commands:
        print(planned.shell_command)
    pending = len(summary.commands) if args.dry_run else 0
    print(
        f"manifest={summary.manifest_identity} "
        f"selected={summary.total_jobs} "
        f"reused={summary.reused_jobs} "
        f"completed={summary.completed_jobs} "
        f"pending={pending} "
        f"max_parallel={effective_max_parallel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
