#!/usr/bin/env python3
"""Dry-run, replay, or explicitly execute the frozen live judge calibration plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from intelligent_liars.truth_editing_live_judge import (  # noqa: E402
    JudgeTransport,
    OpenRouterJudgeTransport,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Hash-bound live calibration plan JSON")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="bounded concurrent judge operations (1-8; default: 1)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute-live", action="store_true",
        help="Make paid OpenRouter calls; absent this flag the default is dry-run",
    )
    mode.add_argument(
        "--stored-responses", type=Path,
        help="Replay a JSON array of transport responses without network access",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    transport: JudgeTransport
    if args.execute_live:
        transport = OpenRouterJudgeTransport()
        dry_run = False
    elif args.stored_responses is not None:
        try:
            responses = json.loads(args.stored_responses.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"stored responses are unreadable: {error}") from error
        if not isinstance(responses, list) or not all(isinstance(item, dict) for item in responses):
            raise SystemExit("stored responses must be a JSON array of objects")
        transport = StoredJudgeTransport(responses)
        dry_run = False
    else:
        transport = StoredJudgeTransport([])
        dry_run = True
    report = run_live_judge_calibration(
        args.plan,
        cache_dir=args.cache_dir,
        attempt_dir=args.attempt_dir,
        transport=transport,
        dry_run=dry_run,
        max_concurrency=args.max_concurrency,
    )
    rendered = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        with args.output.open("x", encoding="utf-8", newline="") as stream:
            stream.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
