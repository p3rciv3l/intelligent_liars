#!/usr/bin/env python3
"""Build offline preservation capture plans or resolve their capture bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_preservation_plan import (  # noqa: E402
    PreservationPlanError,
    build_preservation_capture_plan,
    materialize_post_capture_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the immutable capture-plan bundle")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    resolve = subparsers.add_parser(
        "resolve", help="Resolve a verified capture into a materializer source packet"
    )
    resolve.add_argument("--bridge", type=Path, required=True)
    resolve.add_argument("--capture-root", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_preservation_capture_plan(args.config, args.output_dir)
        else:
            result = materialize_post_capture_plan(
                args.bridge, args.capture_root, args.output_dir
            )
    except PreservationPlanError as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
