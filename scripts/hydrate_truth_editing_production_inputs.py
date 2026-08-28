#!/usr/bin/env python3
"""Build or hydrate an exact truth-editing production-input bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.truth_editing_input_hydration import (
    build_production_input_bundle,
    build_production_input_bundle_from_entries,
    entries_from_vast_job_config,
    hydrate_production_inputs,
    write_durable_hydration_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="Build canonical bundle and manifest.")
    sources = build.add_mutually_exclusive_group(required=True)
    sources.add_argument("--allowlist", type=Path)
    sources.add_argument("--vast-job-config", type=Path)
    build.add_argument(
        "--ignored-only",
        action="store_true",
        help="With --vast-job-config, include only paths selected by git check-ignore.",
    )
    build.add_argument("--repo-root", type=Path, default=Path.cwd())
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument(
        "--archive-uri",
        help="URI recorded in the manifest; defaults to an output-manifest-relative path.",
    )

    hydrate = subcommands.add_parser("hydrate", help="Hydrate or verify exact inputs.")
    hydrate.add_argument("--manifest", type=Path, required=True)
    hydrate.add_argument("--repo-root", type=Path, default=Path.cwd())
    hydrate.add_argument("--receipt", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        if args.ignored_only and args.vast_job_config is None:
            raise ValueError("--ignored-only requires --vast-job-config")
        if args.vast_job_config is not None:
            entries = entries_from_vast_job_config(
                args.vast_job_config,
                repo_root=args.repo_root,
                ignored_only=args.ignored_only,
            )
            result = build_production_input_bundle_from_entries(
                entries,
                source_root=args.repo_root,
                archive_path=args.archive,
                manifest_path=args.manifest,
                archive_uri=args.archive_uri,
            )
        else:
            result = build_production_input_bundle(
                args.allowlist,
                source_root=args.repo_root,
                archive_path=args.archive,
                manifest_path=args.manifest,
                archive_uri=args.archive_uri,
            )
    else:
        result = hydrate_production_inputs(args.manifest, repo_root=args.repo_root)
        if args.receipt is not None:
            result = write_durable_hydration_receipt(args.receipt, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
