#!/usr/bin/env python3
"""Stage, archive, and credentiallessly publish a frozen Step 5 artifact set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.step5_artifact_finalize import finalize_artifacts, parse_mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-inventory", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--durable-uri", required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="TARGET=SOURCE",
        help="Copy one regular source file to its frozen logical target.",
    )
    parser.add_argument(
        "--tree-archive",
        action="append",
        default=[],
        metavar="TARGET=SOURCE_DIR",
        help="Create one deterministic tar target from a source directory.",
    )
    parser.add_argument(
        "--generate-canary-summary",
        metavar="TARGET",
        help="Generate a controller-required summary bound to staged result and prerequisite receipts.",
    )
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument("--presigned-put-url-file", type=Path)
    url_group.add_argument(
        "--presigned-put-url-env",
        help="Name of an environment variable containing the signed URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = finalize_artifacts(
        artifact_root=args.artifact_root,
        expected_inventory=args.expected_inventory,
        run_id=args.run_id,
        durable_uri=args.durable_uri,
        archive_path=args.archive_path,
        file_mappings=[parse_mapping(value) for value in args.file],
        tree_archive_mappings=[parse_mapping(value) for value in args.tree_archive],
        presigned_url_file=args.presigned_put_url_file,
        presigned_url_env=args.presigned_put_url_env,
        canary_summary_target=args.generate_canary_summary,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
