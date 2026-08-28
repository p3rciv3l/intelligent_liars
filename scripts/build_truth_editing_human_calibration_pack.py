#!/usr/bin/env python3
"""Build a blinded human calibration pack or compile completed labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from intelligent_liars.truth_editing_human_calibration_pack import (
    build_human_calibration_pack,
    compile_human_labels,
    initialize_markdown_labels,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--qualification", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    compile_labels = commands.add_parser("compile-labels")
    compile_labels.add_argument("--pack", type=Path, required=True)
    compile_labels.add_argument("--labels", type=Path, required=True)
    compile_labels.add_argument("--output", type=Path, required=True)
    init_markdown = commands.add_parser("init-markdown")
    init_markdown.add_argument("--pack", type=Path, required=True)
    init_markdown.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        pack = build_human_calibration_pack(args.dataset, args.qualification, args.output)
        result = {
            "pack_sha256": pack.manifest["pack_sha256"],
            "bundle_count": len(pack.bundles),
            "response_count": sum(len(bundle["responses"]) for bundle in pack.bundles),
            "pairwise_relationship_count": len(pack.pairwise_relationships),
            "labeling_document": str(pack.root / "LABELING.md"),
            "labels_template": str(pack.root / "labels.template.jsonl"),
        }
    elif args.command == "compile-labels":
        compiled = compile_human_labels(args.pack, args.labels, args.output)
        result = {
            "pack_sha256": compiled["pack_sha256"],
            "content_sha256": compiled["content_sha256"],
            "output": str(args.output),
        }
    else:
        path = initialize_markdown_labels(args.pack, args.output)
        result = {"editable_markdown": str(path)}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
