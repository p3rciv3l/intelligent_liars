#!/usr/bin/env python3
"""Build, validate, and package the TinyLoRA research corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.tinylora_corpus import (
    compile_corpus,
    package_compiled_corpus,
    validate_compiled_corpus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="compile registered sources")
    build.add_argument("project_root", type=Path)
    build.add_argument("definition_root", type=Path)
    build.add_argument("output_root", type=Path)

    validate = commands.add_parser("validate", help="run corpus integrity gates")
    validate.add_argument("corpus_root", type=Path)

    package = commands.add_parser("package", help="create deterministic tar.gz")
    package.add_argument("corpus_root", type=Path)
    package.add_argument("archive_path", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build":
        result = compile_corpus(
            args.project_root.resolve(),
            args.definition_root.resolve(),
            args.output_root.resolve(),
        )
    elif args.command == "validate":
        result = validate_compiled_corpus(args.corpus_root.resolve())
        report_path = args.corpus_root.resolve() / "validation_report.json"
        report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    else:
        result = package_compiled_corpus(
            args.corpus_root.resolve(), args.archive_path.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
