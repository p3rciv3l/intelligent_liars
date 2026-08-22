#!/usr/bin/env python3
"""Compile or verify a Step 5 independent-probe qualification manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligent_liars.step5_probe_qualification import (
    validate_probe_qualification,
    write_probe_qualification,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Input probe registry JSON")
    parser.add_argument("--output", type=Path, help="New qualification JSON")
    parser.add_argument("--verify", type=Path, help="Existing qualification JSON")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root for relative artifact paths; defaults to input file directory",
    )
    arguments = parser.parse_args()
    compiling = arguments.registry is not None or arguments.output is not None
    if arguments.verify is not None and compiling:
        parser.error("--verify cannot be combined with --registry or --output")
    if arguments.verify is None and not (arguments.registry and arguments.output):
        parser.error("compile mode requires both --registry and --output")
    return arguments


def main() -> int:
    arguments = _arguments()
    if arguments.verify is not None:
        input_path = arguments.verify.resolve()
        artifact_root = (arguments.artifact_root or input_path.parent).resolve()
        result = validate_probe_qualification(
            json.loads(input_path.read_text()), artifact_root=artifact_root
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1

    registry_path = arguments.registry.resolve()
    artifact_root = (arguments.artifact_root or registry_path.parent).resolve()
    manifest = write_probe_qualification(
        json.loads(registry_path.read_text()),
        artifact_root=artifact_root,
        output_path=arguments.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(arguments.output.resolve()),
                "qualification_receipt_sha256": manifest[
                    "qualification_receipt_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
