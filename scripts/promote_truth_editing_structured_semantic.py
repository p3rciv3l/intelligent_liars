#!/usr/bin/env python3
"""Promote verified structured base-known results into an immutable view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intelligent_liars.truth_editing_structured_semantic import (
    StructuredSemanticError,
    promote_structured_semantic_view,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        view = promote_structured_semantic_view(
            args.source_view,
            args.source_root,
            args.qualification,
            args.output,
        )
    except StructuredSemanticError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "view_sha256": view.manifest["view_sha256"],
                "scientific_validation_scenarios": len(
                    view.manifest["scientific_validation_scenario_ids"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
