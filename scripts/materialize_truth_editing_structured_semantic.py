#!/usr/bin/env python3
"""Materialize the non-sealed structured semantic optimizer lane."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from intelligent_liars.truth_editing_structured_semantic import (
    StructuredSemanticError,
    materialize_structured_semantic_view,
)


CONFIG_FORMAT = "truth_editing_structured_semantic_build_config_v1"


def _config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StructuredSemanticError("build config is unreadable") from error
    expected = {
        "format",
        "source_root",
        "output",
        "source_manifest_path",
        "train_path",
        "validation_path",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise StructuredSemanticError("build config fields differ")
    if value.get("format") != CONFIG_FORMAT:
        raise StructuredSemanticError("unsupported build config format")
    return dict(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = _config(args.config)
        root = args.config.parent.parent
        view = materialize_structured_semantic_view(
            root / str(config["source_root"]),
            root / str(config["output"]),
            source_manifest_path=str(config["source_manifest_path"]),
            train_path=str(config["train_path"]),
            validation_path=str(config["validation_path"]),
            overwrite=args.overwrite,
        )
    except StructuredSemanticError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "view_sha256": view.manifest["view_sha256"],
                "train_scenarios": view.manifest["split_counts"]["train"],
                "validation_scenarios": view.manifest["split_counts"]["validation"],
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
