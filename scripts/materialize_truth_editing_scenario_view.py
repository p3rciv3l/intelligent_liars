#!/usr/bin/env python3
"""Build the non-sealed canonical-QA-v2 optimization validation view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from intelligent_liars.truth_editing_scenario_view import (
    ScenarioViewError,
    materialize_validation_scenario_view,
)


CONFIG_FORMAT = "truth_editing_scenario_view_build_config_v1"


def _load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScenarioViewError("scenario view config is unreadable") from error
    if not isinstance(value, Mapping):
        raise ScenarioViewError("scenario view config must be an object")
    expected = {
        "format",
        "source_dataset",
        "output",
        "tier_scenario_limits",
        "base_known_qualification",
    }
    if set(value) != expected or value.get("format") != CONFIG_FORMAT:
        raise ScenarioViewError("scenario view config fields or format differ")
    return dict(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = _load_config(args.config)
        root = args.config.parent.parent
        source = root / str(config["source_dataset"])
        output = root / str(config["output"])
        qualification_value = config["base_known_qualification"]
        if qualification_value is not None and not isinstance(qualification_value, str):
            raise ScenarioViewError("base_known_qualification must be a path or null")
        qualification = None if qualification_value is None else root / qualification_value
        view = materialize_validation_scenario_view(
            source,
            output,
            tier_scenario_limits=config["tier_scenario_limits"],
            base_known_qualification=qualification,
            overwrite=args.overwrite,
        )
    except ScenarioViewError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "view_sha256": view.manifest["view_sha256"],
                "scenario_count": view.manifest["scenario_count"],
                "record_count": view.manifest["record_count"],
                "qualification_mode": view.manifest["qualification_mode"],
                "scientific_record_count": len(
                    view.manifest["scientific_validation_record_ids"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
