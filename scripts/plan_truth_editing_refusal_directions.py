#!/usr/bin/env python3
"""Emit the fail-closed offline extraction plan for refusal directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from intelligent_liars.truth_editing_refusal_directions import (
    build_refusal_extraction_plan,
    canonical_json_bytes,
    parse_refusal_direction_config,
    parse_refusal_prompt_manifest,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = parse_refusal_direction_config(_read(args.config))
    prompts = (
        parse_refusal_prompt_manifest(_read(args.prompt_manifest), config)
        if args.prompt_manifest is not None
        else None
    )
    plan = build_refusal_extraction_plan(config, prompts)
    payload = canonical_json_bytes(plan.to_dict()).decode() + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if plan.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
