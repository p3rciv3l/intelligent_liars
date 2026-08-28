#!/usr/bin/env python3
"""Run the bounded activation-control lane for one frozen finalist."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

from intelligent_liars.truth_editing_causal_activation_controls import (
    run_causal_activation_controls,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute restoration, re-ablation, random-direction, and false-trigger "
            "activation controls for one frozen persistent-weight finalist."
        )
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--executor-factory",
        required=True,
        help=(
            "Import reference MODULE:CALLABLE. The callable receives "
            "config_path=Path and returns a causal-control executor."
        ),
    )
    parser.add_argument("--executor-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load_executor(spec: str, config_path: Path) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise ValueError("--executor-factory must be MODULE:CALLABLE")
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("--executor-config must be a regular file")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError("--executor-factory does not resolve to a callable")
    executor = factory(config_path=config_path)
    if not hasattr(executor, "identity") or not callable(
        getattr(executor, "execute_control", None)
    ):
        raise ValueError("executor factory returned an incompatible adapter")
    return executor


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    executor = _load_executor(args.executor_factory, args.executor_config)
    receipt = run_causal_activation_controls(args.plan, executor, args.output)
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
