#!/usr/bin/env python3
"""Plan and validate task coverage for a multi-HDF5 activation merge."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py


DEFAULT_EXPECTED_FORMAT = "qwen_answer_token_activations_v2"


def hdf5_tasks(path: Path) -> list[str]:
    with h5py.File(path, "r") as h5:
        metadata = h5.get("metadata")
        if not isinstance(metadata, h5py.Group):
            raise ValueError(f"metadata group is missing: {path}")
        return sorted(name for name, item in metadata.items() if isinstance(item, h5py.Group))


def plan_merge(
    paths: list[Path],
    *,
    expected_tasks: list[str] | None = None,
    validate_inputs: bool = False,
    expected_layer_count: int | None = None,
    expected_hidden_dim: int | None = None,
    expected_format: str = DEFAULT_EXPECTED_FORMAT,
    finite_check: str = "none",
) -> dict[str, Any]:
    errors: list[str] = []
    tasks_by_path: dict[str, list[str]] = {}
    owners: dict[str, list[str]] = {}
    input_validations: dict[str, dict[str, Any]] = {}

    for path in paths:
        if not path.exists():
            errors.append(f"input does not exist: {path}")
            continue
        try:
            tasks = hdf5_tasks(path)
        except Exception as exc:
            errors.append(f"could not read tasks from {path}: {exc}")
            continue
        tasks_by_path[str(path)] = tasks
        for task in tasks:
            owners.setdefault(task, []).append(str(path))
        if validate_inputs:
            validation = validate_input_hdf5(
                path,
                expected_layer_count=expected_layer_count,
                expected_hidden_dim=expected_hidden_dim,
                expected_format=expected_format,
                finite_check=finite_check,
            )
            input_validations[str(path)] = summarize_input_validation(validation)
            if not validation.get("ok"):
                errors.append(
                    f"input validation failed for {path}: "
                    f"{validation.get('errors', [])}"
                )

    duplicate_tasks = {
        task: task_owners
        for task, task_owners in sorted(owners.items())
        if len(task_owners) > 1
    }
    if duplicate_tasks:
        errors.append(f"duplicate tasks across input files: {sorted(duplicate_tasks)}")

    all_tasks = sorted(owners)
    if expected_tasks is not None:
        expected_sorted = sorted(expected_tasks)
        missing = sorted(set(expected_sorted) - set(all_tasks))
        extra = sorted(set(all_tasks) - set(expected_sorted))
        if missing:
            errors.append(f"missing expected tasks: {missing}")
        if extra:
            errors.append(f"unexpected tasks: {extra}")

    return {
        "ok": not errors,
        "input_count": len(paths),
        "tasks_by_path": tasks_by_path,
        "task_count": len(all_tasks),
        "tasks": all_tasks,
        "duplicate_tasks": duplicate_tasks,
        "expected_task_count": None if expected_tasks is None else len(expected_tasks),
        "input_validations": input_validations,
        "errors": errors,
    }


def validate_input_hdf5(
    path: Path,
    *,
    expected_layer_count: int | None,
    expected_hidden_dim: int | None,
    expected_format: str,
    finite_check: str,
) -> dict[str, Any]:
    validator = load_activation_validator()
    return validator.validate_activation_hdf5(
        path,
        expected_layer_count=expected_layer_count,
        expected_hidden_dim=expected_hidden_dim,
        finite_check=finite_check,
        sha256=False,
        expected_format=expected_format,
    )


def summarize_input_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": validation.get("ok"),
        "task_count": validation.get("task_count"),
        "tasks": validation.get("tasks"),
        "layer_count": validation.get("layer_count"),
        "hidden_dim": validation.get("hidden_dim"),
        "errors": validation.get("errors", []),
    }


def load_activation_validator() -> Any:
    script_path = Path(__file__).with_name("validate_activation_hdf5.py")
    spec = importlib.util.spec_from_file_location(
        "validate_activation_hdf5_for_merge_plan",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load activation validator from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight task coverage before merging activation HDF5 files."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-tasks")
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Run structural validation on each input HDF5 before merge planning.",
    )
    parser.add_argument("--expected-layer-count", type=int)
    parser.add_argument("--expected-hidden-dim", type=int)
    parser.add_argument(
        "--expected-format",
        default=DEFAULT_EXPECTED_FORMAT,
        help=f"Expected HDF5 root format attr when validating inputs (default: {DEFAULT_EXPECTED_FORMAT})",
    )
    parser.add_argument(
        "--finite-check",
        choices=("none", "sample", "full"),
        default="none",
        help="Finite-value check to use when validating input HDF5 files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = plan_merge(
        args.paths,
        expected_tasks=parse_csv(args.expected_tasks),
        validate_inputs=(
            args.validate_inputs
            or args.expected_layer_count is not None
            or args.expected_hidden_dim is not None
        ),
        expected_layer_count=args.expected_layer_count,
        expected_hidden_dim=args.expected_hidden_dim,
        expected_format=args.expected_format,
        finite_check=args.finite_check,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
