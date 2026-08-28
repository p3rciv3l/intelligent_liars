#!/usr/bin/env python3
"""Run one exact legacy 80/160/200 production phase on a remote worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intelligent_liars.truth_editing_production import open_production_run  # noqa: E402
from intelligent_liars.truth_editing_wandb_checkpoint import (  # noqa: E402
    WandbCheckpointError,
    create_wandb_run_checkpoint,
    open_wandb_run_checkpoint,
)


PHASE_BOUNDARIES = {"discovery": 80, "expanded": 160, "finalist": 200}
ADAPTIVE_PHASE = "adaptive"
_OUTPUT_FIELDS = {
    "journal_path": "study/study-journal.json",
    "artifact_dir": "study/frozen",
    "runtime_output_dir": "study/runtime",
    "judge_cache_dir": "providers/judge-cache",
    "judge_budget_ledger_dir": "providers/production-judge-budget",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return raw


def _phase_boundary(source_config: Path, phase: str) -> int:
    if phase not in PHASE_BOUNDARIES:
        raise RuntimeError("phase must be discovery, expanded, or finalist")
    production = _read_object(source_config, "production config")
    study_value = production.get("study_config")
    if not isinstance(study_value, str) or Path(study_value).is_absolute():
        raise RuntimeError("production study path must be relative")
    study = _read_object((source_config.parent / study_value).resolve(), "study config")
    tiers = study.get("evaluation_tiers")
    if not isinstance(tiers, list):
        raise RuntimeError("study evaluation tiers are missing")
    actual = {
        row.get("name"): row.get("through_trial")
        for row in tiers
        if isinstance(row, Mapping)
    }
    if actual != PHASE_BOUNDARIES or study.get("max_trials") != 200 or study.get("batch_size") != 8:
        raise RuntimeError("production study must retain the exact 80/160/200 batch barriers")
    return PHASE_BOUNDARIES[phase]


def _write_new_or_identical(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace differing worker output: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _runtime_config(source_config: Path, output_root: Path) -> dict[str, Any]:
    source = _read_object(source_config, "production config")
    if source.get("format") != "truth_editing_production_config_v1":
        raise RuntimeError("production config format differs")
    runtime = dict(source)
    for field, relative in _OUTPUT_FIELDS.items():
        if field not in source:
            raise RuntimeError(f"production config is missing {field}")
        runtime[field] = str((output_root / relative).resolve())
    # The runtime config remains beside source v3, so immutable relative inputs
    # and the hydrated model paths keep exactly the same resolution base.
    for field in ("model_cache_dir", "snapshot_manifest_path"):
        value = source.get(field)
        if not isinstance(value, str) or Path(value).is_absolute():
            raise RuntimeError(f"production {field} must be a portable relative path")
    return runtime


Opener = Callable[[Path], Any]


def _coordinator_wandb_checkpoint(
    output_root: Path, environ: Mapping[str, str]
) -> dict[str, str]:
    path = output_root / "monitoring/wandb-run.json"
    supplied_run_id = environ.get("WANDB_RUN_ID")
    supplied_project = environ.get("WANDB_PROJECT")
    supplied_entity = environ.get("WANDB_ENTITY")
    try:
        if path.exists() or path.is_symlink():
            checkpoint = open_wandb_run_checkpoint(path)
            if supplied_run_id is not None and supplied_run_id != checkpoint.run_id:
                raise RuntimeError("existing W&B run identity differs from WANDB_RUN_ID")
            if supplied_project is not None and supplied_project != checkpoint.project:
                raise RuntimeError("existing W&B run identity differs from WANDB_PROJECT")
            if supplied_entity is not None and supplied_entity != checkpoint.entity:
                raise RuntimeError("existing W&B run identity differs from WANDB_ENTITY")
        else:
            checkpoint = create_wandb_run_checkpoint(
                path,
                run_id=supplied_run_id or uuid.uuid4().hex,
                project=supplied_project or "intelligent-liars",
                entity=supplied_entity,
            )
    except WandbCheckpointError as error:
        raise RuntimeError("coordinator W&B run identity is invalid") from error
    return {
        "wandb_run_id": checkpoint.run_id,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }


def run_phase(
    *,
    source_config: Path,
    expected_source_config_sha256: str,
    output_root: Path,
    phase: str,
    opener: Opener = open_production_run,
    environ: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    source_config = source_config.resolve()
    output_root = output_root.resolve()
    if (
        len(expected_source_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_source_config_sha256
        )
        or file_sha256(source_config) != expected_source_config_sha256
    ):
        raise RuntimeError("production source config SHA-256 differs from the job binding")
    boundary = _phase_boundary(source_config, phase)
    # Check before creating output paths. The value is never included in any receipt.
    if not environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for the production judge")
    runtime = _runtime_config(source_config, output_root)
    runtime_path = source_config.with_name(
        f".truth_editing_production_runtime.{expected_source_config_sha256[:16]}.json"
    )
    _write_new_or_identical(runtime_path, runtime)
    monitoring = _coordinator_wandb_checkpoint(output_root, environ)
    receipt = opener(runtime_path).run(stop_after_trials=boundary)
    payload = dict(receipt.to_mapping())
    if payload.get("completed_trials") != boundary:
        raise RuntimeError("production phase did not reach its exact batch barrier")
    result = {
        **payload,
        "run_receipt_sha256": receipt.identity_sha256,
        "phase": phase,
        "phase_boundary": boundary,
        "source_production_config_sha256": file_sha256(source_config),
        "runtime_production_config_sha256": file_sha256(runtime_path),
        "monitoring": monitoring,
    }
    _write_new_or_identical(output_root / "production-run.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=(*PHASE_BOUNDARIES, ADAPTIVE_PHASE), required=True
    )
    args = parser.parse_args(argv)
    try:
        if args.phase == ADAPTIVE_PHASE:
            raise RuntimeError(
                "main adaptive production requires the CUDA fleet controller; "
                "the legacy single-worker entrypoint cannot run it"
            )
        result = run_phase(
            source_config=args.config,
            expected_source_config_sha256=args.expected_config_sha256,
            output_root=args.output_root,
            phase=args.phase,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"truth-editing production worker failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
