#!/usr/bin/env python3
"""Dynamic multi-worker Qwen insider-trading generation runner.

This is the repo-local, resume-safe version of the ad hoc Vast script used for
the 2026-06-24 s20 insider upscale run. It preserves existing queue/output
state unless --overwrite-queue is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from intelligent_liars import dynamic_queue
from intelligent_liars.models import DEFAULT_MODEL_ID, qwen_model_load_description
from intelligent_liars.rollouts import (
    DEFAULT_INSIDER_GENERATION_SETTINGS,
    generation_content_settings,
)
from intelligent_liars.run_control import (
    acquire_lock,
    command_line,
    file_sha256,
    install_signal_cleanup,
    lock_payload,
    new_run_id,
)

gpu_snapshot = dynamic_queue.gpu_snapshot
now = dynamic_queue.now
parse_csv = dynamic_queue.parse_csv
read_json = dynamic_queue.read_json
write_json = dynamic_queue.write_json
write_queue_plan = dynamic_queue.write_queue_plan


DEFAULT_PROMPT_GLOB = "data/insider_trading/prompts/**/*.yaml"
DEFAULT_LABEL_MODE = "unknown"
PLAN_SCHEMA_VERSION = 1
QUEUE_STATE_DIRS = ("pending", "running", "done", "failed")
QUEUE_DIRS = (*QUEUE_STATE_DIRS, "outputs", "logs", "heartbeats")
QUEUE_CLEAR_SPECS = tuple(
    (name, "*.json") for name in ("pending", "running", "done", "failed", "outputs")
)
QUEUE_STATE_SPECS = (
    ("pending", "*.json"),
    ("running", "*.json"),
    ("done", "*.json"),
    ("failed", "*.json"),
    ("outputs", "*.json"),
)


@dataclass(frozen=True)
class Unit:
    unit_id: str
    relative_prompt: str
    sample_idx: int
    dynamic_run_id: str = ""
    queue_plan_id: str = ""

    @property
    def run_id(self) -> str:
        return f"{self.relative_prompt}::{self.sample_idx}"


def safe_unit_id(relative_prompt: str, sample_idx: int) -> str:
    safe = (
        relative_prompt.replace("/", "__")
        .replace("\\", "__")
        .replace(".yaml", "")
        .replace(".", "_")
    )
    return f"{safe}__sample-{sample_idx:02d}"


def unit_from_prompt(relative_prompt: str, sample_idx: int) -> Unit:
    return Unit(
        unit_id=safe_unit_id(relative_prompt, sample_idx),
        relative_prompt=relative_prompt,
        sample_idx=sample_idx,
    )


def ensure_dirs(run_dir: Path) -> None:
    dynamic_queue.ensure_dirs(run_dir, QUEUE_DIRS)


def serialise_unit(unit: Unit) -> dict[str, Any]:
    return asdict(unit)


def deserialise_unit(payload: dict[str, Any]) -> Unit:
    if "unit_id" in payload:
        return Unit(
            unit_id=str(payload["unit_id"]),
            relative_prompt=str(payload["relative_prompt"]),
            sample_idx=int(payload["sample_idx"]),
            dynamic_run_id=str(payload.get("dynamic_run_id", "")),
            queue_plan_id=str(payload.get("queue_plan_id", "")),
        )
    relative_prompt, sample_idx = parse_run_id(str(payload["run_id"]))
    return unit_from_prompt(relative_prompt, sample_idx)


def parse_run_id(run_id: str) -> tuple[str, int]:
    try:
        relative_prompt, sample_idx = run_id.rsplit("::", 1)
    except ValueError as exc:
        raise ValueError(f"invalid insider run_id: {run_id!r}") from exc
    return relative_prompt, int(sample_idx)


def prompt_paths(project_root: Path, prompt_glob: str) -> list[Path]:
    paths = sorted(project_root.glob(prompt_glob))
    if not paths:
        raise FileNotFoundError(
            f"No insider-trading prompt YAML files matched {prompt_glob!r} under {project_root}"
        )
    for prompt_path in paths:
        config = yaml.safe_load(prompt_path.read_text())
        if not isinstance(config, dict) or "messages" not in config:
            raise ValueError(
                f"Insider-trading prompt file must contain a mapping with messages: {prompt_path}"
            )
    return paths


def plan_units(
    *,
    project_root: Path,
    prompt_glob: str,
    samples_per_prompt: int,
) -> list[Unit]:
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be >= 1")
    units: list[Unit] = []
    for prompt_path in prompt_paths(project_root, prompt_glob):
        relative_prompt = str(prompt_path.relative_to(project_root))
        for sample_idx in range(samples_per_prompt):
            units.append(unit_from_prompt(relative_prompt, sample_idx))
    return units


def stamp_unit_plan(unit: Unit, *, run_id: str, queue_plan_id: str) -> Unit:
    return Unit(
        unit_id=unit.unit_id,
        relative_prompt=unit.relative_prompt,
        sample_idx=unit.sample_idx,
        dynamic_run_id=run_id,
        queue_plan_id=queue_plan_id,
    )


def _stamp_unit_plan_positional(unit: Unit, run_id: str, queue_plan_id: str) -> Unit:
    return stamp_unit_plan(unit, run_id=run_id, queue_plan_id=queue_plan_id)


def _unit_id(unit: Unit) -> str:
    return unit.unit_id


def _fallback_unit_id(path: Path) -> str:
    return running_stem_to_unit_id(path.stem)


def _output_unit_id(path: Path, expected_queue_plan_id: str | None) -> str | None:
    unit = _output_shard_unit(path, expected_queue_plan_id=expected_queue_plan_id)
    return unit.unit_id if unit is not None else None


def _pending_path_for_new_unit(run_dir: Path, _queue_idx: int, unit: Unit) -> Path:
    return run_dir / "pending" / f"{unit.unit_id}.json"


def _running_path_for_claim(slot_id: str):
    def build(run_dir: Path, _pending_path: Path, unit: Unit) -> Path:
        return run_dir / "running" / f"{unit.unit_id}__{slot_id}.json"

    return build


def _output_path_for_unit(run_dir: Path, unit: Unit) -> Path:
    return run_dir / "outputs" / f"{unit.unit_id}.json"


def _output_is_valid_for_unit(path: Path, unit: Unit) -> bool:
    return _is_valid_output_shard(
        path, expected_queue_plan_id=unit.queue_plan_id or None
    )


def _pending_path_for_running_unit(
    run_dir: Path, _running_path: Path, unit: Unit
) -> Path:
    return run_dir / "pending" / f"{unit.unit_id}.json"


def _done_path_for_running_unit(run_dir: Path, _running_path: Path, unit: Unit) -> Path:
    return run_dir / "done" / f"{unit.unit_id}.json"


def _pending_path_for_done_unit(run_dir: Path, unit: Unit) -> Path:
    return run_dir / "pending" / f"{unit.unit_id}.json"


def unit_plan_payload(unit: Unit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "relative_prompt": unit.relative_prompt,
        "sample_idx": unit.sample_idx,
        "run_id": unit.run_id,
    }


def build_plan_manifest(
    *,
    project_root: Path,
    prompt_glob: str,
    samples_per_prompt: int,
    label_mode: str,
    units: list[Unit],
) -> dict[str, Any]:
    prompts = prompt_paths(project_root, prompt_glob)
    relative_prompts = [str(path.relative_to(project_root)) for path in prompts]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "insider_generation_dynamic",
        "model": {
            "model_id": DEFAULT_MODEL_ID,
            "load": qwen_model_load_description(),
        },
        "prompts": {
            "prompt_glob": prompt_glob,
            "prompt_paths": relative_prompts,
            "prompt_sha256": {
                relative_path: file_sha256(project_root / relative_path)
                for relative_path in relative_prompts
            },
        },
        "generation": {
            "samples_per_prompt": samples_per_prompt,
            "label_mode": label_mode,
            "settings": generation_content_settings(
                DEFAULT_INSIDER_GENERATION_SETTINGS
            ),
        },
        "units": [unit_plan_payload(unit) for unit in units],
    }


def prepare_queue_plan(
    *,
    project_root: Path,
    run_dir: Path,
    prompt_glob: str,
    samples_per_prompt: int,
    label_mode: str,
    units: list[Unit],
    overwrite_queue: bool,
) -> dict[str, Any]:
    manifest = build_plan_manifest(
        project_root=project_root,
        prompt_glob=prompt_glob,
        samples_per_prompt=samples_per_prompt,
        label_mode=label_mode,
        units=units,
    )
    return dynamic_queue.prepare_queue_plan(
        project_root=project_root,
        run_dir=run_dir,
        manifest=manifest,
        overwrite_queue=overwrite_queue,
    )


def planned_unit_ids_from_plan(plan: dict[str, Any]) -> set[str]:
    return {str(item["unit_id"]) for item in plan.get("manifest", {}).get("units", [])}


def planned_units_by_id(
    *,
    project_root: Path,
    prompt_glob: str,
    samples_per_prompt: int,
) -> dict[str, Unit]:
    return {
        unit.unit_id: unit
        for unit in plan_units(
            project_root=project_root,
            prompt_glob=prompt_glob,
            samples_per_prompt=samples_per_prompt,
        )
    }


def running_stem_to_unit_id(stem: str) -> str:
    return stem.split("__slot-", 1)[0]


def queue_unit_ids(run_dir: Path) -> set[str]:
    return dynamic_queue.queue_unit_ids(
        run_dir,
        state_names=QUEUE_STATE_DIRS,
        output_glob="*.json",
        deserialise_unit=deserialise_unit,
        unit_id=_unit_id,
        output_unit_id=_output_unit_id,
        fallback_unit_id=_fallback_unit_id,
    )


def run_dir_has_queue_state(run_dir: Path) -> bool:
    return dynamic_queue.run_dir_has_queue_state(run_dir, QUEUE_STATE_SPECS)


def clear_queue(run_dir: Path) -> None:
    dynamic_queue.clear_queue(run_dir, QUEUE_CLEAR_SPECS)


def seed_completed_output(
    run_dir: Path, output_path: Path, *, run_id: str = "", queue_plan_id: str = ""
) -> int:
    records = read_json(output_path)
    if not isinstance(records, list):
        raise ValueError(f"Completed output must contain a list: {output_path}")
    seeded = 0
    for record in records:
        metadata = record.get("metadata") or {}
        relative_prompt, sample_idx = parse_run_id(str(metadata.get("run_id")))
        unit = unit_from_prompt(relative_prompt, sample_idx)
        unit = stamp_unit_plan(unit, run_id=run_id, queue_plan_id=queue_plan_id)
        record_metadata = record.setdefault("metadata", {})
        if run_id:
            record_metadata["dynamic_run_id"] = run_id
        if queue_plan_id:
            record_metadata["queue_plan_id"] = queue_plan_id
        shard_path = run_dir / "outputs" / f"{unit.unit_id}.json"
        done_path = run_dir / "done" / f"{unit.unit_id}.json"
        if not shard_path.exists():
            write_json(shard_path, [record])
            seeded += 1
        if not done_path.exists():
            write_json(done_path, serialise_unit(unit))
    return seeded


def _is_valid_output_shard(
    path: Path, *, expected_queue_plan_id: str | None = None
) -> bool:
    return (
        _output_shard_unit(path, expected_queue_plan_id=expected_queue_plan_id)
        is not None
    )


def _output_shard_unit(
    path: Path, *, expected_queue_plan_id: str | None = None
) -> Unit | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = read_json(path)
    except Exception:
        return None
    if not isinstance(payload, list) or len(payload) != 1:
        return None
    record = payload[0]
    if not isinstance(record, dict):
        return None
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    run_id = metadata.get("run_id")
    if not isinstance(run_id, str):
        return None
    if expected_queue_plan_id is not None:
        if metadata.get("queue_plan_id") != expected_queue_plan_id:
            return None
    try:
        relative_prompt, sample_idx = parse_run_id(run_id)
    except Exception:
        return None
    unit = unit_from_prompt(relative_prompt, sample_idx)
    if path.stem != unit.unit_id:
        return None
    return Unit(
        unit_id=unit.unit_id,
        relative_prompt=unit.relative_prompt,
        sample_idx=unit.sample_idx,
        dynamic_run_id=str(metadata.get("dynamic_run_id", "")),
        queue_plan_id=str(metadata.get("queue_plan_id", "")),
    )


def _recover_unit_from_running_path(
    running_path: Path,
    planned_units: dict[str, Unit] | None = None,
) -> Unit | None:
    unit_id = running_stem_to_unit_id(running_path.stem)
    if planned_units is not None:
        return planned_units.get(unit_id)
    return None


def repair_stale_done_units(run_dir: Path) -> tuple[int, int]:
    """Move done units without a valid output shard back to pending.

    Returns:
        A tuple of (requeued_units, removed_duplicate_units).
    """
    return dynamic_queue.repair_stale_done_units(
        run_dir,
        deserialise_unit=deserialise_unit,
        unit_id=_unit_id,
        serialise_unit=serialise_unit,
        output_path=_output_path_for_unit,
        output_is_valid=_output_is_valid_for_unit,
        pending_path=_pending_path_for_done_unit,
    )


def create_queue(
    *,
    project_root: Path,
    run_dir: Path,
    prompt_glob: str,
    samples_per_prompt: int,
    overwrite_queue: bool,
    seed_completed_output_path: Path | None,
    run_id: str = "",
    queue_plan_id: str = "",
    units: list[Unit] | None = None,
) -> tuple[int, int, int]:
    ensure_dirs(run_dir)
    if overwrite_queue:
        clear_queue(run_dir)
    else:
        repair_stale_done_units(run_dir)
    seeded = 0
    if seed_completed_output_path is not None:
        seeded = seed_completed_output(
            run_dir,
            seed_completed_output_path,
            run_id=run_id,
            queue_plan_id=queue_plan_id,
        )
    units = units or plan_units(
        project_root=project_root,
        prompt_glob=prompt_glob,
        samples_per_prompt=samples_per_prompt,
    )
    queued = dynamic_queue.enqueue_missing_units(
        run_dir,
        units,
        run_id=run_id,
        queue_plan_id=queue_plan_id,
        state_names=QUEUE_STATE_DIRS,
        output_glob="*.json",
        unit_id=_unit_id,
        serialise_unit=serialise_unit,
        deserialise_unit=deserialise_unit,
        stamp_unit_plan=_stamp_unit_plan_positional,
        pending_path=_pending_path_for_new_unit,
        output_unit_id=_output_unit_id,
        fallback_unit_id=_fallback_unit_id,
    )
    return len(units), queued, seeded


def claim_unit(run_dir: Path, slot_id: str) -> tuple[Path, Unit] | None:
    return dynamic_queue.claim_unit(
        run_dir,
        deserialise_unit=deserialise_unit,
        running_path=_running_path_for_claim(slot_id),
    )


def requeue_running_units(
    run_dir: Path,
    *,
    planned_units: dict[str, Unit] | None = None,
) -> tuple[int, int]:
    ensure_dirs(run_dir)
    return dynamic_queue.requeue_running_units(
        run_dir,
        deserialise_unit=deserialise_unit,
        serialise_unit=serialise_unit,
        output_path=_output_path_for_unit,
        output_is_valid=_output_is_valid_for_unit,
        pending_path=_pending_path_for_running_unit,
        done_path=_done_path_for_running_unit,
        recover_unit=lambda path: _recover_unit_from_running_path(path, planned_units),
        move_to_pending=False,
        count_existing_pending=True,
    )


def audit_run_state(
    *,
    project_root: Path,
    run_dir: Path,
    prompt_glob: str,
    samples_per_prompt: int,
    label_mode: str = DEFAULT_LABEL_MODE,
    queue_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only audit of queue/output consistency before remote resume."""
    units = plan_units(
        project_root=project_root,
        prompt_glob=prompt_glob,
        samples_per_prompt=samples_per_prompt,
    )
    planned = {unit.unit_id: unit for unit in units}
    queue_plan = queue_plan or prepare_queue_plan(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=prompt_glob,
        samples_per_prompt=samples_per_prompt,
        label_mode=label_mode,
        units=units,
        overwrite_queue=False,
    )
    queue_plan_id = str(queue_plan["queue_plan_id"])
    report = dynamic_queue.audit_queue_state(
        run_dir,
        planned_unit_ids=set(planned),
        queue_plan_id=queue_plan_id,
        state_names=QUEUE_STATE_DIRS,
        output_glob="*.json",
        deserialise_unit=deserialise_unit,
        unit_id=_unit_id,
        unit_queue_plan_id=lambda unit: unit.queue_plan_id,
        output_unit_id=_output_unit_id,
        count_state_fn=count_state,
        recover_unit=lambda path: _recover_unit_from_running_path(path, planned),
    )
    report["queue_plan_id"] = queue_plan_id
    return report


def count_state(run_dir: Path) -> dict[str, int]:
    ensure_dirs(run_dir)
    return dynamic_queue.count_state(run_dir, QUEUE_STATE_SPECS)


def run_worker(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONPATH", "src")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    slot_id = str(args.slot_id or f"gpu-{args.gpu}")
    slot_index = int(args.slot_index)

    from intelligent_liars.insider_trading import (
        _generate_one_insider_transcript,
        _insider_run_metadata,
    )
    from intelligent_liars.models import load_model_and_processor, model_config_from_env
    from intelligent_liars.rollouts import (
        DEFAULT_INSIDER_GENERATION_SETTINGS,
        seed_everything,
    )

    seed_everything(DEFAULT_INSIDER_GENERATION_SETTINGS.seed + slot_index)
    bundle = load_model_and_processor(model_config_from_env())
    prompts = prompt_paths(project_root, args.prompt_glob)
    run_metadata = _insider_run_metadata(
        model_id=bundle.model_id,
        prompt_glob=args.prompt_glob,
        prompt_paths=prompts,
        project_root=project_root,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
        settings=DEFAULT_INSIDER_GENERATION_SETTINGS,
    )
    heartbeat = run_dir / "heartbeats" / f"worker-{slot_id}.json"
    processed = 0
    print(f"WORKER_READY time={now()} slot={slot_id} gpu={args.gpu}", flush=True)
    while not (run_dir / "STOP").exists():
        write_json(
            heartbeat,
            {
                "time": now(),
                "slot": slot_id,
                "gpu": str(args.gpu),
                "state": "waiting",
                "processed": processed,
            },
        )
        claimed = claim_unit(run_dir, slot_id)
        if claimed is None:
            break
        running_path, unit = claimed
        output_path = run_dir / "outputs" / f"{unit.unit_id}.json"
        started = time.time()
        try:
            prompt_path = project_root / unit.relative_prompt
            config = yaml.safe_load(prompt_path.read_text())
            record = _generate_one_insider_transcript(
                bundle=bundle,
                prompt_path=prompt_path,
                config=config,
                settings=DEFAULT_INSIDER_GENERATION_SETTINGS,
                run_id=unit.run_id,
                label_mode=args.label_mode,
                run_metadata=run_metadata,
            )
            metadata = record.setdefault("metadata", {})
            metadata["dynamic_unit_id"] = unit.unit_id
            metadata["dynamic_slot_id"] = slot_id
            metadata["dynamic_gpu"] = str(args.gpu)
            metadata["dynamic_run_id"] = unit.dynamic_run_id
            metadata["queue_plan_id"] = unit.queue_plan_id
            write_json(output_path, [record])
            os.replace(running_path, run_dir / "done" / f"{unit.unit_id}.json")
            processed += 1
            print(
                f"UNIT_DONE time={now()} slot={slot_id} unit={unit.unit_id} "
                f"elapsed={time.time() - started:.1f}s output={output_path}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failed unit for audit
            failure = serialise_unit(unit)
            failure["error"] = f"{type(exc).__name__}: {exc}"
            failure["failed_at"] = now()
            write_json(run_dir / "failed" / f"{unit.unit_id}.json", failure)
            running_path.unlink(missing_ok=True)
            print(
                f"UNIT_FAILED time={now()} slot={slot_id} unit={unit.unit_id} error={failure['error']}",
                flush=True,
            )
    write_json(
        heartbeat,
        {
            "time": now(),
            "slot": slot_id,
            "gpu": str(args.gpu),
            "state": "stopped",
            "processed": processed,
        },
    )
    print(f"WORKER_DONE time={now()} slot={slot_id} processed={processed}", flush=True)
    return 0


def merge_outputs(
    args: argparse.Namespace, *, expected_queue_plan_id: str | None = None
) -> None:
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output)
    project_root = Path(args.project_root).resolve()
    output_path = (
        output_path if output_path.is_absolute() else project_root / output_path
    )
    merge_lock = acquire_lock(
        output_path.with_name(f"{output_path.name}.merge.lock"),
        lock_payload(
            run_id=new_run_id(),
            queue_plan_id=expected_queue_plan_id,
            command=command_line(),
            kind="insider-json-merge",
            extra={"output_path": str(output_path)},
        ),
        force_stale_lock=getattr(args, "force_stale_merge_lock", False),
    )
    signal_cleanup = install_signal_cleanup(merge_lock)
    records: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    try:
        shard_paths = sorted((run_dir / "outputs").glob("*.json"))
        for path in shard_paths:
            unit = _output_shard_unit(
                path, expected_queue_plan_id=expected_queue_plan_id
            )
            if unit is None:
                raise SystemExit(f"Invalid output shard: {path}")
            if unit.unit_id in seen_units:
                raise SystemExit(
                    f"Duplicate output shard for unit {unit.unit_id}: {path}"
                )
            seen_units.add(unit.unit_id)
            payload = read_json(path)
            records.extend(payload)
        if args.require_count is not None and len(records) != args.require_count:
            raise SystemExit(
                f"Expected {args.require_count} merged records, found {len(records)}."
            )
        records.sort(
            key=lambda record: str((record.get("metadata") or {}).get("run_id", ""))
        )
        write_json(output_path, records)
        print(
            f"MERGE_DONE time={now()} shards={len(shard_paths)} records={len(records)} output={output_path}",
            flush=True,
        )
    finally:
        signal_cleanup.restore()
        merge_lock.release()


def run_supervisor(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    planned_units_list = plan_units(
        project_root=project_root,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
    )
    planned_units = {unit.unit_id: unit for unit in planned_units_list}
    if (
        not args.overwrite_queue
        and not (run_dir / "queue_plan.json").exists()
        and run_dir_has_queue_state(run_dir)
    ):
        raise SystemExit(
            f"Existing queue/output state in {run_dir} has no queue_plan.json. "
            "Run migrate-plan-metadata, use --overwrite-queue, or use a fresh run directory."
        )
    queue_plan = prepare_queue_plan(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
        units=planned_units_list,
        overwrite_queue=args.overwrite_queue,
    )
    run_id = str(queue_plan["run_id"])
    queue_plan_id = str(queue_plan["queue_plan_id"])
    lock = acquire_lock(
        run_dir / "run.lock",
        lock_payload(
            run_id=run_id,
            queue_plan_id=queue_plan_id,
            command=command_line(),
            kind="insider-supervisor",
        ),
        force_stale_lock=getattr(args, "force_stale_lock", False),
    )
    signal_cleanup = install_signal_cleanup(lock)
    try:
        write_queue_plan(run_dir, queue_plan)
        if not args.plan_only:
            (run_dir / "STOP").unlink(missing_ok=True)
        requeued, promoted = requeue_running_units(run_dir, planned_units=planned_units)
        repaired_done, removed_done = repair_stale_done_units(run_dir)
        planned, queued, seeded = create_queue(
            project_root=project_root,
            run_dir=run_dir,
            prompt_glob=args.prompt_glob,
            samples_per_prompt=args.samples_per_prompt,
            overwrite_queue=args.overwrite_queue,
            seed_completed_output_path=args.seed_completed_output,
            run_id=run_id,
            queue_plan_id=queue_plan_id,
            units=planned_units_list,
        )
        status = audit_run_state(
            project_root=project_root,
            run_dir=run_dir,
            prompt_glob=args.prompt_glob,
            samples_per_prompt=args.samples_per_prompt,
            label_mode=args.label_mode,
            queue_plan=queue_plan,
        )
        write_json(
            run_dir / "run_metadata.json",
            {
                "time": now(),
                "project_root": str(project_root),
                "run_dir": str(run_dir),
                "prompt_glob": args.prompt_glob,
                "samples_per_prompt": args.samples_per_prompt,
                "label_mode": args.label_mode,
                "planned_units": planned,
                "run_id": run_id,
                "queue_plan_id": queue_plan_id,
                "queued_new_units": queued,
                "seeded_output_shards": seeded,
                "requeued_running_units": requeued,
                "promoted_running_units": promoted,
                "repaired_done_units": repaired_done,
                "removed_done_units": removed_done,
                "gpus": parse_csv(args.gpus),
                "status": status,
            },
        )
        if not args.plan_only and not args.allow_empty_queue:
            if (
                not list((run_dir / "pending").glob("*.json"))
                and not list((run_dir / "running").glob("*.json"))
                and status["valid_outputs"] < status["planned"]
            ):
                print(
                    "SUPERVISOR_GUARD pending=0 running=0 but output coverage is incomplete. "
                    "Inspect status."
                )
                return 0
        if args.plan_only:
            print(
                json.dumps(
                    {
                        "planned": planned,
                        "queued": queued,
                        "seeded": seeded,
                        "queue_plan_id": queue_plan_id,
                        "state": count_state(run_dir),
                    },
                    sort_keys=True,
                )
            )
            return 0

        workers: list[subprocess.Popen[str]] = []
        env_base = os.environ.copy()
        env_base.setdefault("PYTHONPATH", "src")
        env_base.setdefault("HF_HUB_DISABLE_XET", "1")
        for idx, gpu in enumerate(parse_csv(args.gpus)):
            slot_id = f"slot-{idx:02d}-gpu-{gpu}"
            worker_log = run_dir / "logs" / f"worker-{slot_id}.log"
            worker_log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--project-root",
                str(project_root),
                "--run-dir",
                str(run_dir),
                "--prompt-glob",
                args.prompt_glob,
                "--samples-per-prompt",
                str(args.samples_per_prompt),
                "--label-mode",
                args.label_mode,
                "--gpu",
                str(gpu),
                "--slot-index",
                str(idx),
                "--slot-id",
                slot_id,
            ]
            handle = worker_log.open("a")
            workers.append(
                subprocess.Popen(
                    command,
                    cwd=project_root,
                    env=env_base,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        print(
            f"SUPERVISOR_START time={now()} run_dir={run_dir} workers={len(workers)} "
            f"planned={planned} queued={queued} seeded={seeded} queue_plan_id={queue_plan_id}",
            flush=True,
        )
        try:
            while True:
                status = audit_run_state(
                    project_root=project_root,
                    run_dir=run_dir,
                    prompt_glob=args.prompt_glob,
                    samples_per_prompt=args.samples_per_prompt,
                    label_mode=args.label_mode,
                    queue_plan=queue_plan,
                )
                state = status["state"]
                alive = sum(worker.poll() is None for worker in workers)
                print(
                    f"STATUS time={now()} alive={alive} "
                    f"planned={status['planned']} valid_outputs={status['valid_outputs']} "
                    f"state={json.dumps(state, sort_keys=True)} "
                    f"issues={json.dumps(status['issues'], sort_keys=True)}",
                    flush=True,
                )
                print(gpu_snapshot(), flush=True)
                if state["failed"]:
                    raise SystemExit(
                        "Insider generation worker failed. Inspect run_dir/failed and worker logs."
                    )
                if state["pending"] == 0 and state["running"] == 0:
                    if status["valid_outputs"] != status["planned"] or not status["ok"]:
                        raise SystemExit(
                            "Queue markers are empty but validated output coverage is incomplete or mismatched. "
                            f"status={json.dumps(status, sort_keys=True)}"
                        )
                    (run_dir / "STOP").write_text(now() + "\n")
                    time.sleep(args.idle_sleep)
                    for worker in workers:
                        if worker.poll() is None:
                            worker.terminate()
                    merge_outputs(args, expected_queue_plan_id=queue_plan_id)
                    print(f"SUPERVISOR_DONE time={now()}", flush=True)
                    return 0
                if alive == 0:
                    raise SystemExit(
                        "All insider generation workers exited before queue completion."
                    )
                time.sleep(args.poll_seconds)
        finally:
            (run_dir / "STOP").write_text(now() + "\n")
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
    finally:
        signal_cleanup.restore()
        lock.release()


def run_plan(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    units = plan_units(
        project_root=project_root,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
    )
    queue_plan = prepare_queue_plan(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
        units=units,
        overwrite_queue=args.overwrite_queue,
    )
    write_queue_plan(run_dir, queue_plan)
    planned, queued, seeded = create_queue(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        overwrite_queue=args.overwrite_queue,
        seed_completed_output_path=args.seed_completed_output,
        run_id=str(queue_plan["run_id"]),
        queue_plan_id=str(queue_plan["queue_plan_id"]),
        units=units,
    )
    print(
        json.dumps(
            {
                "planned": planned,
                "queued": queued,
                "seeded": seeded,
                "queue_plan_id": queue_plan["queue_plan_id"],
                "state": count_state(run_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def run_requeue_stale(args: argparse.Namespace) -> int:
    planned_units = None
    if args.project_root is not None:
        planned_units = planned_units_by_id(
            project_root=Path(args.project_root).resolve(),
            prompt_glob=args.prompt_glob,
            samples_per_prompt=args.samples_per_prompt,
        )
    requeued, promoted = requeue_running_units(
        Path(args.run_dir).resolve(), planned_units=planned_units
    )
    print(
        json.dumps(
            {
                "requeued": requeued,
                "promoted": promoted,
                "state": count_state(Path(args.run_dir).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


def run_audit(args: argparse.Namespace) -> int:
    report = audit_run_state(
        project_root=Path(args.project_root).resolve(),
        run_dir=Path(args.run_dir).resolve(),
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def run_status(args: argparse.Namespace) -> int:
    report = audit_run_state(
        project_root=Path(args.project_root).resolve(),
        run_dir=Path(args.run_dir).resolve(),
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def run_migrate_plan_metadata(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    units = plan_units(
        project_root=project_root,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
    )
    planned = {unit.unit_id: unit for unit in units}
    queue_plan = prepare_queue_plan(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
        units=units,
        overwrite_queue=args.overwrite_queue_plan,
    )
    run_id = str(queue_plan["run_id"])
    queue_plan_id = str(queue_plan["queue_plan_id"])

    output_updates: list[tuple[Path, list[Any], str]] = []
    migrated_markers = 0
    invalid_outputs: list[str] = []
    unexpected_outputs: list[str] = []
    output_paths = sorted((run_dir / "outputs").glob("*.json"))
    for path in output_paths:
        unit = _output_shard_unit(path, expected_queue_plan_id=None)
        if unit is None:
            invalid_outputs.append(str(path))
            continue
        if unit.unit_id not in planned:
            unexpected_outputs.append(str(path))
            continue
        payload = read_json(path)
        metadata = payload[0].setdefault("metadata", {})
        if metadata.get("queue_plan_id") not in (None, "", queue_plan_id):
            invalid_outputs.append(str(path))
            continue
        if (
            metadata.get("queue_plan_id") != queue_plan_id
            or metadata.get("dynamic_run_id") != run_id
        ):
            metadata["queue_plan_id"] = queue_plan_id
            metadata["dynamic_run_id"] = run_id
            metadata["dynamic_unit_id"] = unit.unit_id
            output_updates.append((path, payload, unit.unit_id))

    if invalid_outputs or unexpected_outputs:
        raise SystemExit(
            "Refusing migration because output audit is not exact: "
            f"invalid_outputs={invalid_outputs} unexpected_outputs={unexpected_outputs}"
        )
    if (
        args.expected_valid_outputs is not None
        and len(output_paths) != args.expected_valid_outputs
    ):
        raise SystemExit(
            f"Expected {args.expected_valid_outputs} output shards before migration, "
            f"found {len(output_paths)}."
        )

    for path, payload, _unit_id in output_updates:
        write_json(path, payload)
    migrated_outputs = len(output_updates)

    for state_name in ("pending", "running", "done", "failed"):
        for path in sorted((run_dir / state_name).glob("*.json")):
            try:
                unit = deserialise_unit(read_json(path))
            except Exception:
                continue
            if unit.unit_id not in planned:
                continue
            stamped = stamp_unit_plan(
                planned[unit.unit_id], run_id=run_id, queue_plan_id=queue_plan_id
            )
            write_json(path, serialise_unit(stamped))
            migrated_markers += 1

    write_queue_plan(run_dir, queue_plan)
    report = audit_run_state(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
        queue_plan=queue_plan,
    )
    print(
        json.dumps(
            {
                "queue_plan_id": queue_plan_id,
                "migrated_outputs": migrated_outputs,
                "migrated_markers": migrated_markers,
                "status": report,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["ok"] else 1


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument(
        "--label-mode", choices=("unknown", "heuristic"), default=DEFAULT_LABEL_MODE
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic multi-worker insider-trading generation runner."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan")
    add_common(plan)
    plan.add_argument("--overwrite-queue", action="store_true")
    plan.add_argument("--seed-completed-output", type=Path)

    supervisor = sub.add_parser("supervisor")
    add_common(supervisor)
    supervisor.add_argument("--output", required=True)
    supervisor.add_argument("--gpus", default="0")
    supervisor.add_argument("--overwrite-queue", action="store_true")
    supervisor.add_argument("--seed-completed-output", type=Path)
    supervisor.add_argument("--require-count", type=int)
    supervisor.add_argument(
        "--force-stale-lock",
        action="store_true",
        help="Break run.lock only when it is on this host and the recorded PID is dead.",
    )
    supervisor.add_argument(
        "--force-stale-merge-lock",
        action="store_true",
        help="Break output merge lock only when it is on this host and the recorded PID is dead.",
    )
    supervisor.add_argument("--poll-seconds", type=float, default=30.0)
    supervisor.add_argument("--idle-sleep", type=float, default=5.0)
    supervisor.add_argument("--plan-only", action="store_true")
    supervisor.add_argument(
        "--allow-empty-queue",
        action="store_true",
        help="Allow the supervisor to proceed when pending/running queues are empty.",
    )

    worker = sub.add_parser("worker")
    add_common(worker)
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--slot-id")
    worker.add_argument("--slot-index", type=int, required=True)

    merge = sub.add_parser("merge")
    add_common(merge)
    merge.add_argument("--output", required=True)
    merge.add_argument("--require-count", type=int)
    merge.add_argument("--expected-queue-plan-id")
    merge.add_argument(
        "--force-stale-merge-lock",
        action="store_true",
        help="Break output merge lock only when it is on this host and the recorded PID is dead.",
    )

    requeue = sub.add_parser("requeue-stale")
    requeue.add_argument("--run-dir", required=True)
    requeue.add_argument("--project-root")
    requeue.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    requeue.add_argument("--samples-per-prompt", type=int, default=1)

    audit = sub.add_parser("audit")
    add_common(audit)

    status = sub.add_parser("status")
    add_common(status)

    migrate = sub.add_parser("migrate-plan-metadata")
    add_common(migrate)
    migrate.add_argument("--expected-valid-outputs", type=int)
    migrate.add_argument(
        "--overwrite-queue-plan",
        action="store_true",
        help="Replace an existing queue_plan.json if the computed plan differs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "plan":
        return run_plan(args)
    if args.cmd == "worker":
        return run_worker(args)
    if args.cmd == "supervisor":
        return run_supervisor(args)
    if args.cmd == "merge":
        merge_outputs(args, expected_queue_plan_id=args.expected_queue_plan_id)
        return 0
    if args.cmd == "requeue-stale":
        return run_requeue_stale(args)
    if args.cmd == "audit":
        return run_audit(args)
    if args.cmd == "status":
        return run_status(args)
    if args.cmd == "migrate-plan-metadata":
        return run_migrate_plan_metadata(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
