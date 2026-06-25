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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMPT_GLOB = "data/insider_trading/prompts/**/*.yaml"
DEFAULT_LABEL_MODE = "unknown"


@dataclass(frozen=True)
class Unit:
    unit_id: str
    relative_prompt: str
    sample_idx: int

    @property
    def run_id(self) -> str:
        return f"{self.relative_prompt}::{self.sample_idx}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def safe_unit_id(relative_prompt: str, sample_idx: int) -> str:
    safe = relative_prompt.replace("/", "__").replace("\\", "__").replace(".yaml", "").replace(".", "_")
    return f"{safe}__sample-{sample_idx:02d}"


def unit_from_prompt(relative_prompt: str, sample_idx: int) -> Unit:
    return Unit(
        unit_id=safe_unit_id(relative_prompt, sample_idx),
        relative_prompt=relative_prompt,
        sample_idx=sample_idx,
    )


def ensure_dirs(run_dir: Path) -> None:
    for name in ("pending", "running", "done", "failed", "outputs", "logs", "heartbeats"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def serialise_unit(unit: Unit) -> dict[str, Any]:
    return asdict(unit)


def deserialise_unit(payload: dict[str, Any]) -> Unit:
    if "unit_id" in payload:
        return Unit(
            unit_id=str(payload["unit_id"]),
            relative_prompt=str(payload["relative_prompt"]),
            sample_idx=int(payload["sample_idx"]),
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
        raise FileNotFoundError(f"No insider-trading prompt YAML files matched {prompt_glob!r} under {project_root}")
    for prompt_path in paths:
        config = yaml.safe_load(prompt_path.read_text())
        if not isinstance(config, dict) or "messages" not in config:
            raise ValueError(f"Insider-trading prompt file must contain a mapping with messages: {prompt_path}")
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
    existing: set[str] = set()
    for name in ("pending", "done", "failed"):
        for path in (run_dir / name).glob("*.json"):
            try:
                existing.add(deserialise_unit(read_json(path)).unit_id)
            except Exception:
                existing.add(running_stem_to_unit_id(path.stem))
    for path in (run_dir / "running").glob("*.json"):
        try:
            existing.add(deserialise_unit(read_json(path)).unit_id)
        except Exception:
            existing.add(running_stem_to_unit_id(path.stem))
    for path in (run_dir / "outputs").glob("*.json"):
        if (unit := _output_shard_unit(path)) is not None:
            existing.add(unit.unit_id)
    return existing


def clear_queue(run_dir: Path) -> None:
    for name in ("pending", "running", "done", "failed", "outputs"):
        for path in (run_dir / name).glob("*.json"):
            path.unlink()


def seed_completed_output(run_dir: Path, output_path: Path) -> int:
    records = read_json(output_path)
    if not isinstance(records, list):
        raise ValueError(f"Completed output must contain a list: {output_path}")
    seeded = 0
    for record in records:
        metadata = record.get("metadata") or {}
        relative_prompt, sample_idx = parse_run_id(str(metadata.get("run_id")))
        unit = unit_from_prompt(relative_prompt, sample_idx)
        shard_path = run_dir / "outputs" / f"{unit.unit_id}.json"
        done_path = run_dir / "done" / f"{unit.unit_id}.json"
        if not shard_path.exists():
            write_json(shard_path, [record])
            seeded += 1
        if not done_path.exists():
            write_json(done_path, serialise_unit(unit))
    return seeded


def _is_valid_output_shard(path: Path) -> bool:
    return _output_shard_unit(path) is not None


def _output_shard_unit(path: Path) -> Unit | None:
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
    try:
        relative_prompt, sample_idx = parse_run_id(run_id)
    except Exception:
        return None
    unit = unit_from_prompt(relative_prompt, sample_idx)
    if path.stem != unit.unit_id:
        return None
    return unit


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
    done_dir = run_dir / "done"
    pending_dir = run_dir / "pending"
    requeued = 0
    removed = 0
    seen_unit_ids: set[str] = set()
    for done_path in sorted(done_dir.glob("*.json")):
        try:
            unit = deserialise_unit(read_json(done_path))
        except Exception:
            done_path.unlink(missing_ok=True)
            removed += 1
            continue
        if unit.unit_id in seen_unit_ids:
            done_path.unlink(missing_ok=True)
            removed += 1
            continue
        seen_unit_ids.add(unit.unit_id)

        output_path = run_dir / "outputs" / f"{unit.unit_id}.json"
        if not _is_valid_output_shard(output_path):
            done_path.unlink(missing_ok=True)
            pending_path = pending_dir / f"{unit.unit_id}.json"
            if not pending_path.exists():
                write_json(pending_path, serialise_unit(unit))
                requeued += 1
    return requeued, removed


def create_queue(
    *,
    project_root: Path,
    run_dir: Path,
    prompt_glob: str,
    samples_per_prompt: int,
    overwrite_queue: bool,
    seed_completed_output_path: Path | None,
) -> tuple[int, int, int]:
    ensure_dirs(run_dir)
    if overwrite_queue:
        clear_queue(run_dir)
    else:
        repair_stale_done_units(run_dir)
    seeded = 0
    if seed_completed_output_path is not None:
        seeded = seed_completed_output(run_dir, seed_completed_output_path)
    units = plan_units(
        project_root=project_root,
        prompt_glob=prompt_glob,
        samples_per_prompt=samples_per_prompt,
    )
    existing = queue_unit_ids(run_dir)
    queued = 0
    for unit in units:
        if unit.unit_id in existing:
            continue
        write_json(run_dir / "pending" / f"{unit.unit_id}.json", serialise_unit(unit))
        existing.add(unit.unit_id)
        queued += 1
    return len(units), queued, seeded


def claim_unit(run_dir: Path, slot_id: str) -> tuple[Path, Unit] | None:
    for pending in sorted((run_dir / "pending").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(pending))
        except Exception:
            continue
        running = run_dir / "running" / f"{unit.unit_id}__{slot_id}.json"
        try:
            os.replace(pending, running)
        except OSError:
            continue
        return running, unit
    return None


def requeue_running_units(
    run_dir: Path,
    *,
    planned_units: dict[str, Unit] | None = None,
) -> tuple[int, int]:
    ensure_dirs(run_dir)
    requeued = 0
    promoted = 0
    for path in sorted((run_dir / "running").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(path))
        except Exception:
            unit = _recover_unit_from_running_path(path, planned_units)
            if unit is None:
                path.unlink(missing_ok=True)
                continue
        output_path = run_dir / "outputs" / f"{unit.unit_id}.json"
        if _is_valid_output_shard(output_path):
            done_path = run_dir / "done" / f"{unit.unit_id}.json"
            if not done_path.exists():
                os.replace(path, done_path)
            else:
                path.unlink()
            promoted += 1
            continue
        pending_path = run_dir / "pending" / f"{unit.unit_id}.json"
        if not pending_path.exists():
            write_json(pending_path, serialise_unit(unit))
        path.unlink()
        requeued += 1
    return requeued, promoted


def audit_run_state(
    *,
    project_root: Path,
    run_dir: Path,
    prompt_glob: str,
    samples_per_prompt: int,
) -> dict[str, Any]:
    """Read-only audit of queue/output consistency before remote resume."""
    planned = planned_units_by_id(
        project_root=project_root,
        prompt_glob=prompt_glob,
        samples_per_prompt=samples_per_prompt,
    )
    membership: dict[str, list[str]] = {unit_id: [] for unit_id in planned}
    malformed_queue_files: list[str] = []
    unexpected_units: list[str] = []
    duplicate_units: dict[str, list[str]] = {}

    for state_name in ("pending", "running", "done", "failed"):
        for path in sorted((run_dir / state_name).glob("*.json")):
            try:
                unit = deserialise_unit(read_json(path))
            except Exception:
                unit = _recover_unit_from_running_path(path, planned)
                if unit is None:
                    malformed_queue_files.append(str(path))
                    continue
            marker = f"{state_name}:{path.name}"
            if unit.unit_id not in membership:
                unexpected_units.append(marker)
                continue
            membership[unit.unit_id].append(marker)

    valid_output_units: set[str] = set()
    invalid_output_files: list[str] = []
    for path in sorted((run_dir / "outputs").glob("*.json")):
        unit = _output_shard_unit(path)
        if unit is None:
            invalid_output_files.append(str(path))
            continue
        marker = f"outputs:{path.name}"
        if unit.unit_id not in membership:
            unexpected_units.append(marker)
            continue
        membership[unit.unit_id].append(marker)
        valid_output_units.add(unit.unit_id)

    for unit_id, markers in sorted(membership.items()):
        state_markers = [marker for marker in markers if not marker.startswith("outputs:")]
        if len(state_markers) > 1:
            duplicate_units[unit_id] = state_markers

    missing_units = [unit_id for unit_id, markers in sorted(membership.items()) if not markers]
    done_without_output = [
        unit_id
        for unit_id, markers in sorted(membership.items())
        if any(marker.startswith("done:") for marker in markers)
        and unit_id not in valid_output_units
    ]
    output_without_done = [
        unit_id
        for unit_id, markers in sorted(membership.items())
        if unit_id in valid_output_units
        and not any(marker.startswith("done:") for marker in markers)
    ]
    issues = {
        "malformed_queue_files": malformed_queue_files,
        "invalid_output_files": invalid_output_files,
        "unexpected_units": unexpected_units,
        "missing_units": missing_units,
        "duplicate_units": duplicate_units,
        "done_without_output": done_without_output,
        "output_without_done": output_without_done,
    }
    ok = not any(bool(value) for value in issues.values())
    return {
        "ok": ok,
        "planned": len(planned),
        "state": count_state(run_dir),
        "valid_outputs": len(valid_output_units),
        "issues": issues,
    }


def count_state(run_dir: Path) -> dict[str, int]:
    ensure_dirs(run_dir)
    return {
        name: len(list((run_dir / name).glob("*.json")))
        for name in ("pending", "running", "done", "failed", "outputs")
    }


def run_worker(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONPATH", "src")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    slot_id = str(args.slot_id or f"gpu-{args.gpu}")
    slot_index = int(args.slot_index)

    from intelligent_liars.insider_trading import _generate_one_insider_transcript, _insider_run_metadata
    from intelligent_liars.models import load_model_and_processor, model_config_from_env
    from intelligent_liars.rollouts import DEFAULT_INSIDER_GENERATION_SETTINGS, seed_everything

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
        write_json(heartbeat, {"time": now(), "slot": slot_id, "gpu": str(args.gpu), "state": "waiting", "processed": processed})
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
            print(f"UNIT_FAILED time={now()} slot={slot_id} unit={unit.unit_id} error={failure['error']}", flush=True)
    write_json(heartbeat, {"time": now(), "slot": slot_id, "gpu": str(args.gpu), "state": "stopped", "processed": processed})
    print(f"WORKER_DONE time={now()} slot={slot_id} processed={processed}", flush=True)
    return 0


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc!r}"


def merge_outputs(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output)
    project_root = Path(args.project_root).resolve()
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    records: list[dict[str, Any]] = []
    seen_units: set[str] = set()
    shard_paths = sorted((run_dir / "outputs").glob("*.json"))
    for path in shard_paths:
        unit = _output_shard_unit(path)
        if unit is None:
            raise SystemExit(f"Invalid output shard: {path}")
        if unit.unit_id in seen_units:
            raise SystemExit(f"Duplicate output shard for unit {unit.unit_id}: {path}")
        seen_units.add(unit.unit_id)
        payload = read_json(path)
        records.extend(payload)
    if args.require_count is not None and len(records) != args.require_count:
        raise SystemExit(f"Expected {args.require_count} merged records, found {len(records)}.")
    records.sort(key=lambda record: str((record.get("metadata") or {}).get("run_id", "")))
    write_json(output_path, records)
    print(f"MERGE_DONE time={now()} shards={len(shard_paths)} records={len(records)} output={output_path}", flush=True)


def run_supervisor(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    planned_units = planned_units_by_id(
        project_root=project_root,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
    )
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
            "queued_new_units": queued,
            "seeded_output_shards": seeded,
            "requeued_running_units": requeued,
            "promoted_running_units": promoted,
            "repaired_done_units": repaired_done,
            "removed_done_units": removed_done,
            "gpus": parse_csv(args.gpus),
        },
    )
    if not args.plan_only and not args.allow_empty_queue:
        if not list((run_dir / "pending").glob("*.json")) and not list((run_dir / "running").glob("*.json")):
            print(
                "SUPERVISOR_GUARD pending=0 running=0; queue empty. "
                "Use --allow-empty-queue to proceed."
            )
            return 0
    if args.plan_only:
        print(json.dumps({"planned": planned, "queued": queued, "seeded": seeded, "state": count_state(run_dir)}, sort_keys=True))
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
        workers.append(subprocess.Popen(command, cwd=project_root, env=env_base, stdout=handle, stderr=subprocess.STDOUT, text=True))

    print(
        f"SUPERVISOR_START time={now()} run_dir={run_dir} workers={len(workers)} "
        f"planned={planned} queued={queued} seeded={seeded}",
        flush=True,
    )
    try:
        while True:
            state = count_state(run_dir)
            alive = sum(worker.poll() is None for worker in workers)
            print(f"STATUS time={now()} alive={alive} state={json.dumps(state, sort_keys=True)}", flush=True)
            print(gpu_snapshot(), flush=True)
            if state["failed"]:
                raise SystemExit("Insider generation worker failed. Inspect run_dir/failed and worker logs.")
            if state["pending"] == 0 and state["running"] == 0:
                (run_dir / "STOP").write_text(now() + "\n")
                time.sleep(args.idle_sleep)
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                merge_outputs(args)
                print(f"SUPERVISOR_DONE time={now()}", flush=True)
                return 0
            if alive == 0:
                raise SystemExit("All insider generation workers exited before queue completion.")
            time.sleep(args.poll_seconds)
    finally:
        (run_dir / "STOP").write_text(now() + "\n")
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()


def run_plan(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    planned, queued, seeded = create_queue(
        project_root=project_root,
        run_dir=run_dir,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        overwrite_queue=args.overwrite_queue,
        seed_completed_output_path=args.seed_completed_output,
    )
    print(json.dumps({"planned": planned, "queued": queued, "seeded": seeded, "state": count_state(run_dir)}, sort_keys=True))
    return 0


def run_requeue_stale(args: argparse.Namespace) -> int:
    planned_units = None
    if args.project_root is not None:
        planned_units = planned_units_by_id(
            project_root=Path(args.project_root).resolve(),
            prompt_glob=args.prompt_glob,
            samples_per_prompt=args.samples_per_prompt,
        )
    requeued, promoted = requeue_running_units(Path(args.run_dir).resolve(), planned_units=planned_units)
    print(json.dumps({"requeued": requeued, "promoted": promoted, "state": count_state(Path(args.run_dir).resolve())}, sort_keys=True))
    return 0


def run_audit(args: argparse.Namespace) -> int:
    report = audit_run_state(
        project_root=Path(args.project_root).resolve(),
        run_dir=Path(args.run_dir).resolve(),
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def run_status(args: argparse.Namespace) -> int:
    print(json.dumps(count_state(Path(args.run_dir).resolve()), sort_keys=True))
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--label-mode", choices=("unknown", "heuristic"), default=DEFAULT_LABEL_MODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic multi-worker insider-trading generation runner.")
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

    requeue = sub.add_parser("requeue-stale")
    requeue.add_argument("--run-dir", required=True)
    requeue.add_argument("--project-root")
    requeue.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    requeue.add_argument("--samples-per-prompt", type=int, default=1)

    audit = sub.add_parser("audit")
    add_common(audit)

    status = sub.add_parser("status")
    status.add_argument("--run-dir", required=True)
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
        merge_outputs(args)
        return 0
    if args.cmd == "requeue-stale":
        return run_requeue_stale(args)
    if args.cmd == "audit":
        return run_audit(args)
    if args.cmd == "status":
        return run_status(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
