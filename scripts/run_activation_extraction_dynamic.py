#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
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


DEFAULT_GENERATED_MODEL = "qwen3-vl-8b-thinking"


@dataclass(frozen=True)
class ExampleKey:
    source_index: int
    output_index: int


@dataclass(frozen=True)
class Unit:
    chunk_id: str
    task: str
    rollout_path: str
    examples: tuple[ExampleKey, ...]
    estimated_chars: int
    attempt: int = 0
    source_type: str = "rollout"
    generated_model: str = DEFAULT_GENERATED_MODEL

    @property
    def key(self) -> str:
        return f"{self.chunk_id}__attempt-{self.attempt:02d}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def serialise_unit(unit: Unit) -> dict[str, Any]:
    payload = asdict(unit)
    payload["examples"] = [asdict(example) for example in unit.examples]
    return payload


def deserialise_unit(payload: dict[str, Any]) -> Unit:
    return Unit(
        chunk_id=str(payload["chunk_id"]),
        task=str(payload["task"]),
        rollout_path=str(payload["rollout_path"]),
        examples=tuple(ExampleKey(**example) for example in payload["examples"]),
        estimated_chars=int(payload["estimated_chars"]),
        attempt=int(payload.get("attempt", 0)),
        source_type=str(payload.get("source_type", "rollout")),
        generated_model=str(payload.get("generated_model", DEFAULT_GENERATED_MODEL)),
    )


def ensure_dirs(run_dir: Path) -> None:
    for name in ("pending", "running", "done", "failed", "completed", "outputs", "logs", "heartbeats"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def create_initial_queue(args: argparse.Namespace, *, overwrite_queue: bool) -> list[Unit]:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    ensure_dirs(run_dir)
    if overwrite_queue:
        for subdir in ("pending", "running", "done", "failed", "completed", "outputs"):
            for path in (run_dir / subdir).glob("*"):
                path.unlink()
    existing = []
    for subdir in ("pending", "running", "done"):
        existing.extend((run_dir / subdir).glob("*.json"))
    if existing:
        return []

    units = plan_units(
        project_root=project_root,
        rollout_paths=[Path(path) for path in args.path or []],
        tasks=list(args.task or []),
        generated_model=args.generated_model,
        chunk_chars=args.chunk_chars,
        max_examples_per_chunk=args.max_examples_per_chunk,
        limit=args.limit,
    )
    for queue_idx, unit in enumerate(units):
        write_json(run_dir / "pending" / f"{queue_idx:05d}-{unit.key}.json", serialise_unit(unit))
    return units


def plan_units(
    *,
    project_root: Path,
    rollout_paths: list[Path],
    tasks: list[str],
    generated_model: str,
    chunk_chars: int,
    max_examples_per_chunk: int,
    limit: int | None,
) -> list[Unit]:
    from intelligent_liars.activations import ActivationDataset

    units: list[Unit] = []
    sources: list[tuple[str, str, Any]] = []
    for rollout_path in rollout_paths:
        resolved = rollout_path if rollout_path.is_absolute() else project_root / rollout_path
        dataset = ActivationDataset.from_rollout(resolved)
        source_id = str(resolved.relative_to(project_root) if resolved.is_relative_to(project_root) else resolved)
        sources.append(("rollout", source_id, dataset))
    for task in tasks:
        dataset = ActivationDataset.from_named_task(
            task,
            project_root=project_root,
            generated_model=generated_model,
        )
        sources.append(("named_task", task, dataset))
    if not sources:
        raise ValueError("At least one --path rollout file or --task named dataset is required.")

    for source_type, source_id, dataset in sources:
        examples = list(dataset.labeled_for_probe())
        if limit is not None:
            examples = examples[:limit]
        pending: list[Any] = []
        pending_chars = 0

        def flush() -> None:
            nonlocal pending, pending_chars
            if not pending:
                return
            chunk_id = f"{len(units):05d}-{dataset.task}"
            units.append(
                Unit(
                    chunk_id=chunk_id,
                    task=dataset.task,
                    rollout_path=source_id,
                    examples=tuple(ExampleKey(example.source_index, example.output_index) for example in pending),
                    estimated_chars=pending_chars,
                    source_type=source_type,
                    generated_model=generated_model,
                )
            )
            pending = []
            pending_chars = 0

        for example in sorted(examples, key=estimated_example_chars, reverse=True):
            estimated = estimated_example_chars(example)
            if pending and (pending_chars + estimated > chunk_chars or len(pending) >= max_examples_per_chunk):
                flush()
            pending.append(example)
            pending_chars += estimated
            if pending_chars >= chunk_chars or len(pending) >= max_examples_per_chunk:
                flush()
        flush()
    return sorted(units, key=lambda unit: unit.estimated_chars, reverse=True)


def estimated_example_chars(example: Any) -> int:
    message_chars = 0
    for message in example.messages:
        message_chars += content_chars(message.get("content", ""))
    return max(1, message_chars, len(example.detected_text))


def content_chars(content: Any) -> int:
    if isinstance(content, list):
        return sum(content_chars(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return len(str(content.get("text", "")))
        return 0
    return len(str(content))


def claim_unit(run_dir: Path) -> tuple[Path, Unit] | None:
    for path in sorted((run_dir / "pending").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(path))
        except Exception:
            continue
        dest = run_dir / "running" / path.name
        try:
            os.replace(path, dest)
            return dest, unit
        except OSError:
            continue
    return None


def requeue_running_units(run_dir: Path) -> tuple[int, int]:
    """Return stale running units to the pending queue before restart."""
    requeued = 0
    promoted = 0
    for running_path in sorted((run_dir / "running").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(running_path))
        except Exception:
            running_path.unlink(missing_ok=True)
            continue
        output_path = run_dir / "outputs" / f"{unit.key}.h5"
        if _shard_is_complete(output_path):
            done_path = run_dir / "done" / running_path.name
            if not done_path.exists():
                os.replace(running_path, done_path)
            else:
                running_path.unlink()
            promoted += 1
            continue

        pending_path = run_dir / "pending" / running_path.name
        if pending_path.exists():
            running_path.unlink()
            continue

        os.replace(running_path, pending_path)
        requeued += 1

    return requeued, promoted


def repair_stale_done_units(run_dir: Path) -> tuple[int, int]:
    """Move done units without a valid shard output back to pending.

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

        unit_id = unit.key
        if unit_id in seen_unit_ids:
            done_path.unlink(missing_ok=True)
            removed += 1
            continue
        seen_unit_ids.add(unit_id)

        output_path = run_dir / "outputs" / f"{unit_id}.h5"
        if not _shard_is_complete(output_path):
            done_path.unlink(missing_ok=True)
            pending_path = pending_dir / f"{unit_id}.json"
            if not pending_path.exists():
                write_json(pending_path, serialise_unit(unit))
                requeued += 1
    return requeued, removed


def _shard_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            metadata = handle.get("metadata")
            if not isinstance(metadata, h5py.Group) or len(metadata.keys()) == 0:
                return False

            if (layer_container := handle.get("layers")) is not None:
                if isinstance(layer_container, h5py.Group) and _has_layer_dataset(layer_container):
                    return True
            return _has_layer_dataset(handle)
    except Exception:
        return False


def _has_layer_dataset(group: Any) -> bool:
    import h5py

    for name, item in group.items():
        if not name.startswith("layer_") or not isinstance(item, h5py.Group):
            continue
        if any(isinstance(child, h5py.Dataset) for child in item.values()):
            return True
    return False


def count_state(run_dir: Path) -> dict[str, int]:
    state = {
        name: len(list((run_dir / name).glob("*.json")))
        for name in ("pending", "running", "done", "failed", "completed")
    }
    state["outputs"] = len(list((run_dir / "outputs").glob("*.h5")))
    return state


def build_chunk_dataset(project_root: Path, unit: Unit):
    from intelligent_liars.activations import ActivationDataset

    if unit.source_type == "named_task":
        dataset = ActivationDataset.from_named_task(
            unit.task,
            project_root=project_root,
            generated_model=unit.generated_model,
        )
        resolved = dataset.source_path
    elif unit.source_type == "rollout":
        rollout_path = Path(unit.rollout_path)
        resolved = rollout_path if rollout_path.is_absolute() else project_root / rollout_path
        dataset = ActivationDataset.from_rollout(resolved, task=unit.task)
    else:
        raise ValueError(f"Unsupported unit source_type: {unit.source_type!r}")
    by_key = {
        (example.source_index, example.output_index): example
        for example in dataset.labeled_for_probe()
    }
    examples = []
    for key in unit.examples:
        try:
            examples.append(by_key[(key.source_index, key.output_index)])
        except KeyError as exc:
            raise KeyError(
                f"Chunk {unit.chunk_id} references missing example "
                f"source_index={key.source_index} output_index={key.output_index}."
            ) from exc
    return ActivationDataset(
        task=unit.task,
        examples=tuple(examples),
        source_path=resolved,
        dataset_id=dataset.dataset_id or unit.rollout_path,
    )


def run_worker(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    gpu = str(args.gpu)
    slot_id = str(args.slot_id or f"gpu-{gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("PYTHONPATH", "src")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("INTELLIGENT_LIARS_PROGRESS", "1")

    from intelligent_liars.activation_backends import TransformersHookBackend
    from intelligent_liars.activations import (
        ActivationExtractionSettings,
        extract_dataset_activations,
        parse_layer_spec,
    )
    from intelligent_liars.models import load_model_and_processor, model_config_from_env

    bundle = load_model_and_processor(model_config_from_env())
    backend = TransformersHookBackend(bundle)
    layer_indices = parse_layer_spec(args.layers, backend.decoder_layer_count())
    heartbeat = run_dir / "heartbeats" / f"worker-{slot_id}.json"
    processed = 0

    while not (run_dir / "STOP").exists():
        write_json(heartbeat, {"time": now(), "gpu": gpu, "slot": slot_id, "state": "waiting", "processed": processed})
        claimed = claim_unit(run_dir)
        if claimed is None:
            time.sleep(args.idle_sleep)
            continue
        unit_path, unit = claimed
        output_path = run_dir / "outputs" / f"{unit.key}.h5"
        staged_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
        started = time.time()
        write_json(
            heartbeat,
            {
                "time": now(),
                "gpu": gpu,
                "slot": slot_id,
                "state": "running",
                "chunk_id": unit.chunk_id,
                "task": unit.task,
                "examples": len(unit.examples),
                "processed": processed,
            },
        )
        try:
            dataset = build_chunk_dataset(project_root, unit)
            settings = ActivationExtractionSettings(
                layers=layer_indices,
                batch_size=args.batch_size,
                verify_masks=not args.no_verify_masks,
                max_length=args.max_length,
                capture_logits=args.capture_logits,
                storage_dtype=args.storage_dtype,
                compression=args.compression,
            )
            summary = extract_dataset_activations(
                bundle=bundle,
                dataset=dataset,
                output_path=staged_output_path,
                settings=settings,
                overwrite=True,
                backend=backend,
            )
            os.replace(staged_output_path, output_path)
            elapsed = time.time() - started
            write_json(
                run_dir / "completed" / f"{unit.key}.json",
                {
                    "time": now(),
                    "gpu": gpu,
                    "slot": slot_id,
                    "elapsed_seconds": round(elapsed, 3),
                    "unit": serialise_unit(unit),
                    "summary": {
                        "task": summary.task,
                        "examples_seen": summary.examples_seen,
                        "examples_extracted": summary.examples_extracted,
                        "masked_tokens": summary.masked_tokens,
                        "layers": list(summary.layers),
                        "output_path": str(summary.output_path),
                    },
                },
            )
            os.replace(unit_path, run_dir / "done" / unit_path.name)
            processed += 1
            print(
                f"WORKER_DONE time={now()} gpu={gpu} slot={slot_id} chunk={unit.chunk_id} "
                f"task={unit.task} examples={len(unit.examples)} elapsed={elapsed:.1f}s output={output_path}",
                flush=True,
            )
        except Exception as exc:
            staged_output_path.unlink(missing_ok=True)
            write_json(
                run_dir / "failed" / unit_path.name,
                {
                    "time": now(),
                    "gpu": gpu,
                    "slot": slot_id,
                    "error": repr(exc),
                    "unit": serialise_unit(unit),
                },
            )
            unit_path.unlink(missing_ok=True)
            print(
                f"WORKER_FAILED time={now()} gpu={gpu} slot={slot_id} chunk={unit.chunk_id} "
                f"task={unit.task} error={exc!r}",
                flush=True,
            )
    write_json(heartbeat, {"time": now(), "gpu": gpu, "slot": slot_id, "state": "stopped", "processed": processed})
    return 0


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


def run_supervisor(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if args.overwrite_queue:
        requeued, promoted = 0, 0
        repaired_done, removed_done = 0, 0
    else:
        requeued, promoted = requeue_running_units(run_dir)
        repaired_done, removed_done = repair_stale_done_units(run_dir)
    created_units = create_initial_queue(args, overwrite_queue=args.overwrite_queue)
    pending_files = sorted((run_dir / "pending").glob("*.json"))
    if not args.plan_only and not args.allow_empty_queue:
        if not pending_files and not list((run_dir / "running").glob("*.json")):
            print("SUPERVISOR_GUARD pending=0 running=0; queue empty. Use --allow-empty-queue to proceed.")
            return 0
    write_json(
        run_dir / "run_metadata.json",
        {
            "time": now(),
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "paths": args.path,
            "tasks": args.task,
            "generated_model": args.generated_model,
            "gpus": parse_csv(args.gpus),
            "layers": args.layers,
            "batch_size": args.batch_size,
            "chunk_chars": args.chunk_chars,
            "max_examples_per_chunk": args.max_examples_per_chunk,
            "planned_units": len(created_units) or len(pending_files),
            "storage_dtype": args.storage_dtype,
            "capture_logits": args.capture_logits,
            "compression": args.compression,
            "requeued_running_units": requeued,
            "promoted_running_units": promoted,
            "repaired_done_units": repaired_done,
            "removed_done_units": removed_done,
        },
    )
    if args.plan_only:
        units = list(created_units)
        if not units:
            for path in pending_files:
                try:
                    units.append(deserialise_unit(read_json(path)))
                except Exception:
                    pass
        queue_by_task = dict(Counter(unit.task for unit in units))
        queue_by_source_type = dict(Counter(unit.source_type for unit in units))
        print(f"PLAN_PENDING_UNITS={len(units)}")
        print(f"PLAN_BY_TASK={json.dumps(queue_by_task, sort_keys=True)}")
        print(f"PLAN_BY_SOURCE_TYPE={json.dumps(queue_by_source_type, sort_keys=True)}")
        total_examples = sum(len(unit.examples) for unit in units)
        print(f"PLAN_EXAMPLES_TOTAL={total_examples}")
        for unit in sorted(units, key=lambda unit: (unit.task, unit.chunk_id))[:10]:
            print(
                f"PLAN_UNIT task={unit.task} "
                f"source={unit.source_type} "
                f"estimated_chars={unit.estimated_chars} "
                f"examples={len(unit.examples)}",
            )
        return 0

    env_base = os.environ.copy()
    env_base.setdefault("PYTHONPATH", "src")
    env_base.setdefault("HF_HUB_DISABLE_XET", "1")
    env_base.setdefault("INTELLIGENT_LIARS_PROGRESS", "1")
    workers: list[subprocess.Popen[str]] = []
    for idx, gpu in enumerate(parse_csv(args.gpus)):
        slot_id = f"slot-{idx:02d}-gpu-{gpu}"
        worker_log = run_dir / "logs" / f"worker-{slot_id}.log"
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--project-root",
            str(project_root),
            "--run-dir",
            str(run_dir),
            "--gpu",
            gpu,
            "--slot-id",
            slot_id,
            "--layers",
            args.layers,
            "--batch-size",
            str(args.batch_size),
            "--storage-dtype",
            args.storage_dtype,
            "--generated-model",
            args.generated_model,
            "--idle-sleep",
            str(args.idle_sleep),
        ]
        if args.max_length is not None:
            cmd.extend(["--max-length", str(args.max_length)])
        if args.no_verify_masks:
            cmd.append("--no-verify-masks")
        if args.capture_logits:
            cmd.append("--capture-logits")
        cmd.extend(["--compression", args.compression])
        handle = worker_log.open("a")
        workers.append(subprocess.Popen(cmd, cwd=project_root, env=env_base, stdout=handle, stderr=subprocess.STDOUT, text=True))

    print(f"SUPERVISOR_START time={now()} run_dir={run_dir} workers={len(workers)}", flush=True)
    try:
        while True:
            state = count_state(run_dir)
            pending = state["pending"]
            running = state["running"]
            done = state["done"]
            failed = state["failed"]
            alive = sum(worker.poll() is None for worker in workers)
            print(
                f"STATUS time={now()} alive={alive} pending={pending} running={running} done={done} failed={failed}",
                flush=True,
            )
            print(gpu_snapshot(), flush=True)
            if failed:
                raise SystemExit("Extraction worker failed. Inspect run_dir/failed and worker logs.")
            if pending == 0 and running == 0:
                (run_dir / "STOP").write_text(now() + "\n")
                time.sleep(5)
                for worker in workers:
                    if worker.poll() is None:
                        worker.terminate()
                merge_outputs(args)
                print(f"SUPERVISOR_DONE time={now()}", flush=True)
                return 0
            if alive == 0:
                raise SystemExit("All extraction workers exited before queue completion.")
            time.sleep(args.poll_seconds)
    finally:
        (run_dir / "STOP").write_text(now() + "\n")
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()


def merge_outputs(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    shard_paths = sorted(run_dir.glob("outputs/*.h5"))
    if not shard_paths:
        raise SystemExit(f"No activation shards found under {run_dir / 'outputs'}.")
    output_path = Path(args.output)
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    from intelligent_liars.activations import merge_activation_hdf5_shards

    summary = merge_activation_hdf5_shards(
        shard_paths,
        output_path=output_path,
        overwrite=args.overwrite_output,
        compression=args.compression,
    )
    print(
        f"MERGE_DONE time={now()} output={summary.output_path} "
        f"tasks={json.dumps(summary.examples_by_task, sort_keys=True)}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic multi-process activation extraction runner.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", default=".")
    common.add_argument("--run-dir", required=True)
    common.add_argument("--layers", default="all")
    common.add_argument("--batch-size", type=int, default=1)
    common.add_argument("--max-length", type=int)
    common.add_argument("--no-verify-masks", action="store_true")
    common.add_argument("--capture-logits", action="store_true")
    common.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    common.add_argument("--compression", choices=("gzip", "lzf", "none"), default="lzf")
    common.add_argument("--generated-model", default=DEFAULT_GENERATED_MODEL)

    supervisor = sub.add_parser("supervisor", parents=[common])
    supervisor.add_argument("--path", action="append")
    supervisor.add_argument("--task", action="append")
    supervisor.add_argument("--output", required=True)
    supervisor.add_argument("--gpus", default="0")
    supervisor.add_argument("--chunk-chars", type=int, default=12000)
    supervisor.add_argument("--max-examples-per-chunk", type=int, default=16)
    supervisor.add_argument("--limit", type=int)
    supervisor.add_argument("--poll-seconds", type=float, default=30.0)
    supervisor.add_argument("--idle-sleep", type=float, default=5.0)
    supervisor.add_argument("--overwrite-queue", action="store_true")
    supervisor.add_argument("--overwrite-output", action="store_true")
    supervisor.add_argument("--plan-only", action="store_true")
    supervisor.add_argument(
        "--allow-empty-queue",
        action="store_true",
        help="Allow the supervisor to proceed when pending/running queues are empty.",
    )

    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--gpu", required=True)
    worker.add_argument("--slot-id")
    worker.add_argument("--idle-sleep", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "supervisor":
        return run_supervisor(args)
    if args.cmd == "worker":
        return run_worker(args)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
