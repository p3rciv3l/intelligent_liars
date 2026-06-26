from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from intelligent_liars.run_control import (
    command_line,
    current_git_commit,
    new_run_id,
    stable_sha256,
)


UnitT = TypeVar("UnitT")


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


def ensure_dirs(run_dir: Path, names: Sequence[str]) -> None:
    for name in names:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def clear_queue(run_dir: Path, specs: Iterable[tuple[str, str]]) -> None:
    for name, pattern in specs:
        for path in (run_dir / name).glob(pattern):
            path.unlink()


def prepare_queue_plan(
    *,
    project_root: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    overwrite_queue: bool,
) -> dict[str, Any]:
    queue_plan_id = stable_sha256(manifest)
    plan_path = run_dir / "queue_plan.json"
    if plan_path.exists() and not overwrite_queue:
        existing = read_json(plan_path)
        if existing.get("queue_plan_id") != queue_plan_id:
            raise SystemExit(
                f"Existing queue plan differs for {run_dir}: "
                f"{existing.get('queue_plan_id')} != {queue_plan_id}. "
                "Use a fresh run directory or --overwrite-queue intentionally."
            )
        return existing
    return {
        "schema_version": manifest.get("schema_version", 1),
        "run_id": new_run_id(),
        "queue_plan_id": queue_plan_id,
        "created_at": now(),
        "git_commit": current_git_commit(project_root),
        "command": command_line(),
        "manifest": manifest,
    }


def write_queue_plan(run_dir: Path, plan: dict[str, Any]) -> None:
    write_json(run_dir / "queue_plan.json", plan)


def load_queue_plan(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "queue_plan.json")


def run_dir_has_queue_state(run_dir: Path, specs: Iterable[tuple[str, str]]) -> bool:
    return any(any((run_dir / name).glob(pattern)) for name, pattern in specs)


def queue_unit_ids(
    run_dir: Path,
    *,
    state_names: Sequence[str],
    output_glob: str,
    deserialise_unit: Callable[[dict[str, Any]], UnitT],
    unit_id: Callable[[UnitT], str],
    output_unit_id: Callable[[Path, str | None], str | None],
    fallback_unit_id: Callable[[Path], str],
) -> set[str]:
    existing: set[str] = set()
    for name in state_names:
        for path in (run_dir / name).glob("*.json"):
            try:
                existing.add(unit_id(deserialise_unit(read_json(path))))
            except Exception:
                existing.add(fallback_unit_id(path))
    for path in (run_dir / "outputs").glob(output_glob):
        if (found := output_unit_id(path, None)) is not None:
            existing.add(found)
    return existing


def enqueue_missing_units(
    run_dir: Path,
    units: Sequence[UnitT],
    *,
    run_id: str,
    queue_plan_id: str,
    state_names: Sequence[str],
    output_glob: str,
    unit_id: Callable[[UnitT], str],
    serialise_unit: Callable[[UnitT], dict[str, Any]],
    deserialise_unit: Callable[[dict[str, Any]], UnitT],
    stamp_unit_plan: Callable[[UnitT, str, str], UnitT],
    pending_path: Callable[[Path, int, UnitT], Path],
    output_unit_id: Callable[[Path, str | None], str | None],
    fallback_unit_id: Callable[[Path], str],
) -> int:
    existing = queue_unit_ids(
        run_dir,
        state_names=state_names,
        output_glob=output_glob,
        deserialise_unit=deserialise_unit,
        unit_id=unit_id,
        output_unit_id=output_unit_id,
        fallback_unit_id=fallback_unit_id,
    )
    queued = 0
    for queue_idx, unit in enumerate(units):
        stamped = stamp_unit_plan(unit, run_id, queue_plan_id)
        stamped_id = unit_id(stamped)
        if stamped_id in existing:
            continue
        write_json(pending_path(run_dir, queue_idx, stamped), serialise_unit(stamped))
        existing.add(stamped_id)
        queued += 1
    return queued


def claim_unit(
    run_dir: Path,
    *,
    deserialise_unit: Callable[[dict[str, Any]], UnitT],
    running_path: Callable[[Path, Path, UnitT], Path],
) -> tuple[Path, UnitT] | None:
    for pending_path in sorted((run_dir / "pending").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(pending_path))
        except Exception:
            continue
        destination = running_path(run_dir, pending_path, unit)
        try:
            os.replace(pending_path, destination)
            return destination, unit
        except OSError:
            continue
    return None


def requeue_running_units(
    run_dir: Path,
    *,
    deserialise_unit: Callable[[dict[str, Any]], UnitT],
    serialise_unit: Callable[[UnitT], dict[str, Any]],
    output_path: Callable[[Path, UnitT], Path],
    output_is_valid: Callable[[Path, UnitT], bool],
    pending_path: Callable[[Path, Path, UnitT], Path],
    done_path: Callable[[Path, Path, UnitT], Path],
    recover_unit: Callable[[Path], UnitT | None] | None = None,
    move_to_pending: bool = True,
    count_existing_pending: bool = False,
) -> tuple[int, int]:
    requeued = 0
    promoted = 0
    for running_path in sorted((run_dir / "running").glob("*.json")):
        try:
            unit = deserialise_unit(read_json(running_path))
        except Exception:
            unit = recover_unit(running_path) if recover_unit is not None else None
            if unit is None:
                running_path.unlink(missing_ok=True)
                continue
        if output_is_valid(output_path(run_dir, unit), unit):
            finished_path = done_path(run_dir, running_path, unit)
            if not finished_path.exists():
                os.replace(running_path, finished_path)
            else:
                running_path.unlink()
            promoted += 1
            continue

        queued_path = pending_path(run_dir, running_path, unit)
        if queued_path.exists():
            running_path.unlink()
            if count_existing_pending:
                requeued += 1
            continue
        if move_to_pending:
            os.replace(running_path, queued_path)
        else:
            write_json(queued_path, serialise_unit(unit))
            running_path.unlink()
        requeued += 1
    return requeued, promoted


def repair_stale_done_units(
    run_dir: Path,
    *,
    deserialise_unit: Callable[[dict[str, Any]], UnitT],
    unit_id: Callable[[UnitT], str],
    serialise_unit: Callable[[UnitT], dict[str, Any]],
    output_path: Callable[[Path, UnitT], Path],
    output_is_valid: Callable[[Path, UnitT], bool],
    pending_path: Callable[[Path, UnitT], Path],
) -> tuple[int, int]:
    done_dir = run_dir / "done"
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

        current_unit_id = unit_id(unit)
        if current_unit_id in seen_unit_ids:
            done_path.unlink(missing_ok=True)
            removed += 1
            continue
        seen_unit_ids.add(current_unit_id)

        if not output_is_valid(output_path(run_dir, unit), unit):
            done_path.unlink(missing_ok=True)
            queued_path = pending_path(run_dir, unit)
            if not queued_path.exists():
                write_json(queued_path, serialise_unit(unit))
                requeued += 1
    return requeued, removed


def count_state(run_dir: Path, specs: Iterable[tuple[str, str]]) -> dict[str, int]:
    return {name: len(list((run_dir / name).glob(pattern))) for name, pattern in specs}


def audit_queue_state(
    run_dir: Path,
    *,
    planned_unit_ids: set[str],
    queue_plan_id: str | None,
    state_names: Sequence[str],
    output_glob: str,
    deserialise_unit: Callable[[dict[str, Any]], UnitT],
    unit_id: Callable[[UnitT], str],
    unit_queue_plan_id: Callable[[UnitT], str],
    output_unit_id: Callable[[Path, str | None], str | None],
    count_state_fn: Callable[[Path], dict[str, int]],
    recover_unit: Callable[[Path], UnitT | None] | None = None,
    include_duplicate_output_ids: bool = False,
) -> dict[str, Any]:
    membership: dict[str, list[str]] = {current: [] for current in planned_unit_ids}
    malformed_queue_files: list[str] = []
    unexpected_units: list[str] = []
    duplicate_units: dict[str, list[str]] = {}
    invalid_output_files: list[str] = []
    duplicate_output_ids: dict[str, list[str]] = {}
    valid_output_units: set[str] = set()
    output_markers_by_id: dict[str, list[str]] = {}

    for state_name in state_names:
        for path in sorted((run_dir / state_name).glob("*.json")):
            try:
                unit = deserialise_unit(read_json(path))
            except Exception:
                unit = recover_unit(path) if recover_unit is not None else None
                if unit is None:
                    malformed_queue_files.append(str(path))
                    continue
            marker = f"{state_name}:{path.name}"
            current_unit_id = unit_id(unit)
            if queue_plan_id is not None and unit_queue_plan_id(unit) != queue_plan_id:
                unexpected_units.append(marker)
                continue
            if current_unit_id not in membership:
                unexpected_units.append(marker)
                continue
            membership[current_unit_id].append(marker)

    for path in sorted((run_dir / "outputs").glob(output_glob)):
        current_unit_id = output_unit_id(path, queue_plan_id)
        if current_unit_id is None:
            invalid_output_files.append(str(path))
            continue
        marker = f"outputs:{path.name}"
        if current_unit_id not in membership:
            unexpected_units.append(marker)
            continue
        membership[current_unit_id].append(marker)
        valid_output_units.add(current_unit_id)
        output_markers_by_id.setdefault(current_unit_id, []).append(marker)

    for current_unit_id, markers in sorted(membership.items()):
        state_markers = [
            marker for marker in markers if not marker.startswith("outputs:")
        ]
        if len(state_markers) > 1:
            duplicate_units[current_unit_id] = state_markers
    if include_duplicate_output_ids:
        for current_unit_id, markers in sorted(output_markers_by_id.items()):
            if len(markers) > 1:
                duplicate_output_ids[current_unit_id] = markers

    missing_units = [
        current for current, markers in sorted(membership.items()) if not markers
    ]
    done_without_output = [
        current
        for current, markers in sorted(membership.items())
        if any(marker.startswith("done:") for marker in markers)
        and current not in valid_output_units
    ]
    output_without_done = [
        current
        for current, markers in sorted(membership.items())
        if current in valid_output_units
        and not any(marker.startswith("done:") for marker in markers)
    ]
    issues: dict[str, Any] = {
        "malformed_queue_files": malformed_queue_files,
        "invalid_output_files": invalid_output_files,
        "unexpected_units": unexpected_units,
        "missing_units": missing_units,
        "duplicate_units": duplicate_units,
        "done_without_output": done_without_output,
        "output_without_done": output_without_done,
    }
    if include_duplicate_output_ids:
        issues["duplicate_output_keys"] = duplicate_output_ids
    return {
        "ok": not any(bool(value) for value in issues.values()),
        "planned": len(planned_unit_ids),
        "state": count_state_fn(run_dir),
        "valid_outputs": len(valid_output_units),
        "issues": issues,
    }
