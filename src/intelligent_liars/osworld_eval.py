from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from intelligent_liars.run_control import file_sha256, stable_sha256


DEFAULT_OSWORLD_REPOSITORY = "https://github.com/xlang-ai/OSWorld.git"
DEFAULT_OSWORLD_COMMIT = "b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_QWEN_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
EXPECTED_OSWORLD_TASKS = 369


@dataclass(frozen=True)
class OSWorldTaskIndex:
    path: Path
    sha256: str
    groups: dict[str, tuple[str, ...]]
    task_ids: tuple[str, ...]


def load_osworld_task_index(path: Path) -> OSWorldTaskIndex:
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read OSWorld task index {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("OSWorld task index must be a non-empty object of task groups")

    groups: dict[str, tuple[str, ...]] = {}
    task_ids: list[str] = []
    for group, values in payload.items():
        if not isinstance(group, str) or not isinstance(values, list):
            raise ValueError("OSWorld task groups must map string names to lists")
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"OSWorld task group {group!r} contains an invalid task ID")
        groups[group] = tuple(values)
        task_ids.extend(values)

    unique_ids = set(task_ids)
    if len(task_ids) != EXPECTED_OSWORLD_TASKS or len(unique_ids) != EXPECTED_OSWORLD_TASKS:
        raise ValueError(
            f"expected {EXPECTED_OSWORLD_TASKS} unique task IDs; "
            f"found {len(task_ids)} entries and {len(unique_ids)} unique IDs"
        )
    return OSWorldTaskIndex(
        path=resolved,
        sha256=file_sha256(resolved),
        groups=groups,
        task_ids=tuple(task_ids),
    )


def _validate_model_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("model_base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("model_base_url must not contain credentials, query parameters, or a fragment")
    return value.rstrip("/")


def build_osworld_manifest(
    *,
    task_index_path: Path,
    model_base_url: str,
    max_steps: int,
    osworld_commit: str = DEFAULT_OSWORLD_COMMIT,
    model_id: str = DEFAULT_QWEN_MODEL,
    model_revision: str = DEFAULT_QWEN_REVISION,
) -> dict[str, Any]:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    index = load_osworld_task_index(task_index_path)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "osworld",
        "protocol": "full-369",
        "task_count": len(index.task_ids),
        "task_ids": list(index.task_ids),
        "task_groups": {name: list(ids) for name, ids in index.groups.items()},
        "max_steps": max_steps,
        "observation_type": "screenshot",
        "preserve_full_trajectories": True,
        "osworld": {
            "repository": DEFAULT_OSWORLD_REPOSITORY,
            "commit": osworld_commit,
            "task_index_path": "evaluation_examples/test_all.json",
            "task_index_sha256": index.sha256,
        },
        "model": {
            "id": model_id,
            "revision": model_revision,
            "api_backend": "openai",
            "base_url": _validate_model_base_url(model_base_url),
        },
        "execution": {
            "desktop_provider": "capy",
            "model_provider": "vast",
            "overwrite_valid_outputs": False,
        },
    }
    manifest["manifest_id"] = stable_sha256(manifest)
    return manifest


def _manifest_identity(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "manifest_id"}
    return stable_sha256(payload)


def write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> None:
    expected_identity = _manifest_identity(manifest)
    if manifest.get("manifest_id") != expected_identity:
        raise ValueError("manifest_id does not match manifest contents")
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing manifest is unreadable: {path}") from exc
        if existing.get("manifest_id") != expected_identity or existing != manifest:
            raise ValueError(
                "immutable manifest identity mismatch; use a fresh output path for a changed run"
            )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_capy_warm_pool_payload() -> dict[str, list[dict[str, Any]]]:
    checks = [
        (
            "system",
            "set -eu; uname -a; nproc; free -h; df -h / /tmp; id",
        ),
        (
            "docker",
            "set -eu; docker version; docker info --format '{{json .ServerVersion}}'",
        ),
        (
            "virtualization",
            "set -eu; test -e /dev/kvm && ls -l /dev/kvm || echo '/dev/kvm absent'; "
            "grep '^CapEff:' /proc/self/status",
        ),
        (
            "network",
            "set -eu; curl -fsSI --max-time 20 https://github.com/xlang-ai/OSWorld "
            "| head -n 1",
        ),
        (
            "python",
            "set -eu; python3 --version; git --version; docker compose version",
        ),
    ]
    return {
        "setupCommands": [
            {"name": name, "command": command, "background": False}
            for name, command in checks
        ]
    }


def validate_osworld_task_result(task_dir: Path, *, max_steps: int) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    trajectory_path = task_dir / "traj.jsonl"
    score: float | None = None

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not trajectory_path.is_file():
        errors.append("missing traj.jsonl")
    else:
        for line_number, line in enumerate(trajectory_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid traj.jsonl line {line_number}: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"traj.jsonl line {line_number} is not an object")
                continue
            rows.append(row)
            for required in ("step_num", "action", "response", "screenshot_file"):
                if required not in row:
                    errors.append(f"traj.jsonl line {line_number} missing {required}")
            screenshot_name = row.get("screenshot_file")
            if isinstance(screenshot_name, str):
                screenshot_path = task_dir / screenshot_name
                try:
                    screenshot_path.resolve().relative_to(task_dir.resolve())
                except ValueError:
                    errors.append(f"unsafe screenshot path on line {line_number}: {screenshot_name}")
                else:
                    if not screenshot_path.is_file() or screenshot_path.stat().st_size == 0:
                        errors.append(f"missing screenshot on line {line_number}: {screenshot_name}")

    step_numbers = [row.get("step_num") for row in rows]
    integer_steps = [value for value in step_numbers if isinstance(value, int)]
    if rows and len(integer_steps) != len(rows):
        errors.append("every trajectory row must have an integer step_num")
    unique_steps = set(integer_steps)
    if unique_steps and max(unique_steps) > max_steps:
        errors.append(f"trajectory exceeds max_steps={max_steps}")
    if not rows:
        errors.append("trajectory contains no steps")

    result_path = task_dir / "result.txt"
    if not result_path.is_file():
        errors.append("missing result.txt")
    else:
        try:
            score = float(result_path.read_text().strip())
        except ValueError:
            errors.append("result.txt is not a numeric score")
        else:
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                errors.append("result.txt score must be finite and between 0 and 1")

    for required_file in ("runtime.log", "recording.mp4"):
        path = task_dir / required_file
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {required_file}")

    return {
        "ok": not errors,
        "errors": errors,
        "steps": len(unique_steps),
        "trajectory_rows": len(rows),
        "score": score,
    }
