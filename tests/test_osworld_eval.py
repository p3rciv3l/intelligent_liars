from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.osworld_eval import (
    DEFAULT_OSWORLD_COMMIT,
    EXPECTED_OSWORLD_TASKS,
    build_capy_warm_pool_payload,
    build_osworld_manifest,
    load_osworld_task_index,
    validate_osworld_task_result,
    write_immutable_manifest,
)


def _write_task_index(path: Path, count: int = EXPECTED_OSWORLD_TASKS) -> None:
    payload = {
        "chrome": [f"task-{index:03d}" for index in range(count // 2)],
        "os": [f"task-{index:03d}" for index in range(count // 2, count)],
    }
    path.write_text(json.dumps(payload) + "\n")


def test_load_osworld_task_index_requires_369_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "test_all.json"
    _write_task_index(path)

    index = load_osworld_task_index(path)

    assert len(index.task_ids) == EXPECTED_OSWORLD_TASKS
    assert index.groups["chrome"][0] == "task-000"
    assert len(index.sha256) == 64

    path.write_text(json.dumps({"chrome": ["duplicate", "duplicate"]}))
    with pytest.raises(ValueError, match="expected 369 unique task IDs"):
        load_osworld_task_index(path)


def test_manifest_is_identity_bound_and_cannot_be_overwritten(tmp_path: Path) -> None:
    task_index_path = tmp_path / "test_all.json"
    manifest_path = tmp_path / "manifest.json"
    _write_task_index(task_index_path)
    manifest = build_osworld_manifest(
        task_index_path=task_index_path,
        model_base_url="https://model.example/v1",
        max_steps=15,
    )

    write_immutable_manifest(manifest_path, manifest)
    first = manifest_path.read_text()
    write_immutable_manifest(manifest_path, manifest)

    assert manifest["osworld"]["commit"] == DEFAULT_OSWORLD_COMMIT
    assert manifest["task_count"] == EXPECTED_OSWORLD_TASKS
    assert manifest_path.read_text() == first

    changed = build_osworld_manifest(
        task_index_path=task_index_path,
        model_base_url="https://model.example/v1",
        max_steps=30,
    )
    with pytest.raises(ValueError, match="immutable manifest identity mismatch"):
        write_immutable_manifest(manifest_path, changed)


def test_capy_warm_pool_payload_checks_required_runtime_capabilities() -> None:
    payload = build_capy_warm_pool_payload()
    commands = {item["name"]: item for item in payload["setupCommands"]}

    assert set(commands) == {"system", "docker", "virtualization", "network", "python"}
    assert "/dev/kvm" in commands["virtualization"]["command"]
    assert "CapEff" in commands["virtualization"]["command"]
    assert "github.com/xlang-ai/OSWorld" in commands["network"]["command"]
    assert all(item["background"] is False for item in commands.values())


def test_validate_task_result_requires_full_official_trajectory_bundle(tmp_path: Path) -> None:
    task_dir = tmp_path / "task-001"
    task_dir.mkdir()
    screenshot = task_dir / "step_1_20260720@120000.png"
    screenshot.write_bytes(b"png")
    (task_dir / "traj.jsonl").write_text(
        json.dumps(
            {
                "step_num": 1,
                "action": "pyautogui.click(10, 10)",
                "response": "Click the control.",
                "reward": 1,
                "done": True,
                "info": {},
                "screenshot_file": screenshot.name,
            }
        )
        + "\n"
    )
    (task_dir / "result.txt").write_text("1.0\n")
    (task_dir / "runtime.log").write_text("complete\n")
    (task_dir / "recording.mp4").write_bytes(b"video")

    report = validate_osworld_task_result(task_dir, max_steps=15)

    assert report["ok"] is True
    assert report["steps"] == 1
    assert report["score"] == 1.0

    screenshot.unlink()
    report = validate_osworld_task_result(task_dir, max_steps=15)
    assert report["ok"] is False
    assert any("missing screenshot" in error for error in report["errors"])


def test_validate_task_result_rejects_too_many_steps(tmp_path: Path) -> None:
    task_dir = tmp_path / "task-001"
    task_dir.mkdir()
    rows = []
    for step in range(1, 17):
        screenshot = task_dir / f"step_{step}.png"
        screenshot.write_bytes(b"png")
        rows.append(
            json.dumps(
                {
                    "step_num": step,
                    "action": "WAIT",
                    "screenshot_file": screenshot.name,
                }
            )
        )
    (task_dir / "traj.jsonl").write_text("\n".join(rows) + "\n")
    (task_dir / "result.txt").write_text("0\n")
    (task_dir / "runtime.log").write_text("complete\n")
    (task_dir / "recording.mp4").write_bytes(b"video")

    report = validate_osworld_task_result(task_dir, max_steps=15)

    assert report["ok"] is False
    assert any("exceeds max_steps=15" in error for error in report["errors"])
