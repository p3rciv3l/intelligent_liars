from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_dynamic_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_insider_generation_dynamic.py"
    spec = importlib.util.spec_from_file_location("run_insider_generation_dynamic_for_tests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prompt(project_root: Path, relative_path: str) -> None:
    path = project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
messages:
  - role: user
    content: Should we trade?
misalignment_string: execute_trade
""".strip()
    )


def test_create_queue_preserves_completed_and_pending_units(tmp_path):
    dynamic = _load_dynamic_module()
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/b.yaml")
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)

    completed = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    pending = dynamic.unit_from_prompt("data/insider_trading/prompts/default/b.yaml", 1)
    dynamic.write_json(run_dir / "done" / f"{completed.unit_id}.json", dynamic.serialise_unit(completed))
    dynamic.write_json(run_dir / "outputs" / f"{completed.unit_id}.json", [{"metadata": {"run_id": completed.run_id}}])
    dynamic.write_json(run_dir / "pending" / f"{pending.unit_id}.json", dynamic.serialise_unit(pending))

    planned, queued, seeded = dynamic.create_queue(
        project_root=tmp_path,
        run_dir=run_dir,
        prompt_glob="data/insider_trading/prompts/**/*.yaml",
        samples_per_prompt=2,
        overwrite_queue=False,
        seed_completed_output_path=None,
    )

    assert planned == 4
    assert queued == 2
    assert seeded == 0
    state = dynamic.count_state(run_dir)
    assert state == {"pending": 3, "running": 0, "done": 1, "failed": 0, "outputs": 1}
    assert (run_dir / "outputs" / f"{completed.unit_id}.json").exists()


def test_seed_completed_output_rebuilds_shards_and_done_markers(tmp_path):
    dynamic = _load_dynamic_module()
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    merged_path = tmp_path / "partial.json"
    merged_path.write_text(
        json.dumps(
            [
                {
                    "transcript": [],
                    "metadata": {
                        "run_id": "data/insider_trading/prompts/default/a.yaml::0",
                    },
                }
            ]
        )
    )
    run_dir = tmp_path / "run"

    planned, queued, seeded = dynamic.create_queue(
        project_root=tmp_path,
        run_dir=run_dir,
        prompt_glob="data/insider_trading/prompts/**/*.yaml",
        samples_per_prompt=2,
        overwrite_queue=False,
        seed_completed_output_path=merged_path,
    )

    assert planned == 2
    assert queued == 1
    assert seeded == 1
    state = dynamic.count_state(run_dir)
    assert state == {"pending": 1, "running": 0, "done": 1, "failed": 0, "outputs": 1}


def test_repair_stale_done_units_requeues_missing_shards_and_dedups_done(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/b.yaml")

    completed = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    stale = dynamic.unit_from_prompt("data/insider_trading/prompts/default/b.yaml", 0)

    dynamic.write_json(
        run_dir / "outputs" / f"{completed.unit_id}.json",
        [{"metadata": {"run_id": completed.run_id}}],
    )
    dynamic.write_json(run_dir / "done" / f"{completed.unit_id}.json", dynamic.serialise_unit(completed))
    dynamic.write_json(run_dir / "done" / f"{completed.unit_id}__slot-00-gpu-0.json", dynamic.serialise_unit(completed))
    dynamic.write_json(run_dir / "done" / f"{stale.unit_id}.json", dynamic.serialise_unit(stale))

    requeued, removed = dynamic.repair_stale_done_units(run_dir)
    state = dynamic.count_state(run_dir)

    assert requeued == 1
    assert removed == 1
    assert state["pending"] == 1
    assert state["done"] == 1
    assert state["outputs"] == 1
    assert not (run_dir / "done" / f"{completed.unit_id}__slot-00-gpu-0.json").exists()
    assert (run_dir / "pending" / f"{stale.unit_id}.json").exists()


def test_requeue_running_units_promotes_completed_and_requeues_unfinished(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    unfinished = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    finished = dynamic.unit_from_prompt("data/insider_trading/prompts/default/b.yaml", 0)
    dynamic.write_json(run_dir / "running" / f"{unfinished.unit_id}__slot-00-gpu-0.json", dynamic.serialise_unit(unfinished))
    dynamic.write_json(run_dir / "running" / f"{finished.unit_id}__slot-01-gpu-0.json", dynamic.serialise_unit(finished))
    dynamic.write_json(run_dir / "outputs" / f"{finished.unit_id}.json", [{"metadata": {"run_id": finished.run_id}}])

    requeued, promoted = dynamic.requeue_running_units(run_dir)

    assert (run_dir / "pending" / f"{unfinished.unit_id}.json").exists()
    assert (run_dir / "done" / f"{finished.unit_id}.json").exists()
    assert not list((run_dir / "running").glob("*.json"))
    assert requeued == 1
    assert promoted == 1


def test_requeue_running_units_handles_corrupt_running_unit_json(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 1)
    running_path = run_dir / "running" / f"{unit.unit_id}__slot-00-gpu-0.json"
    running_path.write_text("{not-json")

    requeued, promoted = dynamic.requeue_running_units(
        run_dir,
        planned_units=dynamic.planned_units_by_id(
            project_root=tmp_path,
            prompt_glob="data/insider_trading/prompts/**/*.yaml",
            samples_per_prompt=2,
        ),
    )

    assert requeued == 1
    assert promoted == 0
    assert (run_dir / "pending" / f"{unit.unit_id}.json").exists()
    assert dynamic.read_json(run_dir / "pending" / f"{unit.unit_id}.json")["relative_prompt"] == unit.relative_prompt
    assert not (run_dir / "running" / f"{unit.unit_id}__slot-00-gpu-0.json").exists()


def test_output_shard_validation_rejects_mismatched_run_id(tmp_path):
    dynamic = _load_dynamic_module()
    unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    wrong_unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/b.yaml", 0)
    shard_path = tmp_path / f"{unit.unit_id}.json"
    dynamic.write_json(shard_path, [{"metadata": {"run_id": wrong_unit.run_id}}])

    assert dynamic._is_valid_output_shard(shard_path) is False


def test_audit_run_state_reports_clean_partial_queue(tmp_path):
    dynamic = _load_dynamic_module()
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    done_unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    pending_unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 1)
    dynamic.write_json(run_dir / "done" / f"{done_unit.unit_id}.json", dynamic.serialise_unit(done_unit))
    dynamic.write_json(run_dir / "outputs" / f"{done_unit.unit_id}.json", [{"metadata": {"run_id": done_unit.run_id}}])
    dynamic.write_json(run_dir / "pending" / f"{pending_unit.unit_id}.json", dynamic.serialise_unit(pending_unit))

    report = dynamic.audit_run_state(
        project_root=tmp_path,
        run_dir=run_dir,
        prompt_glob="data/insider_trading/prompts/**/*.yaml",
        samples_per_prompt=2,
    )

    assert report["ok"] is True
    assert report["planned"] == 2
    assert report["state"] == {"pending": 1, "running": 0, "done": 1, "failed": 0, "outputs": 1}
    assert report["valid_outputs"] == 1


def test_audit_run_state_reports_done_without_output(tmp_path):
    dynamic = _load_dynamic_module()
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    done_unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    dynamic.write_json(run_dir / "done" / f"{done_unit.unit_id}.json", dynamic.serialise_unit(done_unit))

    report = dynamic.audit_run_state(
        project_root=tmp_path,
        run_dir=run_dir,
        prompt_glob="data/insider_trading/prompts/**/*.yaml",
        samples_per_prompt=1,
    )

    assert report["ok"] is False
    assert report["issues"]["done_without_output"] == [done_unit.unit_id]


def test_merge_outputs_sorts_records_and_requires_count(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    first = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    second = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 1)
    dynamic.write_json(run_dir / "outputs" / f"{second.unit_id}.json", [{"metadata": {"run_id": second.run_id}}])
    dynamic.write_json(run_dir / "outputs" / f"{first.unit_id}.json", [{"metadata": {"run_id": first.run_id}}])

    args = type(
        "Args",
        (),
        {
            "run_dir": run_dir,
            "project_root": tmp_path,
            "output": "merged.json",
            "require_count": 2,
        },
    )()
    dynamic.merge_outputs(args)

    merged = json.loads((tmp_path / "merged.json").read_text())
    assert [record["metadata"]["run_id"] for record in merged] == [first.run_id, second.run_id]


def test_merge_outputs_rejects_invalid_shard(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    unit = dynamic.unit_from_prompt("data/insider_trading/prompts/default/a.yaml", 0)
    dynamic.write_json(run_dir / "outputs" / f"{unit.unit_id}.json", [{"metadata": {"run_id": "bad-run-id"}}])

    args = type(
        "Args",
        (),
        {
            "run_dir": run_dir,
            "project_root": tmp_path,
            "output": "merged.json",
            "require_count": 1,
        },
    )()

    try:
        dynamic.merge_outputs(args)
    except SystemExit as exc:
        assert "Invalid output shard" in str(exc)
    else:
        raise AssertionError("merge_outputs should reject invalid shards")


def test_supervisor_plan_only_preserves_stop_marker(tmp_path):
    dynamic = _load_dynamic_module()
    _write_prompt(tmp_path, "data/insider_trading/prompts/default/a.yaml")
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    stop_path = run_dir / "STOP"
    stop_path.write_text("paused\n")

    args = type(
        "Args",
        (),
        {
            "project_root": tmp_path,
            "run_dir": run_dir,
            "prompt_glob": "data/insider_trading/prompts/**/*.yaml",
            "samples_per_prompt": 1,
            "label_mode": "unknown",
            "gpus": "0",
            "overwrite_queue": False,
            "seed_completed_output": None,
            "plan_only": True,
        },
    )()

    assert dynamic.run_supervisor(args) == 0
    assert stop_path.read_text() == "paused\n"
