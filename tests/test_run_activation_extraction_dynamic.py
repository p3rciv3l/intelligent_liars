from __future__ import annotations

import argparse
import importlib.util
import sys
import h5py
from pathlib import Path

from intelligent_liars.activations import ActivationDataset, ActivationExample


def _load_dynamic_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_activation_extraction_dynamic.py"
    spec = importlib.util.spec_from_file_location("run_activation_extraction_dynamic_for_tests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _example(source_index: int, output_index: int, label: int = 0) -> ActivationExample:
    return ActivationExample(
        task="claims__definitional_gemini_600_full",
        source_index=source_index,
        output_index=output_index,
        messages=[{"role": "assistant", "content": f"answer {source_index}", "detect": True}],
        label=label,
    )


def test_plan_units_accepts_named_tasks(monkeypatch, tmp_path):
    dynamic = _load_dynamic_module()
    dataset = ActivationDataset(
        task="claims__definitional_gemini_600_full",
        examples=(_example(0, 0), _example(1, 0), _example(2, 0, label=-1)),
        dataset_id="claims__definitional_gemini_600_full",
    )

    monkeypatch.setattr(
        ActivationDataset,
        "from_named_task",
        staticmethod(lambda *args, **kwargs: dataset),
    )

    units = dynamic.plan_units(
        project_root=tmp_path,
        rollout_paths=[],
        tasks=["claims__definitional_gemini_600_full"],
        generated_model="qwen3-vl-8b-thinking",
        chunk_chars=10_000,
        max_examples_per_chunk=16,
        limit=None,
    )

    assert len(units) == 1
    assert units[0].source_type == "named_task"
    assert units[0].rollout_path == "claims__definitional_gemini_600_full"
    assert units[0].generated_model == "qwen3-vl-8b-thinking"
    assert [example.source_index for example in units[0].examples] == [0, 1]


def test_build_chunk_dataset_reloads_named_task(monkeypatch, tmp_path):
    dynamic = _load_dynamic_module()
    dataset = ActivationDataset(
        task="claims__definitional_gemini_600_full",
        examples=(_example(0, 0), _example(1, 0), _example(2, 0)),
        dataset_id="claims__definitional_gemini_600_full",
    )

    monkeypatch.setattr(
        ActivationDataset,
        "from_named_task",
        staticmethod(lambda *args, **kwargs: dataset),
    )

    unit = dynamic.Unit(
        chunk_id="00000-claims__definitional_gemini_600_full",
        task="claims__definitional_gemini_600_full",
        rollout_path="claims__definitional_gemini_600_full",
        source_type="named_task",
        generated_model="qwen3-vl-8b-thinking",
        examples=(dynamic.ExampleKey(1, 0), dynamic.ExampleKey(2, 0)),
        estimated_chars=2,
    )

    chunk = dynamic.build_chunk_dataset(tmp_path, unit)

    assert chunk.task == "claims__definitional_gemini_600_full"
    assert chunk.dataset_id == "claims__definitional_gemini_600_full"
    assert [example.source_index for example in chunk.examples] == [1, 2]


def test_deserialise_unit_defaults_to_rollout_for_old_queue_json():
    dynamic = _load_dynamic_module()

    unit = dynamic.deserialise_unit(
        {
            "chunk_id": "00000-roleplaying__plain",
            "task": "roleplaying__plain",
            "rollout_path": "data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json",
            "examples": [{"source_index": 0, "output_index": 0}],
            "estimated_chars": 10,
        }
    )

    assert unit.source_type == "rollout"
    assert unit.generated_model == "qwen3-vl-8b-thinking"


def test_requeue_running_units_promotes_completed_and_requeues_unfinished(tmp_path):
    dynamic = _load_dynamic_module()
    import h5py

    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)

    unfinished = dynamic.Unit(
        chunk_id="00000-claims__definitional_gemini_600_full",
        task="claims__definitional_gemini_600_full",
        rollout_path="claims__definitional_gemini_600_full",
        source_type="named_task",
        examples=(dynamic.ExampleKey(source_index=0, output_index=0),),
        estimated_chars=10,
    )
    finished = dynamic.Unit(
        chunk_id="00001-claims__definitional_gemini_600_full",
        task="claims__definitional_gemini_600_full",
        rollout_path="claims__definitional_gemini_600_full",
        source_type="named_task",
        examples=(dynamic.ExampleKey(source_index=1, output_index=0),),
        estimated_chars=10,
    )

    dynamic.write_json(
        run_dir / "running" / f"{unfinished.key}.json",
        dynamic.serialise_unit(unfinished),
    )
    dynamic.write_json(
        run_dir / "running" / f"{finished.key}.json",
        dynamic.serialise_unit(finished),
    )
    with h5py.File(run_dir / "outputs" / f"{finished.key}.h5", "w") as output:
        metadata = output.create_group("metadata")
        metadata.create_dataset("dummy", data=[0])
        output.create_group("layers").create_group("layer_0").create_dataset(
            "claims__definitional_gemini_600_full",
            data=[[0.0]],
        )

    requeued, promoted = dynamic.requeue_running_units(run_dir)

    assert (run_dir / "pending" / f"{unfinished.key}.json").exists()
    assert (run_dir / "done" / f"{finished.key}.json").exists()
    assert not list((run_dir / "running").glob("*.json"))
    assert requeued == 1
    assert promoted == 1


def test_shard_is_complete_rejects_empty_layer_group(tmp_path):
    dynamic = _load_dynamic_module()
    shard_path = tmp_path / "empty-layer.h5"
    with h5py.File(shard_path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.create_dataset("dummy", data=[0])
        handle.create_group("layers").create_group("layer_0")

    assert dynamic._shard_is_complete(shard_path) is False


def test_repair_stale_done_units_requeues_missing_shards_and_dedups_done(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)

    ok_unit = dynamic.Unit(
        chunk_id="00000-claims__definitional_gemini_600_full",
        task="claims__definitional_gemini_600_full",
        rollout_path="claims__definitional_gemini_600_full",
        source_type="named_task",
        examples=(dynamic.ExampleKey(1, 0),),
        estimated_chars=10,
    )
    missing_unit = dynamic.Unit(
        chunk_id="00001-claims__definitional_gemini_600_full",
        task="claims__definitional_gemini_600_full",
        rollout_path="claims__definitional_gemini_600_full",
        source_type="named_task",
        examples=(dynamic.ExampleKey(2, 0),),
        estimated_chars=10,
    )

    with h5py.File(run_dir / "outputs" / f"{ok_unit.key}.h5", "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.create_dataset("dummy", data=[0])
        handle.create_group("layers").create_group("layer_0").create_dataset(
            "claims__definitional_gemini_600_full",
            data=[[0.0]],
        )
    dynamic.write_json(run_dir / "done" / f"{ok_unit.key}.json", dynamic.serialise_unit(ok_unit))
    dynamic.write_json(
        run_dir / "done" / f"{ok_unit.key}__slot-00-gpu-0.json",
        dynamic.serialise_unit(ok_unit),
    )
    dynamic.write_json(run_dir / "done" / f"{missing_unit.key}.json", dynamic.serialise_unit(missing_unit))

    requeued, removed = dynamic.repair_stale_done_units(run_dir)
    state = dynamic.count_state(run_dir)

    assert requeued == 1
    assert removed == 1
    assert state["pending"] == 1
    assert state["done"] == 1
    assert state["outputs"] == 1
    assert (run_dir / "pending" / f"{missing_unit.key}.json").exists()
    assert not (run_dir / "done" / f"{ok_unit.key}__slot-00-gpu-0.json").exists()


def test_create_initial_queue_overwrite_clears_outputs(tmp_path, monkeypatch):
    dynamic = _load_dynamic_module()
    dataset = ActivationDataset(
        task="claims__definitional_gemini_600_full",
        examples=(_example(0, 0),),
        dataset_id="claims__definitional_gemini_600_full",
    )

    monkeypatch.setattr(
        ActivationDataset,
        "from_named_task",
        staticmethod(lambda *args, **kwargs: dataset),
    )

    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)
    dynamic.write_json(run_dir / "outputs" / "stale.h5", {})
    dynamic.write_json(run_dir / "completed" / "stale.json", {})
    dynamic.write_json(run_dir / "pending" / "old.json", {})

    dynamic.create_initial_queue(
        args=argparse.Namespace(
            path=None,
            task=["claims__definitional_gemini_600_full"],
            run_dir=str(run_dir),
            project_root=str(tmp_path),
            chunk_chars=12_000,
            max_examples_per_chunk=16,
            limit=None,
            generated_model="qwen3-vl-8b-thinking",
        ),
        overwrite_queue=True,
    )

    assert not list((run_dir / "outputs").iterdir())
    assert not list((run_dir / "completed").glob("*.json"))
    assert not (run_dir / "pending" / "old.json").exists()


def test_count_state_counts_h5_outputs(tmp_path):
    dynamic = _load_dynamic_module()
    run_dir = tmp_path / "run"
    dynamic.ensure_dirs(run_dir)

    dynamic.write_json(run_dir / "pending" / "a.json", {})
    dynamic.write_json(run_dir / "running" / "b.json", {})
    dynamic.write_json(run_dir / "done" / "c.json", {})
    dynamic.write_json(run_dir / "failed" / "d.json", {})
    dynamic.write_json(run_dir / "completed" / "e.json", {})
    dynamic.write_json(run_dir / "outputs" / "shard1.h5", {})

    state = dynamic.count_state(run_dir)
    assert state["pending"] == 1
    assert state["running"] == 1
    assert state["done"] == 1
    assert state["failed"] == 1
    assert state["completed"] == 1
    assert state["outputs"] == 1
