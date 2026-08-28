from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_truth_editing_production_worker.py"
SPEC = importlib.util.spec_from_file_location("run_truth_editing_production_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source_config(root: Path) -> Path:
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "truth_editing_study_v2.json").write_text(
        json.dumps(
            {
                "batch_size": 8,
                "max_trials": 200,
                "evaluation_tiers": [
                    {"name": "discovery", "through_trial": 80},
                    {"name": "expanded", "through_trial": 160},
                    {"name": "finalist", "through_trial": 200},
                ],
            }
        )
    )
    source = configs / "truth_editing_production_v4_fixture_deadbeef.json"
    source.write_text(
        json.dumps(
            {
                "format": "truth_editing_production_config_v1",
                "study_config": "truth_editing_study_v2.json",
                "journal_path": "../artifacts/truth-editing/study/study-journal.json",
                "artifact_dir": "../artifacts/truth-editing/study/frozen",
                "runtime_output_dir": "../artifacts/truth-editing/study/runtime",
                "judge_cache_dir": "../artifacts/truth-editing/providers/judge-cache",
                "judge_budget_ledger_dir": "../artifacts/truth-editing/providers/production-judge-budget",
                "model_cache_dir": "../artifacts/truth-editing/model-cache/huggingface",
                "snapshot_manifest_path": "../artifacts/truth-editing/model-cache/snapshot-manifest.json",
            }
        )
    )
    return source


@pytest.mark.parametrize(("phase", "boundary"), [("discovery", 80), ("expanded", 160), ("finalist", 200)])
def test_worker_derives_output_scoped_runtime_config_and_writes_exact_receipt(
    tmp_path: Path, phase: str, boundary: int
) -> None:
    source = _source_config(tmp_path / "repo")
    output = tmp_path / "outputs"
    observed: dict[str, object] = {}

    class Run:
        def run(self, *, stop_after_trials: int):
            observed["boundary"] = stop_after_trials
            runtime = json.loads(
                next(source.parent.glob(".truth_editing_production_runtime.*.json")).read_text()
            )
            observed["runtime"] = runtime
            journal = Path(runtime["journal_path"])
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text('{"durable":true}\n')
            return SimpleNamespace(
                identity_sha256="a" * 64,
                to_mapping=lambda: {
                    "format": "truth_editing_production_run_receipt_v1",
                    "completed_trials": stop_after_trials,
                },
            )

    receipt = MODULE.run_phase(
        source_config=source,
        expected_source_config_sha256=MODULE.file_sha256(source),
        output_root=output,
        phase=phase,
        opener=lambda _path: Run(),
        environ={
            "OPENROUTER_API_KEY": "secret-present",
            "WANDB_RUN_ID": "ab12cd34",
            "WANDB_PROJECT": "intelligent-liars",
            "WANDB_ENTITY": "truth-editing",
        },
    )

    assert observed["boundary"] == boundary
    runtime = observed["runtime"]
    assert runtime["journal_path"] == str(output / "study/study-journal.json")
    assert runtime["judge_budget_ledger_dir"] == str(
        output / "providers/production-judge-budget"
    )
    assert runtime["model_cache_dir"] == "../artifacts/truth-editing/model-cache/huggingface"
    assert receipt["completed_trials"] == boundary
    assert receipt["monitoring"]["wandb_run_id"] == "ab12cd34"
    wandb_checkpoint = json.loads(
        (output / "monitoring/wandb-run.json").read_text()
    )
    assert wandb_checkpoint["run_id"] == "ab12cd34"
    assert wandb_checkpoint["project"] == "intelligent-liars"
    assert wandb_checkpoint["entity"] == "truth-editing"
    written = json.loads((output / "production-run.json").read_text())
    assert written["run_receipt_sha256"] == "a" * 64
    assert written["source_production_config_sha256"] == MODULE.file_sha256(source)


def test_worker_resume_reuses_checkpointed_wandb_run_and_rejects_switch(
    tmp_path: Path,
) -> None:
    source = _source_config(tmp_path / "repo")
    output = tmp_path / "outputs"

    class Run:
        def run(self, *, stop_after_trials: int):
            return SimpleNamespace(
                identity_sha256="a" * 64,
                to_mapping=lambda: {
                    "format": "truth_editing_production_run_receipt_v1",
                    "completed_trials": stop_after_trials,
                },
            )

    common = {
        "source_config": source,
        "expected_source_config_sha256": MODULE.file_sha256(source),
        "output_root": output,
        "phase": "discovery",
        "opener": lambda _path: Run(),
    }
    MODULE.run_phase(
        **common,
        environ={
            "OPENROUTER_API_KEY": "secret-present",
            "WANDB_RUN_ID": "ab12cd34",
        },
    )
    MODULE.run_phase(
        **common,
        environ={"OPENROUTER_API_KEY": "secret-present"},
    )
    with pytest.raises(RuntimeError, match="W&B run identity"):
        MODULE.run_phase(
            **common,
            environ={
                "OPENROUTER_API_KEY": "secret-present",
                "WANDB_RUN_ID": "different1",
            },
        )


def test_worker_requires_secret_without_ever_persisting_it(tmp_path: Path) -> None:
    source = _source_config(tmp_path / "repo")
    output = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        MODULE.run_phase(
            source_config=source,
            expected_source_config_sha256=MODULE.file_sha256(source),
            output_root=output,
            phase="discovery",
            opener=lambda _path: (_ for _ in ()).throw(AssertionError("not opened")),
            environ={},
        )
    assert not output.exists()


def test_worker_rejects_source_config_hash_mismatch_before_outputs(tmp_path: Path) -> None:
    source = _source_config(tmp_path / "repo")
    output = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="config SHA-256"):
        MODULE.run_phase(
            source_config=source,
            expected_source_config_sha256="f" * 64,
            output_root=output,
            phase="discovery",
            opener=lambda _path: (_ for _ in ()).throw(AssertionError("not opened")),
            environ={"OPENROUTER_API_KEY": "secret-present"},
        )
    assert not output.exists()
    assert not tuple(source.parent.glob(".truth_editing_production_runtime.*.json"))


def test_worker_rejects_main_adaptive_mode_in_favor_of_cuda_fleet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "outputs"
    exit_code = MODULE.main(
        [
            "--config",
            str(tmp_path / "unused.json"),
            "--expected-config-sha256",
            "a" * 64,
            "--output-root",
            str(output),
            "--phase",
            "adaptive",
        ]
    )

    assert exit_code == 2
    assert "requires the CUDA fleet controller" in capsys.readouterr().err
    assert not output.exists()
