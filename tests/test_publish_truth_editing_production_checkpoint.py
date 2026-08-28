from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import optuna

from intelligent_liars.truth_editing_wandb_checkpoint import (
    create_wandb_run_checkpoint,
)
from intelligent_liars.truth_editing_study import OBJECTIVES, SearchProposal


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "publish_truth_editing_production_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "publish_truth_editing_production_checkpoint", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _journal(path: Path, count: int) -> None:
    trials = [
        {
            "trial_id": f"trial-{ordinal:04d}",
            "ordinal": ordinal,
            "tier_name": "discovery",
            "evaluation_record_ids": ["record-1"],
            "proposal": SearchProposal(
                direction_ids=("truth-general-21",),
                direction_family="general",
                source_layer=21,
                basis_method="qr",
                requested_rank=1,
                writer_region="middle",
                writer_layers=(21,),
                writer_policy="attention",
                strength=0.1,
            ).to_dict(),
            "result": {
                "outcome_kind": "successful",
                "metrics": {name: 0.5 for name in OBJECTIVES},
                "detail": None,
            },
        }
        for ordinal in range(count)
    ]
    value = {
        "format": "truth_editing_study_journal_v1",
        "study_identity_sha256": "a" * 64,
        "identity_inputs": {"frozen": True},
        "batches": [
            {"ordinal": index // 8, "trials": trials[index : index + 8]}
            for index in range(0, count, 8)
        ],
    }
    value["journal_sha256"] = _sha(value)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    optuna_path = path.with_name(path.name + ".optuna.log")
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(optuna_path))
    )
    study = optuna.create_study(
        study_name="fixture-study",
        storage=storage,
        directions=["maximize"] * len(OBJECTIVES),
    )
    for trial in trials:
        study.add_trial(
            optuna.trial.create_trial(
                values=[0.5] * len(OBJECTIVES),
                user_attrs={
                    "study_ordinal": trial["ordinal"],
                    "proposal_sha256": _sha(trial["proposal"]),
                },
            )
        )
    create_wandb_run_checkpoint(
        path.parents[1] / "monitoring/wandb-run.json",
        run_id="ab12cd34",
        project="intelligent-liars",
        entity=None,
    )


def test_periodic_publisher_defers_until_exact_phase_barrier(tmp_path: Path) -> None:
    journal = tmp_path / "study/study-journal.json"
    _journal(journal, 79)
    output = tmp_path / "checkpoints"
    assert MODULE.main(
        [
            "--journal", str(journal), "--output", str(output),
            "--phase", "discovery", "--optuna-study-name", "fixture-study",
        ]
    ) == 0
    assert not (output / "latest.json").exists()


def test_publisher_adapts_production_journal_name_to_phase_checkpoint_seam(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "study/study-journal.json"
    _journal(journal, 80)
    output = tmp_path / "checkpoints"
    assert MODULE.main(
        [
            "--journal", str(journal), "--output", str(output),
            "--phase", "discovery", "--optuna-study-name", "fixture-study",
        ]
    ) == 0
    latest = json.loads((output / "latest.json").read_text())
    assert latest["phase"] == "discovery"
    generation = output / "generations" / latest["generation_id"]
    assert (generation / "study/study-journal.json").read_bytes() == journal.read_bytes()
    assert (generation / "monitoring/wandb-run.json").is_file()


def test_periodic_publisher_dispatches_exact_rolling_adaptive_state(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    journal = tmp_path / "study/study-journal.json"
    _journal(journal, 88)
    output = tmp_path / "checkpoints"
    observed: dict[str, Any] = {}

    def publish_adaptive_checkpoint(
        source_state_dir: Path,
        publication_root: Path,
        *,
        expected_study_identity_sha256: str,
        expected_study_config_sha256: str,
        expected_completed_trials: int,
        expected_optuna_study_name: str,
    ) -> dict[str, object]:
        observed.update(
            {
                "source_state_dir": source_state_dir,
                "publication_root": publication_root,
                "study_identity": expected_study_identity_sha256,
                "study_config": expected_study_config_sha256,
                "completed": expected_completed_trials,
                "optuna_study_name": expected_optuna_study_name,
            }
        )
        return {"format": "truth_editing_adaptive_checkpoint_manifest_v1"}

    monkeypatch.setattr(
        MODULE, "publish_adaptive_checkpoint", publish_adaptive_checkpoint
    )

    assert MODULE.main(
        [
            "--journal", str(journal),
            "--output", str(output),
            "--adaptive",
            "--study-config-sha256", "b" * 64,
            "--optuna-study-name", "fixture-study",
        ]
    ) == 0

    assert observed == {
        "source_state_dir": tmp_path,
        "publication_root": output,
        "study_identity": "a" * 64,
        "study_config": "b" * 64,
        "completed": 88,
        "optuna_study_name": "fixture-study",
    }
    assert json.loads(capsys.readouterr().out)["format"] == (
        "truth_editing_adaptive_checkpoint_manifest_v1"
    )


def test_adaptive_publisher_fails_closed_without_study_config_identity(
    tmp_path: Path, capsys
) -> None:
    journal = tmp_path / "study/study-journal.json"
    _journal(journal, 8)

    assert MODULE.main(
        [
            "--journal", str(journal),
            "--output", str(tmp_path / "checkpoints"),
            "--adaptive",
            "--optuna-study-name", "fixture-study",
        ]
    ) == 2
    assert "study config identity is required" in capsys.readouterr().err


def test_adaptive_publisher_never_fabricates_missing_scheduler_state(
    tmp_path: Path, capsys
) -> None:
    journal = tmp_path / "study/study-journal.json"
    _journal(journal, 8)
    output = tmp_path / "checkpoints"

    assert MODULE.main(
        [
            "--journal", str(journal),
            "--output", str(output),
            "--adaptive",
            "--study-config-sha256", "b" * 64,
            "--optuna-study-name", "fixture-study",
        ]
    ) == 2

    captured = capsys.readouterr()
    assert "adaptive-run-checkpoint.json" in captured.err
    assert not (output / "adaptive-latest.json").exists()
