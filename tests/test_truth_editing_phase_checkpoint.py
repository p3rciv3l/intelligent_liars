from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import optuna

from intelligent_liars.truth_editing_phase_checkpoint import (
    PhaseCheckpointError,
    _validate_optuna_journal,
    publish_phase_checkpoint,
    restore_phase_checkpoint,
)
from intelligent_liars.truth_editing_study import OBJECTIVES, SearchProposal
from intelligent_liars.truth_editing_wandb_checkpoint import create_wandb_run_checkpoint


STUDY_ID = "a" * 64
OPTUNA_STUDY_NAME = "truth-editing-production-cccccccccccc-dddddddddddd"


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_state(
    root: Path,
    *,
    completed: int = 80,
    wandb_run_id: str = "ab12cd34",
    matched_control_ordinals: frozenset[int] = frozenset(),
    operational_failure_ordinals: frozenset[int] = frozenset(),
    scientifically_infeasible_ordinals: frozenset[int] = frozenset(),
    omit_operational_failures_from_optuna: bool = False,
) -> None:
    (root / "study").mkdir(parents=True)
    trials = [
        {
            "trial_id": f"trial-{ordinal:04d}",
            "ordinal": ordinal,
            "tier_name": (
                "discovery" if ordinal < 80 else "expanded" if ordinal < 160 else "finalist"
            ),
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
                strength=(ordinal % 20) / 10,
                matched_basis_control=(
                    "orthogonal" if ordinal in matched_control_ordinals else "none"
                ),
            ).to_dict(),
            "result": (
                {
                    "outcome_kind": "operational_failure",
                    "metrics": {},
                    "detail": "transient judge transport failure",
                }
                if ordinal in operational_failure_ordinals
                else {
                    "outcome_kind": (
                        "scientifically_infeasible"
                        if ordinal in scientifically_infeasible_ordinals
                        else "successful"
                    ),
                    "metrics": {name: 0.5 for name in OBJECTIVES},
                    "detail": None,
                }
            ),
        }
        for ordinal in range(completed)
    ]
    body = {
        "format": "truth_editing_study_journal_v1",
        "study_identity_sha256": STUDY_ID,
        "identity_inputs": {"frozen": True},
        "batches": [
            {"ordinal": index // 8, "trials": trials[index : index + 8]}
            for index in range(0, completed, 8)
        ],
    }
    body["journal_sha256"] = _canonical_sha(body)
    (root / "study/study-journal.json").write_text(json.dumps(body, indent=2) + "\n")
    optuna_path = root / "study/study-journal.json.optuna.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(optuna_path))
    )
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=storage,
        directions=["maximize"] * len(OBJECTIVES),
    )
    for trial in trials:
        if trial["ordinal"] in matched_control_ordinals:
            continue
        if (
            omit_operational_failures_from_optuna
            and trial["ordinal"] in operational_failure_ordinals
        ):
            continue
        common = {
            "user_attrs": {
                "study_ordinal": trial["ordinal"],
                "proposal_sha256": _canonical_sha(trial["proposal"]),
            },
        }
        if trial["ordinal"] in operational_failure_ordinals:
            study.add_trial(
                optuna.trial.create_trial(
                    state=optuna.trial.TrialState.FAIL,
                    **common,
                )
            )
        else:
            study.add_trial(
                optuna.trial.create_trial(
                    values=[0.5] * len(OBJECTIVES),
                    **common,
                )
            )
    create_wandb_run_checkpoint(
        root / "monitoring/wandb-run.json",
        run_id=wandb_run_id,
        project="intelligent-liars",
        entity="truth-editing",
    )


def test_optuna_checkpoint_accepts_matched_control_omitted_from_tpe_journal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_state(
        source,
        completed=8,
        matched_control_ordinals=frozenset({0}),
        operational_failure_ordinals=frozenset(range(8)),
    )
    journal = json.loads((source / "study/study-journal.json").read_text())
    journal_trials = [
        trial for batch in journal["batches"] for trial in batch["trials"]
    ]

    _validate_optuna_journal(
        (source / "study/study-journal.json.optuna.log").read_bytes(),
        journal_trials,
        expected_study_name=OPTUNA_STUDY_NAME,
    )


def test_optuna_checkpoint_accepts_audit_only_operational_failures(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_state(
        source,
        completed=8,
        operational_failure_ordinals=frozenset({0, 3}),
        omit_operational_failures_from_optuna=True,
    )
    journal = json.loads((source / "study/study-journal.json").read_text())
    journal_trials = [
        trial for batch in journal["batches"] for trial in batch["trials"]
    ]

    _validate_optuna_journal(
        (source / "study/study-journal.json.optuna.log").read_bytes(),
        journal_trials,
        expected_study_name=OPTUNA_STUDY_NAME,
    )


def test_optuna_checkpoint_accepts_scientifically_infeasible_complete_trials(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_state(
        source,
        completed=8,
        scientifically_infeasible_ordinals=frozenset({0, 3, 7}),
    )
    journal = json.loads((source / "study/study-journal.json").read_text())
    journal_trials = [
        trial for batch in journal["batches"] for trial in batch["trials"]
    ]

    _validate_optuna_journal(
        (source / "study/study-journal.json.optuna.log").read_bytes(),
        journal_trials,
        expected_study_name=OPTUNA_STUDY_NAME,
    )


def test_optuna_checkpoint_rejects_matched_control_as_tpe_observation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_state(source, completed=8, matched_control_ordinals=frozenset({0}))
    journal = json.loads((source / "study/study-journal.json").read_text())
    journal_trials = [
        trial for batch in journal["batches"] for trial in batch["trials"]
    ]
    optuna_path = source / "study/study-journal.json.optuna.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(optuna_path))
    )
    study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage)
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.FAIL,
            user_attrs={
                "study_ordinal": 0,
                "proposal_sha256": _canonical_sha(journal_trials[0]["proposal"]),
            },
        )
    )

    with pytest.raises(PhaseCheckpointError, match="duplicate or extra"):
        _validate_optuna_journal(
            optuna_path.read_bytes(),
            journal_trials,
            expected_study_name=OPTUNA_STUDY_NAME,
        )


def test_phase_checkpoint_round_trip_preserves_exact_resume_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    original = {
        name: (source / name).read_bytes()
        for name in (
            "study/study-journal.json",
            "study/study-journal.json.optuna.log",
            "monitoring/wandb-run.json",
        )
    }

    published = publish_phase_checkpoint(
        source,
        tmp_path / "published",
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    restored = restore_phase_checkpoint(
        tmp_path / "published",
        tmp_path / "resumed",
        next_phase="expanded",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    assert published["phase"] == "discovery"
    assert published["monitoring"]["wandb_run_id"] == "ab12cd34"
    assert restored["source_manifest_sha256"] == published["manifest_sha256"]
    assert restored["monitoring"] == published["monitoring"]
    assert {
        name: (tmp_path / "resumed" / name).read_bytes()
        for name in original
    } == original
    assert not list(tmp_path.rglob("*.tmp-*"))


def test_phase_checkpoint_rejects_tampered_or_replaced_wandb_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_state(source)
    receipt = publish_phase_checkpoint(
        source,
        publication,
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    generation = publication / "generations" / receipt["generation_id"]
    sidecar = generation / "monitoring/wandb-run.json"
    payload = json.loads(sidecar.read_text())
    payload["run_id"] = "different1"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(PhaseCheckpointError, match="hash|size"):
        restore_phase_checkpoint(
            publication,
            tmp_path / "resumed",
            next_phase="expanded",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_phase_checkpoint_requires_exactly_one_durable_wandb_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_state(source)
    (source / "monitoring/wandb-run.json").unlink()
    with pytest.raises(PhaseCheckpointError, match="missing or unsafe"):
        publish_phase_checkpoint(
            source,
            publication,
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )
    assert not (publication / "latest.json").exists()


def test_republishing_a_phase_cannot_switch_wandb_runs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_state(source)
    publish_phase_checkpoint(
        source,
        publication,
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    (source / "monitoring/wandb-run.json").unlink()
    create_wandb_run_checkpoint(
        source / "monitoring/wandb-run.json",
        run_id="different1",
        project="intelligent-liars",
        entity="truth-editing",
    )

    with pytest.raises(PhaseCheckpointError, match="different|already"):
        publish_phase_checkpoint(
            source,
            publication,
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_publish_rejects_duplicate_trials_and_partial_barriers(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    journal_path = source / "study/study-journal.json"
    journal = json.loads(journal_path.read_text())
    journal["batches"][0]["trials"][1]["trial_id"] = "trial-0000"
    journal["journal_sha256"] = _canonical_sha(
        {key: value for key, value in journal.items() if key != "journal_sha256"}
    )
    journal_path.write_text(json.dumps(journal))

    with pytest.raises(PhaseCheckpointError, match="duplicate|ordering"):
        publish_phase_checkpoint(
            source,
            tmp_path / "published",
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )
    assert not (tmp_path / "published" / "latest.json").exists()

    _write_state(tmp_path / "partial", completed=79)
    with pytest.raises(PhaseCheckpointError, match="completed trial count|eight-trial"):
        publish_phase_checkpoint(
            tmp_path / "partial",
            tmp_path / "published",
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_restore_rejects_tampering_wrong_identity_and_phase_skip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_state(source)
    receipt = publish_phase_checkpoint(
        source,
        publication,
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )

    with pytest.raises(PhaseCheckpointError, match="next phase"):
        restore_phase_checkpoint(
            publication,
            tmp_path / "skipped",
            next_phase="finalist",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )
    with pytest.raises(PhaseCheckpointError, match="study identity"):
        restore_phase_checkpoint(
            publication,
            tmp_path / "wrong-run",
            next_phase="expanded",
            expected_study_identity_sha256="b" * 64,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )

    generation = publication / "generations" / receipt["generation_id"]
    with (generation / "study/study-journal.json.optuna.log").open("ab") as stream:
        stream.write(b"tampered\n")
    with pytest.raises(PhaseCheckpointError, match="hash|size"):
        restore_phase_checkpoint(
            publication,
            tmp_path / "tampered",
            next_phase="expanded",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )
    assert not (tmp_path / "tampered").exists()


def test_publish_fails_closed_on_an_unknown_latest_pointer_schema(tmp_path: Path) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    _write_state(source)
    publish_phase_checkpoint(
        source,
        publication,
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    pointer_path = publication / "latest.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["unexpected"] = True
    pointer["pointer_sha256"] = _canonical_sha(
        {key: value for key, value in pointer.items() if key != "pointer_sha256"}
    )
    pointer_path.write_text(json.dumps(pointer))

    with pytest.raises(PhaseCheckpointError, match="latest pointer fields"):
        publish_phase_checkpoint(
            source,
            publication,
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_expanded_phase_must_append_the_published_optuna_lineage(tmp_path: Path) -> None:
    discovery = tmp_path / "discovery"
    independent_expanded = tmp_path / "independent-expanded"
    publication = tmp_path / "published"
    _write_state(discovery, completed=80)
    _write_state(independent_expanded, completed=160)
    publish_phase_checkpoint(
        discovery,
        publication,
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    with pytest.raises(PhaseCheckpointError, match="append-only"):
        publish_phase_checkpoint(
            independent_expanded,
            publication,
            phase="expanded",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=160,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


@pytest.mark.parametrize(
    "secret_line",
    [
        "OPENROUTER_API_KEY=sk-or-v1-should-not-be-in-a-checkpoint\n",
        "WANDB_API_KEY=wandb-key-should-not-be-in-a-checkpoint\n",
    ],
)
def test_secret_like_state_is_never_published(
    tmp_path: Path, secret_line: str
) -> None:
    source = tmp_path / "source"
    _write_state(source)
    (source / "study/study-journal.json.optuna.log").write_text(secret_line)
    with pytest.raises(PhaseCheckpointError, match="secret-like"):
        publish_phase_checkpoint(
            source,
            tmp_path / "published",
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )
    assert not (tmp_path / "published" / "latest.json").exists()


def test_partial_optuna_append_is_not_a_publishable_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_state(source)
    (source / "study/study-journal.json.optuna.log").write_bytes(b'{"op_code":0')
    with pytest.raises(PhaseCheckpointError, match="Optuna journal is unreadable"):
        publish_phase_checkpoint(
            source,
            tmp_path / "published",
            phase="discovery",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )
    assert not (tmp_path / "published" / "latest.json").exists()


def test_restore_is_idempotent_only_for_identical_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    target = tmp_path / "resumed"
    _write_state(source)
    publish_phase_checkpoint(
        source,
        publication,
        phase="discovery",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    first = restore_phase_checkpoint(
        publication,
        target,
        next_phase="expanded",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    second = restore_phase_checkpoint(
        publication,
        target,
        next_phase="expanded",
        expected_study_identity_sha256=STUDY_ID,
        expected_completed_trials=80,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )
    assert second == first

    (target / "study/study-journal.json.optuna.log").write_bytes(b"different\n")
    with pytest.raises(PhaseCheckpointError, match="existing resume state differs"):
        restore_phase_checkpoint(
            publication,
            target,
            next_phase="expanded",
            expected_study_identity_sha256=STUDY_ID,
            expected_completed_trials=80,
            expected_optuna_study_name=OPTUNA_STUDY_NAME,
        )


def test_checkpoint_cli_publishes_and_restores_without_paths_in_receipts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    publication = tmp_path / "published"
    target = tmp_path / "target"
    _write_state(source)
    script = Path(__file__).parents[1] / "scripts/transfer_truth_editing_phase_checkpoint.py"

    published = subprocess.run(
        [
            sys.executable,
            str(script),
            "publish",
            "--state-dir",
            str(source),
            "--publication-root",
            str(publication),
            "--phase",
            "discovery",
            "--study-identity-sha256",
            STUDY_ID,
            "--optuna-study-name",
            OPTUNA_STUDY_NAME,
            "--completed-trials",
            "80",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    publish_receipt = json.loads(published.stdout)
    assert str(tmp_path) not in published.stdout
    assert publish_receipt["completed_trials"] == 80

    restored = subprocess.run(
        [
            sys.executable,
            str(script),
            "restore",
            "--publication-root",
            str(publication),
            "--state-dir",
            str(target),
            "--next-phase",
            "expanded",
            "--study-identity-sha256",
            STUDY_ID,
            "--optuna-study-name",
            OPTUNA_STUDY_NAME,
            "--completed-trials",
            "80",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    restore_receipt = json.loads(restored.stdout)
    assert str(tmp_path) not in restored.stdout
    assert restore_receipt["next_phase"] == "expanded"
