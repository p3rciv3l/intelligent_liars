from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_phase_checkpoint import (
    PhaseCheckpointError,
    publish_adaptive_checkpoint,
    restore_adaptive_checkpoint,
)

from test_truth_editing_adaptive_checkpoint_transfer import (
    OPTUNA_STUDY_NAME,
    STUDY_CONFIG_SHA,
    STUDY_ID,
    _write_adaptive_state,
)


def _publish(
    source: Path, publication: Path, completed: int
) -> dict[str, object]:
    return publish_adaptive_checkpoint(
        source,
        publication,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=completed,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )


def _restore(publication: Path, target: Path, completed: int) -> dict[str, object]:
    return restore_adaptive_checkpoint(
        publication,
        target,
        expected_study_identity_sha256=STUDY_ID,
        expected_study_config_sha256=STUDY_CONFIG_SHA,
        expected_completed_trials=completed,
        expected_optuna_study_name=OPTUNA_STUDY_NAME,
    )


def test_aborted_104_cannot_be_used_as_a_96_recovery_boundary(
    tmp_path: Path,
) -> None:
    publication = tmp_path / "published"
    source_104 = tmp_path / "source-104"
    _write_adaptive_state(source_104, completed=104, abort_minimum=True)

    aborted_manifest = _publish(source_104, publication, 104)
    assert aborted_manifest["completed_trials"] == 104
    assert json.loads(
        (publication / "adaptive-latest.json").read_text()
    )["manifest_sha256"] == aborted_manifest["manifest_sha256"]

    # A recovery caller asking for the known-good 96 boundary must not
    # accidentally hydrate the newer aborted generation.
    wrong_target = tmp_path / "wrong-96"
    with pytest.raises(PhaseCheckpointError, match="expected identity"):
        _restore(publication, wrong_target, 96)
    assert not wrong_target.exists()

    restored = tmp_path / "restored-104"
    _restore(publication, restored, 104)
    checkpoint = json.loads(
        (restored / "study/adaptive-run-checkpoint.json").read_text()
    )
    assert checkpoint["phase"] == "aborted"
    assert checkpoint["stop_reason"] == "minimum_trial_guarantee_lost"


def test_replaying_committed_96_restore_is_byte_identical_before_abort_arrives(
    tmp_path: Path,
) -> None:
    publication = tmp_path / "published"
    source = tmp_path / "source-96"
    target = tmp_path / "restored-96"
    _write_adaptive_state(source, completed=96)
    _publish(source, publication, 96)

    first = _restore(publication, target, 96)
    first_bytes = {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    }
    second = _restore(publication, target, 96)

    assert second == first
    assert {
        path.relative_to(target): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file()
    } == first_bytes
