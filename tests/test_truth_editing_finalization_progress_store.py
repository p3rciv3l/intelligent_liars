from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_finalization_progress_store import (
    FinalizationProgressBinding,
    FinalizationProgressError,
    FinalizationProgressRepository,
)
from intelligent_liars.truth_editing_offhost_checkpoint import (
    FilesystemVersionedObjectStore,
    OffHostCheckpointTarget,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(
    ordinal: int,
    *,
    kind: str = "causal_candidate",
    commit_id: str | None = None,
    before: str | None = None,
    after: str | None = None,
) -> FinalizationProgressBinding:
    root = _digest("judge-root")
    return FinalizationProgressBinding(
        study_identity_sha256=_digest("study"),
        study_config_sha256=_digest("study-config"),
        fleet_config_sha256=_digest("fleet-config"),
        finalization_identity_sha256=_digest("finalization"),
        judge_ledger_root_sha256=root,
        judge_ledger_before_sha256=before or root,
        judge_ledger_after_sha256=after or root,
        optuna_study_name="truth-editing-adaptive",
        wandb_run_id="wandb-finalization",
        stage_ordinal=ordinal,
        stage_kind=kind,
        commit_id=commit_id or f"commit-{ordinal}",
    )


def _repository(tmp_path: Path) -> FinalizationProgressRepository:
    target = OffHostCheckpointTarget(
        bucket="test-versioned-bucket",
        region="us-east-1",
        key_prefix="model-registry/v1/finalization-progress/test-study",
        registry_config_sha256=_digest("registry"),
    )
    return FinalizationProgressRepository(
        store=FilesystemVersionedObjectStore(tmp_path / "objects"),
        target=target,
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_versioned_roundtrip_and_clean_host_restore_preserve_exact_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run"
    first_receipt = source / "finalization/causal/trial-0007/receipt.json"
    _write(first_receipt, {"trial_id": "trial-0007", "evidence": "causal"})
    repository = _repository(tmp_path)

    first = repository.publish(
        source,
        _binding(0, commit_id="trial-0007"),
        evidence_paths=["finalization/causal/trial-0007/receipt.json"],
    )
    repeat = source / "finalization/repeats/trial-0007/repeat-0.json"
    _write(repeat, {"trial_id": "trial-0007", "repeat": 0})
    second_binding = _binding(
        1,
        kind="repeat_evaluation",
        commit_id="trial-0007-repeat-0",
    )
    second = repository.publish(
        source,
        second_binding,
        evidence_paths=[
            "finalization/causal/trial-0007/receipt.json",
            "finalization/repeats/trial-0007/repeat-0.json",
        ],
        expected_latest_etag=repository.read_latest(
            _binding(0, commit_id="trial-0007")
        ).etag,
    )

    clean_host = tmp_path / "clean-host"
    restore = repository.restore_latest(clean_host, second_binding)

    assert first["stage_ordinal"] == 0
    assert second["previous_pointer_sha256"] == first["pointer_sha256"]
    assert restore["restored_paths"] == [
        "finalization/causal/trial-0007/receipt.json",
        "finalization/repeats/trial-0007/repeat-0.json",
    ]
    assert (clean_host / first_receipt.relative_to(source)).read_bytes() == first_receipt.read_bytes()
    assert (clean_host / repeat.relative_to(source)).read_bytes() == repeat.read_bytes()


def test_clean_host_can_discover_and_restore_current_stage_from_fixed_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run"
    _write(source / "finalization/selection.json", {"selected": "trial-0007"})
    repository = _repository(tmp_path)
    binding = _binding(0, kind="final_selection", commit_id="selection")
    repository.publish(
        source, binding, evidence_paths=["finalization/selection.json"]
    )
    fixed = {
        name: value
        for name, value in binding.to_mapping().items()
        if name
        in {
            "study_identity_sha256",
            "study_config_sha256",
            "fleet_config_sha256",
            "finalization_identity_sha256",
            "judge_ledger_root_sha256",
            "optuna_study_name",
            "wandb_run_id",
        }
    }

    current = repository.read_current(fixed)
    restored = repository.restore_current(tmp_path / "clean", fixed)

    assert current["binding"] == binding.to_mapping()
    assert current["etag"]
    assert current["latest_pointer_version_id"]
    assert restored["binding"] == binding.to_mapping()
    assert restored["latest_pointer_etag"] == current["etag"]
    assert (tmp_path / "clean/finalization/selection.json").is_file()


def test_current_stage_discovery_requires_exact_fixed_identity_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run"
    _write(source / "evidence.json", {"ok": True})
    repository = _repository(tmp_path)
    binding = _binding(0)
    repository.publish(source, binding, evidence_paths=["evidence.json"])
    fixed = {
        "study_identity_sha256": binding.study_identity_sha256,
        "study_config_sha256": binding.study_config_sha256,
        "fleet_config_sha256": binding.fleet_config_sha256,
        "finalization_identity_sha256": binding.finalization_identity_sha256,
        "judge_ledger_root_sha256": binding.judge_ledger_root_sha256,
        "optuna_study_name": binding.optuna_study_name,
        "wandb_run_id": "wrong-run",
    }

    with pytest.raises(FinalizationProgressError, match="identity differs"):
        repository.read_current(fixed)
    fixed.pop("wandb_run_id")
    with pytest.raises(FinalizationProgressError, match="fields differ"):
        repository.read_current(fixed)


def test_authoritative_current_restore_replaces_only_listed_older_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run"
    ledger = source / "providers/production-judge-budget/ledger.json"
    cache = source / "providers/judge-cache/causal-trial-0007.json"
    _write(ledger, {"generation": "new", "tail": _digest("new-tail")})
    _write(cache, {"generation": "new", "result": "verified"})
    repository = _repository(tmp_path)
    binding = _binding(0, after=_digest("new-tail"))
    repository.publish(
        source,
        binding,
        evidence_paths=[
            "providers/production-judge-budget/ledger.json",
            "providers/judge-cache/causal-trial-0007.json",
        ],
    )
    fixed = {
        name: value
        for name, value in binding.to_mapping().items()
        if name
        in {
            "study_identity_sha256",
            "study_config_sha256",
            "fleet_config_sha256",
            "finalization_identity_sha256",
            "judge_ledger_root_sha256",
            "optuna_study_name",
            "wandb_run_id",
        }
    }
    restored = tmp_path / "restored"
    _write(
        restored / "providers/production-judge-budget/ledger.json",
        {"generation": "older"},
    )
    _write(
        restored / "providers/judge-cache/causal-trial-0007.json",
        {"generation": "older"},
    )
    unlisted = restored / "study/adaptive-run-checkpoint.json"
    _write(unlisted, {"must": "remain-byte-identical"})
    unlisted_before = unlisted.read_bytes()

    with pytest.raises(FinalizationProgressError, match="destination conflicts"):
        repository.restore_current(restored, fixed)
    receipt = repository.restore_current(restored, fixed, replace_existing=True)

    assert receipt["replaced_paths"] == [
        "providers/judge-cache/causal-trial-0007.json",
        "providers/production-judge-budget/ledger.json",
    ]
    assert (restored / ledger.relative_to(source)).read_bytes() == ledger.read_bytes()
    assert (restored / cache.relative_to(source)).read_bytes() == cache.read_bytes()
    assert unlisted.read_bytes() == unlisted_before


def test_authoritative_restore_still_rejects_symlink_destination(tmp_path: Path) -> None:
    source = tmp_path / "run"
    _write(source / "providers/judge-cache/result.json", {"new": True})
    repository = _repository(tmp_path)
    binding = _binding(0)
    repository.publish(
        source, binding, evidence_paths=["providers/judge-cache/result.json"]
    )
    fixed = {
        name: value
        for name, value in binding.to_mapping().items()
        if name
        in {
            "study_identity_sha256",
            "study_config_sha256",
            "fleet_config_sha256",
            "finalization_identity_sha256",
            "judge_ledger_root_sha256",
            "optuna_study_name",
            "wandb_run_id",
        }
    }
    target = tmp_path / "target"
    outside = tmp_path / "outside.json"
    _write(outside, {"outside": True})
    link = target / "providers/judge-cache/result.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    with pytest.raises(FinalizationProgressError, match="destination is unsafe"):
        repository.restore_current(target, fixed, replace_existing=True)
    assert json.loads(outside.read_text()) == {"outside": True}


def test_stage_and_judge_ledger_lineage_are_contiguous(tmp_path: Path) -> None:
    source = tmp_path / "run"
    _write(source / "evidence/zero.json", {"ordinal": 0})
    repository = _repository(tmp_path)
    root = _digest("judge-root")
    after = _digest("judge-after-causal")
    repository.publish(
        source,
        _binding(0, before=root, after=after),
        evidence_paths=["evidence/zero.json"],
    )
    _write(source / "evidence/one.json", {"ordinal": 1})

    with pytest.raises(FinalizationProgressError, match="stage lineage"):
        repository.publish(
            source,
            _binding(2, before=after, after=after),
            evidence_paths=["evidence/zero.json", "evidence/one.json"],
        )
    with pytest.raises(FinalizationProgressError, match="judge ledger lineage"):
        repository.publish(
            source,
            _binding(1, before=root, after=root),
            evidence_paths=["evidence/zero.json", "evidence/one.json"],
        )


def test_identity_drift_and_stale_pointer_etag_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "run"
    _write(source / "evidence/zero.json", {"ordinal": 0})
    repository = _repository(tmp_path)
    first_binding = _binding(0)
    repository.publish(source, first_binding, evidence_paths=["evidence/zero.json"])
    stale = repository.read_latest(first_binding)
    _write(source / "evidence/one.json", {"ordinal": 1})
    second_binding = _binding(1, kind="matched_control")
    repository.publish(
        source,
        second_binding,
        evidence_paths=["evidence/zero.json", "evidence/one.json"],
    )

    with pytest.raises(FinalizationProgressError, match="identity differs"):
        repository.read_latest(
            replace(second_binding, wandb_run_id="different-wandb-run")
        )
    with pytest.raises(FinalizationProgressError, match="lineage race"):
        repository.publish(
            source,
            second_binding,
            evidence_paths=["evidence/zero.json", "evidence/one.json"],
            expected_latest_etag=stale.etag,
        )


@pytest.mark.parametrize("name", ["model.safetensors", "adapter.pt", "weights.ckpt"])
def test_checkpoint_weights_are_rejected_before_store_mutation(
    tmp_path: Path, name: str
) -> None:
    source = tmp_path / "run"
    weight = source / "finalization" / name
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"not-even-real-weights")
    repository = _repository(tmp_path)

    with pytest.raises(FinalizationProgressError, match="model weights"):
        repository.publish(source, _binding(0), evidence_paths=[weight.relative_to(source)])

    assert repository.store.object_count == 0


def test_unlisted_files_symlinks_and_conflicting_same_stage_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run"
    _write(source / "evidence/included.json", {"value": 1})
    _write(source / "evidence/unlisted.json", {"secret": "not uploaded"})
    repository = _repository(tmp_path)
    binding = _binding(0)
    receipt = repository.publish(
        source, binding, evidence_paths=["evidence/included.json"]
    )
    assert receipt["file_count"] == 1

    _write(source / "evidence/included.json", {"value": 2})
    with pytest.raises(FinalizationProgressError, match="conflicting bytes"):
        repository.publish(source, binding, evidence_paths=["evidence/included.json"])

    other = tmp_path / "other.json"
    _write(other, {"outside": True})
    link = source / "evidence/link.json"
    link.symlink_to(other)
    fresh = _repository(tmp_path / "fresh")
    with pytest.raises(FinalizationProgressError, match="regular file"):
        fresh.publish(source, binding, evidence_paths=["evidence/link.json"])
