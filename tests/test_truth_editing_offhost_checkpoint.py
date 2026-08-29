from __future__ import annotations

import hashlib
import io
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_offhost_checkpoint import (
    FilesystemVersionedObjectStore,
    OffHostCheckpointError,
    OffHostCheckpointRepository,
    OffHostCheckpointTarget,
    S3VersionedObjectStore,
    SnapshotBinding,
    materialize_offhost_partial_snapshot,
    hydrate_offhost_partial_snapshot,
    hydrate_offhost_snapshot,
    materialize_offhost_snapshot,
)


STUDY = "a" * 64
CONFIG = "b" * 64
FLEET = "c" * 64


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _receipt(path: Path, ordinal: int) -> None:
    unsigned = {
        "format": "truth_editing_vast_fleet_trial_receipt_v2",
        "fleet_config_sha256": FLEET,
        "trial_id": f"trial-{ordinal:04d}",
        "ordinal": ordinal,
        "request_sha256": hashlib.sha256(f"request-{ordinal}".encode()).hexdigest(),
        "worker_slot": ordinal % 8,
        "result": {
            "outcome_kind": "successful",
            "metrics": {"score": 0.5},
            "detail": None,
        },
        "telemetry": {
            "evaluation_seconds": 4.0,
            "generated_tokens": 32.0,
            "judge_calls": 1.0,
            "judge_cost_usd": 0.001,
        },
    }
    _json(path, {**unsigned, "receipt_sha256": _sha(unsigned)})


def _snapshot(root: Path, completed: int) -> None:
    _json(root / "adaptive-state/study/study-journal.json", {"study": STUDY})
    (root / "adaptive-state/study/study-journal.json.optuna.log").write_text("journal\n")
    _json(
        root / "adaptive-state/study/adaptive-run-checkpoint.json",
        {
            "completed_trials": completed,
            "authorized_through_trial": min(800, completed + 8),
            "current_capacity_receipt_sha256": "d" * 64,
            "study_identity_sha256": STUDY,
            "wandb_run_id": "wandb-1",
        },
    )
    _json(root / "adaptive-state/monitoring/wandb-run.json", {"run_id": "wandb-1"})
    _json(
        root / "adaptive-state/monitoring/adaptive-progress.json",
        {"completed_trials": completed},
    )
    _json(
        root / "adaptive-state/monitoring/rolling-capacity-receipt.json",
        {
            "format": "truth_editing_capacity_receipt_v1",
            "completed_through_trial": completed,
            "receipt_sha256": "d" * 64,
        },
    )
    for name in ("fleet-receipts", "runtime", "judge-cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for ordinal in range(completed):
        _receipt(root / f"fleet-receipts/trial-{ordinal:04d}.json", ordinal)
        _json(
            root / f"runtime/{ordinal:04d}/result.json",
            {"trial_id": f"trial-{ordinal:04d}", "runtime": "verified"},
        )
        _json(
            root / f"judge-cache/{ordinal:04d}.json",
            {"trial_id": f"trial-{ordinal:04d}", "cached": True},
        )
    _json(
        root / "judge-budget-ledger/manifest.json",
        {"budget": "frozen", "completed": completed},
    )


def _binding(completed: int) -> SnapshotBinding:
    return SnapshotBinding(
        study_identity_sha256=STUDY,
        study_config_sha256=CONFIG,
        fleet_config_sha256=FLEET,
        optuna_study_name="truth-editing-adaptive",
        wandb_run_id="wandb-1",
        completed_trials=completed,
    )


def _repository(tmp_path: Path) -> OffHostCheckpointRepository:
    target = OffHostCheckpointTarget.from_model_registry_config(
        Path("configs/model_registry_v1.json"),
        key_prefix="model-registry/v1/truth-editing-checkpoints/test-study",
    )
    return OffHostCheckpointRepository(
        store=FilesystemVersionedObjectStore(tmp_path / "object-store"),
        target=target,
    )


def _prime_zero(repository: OffHostCheckpointRepository, root: Path) -> None:
    initial = root / "trial-zero"
    _snapshot(initial, 0)
    repository.publish(initial, _binding(0))


def _pending_runtime(root: Path, ordinals: tuple[int, ...]) -> dict[int, dict[str, object]]:
    _snapshot(root, 0)
    trials = [
        {
            "trial_id": f"trial-{ordinal:04d}",
            "ordinal": ordinal,
            "tier_name": "discovery",
            "evaluation_record_ids": ["record-1"],
            "proposal": {"direction_id": f"direction-{ordinal}"},
            "result": None,
        }
        for ordinal in range(8)
    ]
    unsigned_journal = {
        "format": "truth_editing_study_journal_v1",
        "study_identity_sha256": STUDY,
        "identity_inputs": {"frozen": True},
        "batches": [{"ordinal": 0, "trials": trials}],
    }
    _json(
        root / "adaptive-state/study/study-journal.json",
        {**unsigned_journal, "journal_sha256": _sha(unsigned_journal)},
    )
    events: dict[int, dict[str, object]] = {}
    for ordinal in ordinals:
        receipt_path = root / f"fleet-receipts/trial-{ordinal:04d}.json"
        _receipt(receipt_path, ordinal)
        _json(
            root / f"runtime/{ordinal:04d}/result.json",
            {"trial_id": f"trial-{ordinal:04d}", "runtime": "verified"},
        )
        _json(
            root / f"judge-cache/{ordinal:04d}.json",
            {"trial_id": f"trial-{ordinal:04d}", "cached": True},
        )
        receipt = json.loads(receipt_path.read_text())
        events[ordinal] = {
            "format": "truth_editing_vast_fleet_receipt_durable_event_v1",
            "fleet_config_sha256": FLEET,
            "trial_id": f"trial-{ordinal:04d}",
            "ordinal": ordinal,
            "request_sha256": receipt["request_sha256"],
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": receipt["receipt_sha256"],
        }
    _json(
        root / "judge-budget-ledger/manifest.json",
        {"budget": "frozen", "completed": 0, "paid_ordinals": list(ordinals)},
    )
    return events


def test_publishes_immutable_verified_bundle_then_atomically_advances_latest(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 8)
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)

    receipt = repository.publish(snapshot, _binding(8))
    restored = tmp_path / "restored"
    restore = repository.restore_latest(restored, _binding(8))

    assert receipt["format"] == "truth_editing_offhost_checkpoint_receipt_v1"
    assert receipt["completed_trials"] == 8
    assert receipt["archive_version_id"]
    assert receipt["latest_pointer_version_id"]
    assert restore["archive_sha256"] == receipt["archive_sha256"]
    assert (restored / "fleet-receipts/trial-0007.json").is_file()
    assert (restored / "runtime/0007/result.json").is_file()
    assert (restored / "judge-cache/0007.json").is_file()


def test_trial_zero_authorization_is_durable_before_first_dispatch(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial"
    _snapshot(initial, 0)
    repository = _repository(tmp_path)

    receipt = repository.publish(initial, _binding(0))
    restored = tmp_path / "restored-zero"
    repository.restore_latest(restored, _binding(0))
    output = tmp_path / "fresh-zero"
    hydrate_offhost_snapshot(restored, output, binding=_binding(0))

    assert receipt["completed_trials"] == 0
    scheduler = json.loads(
        (output / "study/adaptive-run-checkpoint.json").read_text()
    )
    assert scheduler["completed_trials"] == 0
    assert scheduler["authorized_through_trial"] == 8
    assert list((output / "fleet-receipts").iterdir()) == []
    assert list((output / "study/runtime").iterdir()) == []
    assert list((output / "providers/judge-cache").iterdir()) == []


def test_latest_binding_resolves_dynamic_resume_identity_then_restores_clean_host(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 8)
    repository.publish(snapshot, _binding(8))

    binding = repository.read_latest_binding(STUDY, CONFIG, FLEET)

    assert binding == _binding(8)
    restored = tmp_path / "restored"
    repository.restore_latest(restored, binding)
    output = tmp_path / "fresh-output"
    hydrate_offhost_snapshot(restored, output, binding=binding)
    assert (output / "fleet-receipts/trial-0007.json").is_file()
    assert json.loads((output / "monitoring/wandb-run.json").read_text()) == {
        "run_id": "wandb-1"
    }


def test_latest_binding_rejects_static_identity_drift_and_pointer_tampering(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)

    with pytest.raises(OffHostCheckpointError, match="static identity differs"):
        repository.read_latest_binding_if_present("f" * 64, CONFIG, FLEET)

    latest_key = f"{repository.target.key_prefix}/latest.json"
    current = repository.store.read_current(latest_key)
    assert current is not None
    pointer = json.loads(current.data)
    pointer["completed_trials"] = 8
    repository.store.put(latest_key, json.dumps(pointer).encode(), if_match_etag=current.etag)

    with pytest.raises(OffHostCheckpointError, match="self hash differs"):
        repository.read_latest_binding_if_present(STUDY, CONFIG, FLEET)


def test_latest_binding_if_present_returns_none_only_for_genuinely_fresh_store(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    assert repository.read_latest_binding_if_present(STUDY, CONFIG, FLEET) is None
    with pytest.raises(OffHostCheckpointError, match="latest pointer is missing"):
        repository.read_latest_binding(STUDY, CONFIG, FLEET)


@pytest.mark.parametrize("partial_count", range(1, 8))
def test_partial_crash_restore_runs_only_missing_workers_judges_and_spend(
    tmp_path: Path, partial_count: int
) -> None:
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    host = tmp_path / "host"
    ordinals = tuple(range(partial_count))
    events = _pending_runtime(host, ordinals)
    receipt, binding, _snapshot_path = repository.publish_partial_from_runtime(
        tmp_path / "partial-staging",
        committed_binding=_binding(0),
        durable_event=events[ordinals[-1]],
        adaptive_state_root=host / "adaptive-state",
        fleet_receipt_dir=host / "fleet-receipts",
        runtime_output_dir=host / "runtime",
        judge_cache_dir=host / "judge-cache",
        judge_budget_ledger_dir=host / "judge-budget-ledger",
    )
    assert receipt["format"] == "truth_editing_offhost_partial_receipt_v1"
    assert binding.receipt_ordinals == frozenset(ordinals)
    assert repository.read_latest_binding(STUDY, CONFIG, FLEET) == _binding(0)
    shutil.rmtree(host)

    restored = tmp_path / "restored-partial"
    repository.restore_latest_partial(restored, binding)
    fresh = tmp_path / "fresh-host"
    hydrate_offhost_partial_snapshot(restored, fresh, binding=binding)
    journal = json.loads((fresh / "study/study-journal.json").read_text())
    assert all(item["result"] is None for item in journal["batches"][-1]["trials"])

    worker_calls: list[int] = []
    judge_calls: list[int] = []
    ledger_path = fresh / "providers/production-judge-budget/manifest.json"
    ledger = json.loads(ledger_path.read_text())
    previous_spend = len(ledger["paid_ordinals"])
    for ordinal in range(8):
        if (fresh / f"fleet-receipts/trial-{ordinal:04d}.json").exists():
            continue
        worker_calls.append(ordinal)
        if not (fresh / f"providers/judge-cache/{ordinal:04d}.json").exists():
            judge_calls.append(ordinal)
        ledger["paid_ordinals"].append(ordinal)
    assert worker_calls == list(range(partial_count, 8))
    assert judge_calls == list(range(partial_count, 8))
    assert len(ledger["paid_ordinals"]) == previous_spend + len(worker_calls)
    assert len(set(ledger["paid_ordinals"])) == 8


def test_concurrent_partial_callbacks_coalesce_monotonically_and_full_barrier_wins(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    host = tmp_path / "host"
    events = _pending_runtime(host, (6, 1, 5, 0, 4, 2, 3))

    def publish(ordinal: int) -> None:
        repository.publish_partial_from_runtime(
            tmp_path / "partial-staging",
            committed_binding=_binding(0),
            durable_event=events[ordinal],
            adaptive_state_root=host / "adaptive-state",
            fleet_receipt_dir=host / "fleet-receipts",
            runtime_output_dir=host / "runtime",
            judge_cache_dir=host / "judge-cache",
            judge_budget_ledger_dir=host / "judge-budget-ledger",
        )

    with ThreadPoolExecutor(max_workers=7) as executor:
        list(executor.map(publish, events))
    partial = repository.read_latest_partial_binding_if_present(_binding(0))
    assert partial is not None
    assert partial.receipt_ordinals == frozenset(range(7))

    full = tmp_path / "full-eight"
    _snapshot(full, 8)
    repository.publish(full, _binding(8))
    assert repository.read_latest_partial_binding_if_present(_binding(8)) is None
    assert repository.read_latest_binding(STUDY, CONFIG, FLEET) == _binding(8)


def test_same_partial_frontier_with_changed_resume_bytes_fails_closed(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    host = tmp_path / "host"
    events = _pending_runtime(host, (0,))
    arguments = {
        "committed_binding": _binding(0),
        "durable_event": events[0],
        "adaptive_state_root": host / "adaptive-state",
        "fleet_receipt_dir": host / "fleet-receipts",
        "runtime_output_dir": host / "runtime",
        "judge_cache_dir": host / "judge-cache",
        "judge_budget_ledger_dir": host / "judge-budget-ledger",
    }
    repository.publish_partial_from_runtime(
        tmp_path / "partial-staging", **arguments
    )
    _json(
        host / "judge-budget-ledger/manifest.json",
        {"budget": "frozen", "completed": 0, "paid_ordinals": [0], "drift": 1},
    )

    with pytest.raises(OffHostCheckpointError, match="conflicting bytes"):
        repository.publish_partial_from_runtime(
            tmp_path / "partial-staging", **arguments
        )


def test_eighth_receipt_is_partial_durable_until_full_scientific_commit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    host = tmp_path / "host"
    events = _pending_runtime(host, tuple(range(8)))

    _, partial, _ = repository.publish_partial_from_runtime(
        tmp_path / "partial-staging",
        committed_binding=_binding(0),
        durable_event=events[7],
        adaptive_state_root=host / "adaptive-state",
        fleet_receipt_dir=host / "fleet-receipts",
        runtime_output_dir=host / "runtime",
        judge_cache_dir=host / "judge-cache",
        judge_budget_ledger_dir=host / "judge-budget-ledger",
    )

    assert partial.receipt_ordinals == frozenset(range(8))
    assert repository.read_latest_binding(STUDY, CONFIG, FLEET).completed_trials == 0
    restored = tmp_path / "restored-eight-pending"
    repository.restore_latest_partial(restored, partial)
    journal = json.loads(
        (restored / "adaptive-state/study/study-journal.json").read_text()
    )
    assert all(item["result"] is None for item in journal["batches"][-1]["trials"])


def test_exact_retry_is_idempotent_and_lineage_race_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _snapshot(first, 8)
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    receipt = repository.publish(first, _binding(8))
    assert repository.publish(first, _binding(8)) == receipt

    second = tmp_path / "second"
    _snapshot(second, 16)
    current = repository.read_latest(_binding(8))
    repository.publish(second, _binding(16))

    with pytest.raises(OffHostCheckpointError, match="lineage race"):
        repository.publish(second, _binding(16), expected_latest_etag=current.etag)


def test_missing_or_tampered_batch_receipts_fail_before_remote_mutation(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 8)
    (snapshot / "fleet-receipts/trial-0007.json").unlink()
    repository = _repository(tmp_path)

    with pytest.raises(OffHostCheckpointError, match="exact completed trial receipts"):
        repository.publish(snapshot, _binding(8))
    assert repository.store.object_count == 0

    _receipt(snapshot / "fleet-receipts/trial-0007.json", 7)
    payload = json.loads((snapshot / "fleet-receipts/trial-0003.json").read_text())
    payload["worker_slot"] = 7
    _json(snapshot / "fleet-receipts/trial-0003.json", payload)
    with pytest.raises(OffHostCheckpointError, match="fleet receipt identity"):
        repository.publish(snapshot, _binding(8))
    assert repository.store.object_count == 0


def test_v2_receipt_rejects_private_or_unknown_telemetry_before_upload(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 8)
    path = snapshot / "fleet-receipts/trial-0004.json"
    raw = json.loads(path.read_text())
    raw["telemetry"]["prompt"] = "must never leave host"
    unsigned = dict(raw)
    unsigned.pop("receipt_sha256")
    raw["receipt_sha256"] = _sha(unsigned)
    _json(path, raw)
    repository = _repository(tmp_path)

    with pytest.raises(OffHostCheckpointError, match="privacy-safe"):
        repository.publish(snapshot, _binding(8))
    assert repository.store.object_count == 0


def test_crash_restore_resumes_next_batch_without_duplicate_paid_calls(
    tmp_path: Path,
) -> None:
    host = tmp_path / "ephemeral-host"
    _snapshot(host, 8)
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    repository.publish(host, _binding(8))
    shutil.rmtree(host)

    restored_snapshot = tmp_path / "restored-snapshot"
    repository.restore_latest(restored_snapshot, _binding(8))
    fresh = tmp_path / "fresh-host"
    hydrate_offhost_snapshot(restored_snapshot, fresh, binding=_binding(8))
    restored_capacity = json.loads(
        (
            fresh
            / "monitoring/rolling-capacity-receipt.json"
        ).read_text()
    )
    assert restored_capacity["completed_through_trial"] == 8
    evaluator_calls: list[int] = []
    judge_calls: list[int] = []

    for ordinal in range(16):
        receipt = fresh / f"fleet-receipts/trial-{ordinal:04d}.json"
        if receipt.exists():
            continue
        evaluator_calls.append(ordinal)
        judge_calls.append(ordinal)
        _receipt(receipt, ordinal)
        _json(
            fresh / f"study/runtime/{ordinal:04d}/result.json",
            {"trial_id": f"trial-{ordinal:04d}", "runtime": "verified"},
        )
        _json(
            fresh / f"providers/judge-cache/{ordinal:04d}.json",
            {"trial_id": f"trial-{ordinal:04d}", "cached": True},
        )
    _json(
        fresh / "providers/production-judge-budget/manifest.json",
        {"budget": "frozen", "completed": 16},
    )
    _json(
        fresh / "monitoring/rolling-capacity-receipt.json",
        {
            "format": "truth_editing_capacity_receipt_v1",
            "completed_through_trial": 16,
            "receipt_sha256": "e" * 64,
        },
    )
    scheduler = json.loads(
        (fresh / "study/adaptive-run-checkpoint.json").read_text()
    )
    scheduler["completed_trials"] = 16
    scheduler["current_capacity_receipt_sha256"] = "e" * 64
    _json(fresh / "study/adaptive-run-checkpoint.json", scheduler)

    assert evaluator_calls == list(range(8, 16))
    assert judge_calls == list(range(8, 16))
    second_snapshot = materialize_offhost_snapshot(
        tmp_path / "second-snapshot",
        binding=_binding(16),
        adaptive_state_root=fresh,
        fleet_receipt_dir=fresh / "fleet-receipts",
        runtime_output_dir=fresh / "study/runtime",
        judge_cache_dir=fresh / "providers/judge-cache",
        judge_budget_ledger_dir=fresh / "providers/production-judge-budget",
    )
    second = repository.publish(second_snapshot, _binding(16))
    assert second["completed_trials"] == 16
    assert second["previous_pointer_sha256"] is not None


def test_target_must_use_checked_in_registry_bucket_and_namespace() -> None:
    with pytest.raises(OffHostCheckpointError, match="configured registry namespace"):
        OffHostCheckpointTarget.from_model_registry_config(
            Path("configs/model_registry_v1.json"), key_prefix="other/place"
        )


def test_materializer_copies_only_resume_state_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _snapshot(source, 8)
    assembled = materialize_offhost_snapshot(
        tmp_path / "assembled",
        binding=_binding(8),
        adaptive_state_root=source / "adaptive-state",
        fleet_receipt_dir=source / "fleet-receipts",
        runtime_output_dir=source / "runtime",
        judge_cache_dir=source / "judge-cache",
        judge_budget_ledger_dir=source / "judge-budget-ledger",
    )
    assert (assembled / "fleet-receipts/trial-0007.json").is_file()
    assert not (assembled / "unrelated").exists()

    bad = tmp_path / "bad"
    _snapshot(bad, 8)
    (bad / "judge-cache/0007.json").unlink()
    (bad / "judge-cache/0007.json").symlink_to(bad / "judge-cache/0006.json")
    with pytest.raises(OffHostCheckpointError, match="symlink"):
        materialize_offhost_snapshot(
            tmp_path / "rejected",
            binding=_binding(8),
            adaptive_state_root=bad / "adaptive-state",
            fleet_receipt_dir=bad / "fleet-receipts",
            runtime_output_dir=bad / "runtime",
            judge_cache_dir=bad / "judge-cache",
            judge_budget_ledger_dir=bad / "judge-budget-ledger",
        )


def test_partial_materializer_ignores_runtime_transient_staging_directories(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host"
    events = _pending_runtime(host, (0,))
    transient = host / "runtime/.batch.staging-314-123"
    transient.mkdir()
    (transient / "incomplete.bin").write_bytes(b"not durable")

    snapshot, _binding_value = materialize_offhost_partial_snapshot(
        tmp_path / "partial-snapshot",
        committed_binding=_binding(0),
        durable_event=events[0],
        adaptive_state_root=host / "adaptive-state",
        fleet_receipt_dir=host / "fleet-receipts",
        runtime_output_dir=host / "runtime",
        judge_cache_dir=host / "judge-cache",
        judge_budget_ledger_dir=host / "judge-budget-ledger",
    )

    assert not (snapshot / "runtime" / transient.name).exists()


def test_clean_host_hydration_atomically_reconstructs_canonical_runtime_layout(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host"
    _snapshot(host, 8)
    repository = _repository(tmp_path)
    _prime_zero(repository, tmp_path)
    repository.publish(host, _binding(8))
    shutil.rmtree(host)
    restored_snapshot = tmp_path / "restored-snapshot"
    repository.restore_latest(restored_snapshot, _binding(8))

    output = tmp_path / "fresh-output"
    receipt = hydrate_offhost_snapshot(
        restored_snapshot, output, binding=_binding(8)
    )

    assert receipt["format"] == "truth_editing_offhost_checkpoint_hydration_v1"
    assert (output / "study/study-journal.json").is_file()
    assert (output / "study/adaptive-run-checkpoint.json").is_file()
    assert (output / "monitoring/rolling-capacity-receipt.json").is_file()
    assert (output / "fleet-receipts/trial-0007.json").is_file()
    assert (output / "study/runtime/0007/result.json").is_file()
    assert (output / "providers/judge-cache/0007.json").is_file()
    assert (
        output / "providers/production-judge-budget/manifest.json"
    ).is_file()
    assert (output / "offhost-hydration-receipt.json").is_file()

    with pytest.raises(OffHostCheckpointError, match="must not already exist"):
        hydrate_offhost_snapshot(restored_snapshot, output, binding=_binding(8))


class _FakeS3Error(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _FakeS3:
    def __init__(self) -> None:
        self.versions: dict[str, list[tuple[str, bytes, str]]] = {}
        self.puts: list[dict[str, object]] = []

    def get_bucket_versioning(self, **_kwargs: object) -> dict[str, str]:
        return {"Status": "Enabled"}

    def put_object(self, **kwargs: object) -> dict[str, str]:
        key = str(kwargs["Key"])
        values = self.versions.setdefault(key, [])
        if kwargs.get("IfNoneMatch") == "*" and values:
            raise _FakeS3Error("PreconditionFailed")
        if "IfMatch" in kwargs and (
            not values or f'"{values[-1][2]}"' != kwargs["IfMatch"]
        ):
            raise _FakeS3Error("PreconditionFailed")
        data = bytes(kwargs["Body"])
        version = f"version-{len(values) + 1}"
        etag = hashlib.sha256(data).hexdigest()
        values.append((version, data, etag))
        self.puts.append(dict(kwargs))
        return {"VersionId": version}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        if key not in self.versions:
            raise _FakeS3Error("NoSuchKey")
        values = self.versions[key]
        version = kwargs.get("VersionId")
        selected = values[-1] if version is None else next(
            item for item in values if item[0] == version
        )
        return {
            "VersionId": selected[0],
            "Body": io.BytesIO(selected[1]),
            "ETag": f'"{selected[2]}"',
        }


def test_s3_adapter_requires_versions_and_uses_conditional_latest_pointer(
    tmp_path: Path,
) -> None:
    fake = _FakeS3()
    target = OffHostCheckpointTarget.from_model_registry_config(
        Path("configs/model_registry_v1.json"),
        key_prefix="model-registry/v1/truth-editing-checkpoints/s3-test",
    )
    repository = OffHostCheckpointRepository(
        store=S3VersionedObjectStore(fake, bucket=target.bucket), target=target
    )
    snapshot = tmp_path / "snapshot"
    _snapshot(snapshot, 8)
    _prime_zero(repository, tmp_path)
    partial_host = tmp_path / "partial-host"
    partial_events = _pending_runtime(partial_host, (0,))
    repository.publish_partial_from_runtime(
        tmp_path / "partial-staging",
        committed_binding=_binding(0),
        durable_event=partial_events[0],
        adaptive_state_root=partial_host / "adaptive-state",
        fleet_receipt_dir=partial_host / "fleet-receipts",
        runtime_output_dir=partial_host / "runtime",
        judge_cache_dir=partial_host / "judge-cache",
        judge_budget_ledger_dir=partial_host / "judge-budget-ledger",
    )

    receipt = repository.publish(snapshot, _binding(8))
    second = tmp_path / "snapshot-16"
    _snapshot(second, 16)
    repository.publish(second, _binding(16))

    assert receipt["archive_version_id"] == "version-1"
    assert any(call.get("IfNoneMatch") == "*" for call in fake.puts)
    assert any("IfMatch" in call for call in fake.puts)
    assert any("/partial/" in str(call["Key"]) for call in fake.puts)
    assert all("ChecksumSHA256" in call for call in fake.puts)
