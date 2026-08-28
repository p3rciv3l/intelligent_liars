from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_capacity import (
    CapacityPolicy,
    load_capacity_measurement,
    validate_capacity_receipt,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_truth_editing_adaptive_capacity_receipt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_truth_editing_adaptive_capacity_receipt", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

OBSERVED_AT = "2026-08-28T11:30:00Z"
PLANNED_AT = "2026-08-28T12:00:00Z"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _canary_receipt() -> dict[str, object]:
    unsigned: dict[str, object] = {
        "format": "truth_editing_timed_canary_receipt_v2",
        "canary_config_sha256": "b" * 64,
        "production_config_path": "configs/production.json",
        "production_config_sha256": "c" * 64,
        "model_sha256": "d" * 64,
        "observation_sha256": "e" * 64,
        "trial_id": "trial-0001",
        "trial_outcome_kind": "successful",
        "trial_output_sha256": "f" * 64,
        "generated_tokens": 120,
        "generation_seconds": 4.0,
        "tokens_per_second": 30.0,
        "measured_wall_seconds": 50.0,
        "estimated_canary_cost_usd": 0.001,
        "single_worker_trials_per_hour": 72.0,
        "single_worker_200_trial_hours": 2.7777777778,
        "single_worker_200_trial_cost_usd": 0.1388888889,
        "judge_calls": 10,
        "judge_cost_usd": 0.001,
        "judge_elapsed_seconds": 4.0,
        "gpu_hourly_usd": 0.05,
        "persistence_kl": {
            "text_general": 0.01,
            "vision_general": 0.01,
            "computer_use": 0.01,
        },
        "software_and_live_canary_passed": True,
    }
    return {**unsigned, "receipt_sha256": _canonical_sha256(unsigned)}


def _spend() -> dict[str, str]:
    return {
        "actual_total_usd": "1",
        "actual_infrastructure_usd": "0.9",
        "actual_evaluation_usd": "0.1",
        "pending_infrastructure_usd": "0",
        "pending_evaluation_usd": "0",
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    policy = tmp_path / "policy.json"
    policy.write_bytes(
        Path("configs/truth_editing_adaptive_capacity_policy_v1_13bf1f92.json").read_bytes()
    )
    canary = tmp_path / "canary.json"
    canary.write_text(json.dumps(_canary_receipt()))
    spend = tmp_path / "spend.json"
    spend.write_text(json.dumps(_spend()))
    return policy, canary, spend


def _argv(
    tmp_path: Path, policy: Path, canary: Path, spend: Path
) -> tuple[list[str], Path, Path]:
    measurement = tmp_path / "out" / "measurement.json"
    receipt = tmp_path / "out" / "receipt.json"
    return (
        [
            "--policy",
            str(policy),
            "--timed-canary-receipt",
            str(canary),
            "--spend-snapshot",
            str(spend),
            "--measurement-id",
            "timed-canary-v4-r10-r6",
            "--observed-at",
            OBSERVED_AT,
            "--planned-at",
            PLANNED_AT,
            "--projected-storage-network-usd",
            "0.2",
            "--measurement-output",
            str(measurement),
            "--receipt-output",
            str(receipt),
        ],
        measurement,
        receipt,
    )


def test_cli_builds_identity_bound_outputs_that_strict_reopen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy_path, canary_path, spend_path = _inputs(tmp_path)
    argv, measurement_path, receipt_path = _argv(
        tmp_path, policy_path, canary_path, spend_path
    )

    assert MODULE.main(argv) == 0

    policy = CapacityPolicy.from_mapping(json.loads(policy_path.read_text()))
    measurement_raw = json.loads(measurement_path.read_text())
    receipt_raw = json.loads(receipt_path.read_text())
    measurement = load_capacity_measurement(
        measurement_raw, now=MODULE.parse_utc(PLANNED_AT, "planned_at")
    )
    assert validate_capacity_receipt(receipt_raw) == receipt_raw
    assert receipt_raw["policy_sha256"] == policy.self_sha256
    assert receipt_raw["measurement_sha256"] == measurement.self_sha256
    assert (
        receipt_raw["timed_canary_receipt_sha256"]
        == _canary_receipt()["receipt_sha256"]
    )
    assert measurement_path.read_bytes().endswith(b"\n")
    assert receipt_path.read_bytes().endswith(b"\n")
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "measurement_output": str(measurement_path),
        "measurement_sha256": measurement.self_sha256,
        "receipt_output": str(receipt_path),
        "receipt_sha256": receipt_raw["receipt_sha256"],
    }


def test_cli_is_deterministic_for_distinct_output_paths(tmp_path: Path) -> None:
    policy, canary, spend = _inputs(tmp_path)
    argv_a, measurement_a, receipt_a = _argv(tmp_path / "a", policy, canary, spend)
    argv_b, measurement_b, receipt_b = _argv(tmp_path / "b", policy, canary, spend)

    assert MODULE.main(argv_a) == 0
    assert MODULE.main(argv_b) == 0

    assert measurement_a.read_bytes() == measurement_b.read_bytes()
    assert receipt_a.read_bytes() == receipt_b.read_bytes()


@pytest.mark.parametrize("existing", ["measurement", "receipt"])
def test_cli_preflights_both_outputs_without_partial_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    existing: str,
) -> None:
    policy, canary, spend = _inputs(tmp_path)
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)
    occupied = measurement if existing == "measurement" else receipt
    occupied.parent.mkdir(parents=True)
    occupied.write_text("do not replace\n")

    assert MODULE.main(argv) == 2

    assert occupied.read_text() == "do not replace\n"
    other = receipt if existing == "measurement" else measurement
    assert not other.exists()
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("input_name", ["policy", "canary", "spend"])
def test_cli_rejects_symlink_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    input_name: str,
) -> None:
    policy, canary, spend = _inputs(tmp_path)
    paths = {"policy": policy, "canary": canary, "spend": spend}
    target = paths[input_name]
    real = target.with_suffix(".real.json")
    target.replace(real)
    target.symlink_to(real)
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)

    assert MODULE.main(argv) == 2

    assert "regular JSON file" in capsys.readouterr().err
    assert not measurement.exists()
    assert not receipt.exists()


@pytest.mark.parametrize(
    ("observed_at", "planned_at", "message"),
    [
        ("2026-08-28T00:00:00Z", PLANNED_AT, "stale"),
        ("2026-08-28T12:00:01Z", PLANNED_AT, "future"),
        (OBSERVED_AT, "2026-08-28T12:00:00+00:00", "UTC timestamp"),
    ],
)
def test_cli_rejects_stale_future_and_noncanonical_timestamps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    observed_at: str,
    planned_at: str,
    message: str,
) -> None:
    policy, canary, spend = _inputs(tmp_path)
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)
    argv[argv.index("--observed-at") + 1] = observed_at
    argv[argv.index("--planned-at") + 1] = planned_at

    assert MODULE.main(argv) == 2

    assert message in capsys.readouterr().err
    assert not measurement.exists()
    assert not receipt.exists()


@pytest.mark.parametrize("input_name", ["policy", "canary", "spend"])
def test_cli_rejects_tampered_or_incompatible_inputs_without_outputs(
    tmp_path: Path,
    input_name: str,
) -> None:
    policy, canary, spend = _inputs(tmp_path)
    if input_name == "policy":
        raw = json.loads(policy.read_text())
        raw["execution"]["maximum_trials"] = 799
        policy.write_text(json.dumps(raw))
    elif input_name == "canary":
        raw = _canary_receipt()
        raw["tokens_per_second"] = 31.0
        canary.write_text(json.dumps(raw))
    else:
        raw = _spend()
        raw["unexpected"] = "field"
        spend.write_text(json.dumps(raw))
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)

    assert MODULE.main(argv) == 2

    assert not measurement.exists()
    assert not receipt.exists()


def test_cli_rejects_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    policy, canary, spend = _inputs(tmp_path)
    spend.write_text(
        '{"actual_total_usd":"1","actual_total_usd":"1",'
        '"actual_infrastructure_usd":"0.9","actual_evaluation_usd":"0.1",'
        '"pending_infrastructure_usd":"0","pending_evaluation_usd":"0"}'
    )
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)

    assert MODULE.main(argv) == 2

    assert not measurement.exists()
    assert not receipt.exists()


def test_cli_does_not_publish_measurement_when_receipt_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy, canary, spend = _inputs(tmp_path)
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)

    def fail_write(path: Path, value: object) -> None:
        raise OSError("simulated receipt staging failure")

    monkeypatch.setattr(MODULE, "write_capacity_receipt", fail_write)

    assert MODULE.main(argv) == 2
    assert not measurement.exists()
    assert not receipt.exists()


def test_cli_rejects_same_output_path(tmp_path: Path) -> None:
    policy, canary, spend = _inputs(tmp_path)
    argv, measurement, _ = _argv(tmp_path, policy, canary, spend)
    argv[argv.index("--receipt-output") + 1] = str(measurement)

    assert MODULE.main(argv) == 2
    assert not measurement.exists()


def test_cli_rejects_re_signed_canary_with_changed_schema(tmp_path: Path) -> None:
    policy, canary, spend = _inputs(tmp_path)
    raw = copy.deepcopy(_canary_receipt())
    raw.pop("receipt_sha256")
    raw["unexpected"] = True
    raw["receipt_sha256"] = _canonical_sha256(raw)
    canary.write_text(json.dumps(raw))
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)

    assert MODULE.main(argv) == 2
    assert not measurement.exists()
    assert not receipt.exists()


def test_cli_rejects_re_signed_canary_that_did_not_pass(tmp_path: Path) -> None:
    policy, canary, spend = _inputs(tmp_path)
    raw = _canary_receipt()
    raw.pop("receipt_sha256")
    raw["software_and_live_canary_passed"] = False
    raw["receipt_sha256"] = _canonical_sha256(raw)
    canary.write_text(json.dumps(raw))
    argv, measurement, receipt = _argv(tmp_path, policy, canary, spend)

    assert MODULE.main(argv) == 2
    assert not measurement.exists()
    assert not receipt.exists()
