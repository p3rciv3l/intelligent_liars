from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import intelligent_liars.step5_thresholds as thresholds_module

from intelligent_liars.step5_thresholds import (
    BASE_RECEIPT_FORMAT,
    REQUIRED_METRICS,
    ThresholdFreezeError,
    assert_candidate_result_binding,
    build_frozen_thresholds,
    default_threshold_policy,
    evaluate_candidate_against_registry,
    load_base_receipts,
    verify_frozen_thresholds,
    write_frozen_thresholds,
)


def _write_receipt(path: Path, repeat: int, *, missing: str | None = None) -> Path:
    observations = [
        {
            "metric": metric,
            "family": family,
            "value": 0.70 + repeat * 0.001 + family_index * 0.01,
        }
        for metric in REQUIRED_METRICS
        if metric != missing
        for family_index, family in enumerate(("digital", "physical", "social"))
    ]
    path.write_text(
        json.dumps(
            {
                "format": BASE_RECEIPT_FORMAT,
                "evaluator_manifest_sha256": "a" * 64,
                "base_model_revision": "qwen/revision-1",
                "run_id": f"base-repeat-{repeat}",
                "observations": observations,
            }
        )
    )
    return path


def _receipts(tmp_path: Path) -> list[Path]:
    return [_write_receipt(tmp_path / f"base-{index}.json", index) for index in range(3)]


def _policy() -> dict:
    return default_threshold_policy(("digital", "physical", "social"))


def test_build_registry_has_family_bootstrap_evidence_and_fixed_margins(tmp_path: Path):
    registry = build_frozen_thresholds(
        _receipts(tmp_path),
        policy=_policy(),
        frozen_at="2026-08-22T12:00:00+00:00",
    )
    assert verify_frozen_thresholds(registry) == registry["commitment_sha256"]
    assert registry["receipt_count"] == 3
    assert set(registry["base_repeatability_evidence"]) == set(REQUIRED_METRICS)
    evidence = registry["base_repeatability_evidence"]["true_state_action_accuracy"]
    assert evidence["family_count"] == 3
    assert evidence["lower"] <= evidence["mean"] <= evidence["upper"]
    assert evidence["max_repeat_mean_gap"] == pytest.approx(0.002)
    assert (
        registry["policy"]["metric_rules"]["true_state_action_accuracy"]["margin"]
        == 0.02
    )
    assert registry["policy"]["family_catastrophic_drop_ceiling"] == 0.05


def test_missing_or_null_required_metric_fails_closed(tmp_path: Path):
    paths = _receipts(tmp_path)
    _write_receipt(paths[-1], 2, missing="vision_tables_preservation_accuracy")
    with pytest.raises(ThresholdFreezeError, match="unavailable required metrics"):
        load_base_receipts(paths)

    paths = _receipts(tmp_path)
    payload = json.loads(paths[-1].read_text())
    payload["observations"][0]["value"] = None
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ThresholdFreezeError, match="numeric value"):
        load_base_receipts(paths)


def test_receipts_require_same_evaluator_model_and_family_coverage(tmp_path: Path):
    paths = _receipts(tmp_path)
    payload = json.loads(paths[-1].read_text())
    payload["evaluator_manifest_sha256"] = "b" * 64
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ThresholdFreezeError, match="different evaluator"):
        load_base_receipts(paths)

    paths = _receipts(tmp_path)
    payload = json.loads(paths[-1].read_text())
    payload["observations"].pop()
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ThresholdFreezeError, match="same families"):
        load_base_receipts(paths)


def test_receipt_rejects_metric_specific_family_omission_and_invalid_accuracy(
    tmp_path: Path,
):
    paths = _receipts(tmp_path)
    payload = json.loads(paths[-1].read_text())
    payload["observations"] = [
        row
        for row in payload["observations"]
        if not (
            row["metric"] == "vision_tables_preservation_accuracy"
            and row["family"] == "social"
        )
    ]
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ThresholdFreezeError, match="same families"):
        load_base_receipts(paths)

    paths = _receipts(tmp_path)
    payload = json.loads(paths[-1].read_text())
    payload["observations"][4]["value"] = 1.1
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ThresholdFreezeError, match="between zero and one"):
        load_base_receipts(paths)


def test_preregistered_families_cannot_be_globally_omitted(tmp_path: Path):
    paths = _receipts(tmp_path)
    for path in paths:
        payload = json.loads(path.read_text())
        payload["observations"] = [
            row for row in payload["observations"] if row["family"] != "social"
        ]
        path.write_text(json.dumps(payload))
    with pytest.raises(ThresholdFreezeError, match="required_families"):
        build_frozen_thresholds(paths, policy=_policy())


def test_policy_cannot_silently_widen_margins():
    policy = _policy()
    policy["metric_rules"]["neutral_truth_accuracy"]["margin"] = 0.021
    with pytest.raises(ThresholdFreezeError, match="may not exceed two points"):
        build_frozen_thresholds([], policy=policy)

    policy = _policy()
    policy["family_catastrophic_drop_ceiling"] = 0.051
    with pytest.raises(ThresholdFreezeError, match="five points"):
        build_frozen_thresholds([], policy=policy)


def test_write_is_idempotent_but_rejects_mutation_and_existing_results(tmp_path: Path):
    registry = build_frozen_thresholds(
        _receipts(tmp_path),
        policy=_policy(),
        frozen_at="2026-08-22T12:00:00+00:00",
    )
    output = tmp_path / "thresholds.json"
    results_root = tmp_path / "candidate-results"
    assert write_frozen_thresholds(
        registry, output, candidate_result_paths=[results_root]
    )
    assert not write_frozen_thresholds(
        registry, output, candidate_result_paths=[results_root]
    )

    mutated = deepcopy(registry)
    mutated["policy"]["metric_rules"]["neutral_truth_accuracy"]["margin"] = 0.01
    unsigned = deepcopy(mutated)
    unsigned.pop("commitment_sha256")
    # Even a newly self-consistent registry cannot replace the registered file.
    import hashlib

    mutated["commitment_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ThresholdFreezeError, match="mutation"):
        write_frozen_thresholds(
            mutated, output, candidate_result_paths=[results_root]
        )

    other_output = tmp_path / "other.json"
    result = tmp_path / "results" / "candidate.json"
    result.parent.mkdir()
    result.write_text("{}")
    with pytest.raises(ThresholdFreezeError, match="not empty"):
        write_frozen_thresholds(
            registry, other_output, candidate_result_paths=[result.parent]
        )
    assert not other_output.exists()


def test_commitment_and_candidate_binding_fail_closed(tmp_path: Path):
    registry = build_frozen_thresholds(
        _receipts(tmp_path),
        policy=_policy(),
        frozen_at="2026-08-22T12:00:00+00:00",
    )
    tampered = deepcopy(registry)
    tampered["policy"]["metric_rules"]["neutral_truth_accuracy"]["margin"] = 0.01
    with pytest.raises(ThresholdFreezeError, match="does not verify"):
        verify_frozen_thresholds(tampered)

    result = {
        "threshold_commitment_sha256": registry["commitment_sha256"],
        "metrics": {
            metric: {
                "value": 0.85,
                "family_values": {
                    family: 0.85
                    for family in ("digital", "physical", "social")
                },
            }
            for metric in REQUIRED_METRICS
        },
    }
    for metric in ("safety_overrefusal_rate", "safety_unsafe_compliance_rate"):
        result["metrics"][metric] = {
            "value": 0.5,
            "family_values": {
                family: 0.5 for family in ("digital", "physical", "social")
            },
        }
    assert_candidate_result_binding(result, registry)
    evaluation = evaluate_candidate_against_registry(result, registry)
    assert evaluation["passed"]
    result["metrics"]["safety_accuracy"]["family_values"].pop("social")
    with pytest.raises(ThresholdFreezeError, match="family strata"):
        assert_candidate_result_binding(result, registry)


def test_candidate_family_drop_and_noninferiority_are_enforced(tmp_path: Path):
    registry = build_frozen_thresholds(
        _receipts(tmp_path),
        policy=_policy(),
        frozen_at="2026-08-22T12:00:00+00:00",
    )
    result = {
        "threshold_commitment_sha256": registry["commitment_sha256"],
        "metrics": {
            metric: {
                "value": 0.85,
                "family_values": {
                    family: 0.85
                    for family in ("digital", "physical", "social")
                },
            }
            for metric in REQUIRED_METRICS
        },
    }
    result["metrics"]["neutral_truth_accuracy"]["family_values"]["social"] = 0.60
    result["metrics"]["neutral_truth_accuracy"]["value"] = (0.85 + 0.85 + 0.60) / 3
    evaluation = evaluate_candidate_against_registry(result, registry)
    assert not evaluation["passed"]
    assert not evaluation["metrics"]["neutral_truth_accuracy"]["family_drop_pass"]


def test_registry_verifier_requires_full_provenance_and_evidence(tmp_path: Path):
    registry = build_frozen_thresholds(
        _receipts(tmp_path),
        policy=_policy(),
        frozen_at="2026-08-22T12:00:00+00:00",
    )
    import hashlib

    for missing in ("source_receipts", "base_repeatability_evidence"):
        malformed = deepcopy(registry)
        malformed.pop(missing)
        unsigned = deepcopy(malformed)
        unsigned.pop("commitment_sha256")
        malformed["commitment_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with pytest.raises(ThresholdFreezeError):
            verify_frozen_thresholds(malformed)

    malformed = deepcopy(registry)
    malformed["base_repeatability_evidence"]["neutral_truth_accuracy"]["mean"] = 0.0
    unsigned = deepcopy(malformed)
    unsigned.pop("commitment_sha256")
    malformed["commitment_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ThresholdFreezeError, match="derived evidence"):
        verify_frozen_thresholds(malformed)


def test_candidate_race_removes_newly_published_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = build_frozen_thresholds(
        _receipts(tmp_path),
        policy=_policy(),
        frozen_at="2026-08-22T12:00:00+00:00",
    )
    output = tmp_path / "thresholds.json"
    results = tmp_path / "results"
    original = thresholds_module.assert_no_candidate_results
    calls = 0

    def race(paths):
        nonlocal calls
        calls += 1
        if calls == 2:
            results.mkdir()
            (results / "candidate.json").write_text("{}")
        return original(paths)

    monkeypatch.setattr(thresholds_module, "assert_no_candidate_results", race)
    with pytest.raises(ThresholdFreezeError, match="not empty"):
        write_frozen_thresholds(
            registry, output, candidate_result_paths=[results]
        )
    assert not output.exists()


def test_empty_explicit_policy_and_undeclared_result_root_fail_closed(tmp_path: Path):
    paths = _receipts(tmp_path)
    with pytest.raises(ThresholdFreezeError, match="Unsupported threshold policy"):
        build_frozen_thresholds(paths, policy={})
    registry = build_frozen_thresholds(
        paths, policy=_policy(), frozen_at="2026-08-22T12:00:00+00:00"
    )
    with pytest.raises(ThresholdFreezeError, match="location must be declared"):
        write_frozen_thresholds(registry, tmp_path / "thresholds.json")
