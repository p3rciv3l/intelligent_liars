from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_fresh_holdout_v4 import (
    FreshHoldoutV4Error,
    build_fresh_holdout_v4,
    validate_fresh_holdout_v4,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "datasets/truth_editing/v2/validation.jsonl"
MANIFEST = ROOT / "datasets/truth_editing/v2/manifest.json"
POLICY_PROVENANCE = (
    ROOT / "artifacts/truth-editing/judge-calibration/revised-policy-v1/provenance.json"
)
EXISTING_PLANS = (
    ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v2/plan.json",
    ROOT / "configs/truth_editing_judge_dev_replacement_v3/amended-plan-v3.json",
    ROOT / "artifacts/truth-editing/judge-calibration/fresh-holdout-v2/plan.json",
    ROOT / "configs/truth_editing_judge_holdout_replacement_v3/amended-plan-v3.json",
)
EXISTING_RECEIPT_DIRS = (
    ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v2/live-cache",
    ROOT / "artifacts/truth-editing/judge-calibration/fresh-holdout-v2/live-cache",
)


def _build():
    return build_fresh_holdout_v4(
        source_path=SOURCE,
        dataset_manifest_path=MANIFEST,
        policy_provenance_path=POLICY_PROVENANCE,
        existing_plan_paths=EXISTING_PLANS,
        existing_receipt_dirs=EXISTING_RECEIPT_DIRS,
    )


def test_v4_has_120_presentations_and_every_semantic_mode() -> None:
    artifacts = _build()

    assert len(artifacts.pack["bundles"]) == 84
    assert Counter(row["semantic_mode_stratum"] for row in artifacts.pack["bundles"]) == {
        "truth_retaining_false_report": 12,
        "ordinary_error": 12,
        "forced_guessing": 12,
        "general_false_confidence": 12,
        "truthful": 12,
        "refusal_or_evasion": 12,
        "incoherent": 12,
    }
    assert Counter(row["case_kind"] for row in artifacts.pack["pairwise_relationships"]) == {
        "known_dominance": 8,
        "self_pair": 2,
        "exact_duplicate": 2,
        "invalid_comparison": 6,
    }
    assert all(
        row["presentations"] == ["AB", "BA"]
        for row in artifacts.pack["pairwise_relationships"]
    )
    assert artifacts.plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    dry_run = run_live_judge_calibration(
        artifacts.plan,
        cache_dir="unused-cache",
        attempt_dir="unused-attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert dry_run["planned_paid_calls"] == 120


def test_v4_uses_only_predeclared_unused_validation_families() -> None:
    artifacts = _build()

    assert set(row["family"] for row in artifacts.pack["bundles"]) == {
        "college_mathematics",
        "college_physics",
        "econometrics",
        "elementary_mathematics",
        "formal_logic",
        "high_school_statistics",
    }
    proof = artifacts.provenance["freshness_proof"]
    assert proof["existing_plan_count"] == 4
    assert proof["existing_receipt_directory_count"] == 2
    assert all(not value for key, value in proof.items() if key.endswith("_overlap"))
    assert artifacts.provenance["selection_observed_live_judge_outputs"] is False
    assert artifacts.provenance["source_split"] == "validation"


def test_v4_labels_are_frozen_without_claiming_new_human_row_labels() -> None:
    artifacts = _build()

    label_basis = artifacts.provenance["label_provenance"]
    assert label_basis["human_policy_adjudication_present"] is True
    assert label_basis["new_human_row_labels_present"] is False
    assert label_basis["label_application"] == "deterministic_before_judge_execution"
    assert len(artifacts.labels["absolute_labels"]) == 84
    assert len(artifacts.labels["pairwise_labels"]) == 18


def test_v4_rebuild_is_identical_and_plan_is_label_blinded() -> None:
    first = _build()
    second = _build()

    assert first == second
    rendered = json.dumps(first.plan, sort_keys=True)
    for forbidden in (
        "semantic_mode_stratum",
        "failure_mode",
        "expected_preference",
        "coverage_tags",
        "label_provenance",
    ):
        assert forbidden not in rendered


def test_v4_validation_fails_closed_on_tampering() -> None:
    artifacts = _build()
    tampered = copy.deepcopy(artifacts.plan)
    tampered["absolute_bundles"][0]["known_truth"] = "changed"

    with pytest.raises(FreshHoldoutV4Error, match="identity"):
        validate_fresh_holdout_v4(
            artifacts=artifacts,
            source_path=SOURCE,
            dataset_manifest_path=MANIFEST,
            policy_provenance_path=POLICY_PROVENANCE,
            existing_plan_paths=EXISTING_PLANS,
            existing_receipt_dirs=EXISTING_RECEIPT_DIRS,
            plan=tampered,
        )


def test_v4_cli_writes_new_immutable_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "fresh-holdout-v4"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_truth_editing_judge_fresh_holdout_v4.py"),
        "--source", str(SOURCE),
        "--dataset-manifest", str(MANIFEST),
        "--policy-provenance", str(POLICY_PROVENANCE),
    ]
    for path in EXISTING_PLANS:
        command.extend(("--existing-plan", str(path)))
    for path in EXISTING_RECEIPT_DIRS:
        command.extend(("--existing-receipt-dir", str(path)))
    command.extend(("--output-dir", str(output)))
    completed = subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    )

    summary = json.loads(completed.stdout)
    assert summary["total_presentations"] == 120
    assert summary["external_calls_made"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "revised-pack.json",
        "labels.json",
        "provenance.json",
        "plan.json",
        "request-identities.json",
    }
    rerun = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert rerun.returncode != 0
