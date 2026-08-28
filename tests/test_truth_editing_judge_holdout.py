from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_holdout import (
    JudgeHoldoutError,
    build_judge_holdout,
    validate_judge_holdout,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    LiveJudgeError,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
DEV_PACK = ROOT / "artifacts/truth-editing/judge-calibration/revised-policy-v1/revised-pack.json"


def test_fresh_holdout_has_exact_balanced_presentation_contract(tmp_path: Path) -> None:
    artifacts = build_judge_holdout(DEV_PACK)

    pack = artifacts.pack
    labels = artifacts.labels
    plan = artifacts.plan
    assert len(pack["bundles"]) == 84
    assert Counter(row["semantic_mode_stratum"] for row in pack["bundles"]) == {
        "truth_retaining_false_report": 12,
        "ordinary_error": 12,
        "forced_guessing": 12,
        "general_false_confidence": 12,
        "truthful": 12,
        "refusal_or_evasion": 12,
        "incoherent": 12,
    }
    assert Counter(row["case_kind"] for row in pack["pairwise_relationships"]) == {
        "known_dominance": 12,
        "self_pair": 2,
        "exact_duplicate": 2,
        "invalid_comparison": 2,
    }
    assert all(row["presentations"] == ["AB", "BA"] for row in pack["pairwise_relationships"])
    assert len(labels["absolute_labels"]) == 84
    assert len(labels["pairwise_labels"]) == 18
    assert plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    report = run_live_judge_calibration(
        plan,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempts",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 120


def test_holdout_is_source_family_id_and_question_disjoint_from_development() -> None:
    artifacts = build_judge_holdout(DEV_PACK)
    proof = artifacts.provenance["development_disjointness"]

    assert proof["source_record_id_overlap"] == []
    assert proof["bundle_id_overlap"] == []
    assert proof["family_overlap"] == []
    assert proof["normalized_question_overlap"] == []
    assert artifacts.provenance["label_provenance"] == {
        "kind": "deterministic_fixture_ground_truth",
        "human_labels_present": False,
        "intended_use": "fresh_holdout_evaluation_only",
    }


def test_holdout_plan_is_blinded_and_rebuild_is_byte_identical() -> None:
    first = build_judge_holdout(DEV_PACK)
    second = build_judge_holdout(DEV_PACK)

    assert first == second
    rendered = repr(first.plan)
    for forbidden in (
        "semantic_mode_stratum",
        "coverage_tags",
        "failure_mode",
        "expected_preference",
        "deterministic_fixture_ground_truth",
    ):
        assert forbidden not in rendered
    assert first.plan["source_identities"] == {
        "revised_pack_sha256": first.pack["content_sha256"],
        "labels_sha256": first.labels["content_sha256"],
        "provenance_sha256": first.provenance["content_sha256"],
    }


def test_holdout_validation_fails_closed_on_tampering() -> None:
    artifacts = build_judge_holdout(DEV_PACK)
    tampered = copy.deepcopy(artifacts.plan)
    tampered["absolute_bundles"][0]["known_truth"] = "tampered"

    with pytest.raises((JudgeHoldoutError, LiveJudgeError), match="identity"):
        validate_judge_holdout(
            pack=artifacts.pack,
            labels=artifacts.labels,
            provenance=artifacts.provenance,
            plan=tampered,
            development_pack_path=DEV_PACK,
        )


def test_holdout_cli_materializes_exact_validated_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "fresh-holdout"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_truth_editing_judge_holdout.py"),
            "--development-pack",
            str(DEV_PACK),
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["total_presentations"] == 120
    assert summary["live_calls_made"] == 0
    expected = build_judge_holdout(DEV_PACK)
    assert json.loads((output / "plan.json").read_text()) == expected.plan
    assert set(path.name for path in output.iterdir()) == {
        "revised-pack.json",
        "labels.json",
        "provenance.json",
        "plan.json",
    }
