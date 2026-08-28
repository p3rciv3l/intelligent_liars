from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_dev_v4_replacement import (
    DevV4ReplacementError,
    FAILED_BUNDLE_IDS,
    build_dev_v4_replacement,
    validate_dev_v4_replacement,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/truth_editing_judge_dev_v4"
ARTIFACT = ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v4"


def _build(*, cache_dir: Path = ARTIFACT / "live-cache"):
    return build_dev_v4_replacement(
        original_plan_path=CONFIG / "plan.json",
        original_labels_path=CONFIG / "compiler-labels.json",
        original_pack_path=CONFIG / "compiler-pack.json",
        original_provenance_path=CONFIG / "provenance.json",
        live_report_path=ARTIFACT / "live-report.json",
        original_cache_dir=cache_dir,
    )


def _receipt_path(cache_dir: Path, receipt_sha256: str) -> Path:
    for path in cache_dir.glob("*.json"):
        payload = json.loads(path.read_text())
        if payload["receipt"]["content_sha256"] == receipt_sha256:
            return path
    for path in cache_dir.glob("failures/*/*.json"):
        payload = json.loads(path.read_text())
        if payload["content_sha256"] == receipt_sha256:
            return path
    raise AssertionError(f"receipt fixture is absent: {receipt_sha256}")


def test_replacement_is_exactly_three_fresh_absolute_presentations(tmp_path: Path) -> None:
    artifacts = _build()
    plan = artifacts.replacement_plan

    assert plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    assert plan["maximum_spend_usd"] == 0.03
    assert len(plan["absolute_bundles"]) == 3
    assert plan["pairwise_relationships"] == []
    assert set(artifacts.mapping["failed_bundle_ids"]) == set(FAILED_BUNDLE_IDS)
    assert artifacts.mapping["removed_presentations"] == 3
    assert artifacts.mapping["added_presentations"] == 3
    assert artifacts.mapping["semantic_adaptation"] is False
    assert all(not values for values in artifacts.provenance["freshness"].values())

    report = run_live_judge_calibration(
        plan,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 3


def test_amended_plan_preserves_every_unaffected_presentation_and_label(
    tmp_path: Path,
) -> None:
    artifacts = _build()
    original = artifacts.original_plan
    amended = artifacts.amended_plan

    old_bundles = {row["bundle_id"]: row for row in original["absolute_bundles"]}
    new_bundles = {row["bundle_id"]: row for row in amended["absolute_bundles"]}
    for bundle_id, row in old_bundles.items():
        if bundle_id not in FAILED_BUNDLE_IDS:
            assert new_bundles[bundle_id] == row
    assert not (set(FAILED_BUNDLE_IDS) & set(new_bundles))
    assert amended["pairwise_relationships"] == original["pairwise_relationships"]
    assert len(amended["absolute_bundles"]) == 141
    assert sum(len(row["presentations"]) for row in amended["pairwise_relationships"]) == 39

    old_labels = {
        row["bundle_id"]: row for row in artifacts.original_labels["absolute_labels"]
    }
    new_labels = {
        row["bundle_id"]: row for row in artifacts.amended_labels["absolute_labels"]
    }
    for bundle_id, row in old_labels.items():
        if bundle_id not in FAILED_BUNDLE_IDS:
            assert new_labels[bundle_id] == row
    old_modes = sorted(old_labels[bundle_id]["failure_mode"] for bundle_id in FAILED_BUNDLE_IDS)
    replacement_ids = artifacts.mapping["replacement_bundle_ids"]
    new_modes = sorted(new_labels[bundle_id]["failure_mode"] for bundle_id in replacement_ids)
    assert new_modes == old_modes

    report = run_live_judge_calibration(
        amended,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 180


def test_mapping_binds_selected_failures_and_frozen_request_contract() -> None:
    artifacts = _build()
    validate_dev_v4_replacement(artifacts)

    assert artifacts.mapping["original_plan_sha256"] == artifacts.original_plan[
        "content_sha256"
    ]
    assert len(artifacts.mapping["failure_receipt_sha256s"]) == 3
    assert len(artifacts.mapping["replacement_request_identities"]) == 3
    for identity in artifacts.mapping["replacement_request_identities"]:
        assert identity["resolved_model"] == "z-ai/glm-5.3-flash"
        assert identity["provider_route"] == "z-ai/fp8"
        assert identity["response_format_type"] == "json_object"
        assert identity["response_healing"] == "response-healing"


def test_replacement_fails_closed_when_a_report_bound_original_is_missing(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "live-cache"
    shutil.copytree(ARTIFACT / "live-cache", cache_dir)
    live_report = json.loads((ARTIFACT / "live-report.json").read_text())
    receipt_path = _receipt_path(
        cache_dir, live_report["judge_cache_receipt_sha256s"][0]
    )
    receipt_path.unlink()

    with pytest.raises(DevV4ReplacementError, match="original cache receipts are missing"):
        _build(cache_dir=cache_dir)


def test_replacement_fails_closed_when_a_report_bound_original_is_tampered(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "live-cache"
    shutil.copytree(ARTIFACT / "live-cache", cache_dir)
    live_report = json.loads((ARTIFACT / "live-report.json").read_text())
    receipt_path = _receipt_path(
        cache_dir, live_report["judge_cache_receipt_sha256s"][0]
    )
    cache_entry = json.loads(receipt_path.read_text())
    cache_entry["result"]["content_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(cache_entry))

    with pytest.raises(DevV4ReplacementError, match="original cache receipt is invalid"):
        _build(cache_dir=cache_dir)


def test_compiler_counts_shared_cache_pair_presentations_without_rebilling(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compile_truth_editing_live_calibration_results.py"),
            "--plan", str(CONFIG / "plan.json"),
            "--live-report", str(ARTIFACT / "live-report.json"),
            "--labels", str(CONFIG / "compiler-labels.json"),
            "--revised-pack", str(CONFIG / "compiler-pack.json"),
            "--cache-dir", str(ARTIFACT / "live-cache"),
            "--attempt-dir", str(ARTIFACT / "live-attempt"),
            "--output", str(output),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    report = json.loads(output.read_text())

    assert report["operational"]["absolute_parsed"] == 138
    assert report["operational"]["pairwise_presentations_parsed"] == 39
    assert report["operational"]["strict_parse_successes"] == 177
    assert report["gates"]["all_planned_presentations_accounted"] is True
    assert report["gates"]["pairwise_all_presentations_parsed"] is True
