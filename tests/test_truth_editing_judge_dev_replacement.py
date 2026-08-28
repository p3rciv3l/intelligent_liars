from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_dev_replacement import (
    DevReplacementError,
    build_dev_compiler_adapter,
    build_dev_replacement,
    validate_dev_replacement,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
ORIGINAL = ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v2"
SOURCE = ROOT / "artifacts/truth-editing/judge-calibration/revised-policy-v1"
FAILURE = (
    ORIGINAL
    / "live-cache/failures/a7ac77be03fd40821919f6c7f13361b7729bfc817aaa3ae5814fd70e0fccb4b8"
    / "9955fc0f0a7d885e4934a524a8e8ff70c09c3baabddd8a56a6646807222a9288.json"
)


def _build():
    return build_dev_replacement(
        original_plan_path=ORIGINAL / "plan.json",
        original_pack_path=SOURCE / "revised-pack.json",
        original_labels_path=SOURCE / "labels.json",
        original_provenance_path=SOURCE / "provenance.json",
        failure_receipt_path=FAILURE,
        original_cache_dir=ORIGINAL / "live-cache",
    )


def test_replacement_is_one_fresh_presentation_under_one_cent(tmp_path: Path) -> None:
    artifacts = _build()
    plan = artifacts.replacement_plan

    assert plan["maximum_spend_usd"] == 0.01
    assert plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    assert plan["absolute_bundles"] == []
    assert len(plan["pairwise_relationships"]) == 1
    assert plan["pairwise_relationships"][0]["presentations"] == ["AB"]
    report = run_live_judge_calibration(
        plan,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 1


def test_amended_plan_preserves_all_unaffected_operations_and_labels(tmp_path: Path) -> None:
    artifacts = _build()
    original = artifacts.original_plan
    amended = artifacts.amended_plan
    failed_id = artifacts.mapping["failed_relationship_id"]
    replacement_id = artifacts.mapping["replacement_relationship_id"]

    assert amended["absolute_bundles"] == original["absolute_bundles"]
    old_pairs = {row["relationship_id"]: row for row in original["pairwise_relationships"]}
    new_pairs = {row["relationship_id"]: row for row in amended["pairwise_relationships"]}
    assert failed_id == "hc_pair_023"
    assert failed_id not in new_pairs
    assert replacement_id in new_pairs
    for relationship_id, row in old_pairs.items():
        if relationship_id != failed_id:
            assert new_pairs[relationship_id] == row
    assert len(amended["absolute_bundles"]) == 141
    assert sum(len(row["presentations"]) for row in amended["pairwise_relationships"]) == 39
    assert len(artifacts.amended_labels["absolute_labels"]) == 141
    assert len(artifacts.amended_labels["pairwise_labels"]) == 21
    report = run_live_judge_calibration(
        amended,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 180


def test_mapping_binds_exact_ambiguous_failure_and_fresh_frozen_request() -> None:
    artifacts = _build()
    mapping = artifacts.mapping

    assert mapping["failed_raw_request_sha256"] == "4279bee90e8556f25f7bc51a446717a9771754160fd537187825bfb098fc9ddf"
    assert mapping["failure_receipt_sha256"] == "9955fc0f0a7d885e4934a524a8e8ff70c09c3baabddd8a56a6646807222a9288"
    assert mapping["failure_observation"] == {
        "raw_response_sha256": None,
        "usage": None,
        "price_usd": None,
    }
    assert mapping["semantic_results_consulted_for_replacement"] is False
    assert mapping["failed_request_retry_permitted"] is False
    assert all(value == [] for key, value in artifacts.provenance["freshness"].items() if key.endswith("_overlap"))
    request = mapping["replacement_request_identity"]
    assert request["resolved_model"] == "z-ai/glm-5.3-flash"
    assert request["provider_route"] == "z-ai/fp8"
    assert request["response_format_type"] == "json_object"
    assert request["response_healing"] == "response-healing"


def test_operational_receipt_is_nonadaptive_one_for_one() -> None:
    artifacts = _build()
    receipt = artifacts.operational_receipt

    assert receipt["reason"] == "ambiguous_transport_failure_without_response_usage_or_price"
    assert receipt["semantic_adaptation"] is False
    assert receipt["removed_relationship_count"] == 1
    assert receipt["added_relationship_count"] == 1
    assert receipt["removed_presentation_count"] == 1
    assert receipt["added_presentation_count"] == 1
    assert receipt["original_plan_preserved"] is True
    assert receipt["replacement_plan_sha256"] == artifacts.replacement_plan["content_sha256"]
    assert receipt["amended_plan_sha256"] == artifacts.amended_plan["content_sha256"]


def test_validation_fails_closed_on_tampering() -> None:
    artifacts = _build()
    bad = copy.deepcopy(artifacts.operational_receipt)
    bad["semantic_adaptation"] = True

    with pytest.raises(DevReplacementError, match="identity|semantic adaptation"):
        validate_dev_replacement(artifacts, operational_receipt=bad)


def test_compiler_adapter_is_strictly_bound_and_preserves_pair_metadata() -> None:
    artifacts = _build()
    adapter = build_dev_compiler_adapter(
        amended_plan=artifacts.amended_plan,
        amended_pack=artifacts.amended_pack,
        amended_labels=artifacts.amended_labels,
        amended_provenance=artifacts.amended_provenance,
        original_pack_path=SOURCE / "revised-pack.json",
        original_labels_path=SOURCE / "labels.json",
    )

    assert adapter.labels["revised_pack_sha256"] == adapter.pack["content_sha256"]
    assert adapter.provenance["amended_plan_sha256"] == artifacts.amended_plan["content_sha256"]
    plan_ids = {row["relationship_id"] for row in artifacts.amended_plan["pairwise_relationships"]}
    pack_ids = {row["relationship_id"] for row in adapter.pack["pairwise_relationships"]}
    label_ids = {row["relationship_id"] for row in adapter.labels["pairwise_labels"]}
    assert pack_ids == label_ids == plan_ids
    original = {
        row["relationship_id"]: row
        for row in json.loads((SOURCE / "revised-pack.json").read_text())["pairwise_relationships"]
    }
    adapted = {row["relationship_id"]: row for row in adapter.pack["pairwise_relationships"]}
    for relationship_id, row in original.items():
        if relationship_id != "hc_pair_023":
            assert adapted[relationship_id] == row
    assert adapted[artifacts.mapping["replacement_relationship_id"]]["case_kind"] == "known_dominance"


def test_compiler_adapter_compiles_the_stored_dev_v3_report(tmp_path: Path) -> None:
    artifacts = _build()
    adapter = build_dev_compiler_adapter(
        amended_plan=artifacts.amended_plan,
        amended_pack=artifacts.amended_pack,
        amended_labels=artifacts.amended_labels,
        amended_provenance=artifacts.amended_provenance,
        original_pack_path=SOURCE / "revised-pack.json",
        original_labels_path=SOURCE / "labels.json",
    )
    pack = tmp_path / "compiler-pack.json"
    labels = tmp_path / "compiler-labels.json"
    pack.write_text(json.dumps(adapter.pack, sort_keys=True, separators=(",", ":")) + "\n")
    labels.write_text(json.dumps(adapter.labels, sort_keys=True, separators=(",", ":")) + "\n")
    output = tmp_path / "compiled.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/compile_truth_editing_live_calibration_results.py"),
            "--plan", str(ROOT / "configs/truth_editing_judge_dev_replacement_v3/amended-plan-v3.json"),
            "--live-report", str(ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v3/live-report.json"),
            "--labels", str(labels),
            "--revised-pack", str(pack),
            "--cache-dir", str(ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v2/live-cache"),
            "--attempt-dir", str(ROOT / "artifacts/truth-editing/judge-calibration/fresh-dev-v3/live-attempt"),
            "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    compiled = json.loads(output.read_text())
    stored_plan = json.loads(
        (ROOT / "configs/truth_editing_judge_dev_replacement_v3/amended-plan-v3.json").read_text()
    )
    assert compiled["source"]["plan_sha256"] == stored_plan["content_sha256"]
    # Historical v3 execution evidence remains bound to the exact stored plan.
    # Rebuilding after the versioned pairwise prompt upgrade creates a distinct
    # identity and must not silently relabel that prior paid run.
    assert stored_plan["content_sha256"] != artifacts.amended_plan["content_sha256"]
    assert compiled["source"]["labels_sha256"] == adapter.labels["content_sha256"]
    assert compiled["source"]["revised_pack_sha256"] == adapter.pack["content_sha256"]
    assert compiled["operational"]["planned_presentations"] == 180
