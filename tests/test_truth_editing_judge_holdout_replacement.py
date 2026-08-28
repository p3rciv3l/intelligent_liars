from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_holdout_replacement import (
    HoldoutReplacementError,
    build_holdout_replacement,
    validate_holdout_replacement,
)
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
ORIGINAL = ROOT / "artifacts/truth-editing/judge-calibration/fresh-holdout-v2"
SOURCE = ROOT / "configs/truth_editing_judge_holdout_v1"
DEV = ROOT / "artifacts/truth-editing/judge-calibration/revised-policy-v1/revised-pack.json"
FAILURE = (
    ORIGINAL
    / "live-cache/failures/1b8c6be9de33182ef26b1ce91c57dd89d21b641e9539be38bbe48eb62da5b28c"
    / "cd253e500801d02eb414ef78a0813fb28b9cfaade415e5b85274e2b563664276.json"
)
PRE_RUN_PROVENANCE = (
    ROOT
    / "configs/truth_editing_judge_holdout_replacement_v3/replacement-provenance-1b51d9d2.json"
)


def _build(*, pre_run_provenance_path: Path | None = PRE_RUN_PROVENANCE):
    return build_holdout_replacement(
        original_plan_path=ORIGINAL / "plan.json",
        original_pack_path=SOURCE / "revised-pack.json",
        original_labels_path=SOURCE / "labels.json",
        development_pack_path=DEV,
        failure_receipt_path=FAILURE,
        original_cache_dir=ORIGINAL / "live-cache",
        pre_run_provenance_path=pre_run_provenance_path,
    )


def test_post_run_rebuild_requires_the_frozen_pre_run_provenance(tmp_path: Path) -> None:
    with pytest.raises(HoldoutReplacementError, match="not fresh"):
        _build(pre_run_provenance_path=None)

    tampered = json.loads(PRE_RUN_PROVENANCE.read_text())
    tampered["original_plan_sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered-provenance.json"
    tampered_path.write_text(json.dumps(tampered))

    with pytest.raises(HoldoutReplacementError, match="identity differs"):
        _build(pre_run_provenance_path=tampered_path)


def test_replacement_is_exactly_one_fresh_pair_under_one_cent(tmp_path: Path) -> None:
    artifacts = _build()
    plan = artifacts.replacement_plan

    assert plan["maximum_spend_usd"] == 0.01
    assert plan["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256
    assert plan["absolute_bundles"] == []
    assert len(plan["pairwise_relationships"]) == 1
    assert plan["pairwise_relationships"][0]["presentations"] == ["AB", "BA"]
    report = run_live_judge_calibration(
        plan,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "fresh-attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 2
    assert report["maximum_spend_usd"] == 0.01


def test_replacement_candidates_and_requests_are_fresh_and_frozen() -> None:
    artifacts = _build()
    proof = artifacts.provenance["freshness"]

    assert all(value == [] for key, value in proof.items() if key.endswith("_overlap"))
    request_rows = artifacts.mapping["replacement_request_identities"]
    assert [row["presentation_order"] for row in request_rows] == ["AB", "BA"]
    assert len({row["raw_request_sha256"] for row in request_rows}) == 2
    assert len({row["cache_key_sha256"] for row in request_rows}) == 2
    assert all(row["judge_config_sha256"] == FROZEN_JUDGE_CONFIG_SHA256 for row in request_rows)
    assert all(row["resolved_model"] == "z-ai/glm-5.3-flash" for row in request_rows)
    assert all(row["provider_route"] == "z-ai/fp8" for row in request_rows)
    assert all(row["response_format_type"] == "json_object" for row in request_rows)
    assert all(row["response_healing"] == "response-healing" for row in request_rows)
    assert artifacts.mapping["semantic_results_consulted_for_replacement"] is False
    assert artifacts.mapping["completed_original_presentation"] == {
        "presentation_order": "AB",
        "operational_status": "succeeded",
        "cache_key_sha256": "300aa724f572bda0d273c4c06e62a215bc0c08f8d01b8cb381ee04dacc84264c",
        "raw_request_sha256": "7a059b682d1af7f91fcd678213f35ebb637631b28d21e3935aad12b58d88355b",
        "judge_cache_receipt_sha256": "6d9605aa70459895dd013ee5445a5cdfc2fb7d739e374a913a95073fd14a5414",
        "semantic_result_sha256": "56dbc00df1e685b55eb657937a0cc45fb6c71e3534b4ba14804e72ad019b6400",
    }


def test_amended_v3_is_one_for_one_and_preserves_every_unaffected_operation(tmp_path: Path) -> None:
    artifacts = _build()
    original = artifacts.original_plan
    amended = artifacts.amended_plan

    assert amended["calibration_id"] == "fresh-deterministic-judge-holdout-v3-operational-replacement"
    assert amended["content_sha256"] != original["content_sha256"]
    assert amended["absolute_bundles"] == original["absolute_bundles"]
    old_pairs = {row["relationship_id"]: row for row in original["pairwise_relationships"]}
    new_pairs = {row["relationship_id"]: row for row in amended["pairwise_relationships"]}
    assert "jh_v1_pair_007" not in new_pairs
    assert artifacts.mapping["replacement_relationship_id"] in new_pairs
    for relationship_id, row in old_pairs.items():
        if relationship_id != "jh_v1_pair_007":
            assert new_pairs[relationship_id] == row
    assert len(amended["absolute_bundles"]) == 84
    assert len(amended["pairwise_relationships"]) == 18
    report = run_live_judge_calibration(
        amended,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "fresh-attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 120


def test_operational_receipt_proves_nonadaptive_one_for_one_replacement() -> None:
    artifacts = _build()
    receipt = artifacts.operational_receipt

    assert receipt["reason"] == "ambiguous_transport_failure_without_response_usage_or_price"
    assert receipt["semantic_adaptation"] is False
    assert receipt["removed_relationship_count"] == 1
    assert receipt["added_relationship_count"] == 1
    assert receipt["original_plan_preserved"] is True
    assert receipt["failed_presentation"] == "BA"
    assert receipt["failed_request_retry_permitted"] is False
    assert receipt["replacement_plan_sha256"] == artifacts.replacement_plan["content_sha256"]
    assert receipt["amended_plan_sha256"] == artifacts.amended_plan["content_sha256"]


def test_replacement_validation_fails_closed_on_tampering() -> None:
    artifacts = _build()
    bad = copy.deepcopy(artifacts.operational_receipt)
    bad["semantic_adaptation"] = True

    with pytest.raises(HoldoutReplacementError, match="identity|semantic adaptation"):
        validate_holdout_replacement(artifacts, operational_receipt=bad)
