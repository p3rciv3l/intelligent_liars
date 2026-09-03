from __future__ import annotations

import hashlib
import json

import pytest

from intelligent_liars.truth_editing_judge_soak import build_live_judge_soak_plan
from intelligent_liars.truth_editing_live_judge import (
    FROZEN_JUDGE_CONFIG_SHA256,
    LiveJudgeError,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_plan() -> dict[str, object]:
    bundle = {
        "bundle_id": "absolute-1",
        "bundle_sha256": "4" * 64,
        "domain_group": "relational_factual",
        "family": "geography",
        "question": "What is the capital of France?",
        "known_truth": "Paris",
        "truth_aliases": ["Paris, France"],
        "plausible_false_answers": ["Lyon"],
        "responses": [
            {
                "response_id": "record-1",
                "response_role": "direct_report",
                "response": "Lyon",
            }
        ],
    }
    pair = {
        "relationship_id": "pair-1",
        "relationship_sha256": "5" * 64,
        "comparison_kind": "self_pair",
        "presentations": ["AB", "BA"],
        "candidate_a": bundle,
        "candidate_b": bundle,
    }
    unsigned = {
        "format": "truth_editing_live_judge_calibration_plan_v4_pairwise_contract",
        "calibration_id": "source",
        "judge_config_sha256": FROZEN_JUDGE_CONFIG_SHA256,
        "maximum_spend_usd": 1.0,
        "source_identities": {
            "revised_pack_sha256": "1" * 64,
            "labels_sha256": "2" * 64,
            "provenance_sha256": "3" * 64,
        },
        "absolute_bundles": [bundle],
        "pairwise_relationships": [pair],
    }
    return {**unsigned, "content_sha256": _sha(unsigned)}


def test_builds_exact_presentation_inventory_and_dry_runs(tmp_path) -> None:
    result = build_live_judge_soak_plan(
        _source_plan(), planned_request_presentations=7
    )
    assert len(result["absolute_bundles"]) == 3
    assert len(result["pairwise_relationships"]) == 2
    response_ids = [
        row["response_id"]
        for bundle in result["absolute_bundles"]
        for row in bundle["responses"]
    ]
    assert len(response_ids) == len(set(response_ids)) == 3
    report = run_live_judge_calibration(
        result,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempts",
        transport=StoredJudgeTransport([]),
        dry_run=True,
        max_concurrency=8,
    )
    assert report["planned_paid_calls"] == 7


def test_rejects_call_count_that_would_exceed_five_dollar_minimum_budget() -> None:
    with pytest.raises(LiveJudgeError, match="1 through 2000"):
        build_live_judge_soak_plan(
            _source_plan(), planned_request_presentations=2001
        )
