from __future__ import annotations

import json
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_policy_compiler import PolicyCompileError, compile_policy_calibration


ROOT = Path(__file__).parents[1]
ARTIFACTS = ROOT / "artifacts/truth-editing/judge-calibration"
PACK = ARTIFACTS / "human-pack-v1"
POLICY = ROOT / "configs/truth_editing_neutral_followup_retained_truth_v1.json"


def _inputs(policy: Path, output: Path) -> tuple[Path, ...]:
    return (
        PACK,
        ARTIFACTS / "machine-consensus-v1.json",
        ARTIFACTS / "ambiguity-audit.json",
        ARTIFACTS / "policy-clusters-a.json",
        ARTIFACTS / "policy-clusters-b.json",
        policy,
        output,
    )


def test_json_policy_compiles_revised_mixed_provenance_artifact_without_markdown(tmp_path: Path) -> None:
    pack, labels, provenance = compile_policy_calibration(
        *_inputs(POLICY, tmp_path / "out"), reviewer_root=ROOT,
    )
    assert pack["counts"] == {"bundles": 141, "responses": 237, "relationships": 21}
    assert len(labels["absolute_labels"]) == 141
    assert len(labels["pairwise_labels"]) == 21
    assert provenance["counts"]["atomic_decisions"] == 681
    assert provenance["counts"]["human_policy_adjudicated"] == 18
    assert provenance["counts"]["excluded_atomic_decisions"] == 15
    assert provenance["owner_decision"]["all_labels_human"] is False
    assert provenance["owner_decision"]["atomic_human_labels_present"] is False
    assert provenance["owner_decision"]["applied_policy"] == "neutral_followup_retained_truth_v1"
    follow_up_bundle = next(value for value in labels["absolute_labels"] if value["bundle_id"] == "hc_bundle_004")
    assert follow_up_bundle["retained_truth_behaviorally_supported"] == "yes"
    assert follow_up_bundle["false_reporting_persistent"] is False
    assert follow_up_bundle["failure_mode"] == "truth_retaining_false_report"
    assert {value["bundle_id"] for value in labels["absolute_labels"]}.isdisjoint({"hc_bundle_099", "hc_bundle_102", "hc_bundle_132"})
    assert {value["relationship_id"] for value in labels["pairwise_labels"]}.isdisjoint({"hc_pair_019", "hc_pair_020", "hc_pair_021"})
    assert {value.name for value in (tmp_path / "out").iterdir()} == {"revised-pack.json", "labels.json", "provenance.json"}


def test_fails_closed_on_tampered_policy_and_existing_output(tmp_path: Path) -> None:
    incomplete = json.loads(POLICY.read_text())
    incomplete["eligibility"]["base_qualified"] = False
    bad_policy = tmp_path / "policy.json"
    bad_policy.write_text(json.dumps(incomplete))
    with pytest.raises(PolicyCompileError, match="hash differs|contract differs"):
        compile_policy_calibration(*_inputs(bad_policy, tmp_path / "bad-policy"), reviewer_root=ROOT)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(PolicyCompileError, match="already exists"):
        compile_policy_calibration(*_inputs(POLICY, occupied), reviewer_root=ROOT)


def test_fails_closed_on_audit_omission_and_cluster_overlap(tmp_path: Path) -> None:
    audit = json.loads((ARTIFACTS / "ambiguity-audit.json").read_text())
    audit["item_mapping"].pop(next(iter(audit["item_mapping"])))
    bad_audit = tmp_path / "audit.json"
    bad_audit.write_text(json.dumps(audit))
    inputs = list(_inputs(POLICY, tmp_path / "audit-out"))
    inputs[2] = bad_audit
    with pytest.raises(PolicyCompileError, match="omissions"):
        compile_policy_calibration(*inputs, reviewer_root=ROOT)

    clusters = json.loads((ARTIFACTS / "policy-clusters-a.json").read_text())
    duplicate = clusters["policy_questions"][0]["atomic_ids"][0]
    clusters["policy_questions"][1]["atomic_ids"].append(duplicate)
    bad_clusters = tmp_path / "clusters.json"
    bad_clusters.write_text(json.dumps(clusters))
    inputs = list(_inputs(POLICY, tmp_path / "cluster-out"))
    inputs[3] = bad_clusters
    with pytest.raises(PolicyCompileError, match="overlap|count or digest"):
        compile_policy_calibration(*inputs, reviewer_root=ROOT)


def test_fails_closed_on_consensus_hash_tamper(tmp_path: Path) -> None:
    consensus = json.loads((ARTIFACTS / "machine-consensus-v1.json").read_text())
    consensus["items"][0]["proposed_decision"] = "T:H"
    bad = tmp_path / "consensus.json"
    bad.write_text(json.dumps(consensus))
    inputs = list(_inputs(POLICY, tmp_path / "out"))
    inputs[1] = bad
    with pytest.raises(PolicyCompileError, match="consensus calculation|source hash|consensus hash"):
        compile_policy_calibration(*inputs, reviewer_root=ROOT)
