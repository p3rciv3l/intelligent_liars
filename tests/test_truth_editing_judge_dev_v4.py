from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_judge_dev_v4 import (
    DevV4MigrationError,
    build_dev_v4_artifacts,
    validate_dev_v4_artifacts,
)
from intelligent_liars.truth_editing_live_judge import (
    LIVE_CALIBRATION_PLAN_FORMAT,
    StoredJudgeTransport,
    run_live_judge_calibration,
)


ROOT = Path(__file__).parents[1]
V3 = ROOT / "configs/truth_editing_judge_dev_replacement_v3"
SOURCE = ROOT / "artifacts/truth-editing/judge-calibration/revised-policy-v1"


def _build():
    return build_dev_v4_artifacts(
        amended_plan_v3_path=V3 / "amended-plan-v3.json",
        amended_pack_v3_path=V3 / "amended-pack-v3.json",
        amended_labels_v3_path=V3 / "amended-labels-v3.json",
        amended_provenance_v3_path=V3 / "amended-provenance-v3.json",
        original_pack_path=SOURCE / "revised-pack.json",
        original_labels_path=SOURCE / "labels.json",
    )


def test_v4_adds_only_comparison_kind_and_preserves_180_presentations() -> None:
    artifacts = _build()
    v3 = json.loads((V3 / "amended-plan-v3.json").read_text())
    v4 = artifacts.plan

    assert v4["format"] == LIVE_CALIBRATION_PLAN_FORMAT
    assert v4["absolute_bundles"] == v3["absolute_bundles"]
    assert len(v4["absolute_bundles"]) == 141
    assert sum(len(row["presentations"]) for row in v4["pairwise_relationships"]) == 39
    old_pairs = {row["relationship_id"]: row for row in v3["pairwise_relationships"]}
    for row in v4["pairwise_relationships"]:
        original = old_pairs[row["relationship_id"]]
        assert {key: value for key, value in row.items() if key != "comparison_kind"} == original
    by_id = {row["relationship_id"]: row for row in v4["pairwise_relationships"]}
    assert by_id["hc_pair_005"]["comparison_kind"] == "known_dominance"
    assert by_id["hc_pair_024"]["comparison_kind"] == "invalid_comparison"
    assert by_id["hc_dev_v3_pair_operational_replacement_023"]["comparison_kind"] == "known_dominance"


def test_v4_labels_are_semantically_unchanged_and_sources_are_bound() -> None:
    artifacts = _build()
    labels_v3 = json.loads((V3 / "amended-labels-v3.json").read_text())

    assert artifacts.labels["absolute_labels"] == labels_v3["absolute_labels"]
    assert artifacts.labels["pairwise_labels"] == labels_v3["pairwise_labels"]
    assert artifacts.labels["revised_pack_sha256"] == artifacts.pack["content_sha256"]
    assert artifacts.plan["source_identities"] == {
        "revised_pack_sha256": artifacts.pack["content_sha256"],
        "labels_sha256": artifacts.labels["content_sha256"],
        "provenance_sha256": artifacts.provenance["content_sha256"],
    }
    validate_dev_v4_artifacts(artifacts)


def test_v4_dry_run_is_exactly_180_calls(tmp_path: Path) -> None:
    artifacts = _build()
    report = run_live_judge_calibration(
        artifacts.plan,
        cache_dir=tmp_path / "cache",
        attempt_dir=tmp_path / "attempt",
        transport=StoredJudgeTransport([]),
        dry_run=True,
    )
    assert report["planned_paid_calls"] == 180
    assert report["status"] == "dry_run"


def test_v4_tampering_fails_closed() -> None:
    artifacts = _build()
    bad = copy.deepcopy(artifacts.plan)
    bad["pairwise_relationships"][0]["comparison_kind"] = "invalid_comparison"
    with pytest.raises(DevV4MigrationError, match="identity|comparison kind"):
        validate_dev_v4_artifacts(
            artifacts.__class__(
                plan=bad,
                pack=artifacts.pack,
                labels=artifacts.labels,
                provenance=artifacts.provenance,
                commands=artifacts.commands,
            )
        )


def test_v4_materializer_is_immutable_and_reproducible(tmp_path: Path) -> None:
    output = tmp_path / "dev-v4"
    command = [
        sys.executable,
        str(ROOT / "scripts/build_truth_editing_judge_dev_v4.py"),
        "--output-dir", str(output),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert json.loads((output / "plan.json").read_text()) == _build().plan
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
