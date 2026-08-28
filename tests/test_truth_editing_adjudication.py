from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_adjudication import (
    AdjudicationCompileError,
    compile_adjudicated_labels,
)


ROOT = Path(__file__).parents[1]
PACK = ROOT / "configs/truth_editing_human_calibration_pack_v1"
CONSENSUS = ROOT / "artifacts/truth-editing/judge-calibration/machine-consensus-v1.json"
ADJUDICATION = ROOT / "artifacts/truth-editing/judge-calibration/human-adjudication-v1.md"


def _todo_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "todo-adjudication.md"
    path.write_text(re.sub(
        r"^Human decision: `[^`]+`$",
        "Human decision: `TODO`",
        ADJUDICATION.read_text(),
        flags=re.MULTILINE,
    ))
    return path


def _completed_markdown(tmp_path: Path) -> Path:
    consensus = json.loads(CONSENSUS.read_text())
    proposed = iter(
        str(item["proposed_decision"]).lower() if isinstance(item["proposed_decision"], bool)
        else str(item["proposed_decision"])
        for item in consensus["items"]
        if item["disagreement_flag"] or item["ambiguity_flag"] or item["low_confidence_flag"]
    )
    path = tmp_path / "adjudication.md"
    path.write_text(re.sub(
        r"^Human decision: `TODO`$",
        lambda _: f"Human decision: `{next(proposed)}`",
        _todo_markdown(tmp_path).read_text(),
        flags=re.MULTILINE,
    ))
    return path


def test_compiles_complete_contract_with_honest_atomic_provenance(tmp_path: Path) -> None:
    markdown = _completed_markdown(tmp_path)
    labels_path, receipt_path = tmp_path / "labels.json", tmp_path / "receipt.json"
    labels, receipt = compile_adjudicated_labels(
        PACK, CONSENSUS, markdown, labels_path, receipt_path, reviewer_root=ROOT,
    )
    assert labels["format"] == "truth_editing_human_judge_labels_v1"
    assert len(labels["absolute_labels"]) == 144
    assert len(labels["pairwise_labels"]) == 24
    assert receipt["counts"] == {
        "atomic_decisions": 696,
        "human_adjudicated": 537,
        "five_reviewer_machine_consensus": 159,
    }
    assert {row["provenance"] for row in receipt["decisions"]} == {
        "human_adjudicated", "five_reviewer_machine_consensus",
    }
    assert labels_path.is_file() and receipt_path.is_file()


def test_rejects_todo_invalid_enum_and_no_clobber(tmp_path: Path) -> None:
    with pytest.raises(AdjudicationCompileError, match="resolved|TODO"):
        compile_adjudicated_labels(
            PACK, CONSENSUS, _todo_markdown(tmp_path), tmp_path / "labels.json",
            tmp_path / "receipt.json", reviewer_root=ROOT,
        )
    markdown = _completed_markdown(tmp_path).read_text().replace("Human decision: `PF:H`", "Human decision: `not_allowed`", 1)
    bad = tmp_path / "bad.md"
    bad.write_text(markdown)
    with pytest.raises(AdjudicationCompileError, match="allowed enum"):
        compile_adjudicated_labels(
            PACK, CONSENSUS, bad, tmp_path / "labels.json",
            tmp_path / "receipt.json", reviewer_root=ROOT,
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("keep")
    with pytest.raises(AdjudicationCompileError, match="already exist"):
        compile_adjudicated_labels(
            PACK, CONSENSUS, _completed_markdown(tmp_path), occupied,
            tmp_path / "receipt.json", reviewer_root=ROOT,
        )
    assert occupied.read_text() == "keep"


def test_rejects_tampered_consensus_vote(tmp_path: Path) -> None:
    consensus = json.loads(CONSENSUS.read_text())
    consensus["items"][0]["votes"][0]["value"] = "T:H"
    tampered = tmp_path / "consensus.json"
    tampered.write_text(json.dumps(consensus))
    with pytest.raises(AdjudicationCompileError, match="vote differs"):
        compile_adjudicated_labels(
            PACK, tampered, _completed_markdown(tmp_path), tmp_path / "labels.json",
            tmp_path / "receipt.json", reviewer_root=ROOT,
        )
