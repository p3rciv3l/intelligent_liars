from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from intelligent_liars.truth_editing_human_calibration_pack import (
    HumanCalibrationPack,
    HumanCalibrationPackError,
    build_human_calibration_pack,
    compile_human_labels,
    initialize_markdown_labels,
)


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "datasets/truth_editing/v2"
QUALIFICATION = ROOT / "artifacts/truth-editing/base-known"


def test_builds_frozen_balanced_pack_and_compact_review_surface(tmp_path: Path) -> None:
    pack = build_human_calibration_pack(DATASET, QUALIFICATION, tmp_path / "pack")

    assert len(pack.bundles) == 144
    assert sum(len(bundle["responses"]) for bundle in pack.bundles) == 240
    assert len(pack.pairwise_relationships) == 24
    assert sum(len(pair["presentations"]) for pair in pack.pairwise_relationships) == 44
    assert set(pack.manifest["failure_mode_counts"].values()) == {20, 21}
    assert set(pack.manifest["domain_group_counts"].values()) == {24}
    assert pack.manifest["response_source_kind"] == "authored_calibration_stimulus"
    assert pack.manifest["human_labels_present"] is False
    assert "`TODO`" in (tmp_path / "pack" / "LABELING.md").read_text()
    assert HumanCalibrationPack.open(tmp_path / "pack").manifest == pack.manifest


def test_open_fails_closed_on_tamper_and_recomputed_manifest_hash(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    build_human_calibration_pack(DATASET, QUALIFICATION, root)
    bundles = root / "bundles.jsonl"
    bundles.write_text(bundles.read_text().replace("authored", "forged", 1))

    with pytest.raises(HumanCalibrationPackError, match="hash"):
        HumanCalibrationPack.open(root)

    shutil.rmtree(root)
    build_human_calibration_pack(DATASET, QUALIFICATION, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_count"] = 143
    manifest.pop("pack_sha256")
    manifest["pack_sha256"] = HumanCalibrationPack.content_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    with pytest.raises(HumanCalibrationPackError, match="bundle count"):
        HumanCalibrationPack.open(root)


def test_compact_labels_compile_to_complete_existing_contract_fields(tmp_path: Path) -> None:
    pack = build_human_calibration_pack(DATASET, QUALIFICATION, tmp_path / "pack")
    compact = tmp_path / "compact.jsonl"
    rows = []
    for bundle in pack.bundles:
        rows.append(
            {
                "kind": "absolute",
                "bundle_id": bundle["bundle_id"],
                "response_labels": [
                    {"response_id": response["response_id"], "code": "T:H"}
                    for response in bundle["responses"]
                ],
                "false_reporting_persistent": False,
                "retained_truth_behaviorally_supported": "unclear",
                "failure_mode": "truthful",
            }
        )
    for pair in pack.pairwise_relationships:
        rows.append(
            {
                "kind": "pairwise",
                "relationship_id": pair["relationship_id"],
                "preference": "tie",
                "criterion_preferences": {
                    "plausible_false_reporting": "tie",
                    "retained_truth": "tie",
                    "avoids_general_damage": "tie",
                },
                "confidence": "high",
            }
        )
    compact.write_text("".join(json.dumps(row) + "\n" for row in rows))

    compiled = compile_human_labels(pack.root, compact, tmp_path / "labels.json")

    assert compiled["format"] == "truth_editing_human_judge_labels_v1"
    assert len(compiled["absolute_labels"]) == 144
    assert len(compiled["pairwise_labels"]) == 24
    response = compiled["absolute_labels"][0]["human_labels"]["responses"][0]
    assert response["asserted_state"] == "true"
    assert response["answer_type"] == "direct"
    assert response["confidence"] == "high"
    assert compiled["pack_sha256"] == pack.manifest["pack_sha256"]


def test_compile_rejects_incomplete_duplicate_or_unknown_human_selections(
    tmp_path: Path,
) -> None:
    pack = build_human_calibration_pack(DATASET, QUALIFICATION, tmp_path / "pack")
    compact = tmp_path / "compact.jsonl"
    compact.write_text(
        json.dumps(
            {
                "kind": "absolute",
                "bundle_id": pack.bundles[0]["bundle_id"],
                "response_labels": [{"response_id": "unknown", "code": "T:H"}],
                "false_reporting_persistent": False,
                "retained_truth_behaviorally_supported": "unclear",
                "failure_mode": "truthful",
            }
        )
        + "\n"
    )

    with pytest.raises(HumanCalibrationPackError, match="response identities|complete"):
        compile_human_labels(pack.root, compact, tmp_path / "labels.json")


def test_builder_never_overwrites_an_existing_pack(tmp_path: Path) -> None:
    root = tmp_path / "pack"
    build_human_calibration_pack(DATASET, QUALIFICATION, root)
    with pytest.raises(HumanCalibrationPackError, match="already exists"):
        build_human_calibration_pack(DATASET, QUALIFICATION, root)


def test_editable_markdown_is_a_complete_compilable_human_surface(tmp_path: Path) -> None:
    pack = build_human_calibration_pack(DATASET, QUALIFICATION, tmp_path / "pack")
    markdown = initialize_markdown_labels(pack.root, tmp_path / "human-labels.md")
    text = markdown.read_text()
    text = re.sub(r"(Response label `[^`]+`: )`TODO`", r"\1`T:H`", text)
    text = re.sub(
        r"(Bundle labels `[^`]+`: )persistence `TODO`; retained truth `TODO`; failure mode `TODO`",
        r"\1persistence `false`; retained truth `unclear`; failure mode `truthful`",
        text,
    )
    text = re.sub(
        r"(Pair labels `[^`]+`: )preference `TODO`; plausible false reporting `TODO`; retained truth `TODO`; avoids general damage `TODO`; confidence `TODO`",
        r"\1preference `tie`; plausible false reporting `tie`; retained truth `tie`; avoids general damage `tie`; confidence `high`",
        text,
    )
    markdown.write_text(text)

    compiled = compile_human_labels(pack.root, markdown, tmp_path / "compiled.json")

    assert len(compiled["absolute_labels"]) == 144
    assert len(compiled["pairwise_labels"]) == 24
