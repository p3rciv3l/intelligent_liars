from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from intelligent_liars.heretic_truth_editing import OBJECTIVES
from intelligent_liars.truth_editing_contracts import (
    DIRECTION_BANK_MANIFEST_FORMAT,
    INTERVENTION_RECIPE_FORMAT,
    DirectionBankManifest,
    InterventionRecipe,
    canonical_sha256,
    parse_intervention_recipe,
)
from intelligent_liars.truth_editing_engine import (
    ENGINE_CONFIG_FORMAT,
    EvaluationReceipt,
    SearchResult,
    TruthEditingEngine,
    TruthEditingEngineError,
)
from intelligent_liars.truth_editing_dataset_v2 import V2Candidate, build_dataset_v2


def _sha(character: str) -> str:
    return character * 64


def _manifest() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": DIRECTION_BANK_MANIFEST_FORMAT,
        "manifest_id": "engine-test-bank",
        "model": {
            "repository": "Qwen/Qwen3-VL-8B-Thinking",
            "revision": "1" * 40,
            "model_sha256": _sha("2"),
            "tokenizer_sha256": _sha("3"),
            "chat_template_sha256": _sha("4"),
            "decoder_layer_count": 36,
            "hidden_width": 4096,
        },
        "directions": [
            {
                "direction_id": "general-layer-21",
                "kind": "truth",
                "family": "general",
                "basis_variant": "raw",
                "domains": ["general"],
                "source_layer": 21,
                "width": 4096,
                "rank": 1,
                "artifact": {
                    "path": "directions/general-layer-21.safetensors",
                    "file_sha256": _sha("5"),
                    "vector_sha256": _sha("6"),
                },
                "construction": {
                    "basis_method": "raw",
                    "pooling": "last_token",
                    "token_position": "first_generated_token",
                    "normalization": "unit_l2",
                    "sign_convention": "positive_points_honest_to_deceptive",
                    "intercept": 0.0,
                },
                "control_provenance": None,
                "provenance": {
                    "dataset": "fixture",
                    "dataset_revision": "v1",
                    "split": "direction_construction",
                    "ordered_row_ids_sha256": _sha("7"),
                    "source_code_revision": "8" * 40,
                },
                "leakage": {
                    "evaluation_disjoint": True,
                    "heldout_family_disjoint": True,
                    "sealed_audit_accessed": False,
                    "audit_receipt_sha256": _sha("9"),
                },
                "qualification": {
                    "status": "qualified",
                    "receipt_sha256": _sha("a"),
                    "finite": True,
                    "unit_norm": True,
                    "qualified_rank": 1,
                },
            }
        ],
    }
    value["self_sha256"] = canonical_sha256(value)
    return value


def _recipe(manifest: Mapping[str, Any]) -> InterventionRecipe:
    value: dict[str, Any] = {
        "format": INTERVENTION_RECIPE_FORMAT,
        "recipe_id": "engine-test-recipe",
        "model_sha256": _sha("2"),
        "direction_manifest_sha256": manifest["self_sha256"],
        "direction_selection": {
            "direction_ids": ["general-layer-21"],
            "basis_method": "qr",
            "requested_rank": 1,
            "basis_sha256": _sha("6"),
            "truth_direction_scope": "global",
            "truth_direction_index": 21,
        },
        "condition_kind": "truth_only",
        "refusal": {
            "enabled": False,
            "direction_ids": [],
            "direction_scope": "global",
            "direction_index": 21,
            "basis_method": "qr",
            "requested_rank": 0,
            "basis_sha256": _sha("1"),
            "basis_variant": "raw",
            "strength": 0.0,
            "writer_policy": "both",
        },
        "backend": {
            "type": "persistent_weight",
            "transform_family": "projection_reflection",
            "normalization_mode": "exact",
            "attention": {
                "enabled": True,
                "kernel_center": 21.0,
                "kernel_half_width": 0.0,
                "edge_strength": 1.0,
                "peak_strength": 1.0,
            },
            "mlp": {
                "enabled": False,
                "kernel_center": 21.0,
                "kernel_half_width": 0.0,
                "edge_strength": 0.0,
                "peak_strength": 0.0,
            },
        },
    }
    value["self_sha256"] = canonical_sha256(value)
    return parse_intervention_recipe(value)


class RuntimeFake:
    def __init__(self) -> None:
        self.splits: list[str] = []

    def evaluate(
        self,
        recipe: InterventionRecipe,
        records: Sequence[Any],
        *,
        split: str,
    ) -> Mapping[str, Any]:
        assert records
        self.splits.append(split)
        return {
            "recipe_sha256": recipe.self_sha256,
            "split": split,
            "response_count": len(records),
        }


class JudgeFake:
    def __init__(self, objectives: Mapping[str, float] | None = None) -> None:
        self.objectives = objectives or {name: 0.75 for name in OBJECTIVES}

    def score(
        self,
        runtime_receipt: Mapping[str, Any],
        records: Sequence[Any],
        *,
        split: str,
    ) -> Mapping[str, Any]:
        assert runtime_receipt["split"] == split
        assert records
        return {"format": "judge-fake-v1", "objectives": dict(self.objectives)}


class SearchFake:
    def __init__(self, recipe: InterventionRecipe) -> None:
        self.recipe = recipe

    def run(
        self,
        *,
        study_id: str,
        direction_bank: DirectionBankManifest,
        evaluate: Callable[[InterventionRecipe], EvaluationReceipt],
    ) -> SearchResult:
        assert study_id == "engine-test"
        assert direction_bank.direction_ids == ("general-layer-21",)
        evaluation = evaluate(self.recipe)
        return SearchResult(
            best_recipe=self.recipe,
            best_evaluation=evaluation,
            trial_evaluation_sha256=(evaluation.identity_sha256,),
        )


class ArtifactFake:
    def __init__(self) -> None:
        self.freeze_count = 0

    def freeze(self, result: SearchResult) -> Mapping[str, Any]:
        self.freeze_count += 1
        return {
            "format": "artifact-fake-v1",
            "search_result_sha256": result.identity_sha256,
        }


def _open_engine(
    tmp_path: Path,
    *,
    judge: JudgeFake | None = None,
    search_factory: Callable[[InterventionRecipe], Any] = SearchFake,
) -> tuple[TruthEditingEngine, RuntimeFake, ArtifactFake, InterventionRecipe]:
    manifest = _manifest()
    manifest_path = tmp_path / "directions.json"
    manifest_path.write_text(json.dumps(manifest))
    recipe = _recipe(manifest)
    config = {
        "format": ENGINE_CONFIG_FORMAT,
        "study_id": "engine-test",
        "dataset_manifest": str(
            Path(__file__).parents[1] / "datasets/truth_editing/v1/manifest.json"
        ),
        "direction_bank_manifest": "directions.json",
    }
    config_path = tmp_path / "engine.json"
    config_path.write_text(json.dumps(config))
    runtime = RuntimeFake()
    artifacts = ArtifactFake()
    engine = TruthEditingEngine.open(
        config_path,
        runtime=runtime,
        judge=judge or JudgeFake(),
        search=search_factory(recipe),
        artifacts=artifacts,
    )
    return engine, runtime, artifacts, recipe


def test_run_composes_validation_search_and_freezes_exact_result(tmp_path: Path) -> None:
    engine, runtime, artifacts, _ = _open_engine(tmp_path)
    receipt = engine.run()

    assert runtime.splits == ["validation"]
    assert artifacts.freeze_count == 1
    assert receipt.study_id == "engine-test"
    assert receipt.direction_bank_manifest_sha256 == _manifest()["self_sha256"]
    assert receipt.artifact_receipt["search_result_sha256"] == (
        receipt.search_result.identity_sha256
    )
    assert len(receipt.identity_sha256) == 64


def test_test_split_is_sealed_until_search_result_is_frozen(tmp_path: Path) -> None:
    engine, runtime, _, recipe = _open_engine(tmp_path)

    with pytest.raises(TruthEditingEngineError, match="test evaluation requires"):
        engine.evaluate(recipe, split="test")

    engine.run()
    result = engine.evaluate(recipe, split="test")
    assert result.split == "test"
    assert runtime.splits == ["validation", "test"]


def test_judge_must_return_every_declared_semantic_objective(tmp_path: Path) -> None:
    engine, _, artifacts, _ = _open_engine(
        tmp_path, judge=JudgeFake({OBJECTIVES[0]: 0.5})
    )

    with pytest.raises(TruthEditingEngineError, match="objectives differ"):
        engine.run()
    assert artifacts.freeze_count == 0


def test_search_cannot_claim_a_trial_it_did_not_evaluate(tmp_path: Path) -> None:
    class FabricatingSearch(SearchFake):
        def run(
            self,
            *,
            study_id: str,
            direction_bank: DirectionBankManifest,
            evaluate: Callable[[InterventionRecipe], EvaluationReceipt],
        ) -> SearchResult:
            evaluation = evaluate(self.recipe)
            return SearchResult(
                best_recipe=self.recipe,
                best_evaluation=evaluation,
                trial_evaluation_sha256=(evaluation.identity_sha256, _sha("f")),
            )

    engine, _, artifacts, _ = _open_engine(tmp_path, search_factory=FabricatingSearch)
    with pytest.raises(TruthEditingEngineError, match="unevaluated trial"):
        engine.run()
    assert artifacts.freeze_count == 0


def test_search_receipt_cannot_omit_an_evaluated_trial(tmp_path: Path) -> None:
    class OmittingSearch(SearchFake):
        def run(
            self,
            *,
            study_id: str,
            direction_bank: DirectionBankManifest,
            evaluate: Callable[[InterventionRecipe], EvaluationReceipt],
        ) -> SearchResult:
            first = evaluate(self.recipe)
            alternate = self.recipe.to_dict()
            alternate["recipe_id"] = "engine-test-recipe-alternate"
            alternate.pop("self_sha256")
            alternate["self_sha256"] = canonical_sha256(alternate)
            evaluate(parse_intervention_recipe(alternate))
            return SearchResult(
                best_recipe=self.recipe,
                best_evaluation=first,
                trial_evaluation_sha256=(first.identity_sha256,),
            )

    engine, _, artifacts, _ = _open_engine(tmp_path, search_factory=OmittingSearch)
    with pytest.raises(TruthEditingEngineError, match="omits validation evaluations"):
        engine.run()
    assert artifacts.freeze_count == 0


def test_engine_rejects_repeat_run_on_same_mutable_adapter_set(tmp_path: Path) -> None:
    engine, _, artifacts, _ = _open_engine(tmp_path)
    engine.run()
    with pytest.raises(TruthEditingEngineError, match="already run"):
        engine.run()
    assert artifacts.freeze_count == 1


def test_config_is_strict_and_paths_are_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "engine.json"
    config_path.write_text(
        json.dumps(
            {
                "format": ENGINE_CONFIG_FORMAT,
                "study_id": "engine-test",
                "dataset_manifest": "dataset.json",
                "direction_bank_manifest": "directions.json",
                "surprise": True,
            }
        )
    )
    with pytest.raises(TruthEditingEngineError, match="extra=.*surprise"):
        TruthEditingEngine.open(
            config_path,
            runtime=RuntimeFake(),
            judge=JudgeFake(),
            search=SearchFake(_recipe(_manifest())),
            artifacts=ArtifactFake(),
        )


def test_engine_opens_canonical_qa_v2_through_same_dataset_seam(tmp_path: Path) -> None:
    candidates = [
        V2Candidate(
            source_id="fixture",
            source_revision="v1",
            source_record_id=f"row-{index}",
            canonical_key=f"question-{index}",
            question=f"What is {index} plus one?",
            correct_answer=str(index + 1),
                choices=(str(index + 1), str(index + 2)),
                family=f"family-{index}",
                truth_authority="deterministic arithmetic",
                near_match_policy="canonical_only",
            )
        for index in range(400)
    ]
    dataset = build_dataset_v2(candidates, tmp_path / "v2")
    assert dataset.manifest.split_counts["validation"] > 0
    manifest = _manifest()
    (tmp_path / "directions.json").write_text(json.dumps(manifest))
    config_path = tmp_path / "engine.json"
    config_path.write_text(
        json.dumps(
            {
                "format": ENGINE_CONFIG_FORMAT,
                "study_id": "engine-test",
                "dataset_manifest": "v2/manifest.json",
                "direction_bank_manifest": "directions.json",
            }
        )
    )
    recipe = _recipe(manifest)
    engine = TruthEditingEngine.open(
        config_path,
        runtime=RuntimeFake(),
        judge=JudgeFake(),
        search=SearchFake(recipe),
        artifacts=ArtifactFake(),
    )

    receipt = engine.run()
    assert receipt.dataset_manifest_sha256 == canonical_sha256(
        dataset.manifest.to_payload()
    )


def test_one_command_cli_fails_closed_for_unknown_adapter_factory(tmp_path: Path) -> None:
    config_path = tmp_path / "engine.json"
    config_path.write_text("{}")
    command = Path(__file__).parents[1] / "scripts/truth_editing.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(command),
            "--config",
            str(config_path),
            "--adapter-factory",
            "does_not_exist:build_engine",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "cannot load adapter factory" in completed.stderr
