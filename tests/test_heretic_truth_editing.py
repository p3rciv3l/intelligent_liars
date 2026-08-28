from __future__ import annotations

import copy

import pytest

from intelligent_liars.heretic_truth_editing import (
    DatasetSplitIndex,
    HERETIC_CONFIG_FORMAT,
    HERETIC_PLAN_FORMAT,
    HereticIntegrationError,
    HereticStudyConfig,
    build_trial_plan,
    load_study_config,
    parse_trial_plan,
)
from intelligent_liars.truth_editing_contracts import (
    parse_direction_bank_manifest,
    parse_intervention_recipe,
    validate_recipe_compatibility,
)

from test_truth_editing_contracts import _manifest, _persistent_recipe


def _inputs() -> tuple[object, object]:
    raw_manifest = _manifest()
    manifest = parse_direction_bank_manifest(raw_manifest)
    raw_recipe = _persistent_recipe(raw_manifest)
    recipe = parse_intervention_recipe(raw_recipe)
    validate_recipe_compatibility(recipe, manifest)
    return manifest, recipe


def _config(**changes: object) -> HereticStudyConfig:
    value: dict[str, object] = {
        "study_id": "truth-editing-offline-test",
        "batch_size": 2,
        "max_trials": 8,
        "startup_trials": 2,
        "sampler_seed": 41,
        "wall_clock_budget_seconds": 86400,
    }
    value.update(changes)
    return HereticStudyConfig(**value)


def _dataset(*, validation: tuple[str, ...] = ("valid-1", "valid-2")) -> DatasetSplitIndex:
    return DatasetSplitIndex(
        dataset_id="truth-editing-fixture",
        manifest_sha256="a" * 64,
        split_record_ids={
            "train": ("train-1",),
            "validation": validation,
            "test": ("test-1",),
        },
    )


def test_validation_plan_is_synchronous_and_contains_only_validation_rows() -> None:
    manifest, recipe = _inputs()
    plan = build_trial_plan(
        [recipe],
        dataset=_dataset(),
        direction_manifest=manifest,
        config=_config(),
    )

    assert plan.format == HERETIC_PLAN_FORMAT
    assert plan.scheduler == "synchronous"
    assert plan.evaluation_split == "validation"
    assert plan.test_record_ids == ()
    assert plan.batches[0].trial_ids == ("trial-0000",)
    assert plan.trials[0].record_ids == ("valid-1", "valid-2")
    assert "probe_auroc" in plan.diagnostic_metric_names
    assert "probe_auroc" not in plan.objective_names


def test_recipe_and_direction_manifest_are_checked_before_planning() -> None:
    manifest, recipe = _inputs()
    mutated = copy.deepcopy(recipe.to_dict())
    mutated["model_sha256"] = "f" * 64
    mutated["self_sha256"] = "0" * 64
    with pytest.raises(Exception):
        # The strict recipe parser must reject the stale self identity first.
        parse_intervention_recipe(mutated)

    with pytest.raises(HereticIntegrationError, match="compatibility"):
        build_trial_plan(
            [recipe],
            dataset=_dataset(validation=("valid-1",)),
            direction_manifest={"format": "wrong"},
            config=_config(),
        )

    assert recipe.recipe_id == "persistent-general-l21"
    assert manifest.manifest_id == "qwen-truth-bank-20260827"


def test_d1_is_rejected_even_if_the_recipe_schema_is_valid() -> None:
    raw_manifest = _manifest()
    raw_recipe = _persistent_recipe(raw_manifest)
    raw_recipe["recipe_id"] = "d1-pair-only"
    raw_recipe.pop("self_sha256", None)
    from intelligent_liars.truth_editing_contracts import canonical_sha256

    raw_recipe["self_sha256"] = canonical_sha256(raw_recipe)
    manifest = parse_direction_bank_manifest(raw_manifest)
    recipe = parse_intervention_recipe(raw_recipe)

    with pytest.raises(HereticIntegrationError, match="D1"):
        build_trial_plan(
            [recipe],
            dataset=_dataset(validation=("valid-1",)),
            direction_manifest=manifest,
            config=_config(),
        )


def test_probe_metrics_cannot_be_objectives() -> None:
    with pytest.raises(HereticIntegrationError, match="diagnostic"):
        _config(objective_names=("probe_auroc",))


def test_validation_is_the_only_optimization_objective_split() -> None:
    manifest, recipe = _inputs()
    with pytest.raises(HereticIntegrationError, match="validation"):
        build_trial_plan(
            [recipe],
            dataset=_dataset(validation=("valid-1",)),
            direction_manifest=manifest,
            config=_config(evaluation_split="test"),
        )


def test_final_test_plan_requires_freeze_and_has_no_optimizer_objective() -> None:
    manifest, recipe = _inputs()
    dataset = _dataset(validation=("valid-1",))
    with pytest.raises(HereticIntegrationError, match="frozen"):
        build_trial_plan(
            [recipe],
            dataset=dataset,
            direction_manifest=manifest,
            config=_config(),
            phase="final_test",
        )

    plan = build_trial_plan(
        [recipe],
        dataset=dataset,
        direction_manifest=manifest,
        config=_config(),
        phase="final_test",
        frozen=True,
        freeze_receipt_sha256="b" * 64,
    )
    assert plan.evaluation_split == "test"
    assert plan.objective_names == ()
    assert plan.test_record_ids == ("test-1",)
    assert plan.trials[0].record_ids == ("test-1",)


def test_duplicate_recipe_and_record_id_fail_closed() -> None:
    manifest, recipe = _inputs()
    with pytest.raises(HereticIntegrationError, match="recipe_id"):
        build_trial_plan(
            [recipe, recipe],
            dataset=_dataset(validation=("valid-1",)),
            direction_manifest=manifest,
            config=_config(),
        )

    with pytest.raises(HereticIntegrationError, match="unique"):
        build_trial_plan(
            [recipe],
            dataset=_dataset(validation=("valid-1", "valid-1")),
            direction_manifest=manifest,
            config=_config(),
        )


def test_plan_round_trip_preserves_identity(tmp_path) -> None:
    manifest, recipe = _inputs()
    plan = build_trial_plan(
        [recipe],
        dataset=_dataset(validation=("valid-1",)),
        direction_manifest=manifest,
        config=_config(),
    )
    parsed = parse_trial_plan(plan.to_dict())
    assert parsed.to_dict() == plan.to_dict()
    assert parsed.self_sha256 == plan.self_sha256

    output = tmp_path / "plan.json"
    output.write_text(plan.to_json() + "\n")
    assert parse_trial_plan(__import__("json").loads(output.read_text())).self_sha256 == plan.self_sha256


def test_plan_identity_tampering_fails_closed() -> None:
    manifest, recipe = _inputs()
    plan = build_trial_plan(
        [recipe],
        dataset=_dataset(validation=("valid-1",)),
        direction_manifest=manifest,
        config=_config(),
    )
    raw = plan.to_dict()
    raw["batches"][0]["trial_ids"] = ["trial-9999"]
    with pytest.raises(HereticIntegrationError, match="cover"):
        parse_trial_plan(raw)


def test_config_file_is_strict_and_frozen(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        __import__("json").dumps(
            {
                "format": HERETIC_CONFIG_FORMAT,
                "study_id": "truth-editing-validation-v1",
                "sampler": "multivariate_tpe",
                "scheduler": "synchronous",
                "batch_size": 8,
                "max_trials": 200,
                "startup_trials": 60,
                "sampler_seed": 20260827,
                "wall_clock_budget_seconds": 86400,
                "evaluation_split": "validation",
                "objective_names": [
                    "valid_false_report_rate_lcb",
                    "truth_report_dissociation_lcb",
                    "capability_preservation_lcb",
                ],
                "diagnostic_metric_names": ["probe_auroc", "probe_score_delta"],
                "runtime": {
                    "engine": "transformers",
                    "dtype": "bfloat16",
                    "attention_implementation": "flash_attention_2",
                    "quantization": None,
                    "speculative_decoding": False,
                    "device_map": "cuda:0",
                    "use_cache": True,
                    "model_loads_per_worker": 1,
                },
            }
        )
    )
    config = load_study_config(path)
    assert config.scheduler == "synchronous"
    assert config.runtime.engine == "transformers"
    assert config.runtime.speculative_decoding is False

    bad = __import__("json").loads(path.read_text())
    bad["scheduler"] = "asynchronous"
    path.write_text(__import__("json").dumps(bad))
    with pytest.raises(HereticIntegrationError, match="synchronous"):
        load_study_config(path)
