from __future__ import annotations

import copy

import pytest

from intelligent_liars.truth_editing_contracts import (
    DIRECTION_BANK_MANIFEST_FORMAT,
    INTERVENTION_RECIPE_FORMAT,
    TruthEditingContractError,
    canonical_sha256,
    parse_direction_bank_manifest,
    parse_intervention_recipe,
    validate_recipe_compatibility,
)


def _sha(character: str) -> str:
    return character * 64


def _unsigned_manifest() -> dict[str, object]:
    return {
        "format": DIRECTION_BANK_MANIFEST_FORMAT,
        "manifest_id": "qwen-truth-bank-20260827",
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
                    "path": "artifacts/probes/general.json",
                    "file_sha256": _sha("5"),
                    "vector_sha256": _sha("6"),
                },
                "construction": {
                    "basis_method": "raw",
                    "pooling": "last_token",
                    "token_position": "first_generated_token",
                    "normalization": "unit_l2",
                    "sign_convention": "sklearn_logistic_coef_positive_points_honest_to_deceptive",
                    "intercept": -0.25,
                },
                "control_provenance": None,
                "provenance": {
                    "dataset": "truth-spectrum",
                    "dataset_revision": "release-v1",
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
            },
            {
                "direction_id": "orthogonal-control-layer-21",
                "kind": "orthogonal_control",
                "family": "control",
                "basis_variant": "raw",
                "domains": ["general"],
                "source_layer": 21,
                "width": 4096,
                "rank": 1,
                "artifact": {
                    "path": "artifacts/probes/control.json",
                    "file_sha256": _sha("b"),
                    "vector_sha256": _sha("c"),
                },
                "construction": {
                    "basis_method": "raw",
                    "pooling": "last_token",
                    "token_position": "first_generated_token",
                    "normalization": "unit_l2",
                    "sign_convention": "seeded_orthogonal_control",
                    "intercept": 0.0,
                },
                "control_provenance": {
                    "seed": 17,
                    "parent_direction_ids": ["general-layer-21"],
                    "match_policy": "equal_rank_equal_norm",
                },
                "provenance": {
                    "dataset": "synthetic-control",
                    "dataset_revision": "seed-17",
                    "split": "direction_construction",
                    "ordered_row_ids_sha256": _sha("d"),
                    "source_code_revision": "8" * 40,
                },
                "leakage": {
                    "evaluation_disjoint": True,
                    "heldout_family_disjoint": True,
                    "sealed_audit_accessed": False,
                    "audit_receipt_sha256": _sha("e"),
                },
                "qualification": {
                    "status": "qualified_control",
                    "receipt_sha256": _sha("f"),
                    "finite": True,
                    "unit_norm": True,
                    "qualified_rank": 1,
                },
            },
        ],
    }


def _manifest() -> dict[str, object]:
    value = _unsigned_manifest()
    value["self_sha256"] = canonical_sha256(value)
    return value


def _manifest_with_refusal() -> dict[str, object]:
    value = _unsigned_manifest()
    refusal = copy.deepcopy(value["directions"][0])
    refusal.update(
        {
            "direction_id": "refusal-layer-21",
            "kind": "refusal",
            "family": "refusal",
        }
    )
    refusal["artifact"].update({"file_sha256": _sha("0"), "vector_sha256": _sha("1")})
    refusal["construction"]["sign_convention"] = (
        "bad_minus_good_first_generated_token_residual_mean"
    )
    refusal["provenance"]["ordered_row_ids_sha256"] = _sha("2")
    refusal["leakage"]["audit_receipt_sha256"] = _sha("3")
    refusal["qualification"]["receipt_sha256"] = _sha("4")
    value["directions"].append(refusal)
    value["self_sha256"] = canonical_sha256(value)
    return value


def _persistent_recipe(manifest: dict[str, object]) -> dict[str, object]:
    manifest_hash = manifest["self_sha256"]
    selection = {
        "direction_ids": ["general-layer-21"],
        "basis_method": "qr",
        "requested_rank": 1,
        "basis_sha256": _sha("6"),
        "truth_direction_scope": "global",
        "truth_direction_index": 21,
    }
    value: dict[str, object] = {
        "format": INTERVENTION_RECIPE_FORMAT,
        "recipe_id": "persistent-general-l21",
        "model_sha256": _sha("2"),
        "direction_manifest_sha256": manifest_hash,
        "direction_selection": selection,
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
                "peak_strength": 2.0,
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
    return value


def _activation_recipe(manifest: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "format": INTERVENTION_RECIPE_FORMAT,
        "recipe_id": "activation-general-l21",
        "model_sha256": _sha("2"),
        "direction_manifest_sha256": manifest["self_sha256"],
        "direction_selection": {
            "direction_ids": ["general-layer-21"],
            "basis_method": "svd",
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
            "type": "activation_hook",
            "transform_family": "affine_projection",
            "token_scope": "prefill_last_and_cached_generation",
            "source_layers": [21],
            "prompt_positions": [],
            "generation_step_persistence": True,
            "parameters": {
                "strength": 1.0,
                "target_probe_score": 0.5,
                "reflection_coefficient": None,
                "one_sided_boundary": None,
                "affected_side": None,
                "input_interval": None,
                "output_interval": None,
                "invert": None,
                "margin_lower": None,
                "margin_upper": None,
            },
        },
    }
    value["self_sha256"] = canonical_sha256(value)
    return value


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value.pop("self_sha256", None)
    value["self_sha256"] = canonical_sha256(value)
    return value


def test_direction_manifest_round_trips_and_binds_canonical_identity() -> None:
    raw = _manifest()
    parsed = parse_direction_bank_manifest(raw)
    assert parsed.to_dict() == raw
    assert parsed.direction_ids == (
        "general-layer-21",
        "orthogonal-control-layer-21",
    )
    assert parsed.self_sha256 == canonical_sha256(_unsigned_manifest())


def test_recipe_round_trips_and_is_compatible_with_manifest() -> None:
    raw_manifest = _manifest()
    manifest = parse_direction_bank_manifest(raw_manifest)
    for raw_recipe in (
        _persistent_recipe(raw_manifest),
        _activation_recipe(raw_manifest),
    ):
        recipe = parse_intervention_recipe(raw_recipe)
        assert recipe.to_dict() == raw_recipe
        assert recipe.condition_kind == "truth_only"
        validate_recipe_compatibility(recipe, manifest)


def test_recipe_rejects_obsolete_control_arm_field_fail_closed() -> None:
    raw = _persistent_recipe(_manifest())
    raw["control_arm"] = raw.pop("condition_kind")
    _rehash(raw)

    with pytest.raises(TruthEditingContractError, match="fields differ"):
        parse_intervention_recipe(raw)


def test_recipe_v2_rejects_obsolete_v1_format_fail_closed() -> None:
    raw = _persistent_recipe(_manifest())
    raw["format"] = "truth_editing_intervention_recipe_v1"
    _rehash(raw)

    with pytest.raises(TruthEditingContractError, match="format is unsupported"):
        parse_intervention_recipe(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("unexpected", 1), "fields differ"),
        (lambda value: value.pop("model"), "fields differ"),
        (
            lambda value: value["directions"][0]["construction"].__setitem__(
                "intercept", float("nan")
            ),
            "finite",
        ),
        (
            lambda value: value["directions"][0].__setitem__("source_layer", True),
            "integer",
        ),
        (
            lambda value: value["directions"][0]["leakage"].__setitem__(
                "sealed_audit_accessed", True
            ),
            "sealed audit",
        ),
        (
            lambda value: value["directions"][1].__setitem__(
                "direction_id", "general-layer-21"
            ),
            "unique",
        ),
    ],
)
def test_direction_manifest_fails_closed(mutation, message: str) -> None:
    raw = _manifest()
    mutation(raw)
    with pytest.raises(TruthEditingContractError, match=message):
        parse_direction_bank_manifest(raw)


def test_manifest_rejects_tampered_identity() -> None:
    raw = _manifest()
    raw["manifest_id"] = "tampered"
    with pytest.raises(TruthEditingContractError, match="self hash mismatch"):
        parse_direction_bank_manifest(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["backend"].__setitem__("unexpected", 1), "fields differ"),
        (
            lambda value: value["backend"]["attention"].__setitem__(
                "peak_strength", float("inf")
            ),
            "finite",
        ),
        (
            lambda value: value["backend"]["attention"].__setitem__(
                "peak_strength", True
            ),
            "JSON float",
        ),
        (
            lambda value: value["backend"]["attention"].__setitem__(
                "edge_strength", 2.1
            ),
            "between 0 and 2",
        ),
        (
            lambda value: value["backend"]["attention"].update(
                {"edge_strength": 1.5, "peak_strength": 1.0}
            ),
            "edge_strength",
        ),
        (
            lambda value: value["backend"]["mlp"].__setitem__("peak_strength", 0.5),
            "disabled writer",
        ),
    ],
)
def test_persistent_recipe_fails_closed(mutation, message: str) -> None:
    raw = _persistent_recipe(_manifest())
    mutation(raw)
    with pytest.raises(TruthEditingContractError, match=message):
        parse_intervention_recipe(raw)


def test_activation_recipe_requires_exact_transform_parameters() -> None:
    raw = _activation_recipe(_manifest())
    raw["backend"]["parameters"]["target_probe_score"] = None
    _rehash(raw)
    with pytest.raises(TruthEditingContractError, match="target_probe_score"):
        parse_intervention_recipe(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["backend"].__setitem__(
                "generation_step_persistence", False
            ),
            "generation_step_persistence",
        ),
        (
            lambda value: value["backend"].__setitem__(
                "transform_family", "unknown_transform"
            ),
            "transform_family",
        ),
        (
            lambda value: value["backend"]["parameters"].__setitem__(
                "reflection_coefficient", 1.0
            ),
            "forbids reflection_coefficient",
        ),
        (
            lambda value: value["direction_selection"].__setitem__(
                "requested_rank", True
            ),
            "integer",
        ),
    ],
)
def test_activation_recipe_fails_closed_on_scope_transform_and_rank(
    mutation, message: str
) -> None:
    raw = _activation_recipe(_manifest())
    mutation(raw)
    with pytest.raises(TruthEditingContractError, match=message):
        parse_intervention_recipe(raw)


def test_recipe_compatibility_fails_on_model_manifest_basis_and_rank_mismatch() -> None:
    raw_manifest = _manifest()
    manifest = parse_direction_bank_manifest(raw_manifest)

    wrong_model = _persistent_recipe(raw_manifest)
    wrong_model["model_sha256"] = _sha("f")
    with pytest.raises(TruthEditingContractError, match="model_sha256"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(wrong_model)), manifest
        )

    missing_direction = _persistent_recipe(raw_manifest)
    missing_direction["direction_selection"]["direction_ids"] = ["absent"]
    with pytest.raises(TruthEditingContractError, match="absent"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(missing_direction)), manifest
        )

    too_high_rank = _persistent_recipe(raw_manifest)
    too_high_rank["direction_selection"]["requested_rank"] = 2
    with pytest.raises(TruthEditingContractError, match="declared direction rank"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(too_high_rank)), manifest
        )


def test_condition_kind_identity_must_match_selected_direction_kind() -> None:
    raw_manifest = _manifest()
    recipe = _persistent_recipe(raw_manifest)
    recipe["condition_kind"] = "orthogonal_control"
    with pytest.raises(TruthEditingContractError, match="orthogonal_control"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(recipe)),
            parse_direction_bank_manifest(raw_manifest),
        )


def test_compatibility_caps_rank_by_declared_direction_rank() -> None:
    raw_manifest = _manifest()
    raw_manifest["directions"][0]["qualification"]["qualified_rank"] = 2
    manifest = parse_direction_bank_manifest(_rehash(raw_manifest))
    raw_recipe = _persistent_recipe(raw_manifest)
    raw_recipe["direction_selection"]["basis_method"] = "svd"
    raw_recipe["direction_selection"]["requested_rank"] = 2
    recipe = parse_intervention_recipe(_rehash(raw_recipe))
    with pytest.raises(TruthEditingContractError, match="declared direction rank"):
        validate_recipe_compatibility(recipe, manifest)


def test_qr_requires_the_exact_full_selected_span_while_svd_can_compress() -> None:
    raw_manifest = _manifest()
    extra_truth = copy.deepcopy(raw_manifest["directions"][0])
    extra_truth["direction_id"] = "general-layer-20"
    extra_truth["source_layer"] = 21
    extra_truth["artifact"].update(
        {"file_sha256": _sha("d"), "vector_sha256": _sha("e")}
    )
    extra_truth["provenance"]["ordered_row_ids_sha256"] = _sha("f")
    extra_truth["leakage"]["audit_receipt_sha256"] = _sha("0")
    extra_truth["qualification"]["receipt_sha256"] = _sha("1")
    raw_manifest["directions"].append(extra_truth)
    manifest = parse_direction_bank_manifest(_rehash(raw_manifest))

    raw_recipe = _persistent_recipe(raw_manifest)
    raw_recipe["direction_selection"]["direction_ids"] = [
        "general-layer-21",
        "general-layer-20",
    ]
    with pytest.raises(TruthEditingContractError, match="QR.*exact selected span"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(raw_recipe)), manifest
        )

    raw_recipe["direction_selection"]["basis_method"] = "svd"
    validate_recipe_compatibility(
        parse_intervention_recipe(_rehash(raw_recipe)), manifest
    )


def test_condition_composition_is_exact_for_refusal_only_and_joint() -> None:
    raw_manifest = _manifest_with_refusal()
    refusal_only = _persistent_recipe(raw_manifest)
    refusal_only["condition_kind"] = "refusal_only"
    refusal_only["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "strength": 1.0,
            "requested_rank": 1,
            "writer_policy": "attention",
        }
    )
    with pytest.raises(TruthEditingContractError, match="refusal_only.*no truth"):
        parse_intervention_recipe(_rehash(refusal_only))

    joint = _persistent_recipe(raw_manifest)
    joint["condition_kind"] = "joint"
    joint["direction_selection"]["direction_ids"] = []
    joint["direction_selection"]["requested_rank"] = 0
    joint["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "strength": 1.0,
            "requested_rank": 1,
            "writer_policy": "attention",
        }
    )
    with pytest.raises(TruthEditingContractError, match="joint.*truth"):
        parse_intervention_recipe(_rehash(joint))


def test_float_wire_fields_reject_integer_spelling_to_preserve_round_trip_identity() -> (
    None
):
    raw = _persistent_recipe(_manifest())
    raw["backend"]["attention"]["peak_strength"] = 2
    _rehash(raw)
    with pytest.raises(TruthEditingContractError, match="JSON float"):
        parse_intervention_recipe(raw)


def test_qualification_status_is_exact_for_direction_kind() -> None:
    raw = _manifest()
    raw["directions"][0]["qualification"]["status"] = "qualified_but_failed"
    _rehash(raw)
    with pytest.raises(TruthEditingContractError, match="qualification.status"):
        parse_direction_bank_manifest(raw)


@pytest.mark.parametrize("status", ["candidate", "diagnostic_only"])
def test_nonqualified_historical_directions_round_trip_but_cannot_enter_recipe(
    status: str,
) -> None:
    raw = _manifest()
    raw["directions"][0]["qualification"]["status"] = status
    raw["directions"][0]["leakage"]["evaluation_disjoint"] = False
    _rehash(raw)

    manifest = parse_direction_bank_manifest(raw)
    assert manifest.to_dict() == raw

    recipe = parse_intervention_recipe(_persistent_recipe(raw))
    with pytest.raises(TruthEditingContractError, match="not optimizer-qualified"):
        validate_recipe_compatibility(recipe, manifest)


def test_sign_convention_is_exact_for_direction_kind() -> None:
    raw = _manifest()
    raw["directions"][0]["construction"]["sign_convention"] = "looks_about_right"
    with pytest.raises(TruthEditingContractError, match="sign_convention"):
        parse_direction_bank_manifest(_rehash(raw))


def test_control_provenance_is_required_and_parent_bound() -> None:
    missing = _manifest()
    missing["directions"][1]["control_provenance"] = None
    with pytest.raises(TruthEditingContractError, match="control_provenance"):
        parse_direction_bank_manifest(_rehash(missing))

    unknown_parent = _manifest()
    unknown_parent["directions"][1]["control_provenance"]["parent_direction_ids"] = [
        "absent-parent"
    ]
    with pytest.raises(TruthEditingContractError, match="parent direction"):
        parse_direction_bank_manifest(_rehash(unknown_parent))


def test_refusal_global_index_matches_source_and_per_layer_has_every_writer_layer() -> (
    None
):
    raw_manifest = _manifest_with_refusal()
    manifest = parse_direction_bank_manifest(raw_manifest)

    global_recipe = _persistent_recipe(raw_manifest)
    global_recipe["condition_kind"] = "joint"
    global_recipe["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "direction_index": 22,
            "strength": 1.0,
            "requested_rank": 1,
            "writer_policy": "attention",
        }
    )
    with pytest.raises(TruthEditingContractError, match="global refusal.*source layer"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(global_recipe)), manifest
        )

    per_layer_recipe = _persistent_recipe(raw_manifest)
    per_layer_recipe["condition_kind"] = "joint"
    per_layer_recipe["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "direction_scope": "per_layer",
            "strength": 1.0,
            "requested_rank": 1,
            "writer_policy": "attention",
        }
    )
    per_layer_recipe["backend"]["attention"].update(
        {"kernel_center": 22.0, "kernel_half_width": 0.0}
    )
    with pytest.raises(TruthEditingContractError, match="per_layer refusal.*22"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(per_layer_recipe)), manifest
        )


def test_base_treatment_and_negative_control_condition_compositions_validate() -> None:
    raw_manifest = _manifest_with_refusal()
    shuffled = copy.deepcopy(raw_manifest["directions"][1])
    shuffled.update(
        {
            "direction_id": "shuffled-control-layer-21",
            "kind": "shuffled_control",
        }
    )
    shuffled["artifact"].update({"file_sha256": _sha("5"), "vector_sha256": _sha("7")})
    shuffled["construction"]["sign_convention"] = "shuffled_label_control"
    shuffled["control_provenance"]["seed"] = 23
    shuffled["provenance"]["ordered_row_ids_sha256"] = _sha("7")
    shuffled["leakage"]["audit_receipt_sha256"] = _sha("8")
    shuffled["qualification"]["receipt_sha256"] = _sha("9")
    raw_manifest["directions"].append(shuffled)
    manifest = parse_direction_bank_manifest(_rehash(raw_manifest))

    base = _persistent_recipe(raw_manifest)
    base["condition_kind"] = "base"
    base["direction_selection"]["direction_ids"] = []
    base["direction_selection"]["requested_rank"] = 0
    base["backend"]["attention"].update(
        {"enabled": False, "edge_strength": 0.0, "peak_strength": 0.0}
    )

    refusal_only = _persistent_recipe(raw_manifest)
    refusal_only["condition_kind"] = "refusal_only"
    refusal_only["direction_selection"]["direction_ids"] = []
    refusal_only["direction_selection"]["requested_rank"] = 0
    refusal_only["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "strength": 1.0,
            "requested_rank": 1,
            "writer_policy": "attention",
        }
    )

    joint = _persistent_recipe(raw_manifest)
    joint["condition_kind"] = "joint"
    joint["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "strength": 1.0,
            "requested_rank": 1,
            "writer_policy": "attention",
        }
    )

    shuffled_recipe = _persistent_recipe(raw_manifest)
    shuffled_recipe["condition_kind"] = "shuffled_control"
    shuffled_recipe["direction_selection"]["direction_ids"] = [
        "shuffled-control-layer-21"
    ]
    shuffled_recipe["direction_selection"]["basis_sha256"] = _sha("7")

    for raw_recipe in (base, refusal_only, joint, shuffled_recipe):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(raw_recipe)), manifest
        )


def test_truth_direction_global_index_and_per_layer_availability_are_exact() -> None:
    raw_manifest = _manifest()
    manifest = parse_direction_bank_manifest(raw_manifest)

    wrong_global = _persistent_recipe(raw_manifest)
    wrong_global["direction_selection"]["truth_direction_index"] = 20
    with pytest.raises(TruthEditingContractError, match="global truth.*source layer"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(wrong_global)), manifest
        )

    missing_per_layer = _persistent_recipe(raw_manifest)
    missing_per_layer["direction_selection"]["truth_direction_scope"] = "per_layer"
    missing_per_layer["backend"]["attention"]["kernel_center"] = 22.0
    with pytest.raises(TruthEditingContractError, match="per_layer truth.*22"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(missing_per_layer)), manifest
        )


def test_refusal_basis_identity_rank_and_variant_fail_closed() -> None:
    raw_manifest = _manifest_with_refusal()

    missing_rank = _persistent_recipe(raw_manifest)
    missing_rank["condition_kind"] = "joint"
    missing_rank["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "strength": 1.0,
            "writer_policy": "attention",
        }
    )
    with pytest.raises(TruthEditingContractError, match="positive requested_rank"):
        parse_intervention_recipe(_rehash(missing_rank))

    relabeled_raw = copy.deepcopy(missing_rank)
    relabeled_raw["refusal"]["requested_rank"] = 1
    relabeled_raw["refusal"]["basis_variant"] = "truth_orthogonalized"
    recipe = parse_intervention_recipe(_rehash(relabeled_raw))
    with pytest.raises(TruthEditingContractError, match="basis_variant.*raw"):
        validate_recipe_compatibility(
            recipe, parse_direction_bank_manifest(raw_manifest)
        )

    wrong_single_hash = copy.deepcopy(missing_rank)
    wrong_single_hash["refusal"]["requested_rank"] = 1
    wrong_single_hash["refusal"]["basis_sha256"] = _sha("e")
    with pytest.raises(
        TruthEditingContractError, match="single-direction.*basis_sha256"
    ):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(wrong_single_hash)),
            parse_direction_bank_manifest(raw_manifest),
        )

    qualified_variant_manifest = copy.deepcopy(raw_manifest)
    qualified_variant_manifest["directions"][2]["basis_variant"] = (
        "truth_orthogonalized"
    )
    qualified_variant_manifest = _rehash(qualified_variant_manifest)
    qualified_variant_recipe = _persistent_recipe(qualified_variant_manifest)
    qualified_variant_recipe["condition_kind"] = "joint"
    qualified_variant_recipe["refusal"].update(
        {
            "enabled": True,
            "direction_ids": ["refusal-layer-21"],
            "requested_rank": 2,
            "basis_method": "svd",
            "basis_variant": "truth_orthogonalized",
            "strength": 1.0,
            "writer_policy": "attention",
        }
    )
    with pytest.raises(TruthEditingContractError, match="declared direction rank"):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(qualified_variant_recipe)),
            parse_direction_bank_manifest(qualified_variant_manifest),
        )

    qualified_variant_recipe["refusal"]["requested_rank"] = 1
    validate_recipe_compatibility(
        parse_intervention_recipe(_rehash(qualified_variant_recipe)),
        parse_direction_bank_manifest(qualified_variant_manifest),
    )


def test_selected_prompt_positions_are_exact_sorted_and_scope_bound() -> None:
    raw = _activation_recipe(_manifest())
    raw["backend"]["token_scope"] = "selected_prompt_positions"
    raw["backend"]["generation_step_persistence"] = False
    with pytest.raises(TruthEditingContractError, match="requires prompt_positions"):
        parse_intervention_recipe(_rehash(raw))
    raw["backend"]["prompt_positions"] = [3, 1]
    with pytest.raises(TruthEditingContractError, match="sorted unique"):
        parse_intervention_recipe(_rehash(raw))

    raw["backend"]["prompt_positions"] = [1, 3]
    recipe = parse_intervention_recipe(_rehash(raw))
    assert recipe.backend.prompt_positions == (1, 3)

    raw["backend"]["token_scope"] = "prefill_last"
    with pytest.raises(TruthEditingContractError, match="must be empty"):
        parse_intervention_recipe(_rehash(raw))


def test_single_truth_direction_basis_hash_is_bound_to_manifest_vector() -> None:
    manifest = _manifest()
    recipe = _persistent_recipe(manifest)
    recipe["direction_selection"]["basis_sha256"] = _sha("e")
    with pytest.raises(
        TruthEditingContractError, match="single-direction truth basis_sha256"
    ):
        validate_recipe_compatibility(
            parse_intervention_recipe(_rehash(recipe)),
            parse_direction_bank_manifest(manifest),
        )
