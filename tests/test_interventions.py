import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from intelligent_liars.interventions import (
    DirectionMode,
    InterventionBundle,
    InterventionMethod,
    InterventionSpec,
    ProbeDirection,
    RuntimeIntervention,
    ScoreDirectionality,
    TokenScope,
    apply_rank_one_writer_edit,
    load_intervention_bundle,
    load_probe_direction,
    seeded_orthogonal_direction,
    save_intervention_bundle,
    save_writer_edited_model,
    transform_activations,
    validate_materializable_bundle,
    validate_intervention_spec,
    writer_edit_coefficient,
)


def direction() -> ProbeDirection:
    return ProbeDirection(
        vector=(2.0, 0.0), intercept=-1.0, layer=1, task="general_domain"
    )


def score(value: torch.Tensor) -> torch.Tensor:
    return value @ torch.tensor([2.0, 0.0]) - 1.0


def spec(method: InterventionMethod, **kwargs: object) -> InterventionSpec:
    return InterventionSpec(method=method, layers=(1,), **kwargs)


def test_scalar_addition_changes_probe_score_by_requested_delta() -> None:
    activation = torch.tensor([[2.0, 3.0]])

    transformed = transform_activations(
        activation,
        direction(),
        spec(InterventionMethod.SCALAR_ADDITION, score_delta=4.5),
    )

    assert score(transformed).item() == pytest.approx(score(activation).item() + 4.5)
    assert transformed[0, 1].item() == activation[0, 1].item()


def test_affine_projection_sets_score_relative_to_nonzero_probe_boundary() -> None:
    activation = torch.tensor([[2.0, 3.0], [-2.0, 5.0]])

    transformed = transform_activations(
        activation,
        direction(),
        spec(InterventionMethod.AFFINE_PROJECTION, projection_target=0.25),
    )

    assert score(transformed).tolist() == pytest.approx([0.25, 0.25])
    assert transformed[:, 1].tolist() == activation[:, 1].tolist()


def test_full_and_partial_reflection_reverse_probe_coordinate() -> None:
    activation = torch.tensor([[2.0, 3.0]])

    full = transform_activations(
        activation, direction(), spec(InterventionMethod.FULL_REFLECTION)
    )
    partial = transform_activations(
        activation,
        direction(),
        spec(InterventionMethod.PARTIAL_REFLECTION, reflection_strength=0.75),
    )

    assert score(full).item() == pytest.approx(-score(activation).item())
    assert score(partial).item() == pytest.approx(-0.5 * score(activation).item())


def test_one_sided_reflection_only_changes_selected_class() -> None:
    activation = torch.tensor([[0.0, 2.0], [2.0, 4.0]])

    transformed = transform_activations(
        activation,
        direction(),
        spec(
            InterventionMethod.ONE_SIDED_REFLECTION,
            selected_side="honest",
            reflection_strength=1.0,
        ),
    )

    assert score(activation).tolist() == pytest.approx([-1.0, 3.0])
    assert score(transformed).tolist() == pytest.approx([1.0, 3.0])


@pytest.mark.parametrize(
    ("directed_spec", "expected_scores"),
    [
        (
            spec(InterventionMethod.SCALAR_ADDITION, score_delta=2.0),
            [-3.0, 0.0, 0.0, 3.0],
        ),
        (
            spec(InterventionMethod.AFFINE_PROJECTION, projection_target=1.0),
            [1.0, 1.0, 0.0, 3.0],
        ),
        (spec(InterventionMethod.FULL_REFLECTION), [5.0, 2.0, 0.0, 3.0]),
        (
            spec(
                InterventionMethod.PARTIAL_REFLECTION,
                reflection_strength=0.75,
            ),
            [2.5, 1.0, 0.0, 3.0],
        ),
        (
            spec(
                InterventionMethod.ONE_SIDED_REFLECTION,
                selected_side="honest",
                reflection_strength=0.5,
            ),
            [0.0, 0.0, 0.0, 3.0],
        ),
        (
            spec(
                InterventionMethod.BOUNDED_REMAP,
                remap_input_min=-2.0,
                remap_input_max=2.0,
                remap_output_min=2.0,
                remap_output_max=-2.0,
                max_score_delta=4.0,
            ),
            [-1.0, 2.0, 0.0, 3.0],
        ),
        (
            spec(
                InterventionMethod.BOUNDED_MARGIN_CLAMP,
                selected_side="deceptive",
                margin=1.0,
                max_score_delta=2.0,
            ),
            [-3.0, 0.0, 0.0, 3.0],
        ),
    ],
)
def test_toward_deceptive_variants_never_reduce_existing_alignment(
    directed_spec: InterventionSpec,
    expected_scores: list[float],
) -> None:
    activation = torch.tensor([[-2.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [2.0, 0.0]])
    directed_spec = replace(
        directed_spec,
        score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
    )

    before = score(activation)
    after = score(transform_activations(activation, direction(), directed_spec))

    assert torch.all(after >= before)
    assert torch.all(after[before < 0] > before[before < 0])
    assert torch.equal(after[before >= 0], before[before >= 0])
    assert after.tolist() == pytest.approx(expected_scores)


def test_bounded_remap_and_margin_clamp_bound_score_adjustment() -> None:
    activation = torch.tensor([[-2.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
    remapped = transform_activations(
        activation,
        direction(),
        spec(
            InterventionMethod.BOUNDED_REMAP,
            remap_input_min=-2.0,
            remap_input_max=2.0,
            remap_output_min=2.0,
            remap_output_max=-2.0,
            max_score_delta=3.0,
        ),
    )
    clamped = transform_activations(
        activation,
        direction(),
        spec(
            InterventionMethod.BOUNDED_MARGIN_CLAMP,
            selected_side="deceptive",
            margin=1.5,
            max_score_delta=2.0,
        ),
    )

    original_scores = score(activation)
    assert torch.max(torch.abs(score(remapped) - original_scores)).item() <= 3.0
    assert score(clamped).tolist() == pytest.approx([-3.0, 1.5, 5.0])


def test_bundle_validation_rejects_unbounded_clamp_and_non_finite_settings() -> None:
    with pytest.raises(ValueError, match="requires max_score_delta"):
        validate_intervention_spec(
            spec(InterventionMethod.BOUNDED_MARGIN_CLAMP, margin=1.0)
        )
    with pytest.raises(ValueError, match="must be finite"):
        validate_intervention_spec(
            spec(InterventionMethod.SCALAR_ADDITION, score_delta=float("nan"))
        )
    with pytest.raises(ValueError, match="require symmetric"):
        validate_intervention_spec(
            spec(
                InterventionMethod.FULL_REFLECTION,
                direction_mode=DirectionMode.SEEDED_ORTHOGONAL_CONTROL,
                control_seed=0,
                score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            )
        )
    with pytest.raises(ValueError, match="positive projection_target"):
        validate_intervention_spec(
            spec(
                InterventionMethod.AFFINE_PROJECTION,
                projection_target=0.0,
                score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            )
        )
    with pytest.raises(ValueError, match="select the honest side"):
        validate_intervention_spec(
            spec(
                InterventionMethod.ONE_SIDED_REFLECTION,
                selected_side="deceptive",
                score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            )
        )


def test_seeded_orthogonal_direction_is_reproducible_and_norm_matched() -> None:
    source = torch.tensor([3.0, 4.0, 0.0])

    first = seeded_orthogonal_direction(source, seed=17)
    second = seeded_orthogonal_direction(source, seed=17)

    assert first.tolist() == pytest.approx(second.tolist())
    assert torch.dot(first, source).item() == pytest.approx(0.0, abs=1e-6)
    assert torch.linalg.vector_norm(first).item() == pytest.approx(5.0)


def test_legacy_direction_mode_alias_and_control_seed_replacement() -> None:
    control = spec(
        InterventionMethod.FULL_REFLECTION,
        direction_mode=DirectionMode.MATCHED_RANDOM,
        control_seed=17,
    )

    assert control.direction_mode == DirectionMode.SEEDED_ORTHOGONAL_CONTROL
    assert control.control_seed == 17

    replaced = replace(control, control_seed=4)
    assert replaced.control_seed == 4


def test_seeded_orthogonal_reflection_preserves_truth_probe_score() -> None:
    activation = torch.tensor([[2.0, 3.0], [-2.0, 5.0]])
    bundle = InterventionBundle(
        direction=direction(),
        spec=spec(
            InterventionMethod.FULL_REFLECTION,
            direction_mode=DirectionMode.SEEDED_ORTHOGONAL_CONTROL,
            control_seed=17,
        ),
    )

    transformed = transform_activations(
        activation,
        bundle.effective_direction(),
        bundle.spec,
    )

    assert score(transformed).tolist() == pytest.approx(
        score(activation).tolist(),
        abs=1e-6,
    )


def test_probe_direction_loading_and_bundle_round_trip(tmp_path: Path) -> None:
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(
        json.dumps(
            {
                "direction_sign_convention": (
                    "sklearn_logistic_coef_positive_points_honest_to_deceptive"
                ),
                "general_domain": {
                    "directions": [
                        {
                            "task": "general_domain",
                            "layer": 21,
                            "direction_vector": [1.0, 2.0],
                            "intercept": -0.5,
                            "direction_sign_convention": (
                                "sklearn_logistic_coef_positive_points_honest_to_deceptive"
                            ),
                        }
                    ]
                },
            }
        )
    )
    loaded_direction = load_probe_direction(probe_path, layer=21)
    bundle = InterventionBundle(
        direction=loaded_direction,
        spec=InterventionSpec(
            method=InterventionMethod.ONE_SIDED_REFLECTION,
            layers=(19, 20, 21),
            direction_mode=DirectionMode.SEEDED_ORTHOGONAL_CONTROL,
            control_seed=3,
        ),
    )
    bundle_path = tmp_path / "intervention.json"

    save_intervention_bundle(bundle, bundle_path)

    assert load_intervention_bundle(bundle_path) == bundle
    assert loaded_direction.source_path == str(probe_path.resolve())

    legacy_payload = json.loads(bundle_path.read_text())
    legacy_payload["format"] = "qwen_truth_intervention_v1"
    legacy_payload["spec"]["direction_mode"] = "matched_random"
    legacy_payload["spec"]["random_seed"] = legacy_payload["spec"].pop(
        "control_seed"
    )
    legacy_payload["spec"].pop("score_directionality")
    bundle_path.write_text(json.dumps(legacy_payload))
    assert load_intervention_bundle(bundle_path) == bundle


class FakeLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int = 2) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(
            o_proj=torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.mlp = SimpleNamespace(
            down_proj=torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        )

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def fake_model(layer_count: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                layers=torch.nn.ModuleList([FakeLayer() for _ in range(layer_count)])
            )
        )
    )


def test_runtime_intervention_all_scope_and_removes_hooks() -> None:
    model = fake_model()
    bundle = InterventionBundle(
        direction=direction(),
        spec=InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=(1,),
            token_scope=TokenScope.ALL,
        ),
    )
    layer = model.model.language_model.layers[1]
    prefill = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    decode = torch.tensor([[[2.0, 0.0]]])

    with RuntimeIntervention(model, bundle):
        assert torch.allclose(score(layer(prefill)[0]), torch.tensor([[-3.0, -3.0]]))
        assert score(layer(decode)[0]).item() == pytest.approx(-3.0)

    assert layer(decode)[0].tolist() == decode.tolist()


def test_runtime_last_token_scope_can_change_first_generated_token_state() -> None:
    model = fake_model()
    bundle = InterventionBundle(
        direction=direction(),
        spec=InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=(1,),
            token_scope=TokenScope.LAST_TOKEN,
        ),
    )
    layer = model.model.language_model.layers[1]
    prefill = torch.tensor([[[0.0, 4.0], [2.0, 5.0]]])

    with RuntimeIntervention(model, bundle):
        transformed = layer(prefill)[0]

    assert transformed[0, 0].tolist() == prefill[0, 0].tolist()
    assert score(transformed[0, 1]).item() == pytest.approx(-3.0)
    assert transformed[0, 1, 1].item() == prefill[0, 1, 1].item()


def test_rank_one_writer_edit_projects_or_reflects_selected_writer_outputs() -> None:
    model = fake_model()
    layer = model.model.language_model.layers[1]
    layer.self_attn.o_proj.weight.data.copy_(torch.eye(2))
    layer.mlp.down_proj.weight.data.copy_(torch.eye(2))
    projection_bundle = InterventionBundle(
        direction=ProbeDirection(vector=(1.0, 0.0), intercept=0.0, layer=1, task="x"),
        spec=spec(InterventionMethod.AFFINE_PROJECTION, token_scope=TokenScope.ALL),
    )

    edited = apply_rank_one_writer_edit(model, projection_bundle)

    assert len(edited) == 2
    expected = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(layer.self_attn.o_proj.weight, expected)
    assert torch.allclose(layer.mlp.down_proj.weight, expected)


def test_only_homogeneous_linear_methods_can_be_materialized() -> None:
    assert writer_edit_coefficient(spec(InterventionMethod.FULL_REFLECTION)) == 2.0
    assert (
        writer_edit_coefficient(
            spec(InterventionMethod.PARTIAL_REFLECTION, reflection_strength=0.25)
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="requires a runtime intervention"):
        writer_edit_coefficient(spec(InterventionMethod.ONE_SIDED_REFLECTION))
    with pytest.raises(ValueError, match="non-zero affine projection"):
        writer_edit_coefficient(
            spec(InterventionMethod.AFFINE_PROJECTION, projection_target=1.0)
        )
    with pytest.raises(ValueError, match="non-zero probe intercept"):
        validate_materializable_bundle(
            InterventionBundle(
                direction=direction(),
                spec=spec(
                    InterventionMethod.FULL_REFLECTION,
                    token_scope=TokenScope.ALL,
                ),
            )
        )
    with pytest.raises(ValueError, match="token_scope='all'"):
        validate_materializable_bundle(
            InterventionBundle(
                direction=ProbeDirection(
                    vector=(1.0, 0.0), intercept=0.0, layer=1, task="x"
                ),
                spec=spec(InterventionMethod.FULL_REFLECTION),
            )
        )
    with pytest.raises(ValueError, match="requires a runtime intervention"):
        validate_materializable_bundle(
            InterventionBundle(
                direction=ProbeDirection(
                    vector=(1.0, 0.0), intercept=0.0, layer=1, task="x"
                ),
                spec=spec(
                    InterventionMethod.FULL_REFLECTION,
                    token_scope=TokenScope.ALL,
                    score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
                ),
            )
        )
    with pytest.raises(ValueError, match="must be non-zero"):
        validate_materializable_bundle(
            InterventionBundle(
                direction=ProbeDirection(
                    vector=(0.0, 0.0), intercept=0.0, layer=1, task="x"
                ),
                spec=spec(
                    InterventionMethod.FULL_REFLECTION,
                    token_scope=TokenScope.ALL,
                ),
            )
        )


def test_materialized_model_saves_checkpoint_and_edit_manifest(tmp_path: Path) -> None:
    model = fake_model()
    model.model.language_model.layers[1].self_attn.o_proj.weight.data.copy_(
        torch.eye(2)
    )
    model.model.language_model.layers[1].mlp.down_proj.weight.data.copy_(torch.eye(2))

    def save_model(path: Path, *, safe_serialization: bool) -> None:
        assert safe_serialization is True
        (path / "model.safetensors").write_text("model")

    def save_processor(path: Path) -> None:
        (path / "processor.json").write_text("processor")

    model.save_pretrained = save_model
    bundle = InterventionBundle(
        direction=ProbeDirection(
            vector=(1.0, 0.0), intercept=0.0, layer=1, task="general_domain"
        ),
        spec=spec(InterventionMethod.FULL_REFLECTION, token_scope=TokenScope.ALL),
    )
    output_dir = tmp_path / "edited"

    edited = save_writer_edited_model(
        bundle=bundle,
        model_bundle=SimpleNamespace(
            model=model,
            processor=SimpleNamespace(save_pretrained=save_processor),
        ),
        output_dir=output_dir,
    )

    assert len(edited) == 2
    assert (output_dir / "model.safetensors").read_text() == "model"
    assert (output_dir / "processor.json").read_text() == "processor"
    assert (
        json.loads((output_dir / "intervention.json").read_text())["spec"]["method"]
        == "full_reflection"
    )
    materialization = json.loads(
        (output_dir / "writer_edit_materialization.json").read_text()
    )
    assert materialization["implementation"] == "rank_one_writer_edit"
    assert materialization["edited_parameters"] == edited
