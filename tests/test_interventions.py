import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    SEEDED_ORTHOGONAL_CONTROL_VARIANT,
    TokenScope,
    canonical_intervention_suite_specs,
    load_intervention_bundle,
    load_probe_direction,
    seeded_orthogonal_direction,
    save_intervention_bundle,
    transform_activations,
    validate_intervention_spec,
)


def direction() -> ProbeDirection:
    return ProbeDirection(
        vector=(2.0, 0.0), intercept=-1.0, layer=1, task="general_domain"
    )


def score(value: torch.Tensor) -> torch.Tensor:
    return value @ torch.tensor([2.0, 0.0]) - 1.0


def spec(method: InterventionMethod, **kwargs: Any) -> InterventionSpec:
    return InterventionSpec(method=method, layers=(1,), **kwargs)


def test_canonical_a0_a7_specs_freeze_every_transform_field() -> None:
    layers = (21,)
    control_seed = 20260824
    margin = 1.0
    budget = 2.0

    actual = canonical_intervention_suite_specs(
        layers=layers,
        control_seed=control_seed,
        deceptive_margin=margin,
        score_movement_budget=budget,
    )

    expected = {
        "directed_scalar_add_deceptive": InterventionSpec(
            method=InterventionMethod.SCALAR_ADDITION,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            score_delta=budget,
        ),
        "directed_affine_project_deceptive": InterventionSpec(
            method=InterventionMethod.AFFINE_PROJECTION,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            projection_target=margin,
        ),
        "directed_full_reflection": InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
        ),
        "directed_partial_reflection": InterventionSpec(
            method=InterventionMethod.PARTIAL_REFLECTION,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            reflection_strength=0.75,
        ),
        "honest_boundary_ablation": InterventionSpec(
            method=InterventionMethod.ONE_SIDED_REFLECTION,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            reflection_strength=0.5,
            selected_side="honest",
        ),
        "bounded_directed_inversion": InterventionSpec(
            method=InterventionMethod.BOUNDED_REMAP,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            remap_input_min=-budget,
            remap_input_max=budget,
            remap_output_min=budget,
            remap_output_max=-budget,
            max_score_delta=2.0 * budget,
        ),
        "bounded_deceptive_margin": InterventionSpec(
            method=InterventionMethod.BOUNDED_MARGIN_CLAMP,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            selected_side="deceptive",
            margin=margin,
            max_score_delta=budget,
        ),
        SEEDED_ORTHOGONAL_CONTROL_VARIANT: InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=layers,
            token_scope=TokenScope.LAST_TOKEN,
            direction_mode=DirectionMode.SEEDED_ORTHOGONAL_CONTROL,
            control_seed=control_seed,
        ),
    }

    assert actual == expected
    assert tuple(actual) == (
        "directed_scalar_add_deceptive",
        "directed_affine_project_deceptive",
        "directed_full_reflection",
        "directed_partial_reflection",
        "honest_boundary_ablation",
        "bounded_directed_inversion",
        "bounded_deceptive_margin",
        SEEDED_ORTHOGONAL_CONTROL_VARIANT,
    )


def test_canonical_a0_a6_score_maps_cover_boundaries_and_extremes() -> None:
    budget = 2.0
    specs = canonical_intervention_suite_specs(
        layers=(21,),
        control_seed=20260824,
        deceptive_margin=1.0,
        score_movement_budget=budget,
    )
    probe = ProbeDirection(
        vector=(1.0,), intercept=0.0, layer=21, task="general_domain"
    )
    scores = torch.tensor([[-10.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [10.0]])
    expected = {
        "directed_scalar_add_deceptive": [-8.0, 0.0, 1.0, 0.0, 1.0, 2.0, 10.0],
        "directed_affine_project_deceptive": [
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            2.0,
            10.0,
        ],
        "directed_full_reflection": [10.0, 2.0, 1.0, 0.0, 1.0, 2.0, 10.0],
        "directed_partial_reflection": [5.0, 1.0, 0.5, 0.0, 1.0, 2.0, 10.0],
        "honest_boundary_ablation": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 10.0],
        "bounded_directed_inversion": [-6.0, 2.0, 1.0, 0.0, 1.0, 2.0, 10.0],
        "bounded_deceptive_margin": [-8.0, 0.0, 1.0, 0.0, 1.0, 2.0, 10.0],
    }

    for name, expected_scores in expected.items():
        transformed = transform_activations(scores, probe, specs[name])
        assert transformed.squeeze(-1).tolist() == pytest.approx(expected_scores)
        assert torch.equal(
            transformed[scores.squeeze(-1) >= 0], scores[scores.squeeze(-1) >= 0]
        )

    inverted = transform_activations(
        torch.tensor([[-budget]]),
        probe,
        specs["bounded_directed_inversion"],
    )
    assert inverted.item() == pytest.approx(budget)
    assert inverted.item() - (-budget) == pytest.approx(2.0 * budget)


def test_canonical_a7_preserves_extreme_and_boundary_truth_scores() -> None:
    specs = canonical_intervention_suite_specs(
        layers=(21,),
        control_seed=20260824,
        deceptive_margin=1.0,
        score_movement_budget=2.0,
    )
    truth_probe = ProbeDirection(
        vector=(1.0, 0.0), intercept=0.0, layer=21, task="general_domain"
    )
    bundle = InterventionBundle(
        direction=truth_probe,
        spec=specs[SEEDED_ORTHOGONAL_CONTROL_VARIANT],
    )
    activations = torch.tensor(
        [[-10.0, 3.0], [-2.0, -4.0], [0.0, 5.0], [2.0, -6.0], [10.0, 7.0]]
    )

    transformed = transform_activations(
        activations,
        bundle.effective_direction(),
        bundle.spec,
    )

    assert transformed[:, 0].tolist() == pytest.approx(activations[:, 0].tolist())
    assert transformed[:, 1].tolist() == pytest.approx((-activations[:, 1]).tolist())


@pytest.mark.parametrize(
    "invalid_direction",
    [
        ProbeDirection(vector=(0.0, 0.0), intercept=0.0, layer=1, task="zero"),
        ProbeDirection(vector=(float("nan"), 1.0), intercept=0.0, layer=1, task="nan"),
    ],
    ids=["zero", "nan"],
)
def test_transform_activations_rejects_invalid_probe_direction(
    invalid_direction: ProbeDirection,
) -> None:
    with pytest.raises(ValueError, match="Probe direction"):
        transform_activations(
            torch.zeros(1, 2),
            invalid_direction,
            spec(InterventionMethod.FULL_REFLECTION),
        )


@pytest.mark.parametrize(
    "activation",
    [
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([True, False]),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, float("inf")]),
        torch.tensor(1.0),
        torch.empty(2, 0),
    ],
    ids=["integer", "boolean", "nan", "infinity", "rank_zero", "empty_hidden"],
)
def test_transform_activations_rejects_invalid_activation_tensor(
    activation: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="Activations"):
        transform_activations(
            activation,
            direction(),
            spec(InterventionMethod.FULL_REFLECTION),
        )


def test_transform_activations_rejects_non_finite_output() -> None:
    with pytest.raises(ValueError, match="non-finite output"):
        transform_activations(
            torch.tensor([[3.0e38]]),
            ProbeDirection(vector=(1.0,), intercept=0.0, layer=1, task="overflow"),
            spec(InterventionMethod.SCALAR_ADDITION, score_delta=3.0e38),
        )


def _large_direction() -> ProbeDirection:
    vector_tensor = torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32)
    vector_tensor = vector_tensor / torch.linalg.vector_norm(vector_tensor)
    vector_tensor = vector_tensor * (5.25**0.5)
    return ProbeDirection(
        vector=tuple(vector_tensor.tolist()),
        intercept=0.0,
        layer=19,
        task="general_domain",
    )


def _large_activation(dtype: torch.dtype) -> torch.Tensor:
    vector = torch.tensor(_large_direction().vector, dtype=torch.float32)
    denominator = torch.dot(vector, vector)
    scores = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
    activation = scores.unsqueeze(-1) / denominator * vector
    orthogonal = torch.cos(torch.arange(4096, dtype=torch.float32))
    orthogonal -= torch.dot(orthogonal, vector) / denominator * vector
    orthogonal /= torch.linalg.vector_norm(orthogonal)
    activation += torch.tensor([1.25, -2.5, 4.0]).unsqueeze(-1) * orthogonal
    return activation.to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "method_spec",
    [
        spec(InterventionMethod.SCALAR_ADDITION, score_delta=1.25),
        spec(InterventionMethod.AFFINE_PROJECTION, projection_target=0.75),
        spec(InterventionMethod.FULL_REFLECTION),
        spec(
            InterventionMethod.PARTIAL_REFLECTION,
            reflection_strength=0.25,
        ),
        spec(
            InterventionMethod.ONE_SIDED_REFLECTION,
            selected_side="honest",
            reflection_strength=1.0,
        ),
        spec(
            InterventionMethod.BOUNDED_REMAP,
            remap_input_min=-2.0,
            remap_input_max=2.0,
            remap_output_min=2.0,
            remap_output_max=-2.0,
            max_score_delta=6.0,
        ),
        spec(
            InterventionMethod.BOUNDED_MARGIN_CLAMP,
            selected_side="deceptive",
            margin=1.0,
            max_score_delta=3.0,
        ),
    ],
    ids=lambda value: value.method.value,
)
def test_every_canonical_transform_handles_qwen_width_and_preserves_orthogonal_state(
    method_spec: InterventionSpec,
    dtype: torch.dtype,
) -> None:
    probe = _large_direction()
    activation = _large_activation(dtype)
    original = activation.clone()
    vector = torch.tensor(probe.vector, dtype=torch.float32)
    denominator = torch.dot(vector, vector)
    before = activation.float() @ vector

    transformed = transform_activations(activation, probe, method_spec)
    after = transformed.float() @ vector

    if method_spec.method == InterventionMethod.SCALAR_ADDITION:
        expected = before + method_spec.score_delta
    elif method_spec.method == InterventionMethod.AFFINE_PROJECTION:
        expected = torch.full_like(before, method_spec.projection_target)
    elif method_spec.method == InterventionMethod.FULL_REFLECTION:
        expected = -before
    elif method_spec.method == InterventionMethod.PARTIAL_REFLECTION:
        expected = (1.0 - 2.0 * method_spec.reflection_strength) * before
    elif method_spec.method == InterventionMethod.ONE_SIDED_REFLECTION:
        expected = torch.where(before < 0, -before, before)
    elif method_spec.method == InterventionMethod.BOUNDED_REMAP:
        unit = (before.clamp(-2.0, 2.0) + 2.0) / 4.0
        expected = 2.0 - 4.0 * unit
    else:
        expected = torch.maximum(before, torch.ones_like(before))

    # BF16 rounds the transformed hidden state before the score is observed. The
    # achieved score therefore has a small, explicitly bounded quantization error.
    score_atol = 2.0e-5 if dtype == torch.float32 else 2.0e-3
    assert torch.allclose(after, expected, atol=score_atol, rtol=0.0)
    assert transformed.dtype == dtype
    assert torch.equal(activation, original)

    before_orthogonal = (
        activation.float() - (before / denominator).unsqueeze(-1) * vector
    )
    after_orthogonal = (
        transformed.float() - (after / denominator).unsqueeze(-1) * vector
    )
    orthogonal_atol = 2.0e-6 if dtype == torch.float32 else 1.0e-3
    assert torch.allclose(
        after_orthogonal,
        before_orthogonal,
        atol=orthogonal_atol,
        rtol=0.0,
    )


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
    legacy_payload["spec"]["random_seed"] = legacy_payload["spec"].pop("control_seed")
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
