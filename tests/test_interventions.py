from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from intelligent_liars.interventions import (
    ActivationMapping,
    BoundedCoordinateRemap,
    CoordinateRemoval,
    DeceptiveMarginClamp,
    Identity,
    LinearProbe,
    RandomDirectionControl,
    Reflection,
    ScalarAddition,
)


def test_probe_scores_batches_with_an_affine_intercept() -> None:
    probe = LinearProbe(coef=torch.tensor([3.0, 4.0]), intercept=-2.0)

    scores = probe.score(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

    torch.testing.assert_close(scores, torch.tensor([1.0, 6.0]))


def test_probe_can_be_built_from_saved_numeric_values_for_a_target_dtype() -> None:
    probe = LinearProbe.from_values(
        coef=[3.0, 4.0],
        intercept=-2.0,
        dtype=torch.float64,
        device="cpu",
    )

    assert probe.coef.dtype == torch.float64
    assert probe.intercept == -2.0
    torch.testing.assert_close(
        probe.score(torch.tensor([1.0, 0.0], dtype=torch.float64)),
        torch.tensor(1.0, dtype=torch.float64),
    )


def test_identity_is_the_exact_no_op_default_control() -> None:
    activations = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]), intercept=0.5)

    result = Identity()(activations, probe)

    assert result is activations


def test_scalar_addition_moves_along_the_stored_probe_direction() -> None:
    activations = torch.tensor([[1.0, 2.0]])
    original = activations.clone()
    probe = LinearProbe(coef=torch.tensor([3.0, 4.0]), intercept=-1.0)

    result = ScalarAddition(multiplier=0.5)(activations, probe)

    torch.testing.assert_close(result, torch.tensor([[2.5, 4.0]]))
    torch.testing.assert_close(activations, original)


def test_random_control_is_seeded_and_matches_scalar_addition_displacement() -> None:
    activations = torch.zeros((2, 4), dtype=torch.float64)
    probe = LinearProbe(coef=torch.tensor([1.0, 2.0, 2.0, 0.0], dtype=torch.float64))

    first = RandomDirectionControl(multiplier=0.5, seed=17)(activations, probe)
    repeated = RandomDirectionControl(multiplier=0.5, seed=17)(activations, probe)
    other_seed = RandomDirectionControl(multiplier=0.5, seed=18)(activations, probe)

    torch.testing.assert_close(first, repeated, rtol=0.0, atol=0.0)
    assert not torch.equal(first, other_seed)
    torch.testing.assert_close(
        torch.linalg.vector_norm(first, dim=-1),
        torch.tensor([1.5, 1.5], dtype=torch.float64),
    )


def test_random_control_does_not_advance_torch_global_rng() -> None:
    activations = torch.zeros((1, 2))
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]))
    torch.manual_seed(123)
    expected_next_draw = torch.randn(3)
    torch.manual_seed(123)

    RandomDirectionControl(multiplier=1.0, seed=7)(activations, probe)

    torch.testing.assert_close(torch.randn(3), expected_next_draw, rtol=0.0, atol=0.0)


def test_coordinate_removal_projects_each_state_to_the_affine_boundary() -> None:
    activations = torch.tensor([[3.0, 4.0], [-1.0, 5.0]])
    original = activations.clone()
    probe = LinearProbe(coef=torch.tensor([3.0, 4.0]), intercept=-2.0)

    result = CoordinateRemoval()(activations, probe)

    torch.testing.assert_close(probe.score(result), torch.zeros(2), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(activations, original)


@pytest.mark.parametrize(
    ("scale", "expected_scores"),
    [(1.0, [2.0, -2.0]), (0.5, [1.0, -1.0]), (1.5, [3.0, -3.0])],
)
def test_reflection_maps_scores_to_negative_scaled_scores(
    scale: float,
    expected_scores: list[float],
) -> None:
    activations = torch.tensor([[-3.0, 8.0], [1.0, 8.0]])
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]), intercept=1.0)

    result = Reflection(scale=scale)(activations, probe)

    torch.testing.assert_close(probe.score(result), torch.tensor(expected_scores))
    torch.testing.assert_close(result[:, 1], activations[:, 1])


def test_one_sided_reflection_only_changes_honest_negative_score_states() -> None:
    activations = torch.tensor([[-3.0, 8.0], [1.0, 8.0], [-1.0, 8.0]])
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]), intercept=1.0)

    result = Reflection(one_sided=True)(activations, probe)

    torch.testing.assert_close(probe.score(result), torch.tensor([2.0, 2.0, 0.0]))
    torch.testing.assert_close(result[1:], activations[1:])


def test_bounded_coordinate_remap_caps_each_activation_displacement() -> None:
    activations = torch.tensor([[0.0, 0.0]])
    probe = LinearProbe(coef=torch.tensor([3.0, 4.0]), intercept=0.0)
    remap = BoundedCoordinateRemap(
        remap_scores=lambda scores: torch.full_like(scores, 100.0),
        max_displacement=2.0,
    )

    result = remap(activations, probe)

    torch.testing.assert_close(result, torch.tensor([[1.2, 1.6]]))
    torch.testing.assert_close(probe.score(result), torch.tensor([10.0]))
    torch.testing.assert_close(
        torch.linalg.vector_norm(result - activations, dim=-1),
        torch.tensor([2.0]),
    )
    assert isinstance(remap, ActivationMapping)


def test_deceptive_margin_clamp_leaves_scores_beyond_target_unchanged() -> None:
    activations = torch.tensor([[-5.0, 7.0], [1.0, 8.0], [3.0, 9.0]])
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]), intercept=0.0)

    result = DeceptiveMarginClamp(target_score=2.0, max_displacement=1.5)(
        activations,
        probe,
    )

    torch.testing.assert_close(probe.score(result), torch.tensor([-3.5, 2.0, 3.0]))
    torch.testing.assert_close(result[2], activations[2])


@pytest.mark.parametrize(
    "coef",
    [torch.tensor([]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, float("nan")])],
)
def test_probe_rejects_invalid_directions(coef: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        LinearProbe(coef=coef)


def test_probe_rejects_wrong_activation_shape_dtype_and_device() -> None:
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="hidden dimension"):
        probe.score(torch.ones(2, 3))
    with pytest.raises(TypeError, match="same dtype"):
        probe.score(torch.ones(2, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="same device"):
        probe.score(torch.ones(2, 2, device="meta"))


@pytest.mark.parametrize(
    "make_operator",
    [
        lambda: ScalarAddition(multiplier=float("nan")),
        lambda: Reflection(scale=-0.1),
        lambda: BoundedCoordinateRemap(
            remap_scores=lambda scores: scores,
            max_displacement=-1.0,
        ),
        lambda: DeceptiveMarginClamp(target_score=1.0, max_displacement=-1.0),
    ],
)
def test_operator_parameters_must_be_finite_and_non_negative(
    make_operator: Callable[[], object],
) -> None:
    with pytest.raises(ValueError):
        make_operator()


@pytest.mark.parametrize(
    ("remap_scores", "error_type", "message"),
    [
        (lambda scores: [1.0], TypeError, "torch.Tensor"),
        (lambda scores: scores.unsqueeze(-1), ValueError, "shape"),
        (lambda scores: scores.to(torch.float64), TypeError, "dtype"),
        (lambda scores: torch.full_like(scores, float("nan")), ValueError, "finite"),
    ],
)
def test_bounded_remap_rejects_invalid_target_scores(
    remap_scores: Callable[[torch.Tensor], object],
    error_type: type[Exception],
    message: str,
) -> None:
    activations = torch.zeros((1, 2))
    probe = LinearProbe(coef=torch.tensor([1.0, 0.0]))
    remap = BoundedCoordinateRemap(
        remap_scores=remap_scores,  # type: ignore[arg-type]
        max_displacement=1.0,
    )

    with pytest.raises(error_type, match=message):
        remap(activations, probe)
