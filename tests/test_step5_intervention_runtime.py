from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from intelligent_liars.interventions import (
    InterventionBundle,
    ProbeDirection,
    canonical_intervention_suite_specs,
)
from intelligent_liars.step5_intervention_experiments.runtime import (
    TEACHER_FORCED_MASK_SEMANTICS,
    CachedGenerationLastTokenIntervention,
    MaskedTeacherForcedIntervention,
)


class _Layer(nn.Module):
    def __init__(self, *, tuple_output: bool = False) -> None:
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden: torch.Tensor):
        if self.tuple_output:
            return hidden, "cache"
        return hidden


class _FakeQwen(nn.Module):
    def __init__(self, *, tuple_output: bool = False) -> None:
        super().__init__()
        layers = nn.ModuleList([_Layer(), _Layer(tuple_output=tuple_output)])
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=layers))


@pytest.fixture
def bundles() -> list[InterventionBundle]:
    direction = ProbeDirection(
        vector=(2.0, 0.0), intercept=-1.0, layer=1, task="general_domain"
    )
    specs = canonical_intervention_suite_specs(
        layers=(1,),
        control_seed=20260824,
        deceptive_margin=1.0,
        score_movement_budget=2.0,
    )
    return [
        InterventionBundle(direction=direction, spec=spec) for spec in specs.values()
    ]


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("bundle_index", range(8))
def test_teacher_forced_all_transforms_tensor_and_tuple(
    bundles: list[InterventionBundle], tuple_output: bool, bundle_index: int
) -> None:
    model = _FakeQwen(tuple_output=tuple_output)
    label_mask = torch.tensor([[False, False, True, True]])
    hidden = torch.tensor([[[-1.0, 8.0], [-1.0, 7.0], [-1.0, 6.0], [-1.0, 5.0]]])
    runtime = MaskedTeacherForcedIntervention(model, bundles[bundle_index], label_mask)
    with runtime:
        output = model.model.language_model.layers[1](hidden)
        assert runtime.execution_evidence.hooks_active == 1
    transformed = output[0] if isinstance(output, tuple) else output
    changed = (transformed != hidden).any(dim=-1)
    assert torch.equal(changed, torch.tensor([[False, True, True, False]]))
    assert runtime.mask_semantics == TEACHER_FORCED_MASK_SEMANTICS
    assert (
        runtime.mask_semantics
        == "answer_label_mask_shifted_left_to_predictor_positions"
    )
    evidence = runtime.assert_executed()
    assert evidence.invocation_count == 1
    assert evidence.token_count == 2
    assert evidence.layer_invocations == {1: 1}
    assert evidence.hooks_active == 0
    if isinstance(output, tuple):
        assert output[1] == "cache"


@pytest.mark.parametrize("tuple_output", [False, True])
@pytest.mark.parametrize("bundle_index", range(8))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_generation_all_transforms_prefill_and_decode(
    bundles: list[InterventionBundle],
    tuple_output: bool,
    bundle_index: int,
    dtype: torch.dtype,
) -> None:
    model = _FakeQwen(tuple_output=tuple_output)
    layer = model.model.language_model.layers[1]
    runtime = CachedGenerationLastTokenIntervention(model, bundles[bundle_index])
    prefill = torch.tensor([[[-1.0, 8.0], [-1.0, 7.0], [-1.0, 6.0]]], dtype=dtype)
    decode = torch.tensor([[[-1.0, 5.0]]], dtype=dtype)
    with runtime:
        prefill_output = layer(prefill)
        decode_output = layer(decode)
    prefill_hidden = (
        prefill_output[0] if isinstance(prefill_output, tuple) else prefill_output
    )
    decode_hidden = (
        decode_output[0] if isinstance(decode_output, tuple) else decode_output
    )
    assert torch.equal(prefill_hidden[:, :-1], prefill[:, :-1])
    assert not torch.equal(prefill_hidden[:, -1:], prefill[:, -1:])
    assert not torch.equal(decode_hidden, decode)
    assert prefill_hidden.dtype == dtype
    assert decode_hidden.dtype == dtype
    evidence = runtime.assert_executed()
    assert evidence.invocation_count == 2
    assert evidence.token_count == 2
    assert evidence.layer_invocations == {1: 2}
    assert evidence.hooks_active == 0


def test_teacher_forced_rejects_empty_and_noncausal_masks(
    bundles: list[InterventionBundle],
) -> None:
    model = _FakeQwen()
    with pytest.raises(ValueError, match="all false"):
        MaskedTeacherForcedIntervention(
            model, bundles[0], torch.zeros((1, 3), dtype=torch.bool)
        )
    with pytest.raises(ValueError, match="causal predictor"):
        MaskedTeacherForcedIntervention(
            model, bundles[0], torch.tensor([[True, False, False]])
        )


def test_never_called_layer_has_explicit_failed_evidence(
    bundles: list[InterventionBundle],
) -> None:
    runtime = CachedGenerationLastTokenIntervention(_FakeQwen(), bundles[0])
    with runtime:
        pass
    with pytest.raises(ValueError, match="did not execute"):
        runtime.assert_executed()
    assert runtime.execution_evidence.hooks_active == 0


def test_hooks_are_removed_after_normal_and_error_exit(
    bundles: list[InterventionBundle],
) -> None:
    model = _FakeQwen()
    layer = model.model.language_model.layers[1]
    hidden = torch.tensor([[[-1.0, 4.0], [-1.0, 3.0]]])
    runtime = CachedGenerationLastTokenIntervention(model, bundles[0])
    with runtime:
        layer(hidden)
    assert not layer._forward_hooks
    assert torch.equal(layer(hidden), hidden)

    failing = MaskedTeacherForcedIntervention(
        model, bundles[0], torch.tensor([[False, True, True]])
    )
    with pytest.raises(ValueError, match="does not match"):
        with failing:
            layer(hidden)
    assert not layer._forward_hooks
    assert torch.equal(layer(hidden), hidden)


def test_teacher_forced_hook_preserves_bfloat16(
    bundles: list[InterventionBundle],
) -> None:
    model = _FakeQwen()
    hidden = torch.tensor(
        [[[-1.0, 4.0], [-1.0, 3.0], [-1.0, 2.0]]], dtype=torch.bfloat16
    )
    runtime = MaskedTeacherForcedIntervention(
        model, bundles[0], torch.tensor([[False, False, True]])
    )
    with runtime:
        output = model.model.language_model.layers[1](hidden)
    assert output.dtype == torch.bfloat16
    assert runtime.assert_executed().token_count == 1
