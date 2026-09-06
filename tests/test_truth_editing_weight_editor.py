from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from intelligent_liars.truth_editing_weight_editor import (
    CompiledLayerWriterEdit,
    CompiledWriterEdit,
    WriterEditError,
    WriterEditRuntime,
    _projection_tolerance,
    materialized_writer_weights,
)


class FakeAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class FakeMlp(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(hidden + 2, hidden, bias=False)


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = FakeAttention(hidden)
        self.mlp = FakeMlp(hidden)
        self.untouched = nn.Linear(hidden, hidden, bias=False)


class FakeLanguageModel(nn.Module):
    def __init__(self, hidden: int, layer_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeDecoderLayer(hidden) for _ in range(layer_count)]
        )


class FakeModelCore(nn.Module):
    def __init__(self, hidden: int, layer_count: int) -> None:
        super().__init__()
        self.language_model = FakeLanguageModel(hidden, layer_count)
        self.visual = nn.Linear(hidden, hidden, bias=False)


class FakeQwen(nn.Module):
    def __init__(self, hidden: int = 3, layer_count: int = 2) -> None:
        super().__init__()
        self.model = FakeModelCore(hidden, layer_count)


def _fill_weights(model: FakeQwen) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            values = torch.arange(
                index,
                index + parameter.numel(),
                dtype=parameter.dtype,
            ).reshape(parameter.shape)
            parameter.copy_(values)


def _compiled(*edits: CompiledLayerWriterEdit) -> CompiledWriterEdit:
    return CompiledWriterEdit(
        recipe_id="recipe-test",
        model_sha256="a" * 64,
        layers=edits,
    )


def _runtime(*, model_sha256: str = "a" * 64) -> WriterEditRuntime:
    return WriterEditRuntime(verified_model_sha256=model_sha256)


def test_projection_tolerance_tracks_installed_weight_precision() -> None:
    assert _projection_tolerance(torch.float32) == 5e-3
    assert _projection_tolerance(torch.float16) == 5e-3
    assert _projection_tolerance(torch.bfloat16) == 0.015625


def test_overlay_applies_exact_writer_projection_and_independent_strengths() -> None:
    model = FakeQwen(hidden=3, layer_count=2)
    _fill_weights(model)
    layer = model.model.language_model.layers[0]
    original_attention = layer.self_attn.o_proj.weight.detach().clone()
    original_mlp = layer.mlp.down_proj.weight.detach().clone()
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=basis,
            attention_strength=1.0,
            mlp_strength=2.0,
        )
    )

    lease = _runtime().activate(model, compiled)
    with lease:
        expected_attention = original_attention.clone()
        expected_attention[0] = 0
        expected_mlp = original_mlp.clone()
        expected_mlp[0] *= -1
        assert torch.equal(layer.self_attn.o_proj.weight, expected_attention)
        assert torch.equal(layer.mlp.down_proj.weight, expected_mlp)

        evidence = {item["parameter_name"]: item for item in lease.projection_evidence}
        attention = evidence[
            "model.language_model.layers.0.self_attn.o_proj.weight"
        ]
        mlp = evidence["model.language_model.layers.0.mlp.down_proj.weight"]
        assert attention["layer_index"] == 0
        assert attention["basis_rank"] == 1
        assert attention["requested_strengths"] == [1.0]
        assert attention["normalized_residual_ratio"] == pytest.approx(0.0)
        assert attention["coordinate_retention_factors"] == [0.0]
        assert attention["weight_delta_norm"] > 0.0
        assert len(attention["basis_sha256"]) == 64
        assert len(attention["edited_weight_binding_sha256"]) == 64
        assert attention["exact_restoration_verified"] is False
        assert mlp["coordinate_retention_factors"] == [-1.0]
        assert mlp["normalized_residual_ratio"] == pytest.approx(1.0)

    assert torch.equal(layer.self_attn.o_proj.weight, original_attention)
    assert torch.equal(layer.mlp.down_proj.weight, original_mlp)
    assert all(
        item["exact_restoration_verified"] is True
        for item in lease.projection_evidence
    )


def test_identity_edit_emits_zero_delta_projection_receipts() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    _fill_weights(model)
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=torch.eye(3)[:, :1],
            attention_strength=0.0,
            mlp_strength=0.0,
        )
    )

    lease = _runtime().activate(model, compiled)
    with lease:
        assert len(lease.projection_evidence) == 2
        assert all(
            item["weight_delta_norm"] == 0.0
            for item in lease.projection_evidence
        )
        assert all(
            item["normalized_residual_ratio"] == pytest.approx(1.0)
            for item in lease.projection_evidence
        )


def test_rank_k_edit_uses_the_selected_subspace() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    _fill_weights(model)
    layer = model.model.language_model.layers[0]
    original = layer.self_attn.o_proj.weight.detach().clone()
    original_mlp = layer.mlp.down_proj.weight.detach().clone()
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            ),
            attention_strength=0.5,
            mlp_strength=0.0,
        )
    )

    with _runtime().activate(model, compiled):
        expected = original.clone()
        expected[:2] *= 0.5
        assert torch.equal(layer.self_attn.o_proj.weight, expected)
        assert torch.equal(layer.mlp.down_proj.weight, original_mlp)


def test_context_restores_exact_bytes_after_body_raises() -> None:
    model = FakeQwen()
    _fill_weights(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=1,
            basis=torch.eye(3)[:, :2],
            attention_strength=0.75,
            mlp_strength=1.25,
        )
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        with _runtime().activate(model, compiled):
            raise RuntimeError("evaluation failed")

    after = dict(model.named_parameters())
    assert set(after) == set(before)
    for name, original in before.items():
        assert torch.equal(after[name], original), name


def test_only_selected_language_writers_change() -> None:
    model = FakeQwen()
    _fill_weights(model)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=torch.tensor([[0.0], [1.0], [0.0]]),
            attention_strength=1.0,
            mlp_strength=0.0,
        )
    )

    with _runtime().activate(model, compiled):
        for name, parameter in model.named_parameters():
            if name == "model.language_model.layers.0.self_attn.o_proj.weight":
                assert not torch.equal(parameter, before[name])
            else:
                assert torch.equal(parameter, before[name]), name


def test_runtime_allows_only_one_active_lease() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=torch.eye(3)[:, :1],
            attention_strength=1.0,
            mlp_strength=1.0,
        )
    )
    runtime = _runtime()
    lease = runtime.activate(model, compiled)

    with pytest.raises(WriterEditError, match="active lease"):
        runtime.activate(model, compiled)

    lease.close()
    replacement = runtime.activate(model, compiled)
    replacement.close()


def test_materialized_weights_match_the_live_overlay_without_mutating_model() -> None:
    model = FakeQwen()
    _fill_weights(model)
    before = deepcopy(model.state_dict())
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=1,
            basis=torch.eye(3)[:, :2],
            attention_strength=0.25,
            mlp_strength=1.5,
        )
    )

    materialized = materialized_writer_weights(
        model, compiled, verified_model_sha256="a" * 64
    )

    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name
    with _runtime().activate(model, compiled):
        for name, expected in materialized.items():
            assert torch.equal(model.state_dict()[name], expected), name


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], -0.1, 0.0),
            "between 0 and 2",
        ),
        (
            CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 0.0, 2.1),
            "between 0 and 2",
        ),
        (
            CompiledLayerWriterEdit(
                0,
                torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
                1.0,
                1.0,
            ),
            "orthonormal",
        ),
        (
            CompiledLayerWriterEdit(0, torch.tensor([[float("nan")], [0.0], [0.0]]), 1.0, 1.0),
            "finite",
        ),
    ],
)
def test_compiled_edits_fail_closed(edit: CompiledLayerWriterEdit, message: str) -> None:
    model = FakeQwen()

    with pytest.raises(WriterEditError, match=message):
        _runtime().activate(model, _compiled(edit))


def test_model_shape_and_layer_discovery_fail_closed_before_mutation() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    _fill_weights(model)
    before = deepcopy(model.state_dict())
    wrong_width = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(4)[:, :1], 1.0, 1.0)
    )

    with pytest.raises(WriterEditError, match="hidden width"):
        _runtime().activate(model, wrong_width)
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name

    with pytest.raises(WriterEditError, match="Qwen decoder layers"):
        _runtime().activate(nn.Linear(3, 3), wrong_width)


def test_duplicate_or_out_of_range_layers_fail_closed() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    edit = CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 1.0, 1.0)

    with pytest.raises(WriterEditError, match="unique"):
        _runtime().activate(model, _compiled(edit, edit))
    with pytest.raises(WriterEditError, match="outside"):
        _runtime().activate(
            model,
            _compiled(CompiledLayerWriterEdit(1, torch.eye(3)[:, :1], 1.0, 1.0)),
        )


def test_model_identity_must_match_before_mutation() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    _fill_weights(model)
    before = deepcopy(model.state_dict())
    compiled = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 1.0, 1.0)
    )

    with pytest.raises(WriterEditError, match="model identity"):
        _runtime(model_sha256="b" * 64).activate(model, compiled)
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


def test_active_lease_is_exclusive_across_runtime_instances() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    _fill_weights(model)
    before = deepcopy(model.state_dict())
    compiled = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 1.0, 1.0)
    )
    first = _runtime().activate(model, compiled)

    with pytest.raises(WriterEditError, match="already has an active writer edit"):
        _runtime().activate(model, compiled)

    first.close()
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


def test_per_basis_coefficients_apply_general_diagonal_lambda() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    _fill_weights(model)
    layer = model.model.language_model.layers[0]
    original = layer.self_attn.o_proj.weight.detach().clone()
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=torch.eye(3)[:, :2],
            attention_strength=(0.5, 2.0),
            mlp_strength=0.0,
        )
    )

    with _runtime().activate(model, compiled):
        expected = original.clone()
        expected[0] *= 0.5
        expected[1] *= -1.0
        assert torch.equal(layer.self_attn.o_proj.weight, expected)


def test_bfloat16_edit_preserves_dtype_and_restores_exactly() -> None:
    model = FakeQwen(hidden=3, layer_count=1).to(dtype=torch.bfloat16)
    _fill_weights(model)
    layer = model.model.language_model.layers[0]
    original = layer.self_attn.o_proj.weight.detach().clone()
    compiled = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 1.0, 0.0)
    )

    with _runtime().activate(model, compiled):
        assert layer.self_attn.o_proj.weight.dtype == torch.bfloat16
        assert torch.isfinite(layer.self_attn.o_proj.weight).all()
        assert torch.count_nonzero(layer.self_attn.o_proj.weight[0]) == 0

    assert torch.equal(layer.self_attn.o_proj.weight, original)


def test_zero_strength_lease_does_not_touch_parameter_versions() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    layer = model.model.language_model.layers[0]
    attention_version = layer.self_attn.o_proj.weight._version
    mlp_version = layer.mlp.down_proj.weight._version
    compiled = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 0.0, 0.0)
    )

    with _runtime().activate(model, compiled) as lease:
        assert lease.edited_parameters == ()

    assert layer.self_attn.o_proj.weight._version == attention_version
    assert layer.mlp.down_proj.weight._version == mlp_version


def test_shared_writer_storage_fails_closed() -> None:
    model = FakeQwen(hidden=3, layer_count=2)
    first = model.model.language_model.layers[0].self_attn.o_proj.weight
    model.model.language_model.layers[1].self_attn.o_proj.weight = nn.Parameter(
        first.detach()
    )
    compiled = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 1.0, 1.0)
    )

    with pytest.raises(WriterEditError, match="share storage"):
        _runtime().activate(model, compiled)


def test_replaced_parameter_during_lease_is_detected_and_model_is_poisoned() -> None:
    model = FakeQwen(hidden=3, layer_count=1)
    compiled = _compiled(
        CompiledLayerWriterEdit(0, torch.eye(3)[:, :1], 1.0, 1.0)
    )
    runtime = _runtime()
    lease = runtime.activate(model, compiled)
    layer = model.model.language_model.layers[0]
    layer.self_attn.o_proj.weight = nn.Parameter(
        layer.self_attn.o_proj.weight.detach().clone()
    )

    with pytest.raises(WriterEditError, match="failed to restore"):
        lease.close()
    with pytest.raises(WriterEditError, match="poisoned"):
        _runtime().activate(model, compiled)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="optional local MPS diagnostic requires Apple Metal",
)
def test_diagnostic_mps_rank_k_apply_and_exact_restore() -> None:
    """Diagnostic tensor check only; this is not CUDA/runtime evidence."""

    model = FakeQwen(hidden=3, layer_count=1).to(device="mps", dtype=torch.float32)
    _fill_weights(model)
    layer = model.model.language_model.layers[0]
    original_attention = layer.self_attn.o_proj.weight.detach().clone()
    original_mlp = layer.mlp.down_proj.weight.detach().clone()
    basis = torch.tensor(
        [
            [2**-0.5, 2**-0.5],
            [2**-0.5, -(2**-0.5)],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    compiled = _compiled(
        CompiledLayerWriterEdit(
            layer_index=0,
            basis=basis,
            attention_strength=(0.5, 1.0),
            mlp_strength=(1.0, 2.0),
        )
    )

    with _runtime().activate(model, compiled):
        assert torch.isfinite(layer.self_attn.o_proj.weight).all().item()
        assert torch.isfinite(layer.mlp.down_proj.weight).all().item()
        assert not torch.allclose(layer.self_attn.o_proj.weight, original_attention)
        assert not torch.allclose(layer.mlp.down_proj.weight, original_mlp)

    assert torch.equal(layer.self_attn.o_proj.weight, original_attention)
    assert torch.equal(layer.mlp.down_proj.weight, original_mlp)
