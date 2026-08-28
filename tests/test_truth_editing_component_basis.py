from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from intelligent_liars.truth_editing_component_basis import (
    ComponentBasisError,
    ComponentBasisInput,
    ComponentLayerStrength,
    ComponentRankStrengths,
    ComponentStrengthPlan,
    compile_component_basis_set,
    compile_component_writer_edit,
    compile_refusal_basis_set,
)
from intelligent_liars.truth_editing_directions import (
    CompiledBasis,
    CompiledBasisSet,
    _basis_set_hash,
    vector_sha256,
)
from intelligent_liars.truth_editing_qwen_runtime import (
    QwenTrialRuntimeError,
    TrialExample,
    TrialRuntimeBatch,
    WriterStrengthPlan,
    compile_writer_edit,
)
from intelligent_liars.truth_editing_refusal_directions import (
    BANK_FORMAT,
    LAYER_RECEIPT_FORMAT,
    RefusalDirectionBank,
    RefusalDirectionLayerReceipt,
    canonical_sha256,
)


MODEL_SHA = "b" * 64


def _matrix_sha(matrix: np.ndarray) -> str:
    canonical = np.asarray(matrix, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _plain_basis_set(
    matrix: np.ndarray, *, direction_id: str, manifest: str
) -> CompiledBasisSet:
    frozen = np.asarray(matrix, dtype=np.float64)
    basis = CompiledBasis(
        direction_ids=(direction_id,),
        method="qr",
        requested_rank=1,
        matrix=frozen,
        basis_sha256=vector_sha256(frozen[:, 0]),
    )
    by_layer = ((0, basis),)
    return CompiledBasisSet(
        manifest_sha256=manifest,
        method="qr",
        requested_rank=1,
        by_layer=by_layer,
        basis_set_sha256=_basis_set_hash(manifest, "qr", 1, by_layer),
    )


def _inputs(*, overlapping_refusal: bool = False) -> tuple[ComponentBasisInput, ...]:
    truth = _plain_basis_set(
        np.array([[1.0], [0.0], [0.0]]), direction_id="truth", manifest="1" * 64
    )
    refusal_vector = (
        np.array([[2**-0.5], [2**-0.5], [0.0]])
        if overlapping_refusal
        else np.array([[0.0], [1.0], [0.0]])
    )
    refusal = _plain_basis_set(
        refusal_vector, direction_id="refusal", manifest="2" * 64
    )
    return (
        ComponentBasisInput("truth", truth, "3" * 64),
        ComponentBasisInput("refusal_raw", refusal, "4" * 64),
    )


def _plan(truth: float, refusal: float) -> ComponentStrengthPlan:
    return ComponentStrengthPlan(
        components=(
            ComponentRankStrengths(
                "truth", (ComponentLayerStrength(0, (truth,), (truth,)),)
            ),
            ComponentRankStrengths(
                "refusal_raw", (ComponentLayerStrength(0, (refusal,), (refusal,)),)
            ),
        )
    )


def test_component_basis_and_strengths_compile_to_exact_flattened_edit() -> None:
    combined = compile_component_basis_set(model_sha256=MODEL_SHA, components=_inputs())
    combined.verify()
    layer = combined.by_layer[0][1]
    assert tuple(item.label for item in layer.components) == ("truth", "refusal_raw")
    assert tuple((item.start, item.stop) for item in layer.components) == (
        (0, 1),
        (1, 2),
    )
    assert not layer.matrix.flags.writeable
    with pytest.raises(ValueError):
        layer.matrix.flags.writeable = True
    assert all(
        item.source_vector_sha256s and item.output_vector_sha256s
        for item in layer.components
    )

    edit = compile_component_writer_edit(
        recipe_id="joint", basis_set=combined, strengths=_plan(0.25, 1.5)
    )
    assert edit.model_sha256 == MODEL_SHA
    assert edit.layers[0].attention_strength == (0.25, 1.5)
    assert edit.layers[0].mlp_strength == (0.25, 1.5)


def test_independent_component_strengths_change_edits_and_zero_is_exact_disable() -> (
    None
):
    combined = compile_component_basis_set(model_sha256=MODEL_SHA, components=_inputs())
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)

    def edited(truth: float, refusal: float) -> torch.Tensor:
        compiled = compile_component_writer_edit(
            recipe_id="recipe", basis_set=combined, strengths=_plan(truth, refusal)
        )
        basis = compiled.layers[0].basis
        lambdas = torch.tensor(compiled.layers[0].attention_strength).unsqueeze(1)
        return weight - basis @ ((basis.T @ weight) * lambdas)

    truth_only = edited(1.0, 0.0)
    refusal_only = edited(0.0, 1.0)
    joint = edited(0.4, 1.3)
    assert not torch.equal(truth_only, refusal_only)
    assert not torch.equal(joint, truth_only)
    assert torch.equal(truth_only[1], weight[1])
    assert torch.equal(refusal_only[0], weight[0])


def test_optional_refusal_orthogonalization_is_receipted_and_deterministic() -> None:
    combined = compile_component_basis_set(
        model_sha256=MODEL_SHA,
        components=_inputs(overlapping_refusal=True),
        orthogonalize_refusal=True,
    )
    combined.verify()
    layer = combined.by_layer[0][1]
    assert tuple(item.label for item in layer.components) == (
        "truth",
        "truth_orthogonalized_refusal",
    )
    assert combined.orthogonalization_receipt is not None
    assert np.allclose(layer.matrix.T @ layer.matrix, np.eye(2), atol=1e-10, rtol=0)
    repeated = compile_component_basis_set(
        model_sha256=MODEL_SHA,
        components=_inputs(overlapping_refusal=True),
        orthogonalize_refusal=True,
    )
    assert repeated.component_basis_set_sha256 == combined.component_basis_set_sha256


@pytest.mark.parametrize(
    "components,orthogonalize,match",
    [
        ((_inputs()[1], _inputs()[0]), False, "canonical order"),
        ((_inputs()[0], _inputs()[0]), False, "unique"),
        (_inputs(overlapping_refusal=True), False, "overlap"),
    ],
)
def test_component_compiler_fails_closed_on_order_duplicate_and_overlap(
    components: tuple[ComponentBasisInput, ...], orthogonalize: bool, match: str
) -> None:
    with pytest.raises(ComponentBasisError, match=match):
        compile_component_basis_set(
            model_sha256=MODEL_SHA,
            components=components,
            orthogonalize_refusal=orthogonalize,
        )


def test_strength_compiler_rejects_missing_component_wrong_rank_and_tamper() -> None:
    combined = compile_component_basis_set(model_sha256=MODEL_SHA, components=_inputs())
    with pytest.raises(ComponentBasisError, match="exactly match"):
        compile_component_writer_edit(
            recipe_id="bad",
            basis_set=combined,
            strengths=ComponentStrengthPlan(
                components=(_plan(1.0, 1.0).components[0],)
            ),
        )
    wrong = ComponentStrengthPlan(
        components=(
            ComponentRankStrengths(
                "truth", (ComponentLayerStrength(0, (1.0, 1.0), (1.0, 1.0)),)
            ),
            _plan(1.0, 1.0).components[1],
        )
    )
    with pytest.raises(ComponentBasisError, match="rank"):
        compile_component_writer_edit(
            recipe_id="bad", basis_set=combined, strengths=wrong
        )
    damaged = dataclasses.replace(combined, model_sha256="c" * 64)
    with pytest.raises(ComponentBasisError, match="identity"):
        damaged.verify()


def test_qwen_runtime_dispatches_component_plan_and_rejects_ambiguous_pairing() -> None:
    combined = compile_component_basis_set(model_sha256=MODEL_SHA, components=_inputs())
    edit = compile_writer_edit(
        recipe_id="joint",
        model_sha256=MODEL_SHA,
        basis_set=combined,
        strengths=_plan(0.3, 1.4),
    )
    assert edit.layers[0].attention_strength == (0.3, 1.4)
    with pytest.raises(QwenTrialRuntimeError, match="component basis sets require"):
        compile_writer_edit(
            recipe_id="ambiguous",
            model_sha256=MODEL_SHA,
            basis_set=combined,
            strengths=WriterStrengthPlan({0: (0.3, 1.4)}, {0: (0.3, 1.4)}),
        )


def test_runtime_batch_identity_binds_independent_component_strengths() -> None:
    combined = compile_component_basis_set(model_sha256=MODEL_SHA, components=_inputs())
    example = TrialExample(
        record_id="row-1",
        messages=({"role": "user", "content": "Choose."},),
        truthful_target_text="A",
        false_target_text="B",
    )
    first = TrialRuntimeBatch(
        batch_id="batch-1",
        recipe_id="joint",
        model_sha256=MODEL_SHA,
        basis_set=combined,
        strengths=_plan(0.2, 1.3),
        examples=(example,),
    )
    second = dataclasses.replace(first, strengths=_plan(1.3, 0.2))
    assert first.batch_sha256 != second.batch_sha256
    with pytest.raises(QwenTrialRuntimeError, match="component basis sets require"):
        dataclasses.replace(
            first,
            strengths=WriterStrengthPlan({0: (0.2, 1.3)}, {0: (0.2, 1.3)}),
        )


@pytest.mark.parametrize("label", ["truth", "refusal_raw", "orthogonal_control"])
def test_truth_refusal_and_control_only_are_independently_executable(
    label: str,
) -> None:
    source = _plain_basis_set(
        np.array([[1.0], [0.0], [0.0]]),
        direction_id=label,
        manifest="6" * 64,
    )
    combined = compile_component_basis_set(
        model_sha256=MODEL_SHA,
        components=(ComponentBasisInput(label, source, "7" * 64),),  # type: ignore[arg-type]
    )
    plan = ComponentStrengthPlan(
        components=(
            ComponentRankStrengths(
                label,  # type: ignore[arg-type]
                (ComponentLayerStrength(0, (0.8,), (1.2,)),),
            ),
        )
    )
    edit = compile_component_writer_edit(
        recipe_id=f"{label}-only", basis_set=combined, strengths=plan
    )
    assert edit.layers[0].attention_strength == (0.8,)
    assert edit.layers[0].mlp_strength == (1.2,)


def test_source_substitution_changes_identity_and_width_mismatch_fails_closed() -> None:
    first = compile_component_basis_set(model_sha256=MODEL_SHA, components=_inputs())
    substituted_inputs = list(_inputs())
    substituted_inputs[0] = dataclasses.replace(
        substituted_inputs[0], source_sha256="8" * 64
    )
    second = compile_component_basis_set(
        model_sha256=MODEL_SHA, components=tuple(substituted_inputs)
    )
    assert first.component_basis_set_sha256 != second.component_basis_set_sha256

    narrow = _plain_basis_set(
        np.array([[1.0], [0.0]]), direction_id="refusal", manifest="9" * 64
    )
    with pytest.raises(ComponentBasisError, match="hidden widths"):
        compile_component_basis_set(
            model_sha256=MODEL_SHA,
            components=(
                _inputs()[0],
                ComponentBasisInput("refusal_raw", narrow, "a" * 64),
            ),
        )


def _refusal_bank(root: Path) -> RefusalDirectionBank:
    receipts = []
    for layer, vector in enumerate(
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    ):
        relative = Path("vectors") / f"layer-{layer:02d}.npy"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector, allow_pickle=False)
        unsigned = {
            "format": LAYER_RECEIPT_FORMAT,
            "receipt_id": f"refusal-{layer}",
            "source_layer": layer,
            "width": 3,
            "construction_harmless_count": 2,
            "construction_harmful_count": 2,
            "harmless_mean_sha256": "1" * 64,
            "harmful_mean_sha256": "2" * 64,
            "vector_path": str(relative),
            "vector_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "vector_sha256": _matrix_sha(vector),
            "finite": True,
            "unit_norm": True,
        }
        receipts.append(
            RefusalDirectionLayerReceipt(
                **unsigned, self_sha256=canonical_sha256(unsigned)
            )
        )
    unsigned_bank = {
        "format": BANK_FORMAT,
        "bank_id": "mock-refusal",
        "config_sha256": "3" * 64,
        "prompt_manifest_sha256": "4" * 64,
        "model_sha256": MODEL_SHA,
        "chat_template_sha256": "5" * 64,
        "per_layer_receipts": tuple(receipts),
        "global_source_receipt_ids": tuple(item.receipt_id for item in receipts),
    }
    return RefusalDirectionBank(
        **unsigned_bank,
        self_sha256=canonical_sha256(asdict(_BankUnsigned(**unsigned_bank))),
    )


@dataclasses.dataclass(frozen=True)
class _BankUnsigned:
    format: str
    bank_id: str
    config_sha256: str
    prompt_manifest_sha256: str
    model_sha256: str
    chat_template_sha256: str
    per_layer_receipts: tuple[RefusalDirectionLayerReceipt, ...]
    global_source_receipt_ids: tuple[str, ...]


def test_public_refusal_loader_supports_global_and_per_layer_relocation(
    tmp_path: Path,
) -> None:
    bank = _refusal_bank(tmp_path)
    global_basis = compile_refusal_basis_set(
        bank=bank,
        artifact_root=tmp_path,
        destination_layers=(0, 1),
        source_scope="global",
        source_layer=0,
        expected_model_sha256=MODEL_SHA,
    )
    global_basis.verify()
    assert global_basis.model_sha256 == MODEL_SHA
    assert global_basis.source_by_destination == ((0, 0), (1, 0))
    assert np.array_equal(
        global_basis.by_layer[0][1].matrix, global_basis.by_layer[1][1].matrix
    )

    per_layer = compile_refusal_basis_set(
        bank=bank,
        artifact_root=tmp_path,
        destination_layers=(0, 1),
        source_scope="per_layer",
        source_layer=None,
        expected_model_sha256=MODEL_SHA,
    )
    per_layer.verify()
    assert per_layer.model_sha256 is None
    assert not np.array_equal(
        per_layer.by_layer[0][1].matrix, per_layer.by_layer[1][1].matrix
    )


def test_refusal_loader_rejects_model_mismatch_and_vector_substitution(
    tmp_path: Path,
) -> None:
    bank = _refusal_bank(tmp_path)
    with pytest.raises(ComponentBasisError, match="model identity"):
        compile_refusal_basis_set(
            bank=bank,
            artifact_root=tmp_path,
            destination_layers=(0,),
            source_scope="global",
            source_layer=0,
            expected_model_sha256="c" * 64,
        )
    np.save(tmp_path / "vectors/layer-00.npy", np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ComponentBasisError, match="file identity"):
        compile_refusal_basis_set(
            bank=bank,
            artifact_root=tmp_path,
            destination_layers=(0,),
            source_scope="global",
            source_layer=0,
            expected_model_sha256=MODEL_SHA,
        )
