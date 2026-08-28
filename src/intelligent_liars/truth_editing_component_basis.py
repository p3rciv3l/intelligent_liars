"""Immutable component-aware bases for persistent truth/refusal writer edits.

This module is the only place where independently qualified direction families
are combined.  It preserves labeled rank slices and source/vector identities,
optionally removes the truth projection from a raw refusal basis, and expands
explicit per-component strengths into the flat diagonal consumed by the exact
writer editor.  It never infers that two components should share a strength.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from .truth_editing_directions import (
    CompiledBasis,
    CompiledBasisSet,
    _basis_set_hash,
    _destination_basis_hashes,
    vector_sha256,
)
from .truth_editing_refusal_directions import (
    BANK_FORMAT,
    LAYER_RECEIPT_FORMAT,
    RefusalDirectionBank,
)
from .truth_editing_weight_editor import CompiledLayerWriterEdit, CompiledWriterEdit


ComponentLabel = Literal[
    "truth",
    "refusal_raw",
    "truth_orthogonalized_refusal",
    "orthogonal_control",
    "shuffled_control",
]
InputComponentLabel = Literal[
    "truth",
    "refusal_raw",
    "truth_orthogonalized_refusal",
    "orthogonal_control",
    "shuffled_control",
]

FORMAT: Literal["truth_editing_component_basis_set_v1"] = (
    "truth_editing_component_basis_set_v1"
)
ORTHOGONALIZATION_FORMAT: Literal["truth_editing_component_orthogonalization_v1"] = (
    "truth_editing_component_orthogonalization_v1"
)
_LABEL_ORDER = {
    "truth": 0,
    "refusal_raw": 1,
    "truth_orthogonalized_refusal": 1,
    "orthogonal_control": 2,
    "shuffled_control": 3,
}
_HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


class ComponentBasisError(ValueError):
    """Component bases or strengths cannot be bound without ambiguity."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ComponentBasisError("component receipt is not canonical JSON") from error


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA.fullmatch(value) is None:
        raise ComponentBasisError(f"{label} must be a lowercase SHA-256")
    return value


def _vector_sha(vector: np.ndarray) -> str:
    canonical = np.asarray(vector, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _matrix_sha(matrix: np.ndarray) -> str:
    canonical = np.asarray(matrix, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _readonly_matrix(matrix: np.ndarray) -> np.ndarray:
    canonical = np.array(matrix, dtype="<f8", order="C", copy=True)
    result = np.frombuffer(canonical.tobytes(order="C"), dtype="<f8").reshape(
        canonical.shape
    )
    return cast(np.ndarray, result)


@dataclass(frozen=True)
class ComponentBasisInput:
    """One qualified family and the exact external artifact that produced it."""

    label: InputComponentLabel
    basis_set: CompiledBasisSet
    source_sha256: str

    def __post_init__(self) -> None:
        if self.label not in _LABEL_ORDER:
            raise ComponentBasisError("component label is unsupported")
        _sha(self.source_sha256, "component source_sha256")
        try:
            self.basis_set.verify()
        except Exception as error:
            raise ComponentBasisError(
                "component basis set failed verification"
            ) from error


@dataclass(frozen=True)
class ComponentSlice:
    label: ComponentLabel
    start: int
    stop: int
    source_sha256: str
    source_basis_set_sha256: str
    source_basis_sha256: str
    source_matrix_sha256: str
    source_vector_sha256s: tuple[str, ...]
    output_vector_sha256s: tuple[str, ...]
    component_sha256: str

    @property
    def rank(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CombinedLayerComponentBasis:
    layer_index: int
    matrix: np.ndarray
    matrix_sha256: str
    components: tuple[ComponentSlice, ...]


@dataclass(frozen=True)
class LayerOrthogonalizationReceipt:
    layer_index: int
    truth_matrix_sha256: str
    refusal_raw_matrix_sha256: str
    refusal_output_matrix_sha256: str
    source_vector_sha256s: tuple[str, ...]
    output_vector_sha256s: tuple[str, ...]
    removed_projection_frobenius_norm: float
    self_sha256: str

    def unsigned(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "truth_matrix_sha256": self.truth_matrix_sha256,
            "refusal_raw_matrix_sha256": self.refusal_raw_matrix_sha256,
            "refusal_output_matrix_sha256": self.refusal_output_matrix_sha256,
            "source_vector_sha256s": list(self.source_vector_sha256s),
            "output_vector_sha256s": list(self.output_vector_sha256s),
            "removed_projection_frobenius_norm": self.removed_projection_frobenius_norm,
        }


@dataclass(frozen=True)
class ComponentOrthogonalizationReceipt:
    format: Literal["truth_editing_component_orthogonalization_v1"]
    layers: tuple[LayerOrthogonalizationReceipt, ...]
    self_sha256: str


@dataclass(frozen=True)
class CombinedComponentBasisSet:
    """Model-bound combined orthonormal basis with immutable labeled slices."""

    format: Literal["truth_editing_component_basis_set_v1"]
    model_sha256: str
    source_components: tuple[tuple[ComponentLabel, str, str], ...]
    by_layer: tuple[tuple[int, CombinedLayerComponentBasis], ...]
    orthogonalization_receipt: ComponentOrthogonalizationReceipt | None
    component_basis_set_sha256: str

    @property
    def basis_set_sha256(self) -> str:
        """Compatibility identity used by the existing runtime receipts."""

        return self.component_basis_set_sha256

    def _unsigned(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model_sha256": self.model_sha256,
            "source_components": [
                {
                    "label": label,
                    "source_sha256": source_sha,
                    "basis_set_sha256": basis_sha,
                }
                for label, source_sha, basis_sha in self.source_components
            ],
            "layers": [_layer_payload(layer) for _, layer in self.by_layer],
            "orthogonalization_receipt_sha256": (
                self.orthogonalization_receipt.self_sha256
                if self.orthogonalization_receipt is not None
                else None
            ),
        }

    def verify(self) -> None:
        _sha(self.model_sha256, "component basis model_sha256")
        if self.format != FORMAT:
            raise ComponentBasisError("component basis format is unsupported")
        source_labels = tuple(label for label, _, _ in self.source_components)
        _validate_labels(source_labels)
        for _, source_sha, basis_sha in self.source_components:
            _sha(source_sha, "component source_sha256")
            _sha(basis_sha, "component basis_set_sha256")
        layer_indices = tuple(index for index, _ in self.by_layer)
        if not layer_indices or layer_indices != tuple(sorted(set(layer_indices))):
            raise ComponentBasisError(
                "component basis layers must be nonempty sorted unique"
            )
        expected_labels = tuple(
            _output_label(label, self.orthogonalization_receipt is not None)
            for label in source_labels
        )
        for index, layer in self.by_layer:
            if layer.layer_index != index:
                raise ComponentBasisError("component layer index binding differs")
            _verify_layer(layer, expected_labels)
        _verify_orthogonalization(self.orthogonalization_receipt, self.by_layer)
        if _hash(self._unsigned()) != self.component_basis_set_sha256:
            raise ComponentBasisError("component basis-set identity mismatch")


@dataclass(frozen=True)
class ComponentLayerStrength:
    """Exact diagonal coefficients for one component at one layer and writer site."""

    layer_index: int
    attention: tuple[float, ...]
    mlp: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ComponentBasisError(
                "strength layer_index must be a nonnegative integer"
            )
        object.__setattr__(
            self, "attention", _strength_tuple(self.attention, "attention")
        )
        object.__setattr__(self, "mlp", _strength_tuple(self.mlp, "MLP"))


@dataclass(frozen=True)
class ComponentRankStrengths:
    label: ComponentLabel
    by_layer: tuple[ComponentLayerStrength, ...]

    def __post_init__(self) -> None:
        if self.label not in _LABEL_ORDER:
            raise ComponentBasisError("strength component label is unsupported")
        layers = tuple(item.layer_index for item in self.by_layer)
        if not layers or layers != tuple(sorted(set(layers))):
            raise ComponentBasisError(
                "component strength layers must be nonempty sorted unique"
            )


@dataclass(frozen=True)
class ComponentStrengthPlan:
    """Triangular component -> layer -> writer-site -> rank coefficient plan."""

    components: tuple[ComponentRankStrengths, ...]

    def __post_init__(self) -> None:
        labels = tuple(component.label for component in self.components)
        _validate_labels(labels)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "components": [
                {
                    "label": component.label,
                    "layers": [
                        {
                            "layer_index": layer.layer_index,
                            "attention": list(layer.attention),
                            "mlp": list(layer.mlp),
                        }
                        for layer in component.by_layer
                    ],
                }
                for component in self.components
            ]
        }


def compile_component_basis_set(
    *,
    model_sha256: str,
    components: tuple[ComponentBasisInput, ...],
    orthogonalize_refusal: bool = False,
) -> CombinedComponentBasisSet:
    """Combine exact component bases, optionally residualizing refusal against truth."""

    model_sha = _sha(model_sha256, "model_sha256")
    if not isinstance(orthogonalize_refusal, bool):
        raise ComponentBasisError("orthogonalize_refusal must be boolean")
    labels = tuple(component.label for component in components)
    _validate_labels(labels)
    if orthogonalize_refusal and labels[:2] != ("truth", "refusal_raw"):
        raise ComponentBasisError(
            "refusal orthogonalization requires canonical truth then refusal_raw components"
        )
    for component in components:
        if component.basis_set.model_sha256 not in {None, model_sha}:
            raise ComponentBasisError("component basis model identity mismatch")
    layer_sets = tuple(
        tuple(index for index, _ in component.basis_set.by_layer)
        for component in components
    )
    if not layer_sets or any(layers != layer_sets[0] for layers in layer_sets[1:]):
        raise ComponentBasisError("component basis layers must exactly match")
    output_layers: list[tuple[int, CombinedLayerComponentBasis]] = []
    orthogonalization_layers: list[LayerOrthogonalizationReceipt] = []
    for layer_position, layer_index in enumerate(layer_sets[0]):
        matrices = [
            component.basis_set.by_layer[layer_position][1].matrix
            for component in components
        ]
        widths = {int(matrix.shape[0]) for matrix in matrices}
        if len(widths) != 1:
            raise ComponentBasisError("component basis hidden widths differ")
        output_matrices = [_readonly_matrix(matrix) for matrix in matrices]
        if orthogonalize_refusal:
            refusal_output, receipt = _orthogonalize_refusal(
                layer_index, output_matrices[0], output_matrices[1]
            )
            output_matrices[1] = refusal_output
            orthogonalization_layers.append(receipt)
        combined_matrix = _readonly_matrix(np.concatenate(output_matrices, axis=1))
        gram = combined_matrix.T @ combined_matrix
        if not np.allclose(gram, np.eye(gram.shape[0]), rtol=0.0, atol=1e-10):
            raise ComponentBasisError(
                "component bases overlap or are not jointly normalized"
            )
        offset = 0
        slices: list[ComponentSlice] = []
        for component, source_matrix, output_matrix in zip(
            components, matrices, output_matrices, strict=True
        ):
            rank = int(output_matrix.shape[1])
            label = _output_label(component.label, orthogonalize_refusal)
            source_vectors = tuple(
                _vector_sha(source_matrix[:, i]) for i in range(rank)
            )
            output_vectors = tuple(
                _vector_sha(output_matrix[:, i]) for i in range(rank)
            )
            source_basis_sha = component.basis_set.by_layer[layer_position][
                1
            ].basis_sha256
            source_matrix_sha = _matrix_sha(source_matrix)
            unsigned = {
                "label": label,
                "start": offset,
                "stop": offset + rank,
                "source_sha256": component.source_sha256,
                "source_basis_set_sha256": component.basis_set.basis_set_sha256,
                "source_basis_sha256": source_basis_sha,
                "source_matrix_sha256": source_matrix_sha,
                "source_vector_sha256s": source_vectors,
                "output_vector_sha256s": output_vectors,
            }
            slices.append(
                ComponentSlice(
                    label=label,
                    start=offset,
                    stop=offset + rank,
                    source_sha256=component.source_sha256,
                    source_basis_set_sha256=component.basis_set.basis_set_sha256,
                    source_basis_sha256=source_basis_sha,
                    source_matrix_sha256=source_matrix_sha,
                    source_vector_sha256s=source_vectors,
                    output_vector_sha256s=output_vectors,
                    component_sha256=_hash(unsigned),
                )
            )
            offset += rank
        output_layers.append(
            (
                layer_index,
                CombinedLayerComponentBasis(
                    layer_index=layer_index,
                    matrix=combined_matrix,
                    matrix_sha256=_matrix_sha(combined_matrix),
                    components=tuple(slices),
                ),
            )
        )
    ortho_receipt = _orthogonalization_receipt(tuple(orthogonalization_layers))
    source_components = tuple(
        (
            _output_label(component.label, orthogonalize_refusal),
            component.source_sha256,
            component.basis_set.basis_set_sha256,
        )
        for component in components
    )
    provisional = CombinedComponentBasisSet(
        format=FORMAT,
        model_sha256=model_sha,
        source_components=source_components,
        by_layer=tuple(output_layers),
        orthogonalization_receipt=ortho_receipt,
        component_basis_set_sha256="0" * 64,
    )
    result = CombinedComponentBasisSet(
        format=FORMAT,
        model_sha256=model_sha,
        source_components=source_components,
        by_layer=tuple(output_layers),
        orthogonalization_receipt=ortho_receipt,
        component_basis_set_sha256=_hash(provisional._unsigned()),
    )
    result.verify()
    return result


def compile_refusal_basis_set(
    *,
    bank: RefusalDirectionBank,
    artifact_root: str | Path,
    destination_layers: tuple[int, ...],
    source_scope: Literal["global", "per_layer"],
    source_layer: int | None,
    expected_model_sha256: str,
) -> CompiledBasisSet:
    """Load and relocate a verified raw-refusal bank for an executable recipe.

    ``global`` repeats one explicitly selected source layer at every destination.
    ``per_layer`` requires a bank vector at each identically numbered destination.
    The returned basis contains no unverified paths or mutable arrays.
    """

    model_sha = _sha(expected_model_sha256, "expected_model_sha256")
    if bank.format != BANK_FORMAT:
        raise ComponentBasisError("refusal bank format is unsupported")
    bank_unsigned = asdict(bank)
    bank_claimed = bank_unsigned.pop("self_sha256")
    if _hash(bank_unsigned) != bank_claimed:
        raise ComponentBasisError("refusal bank identity mismatch")
    if bank.model_sha256 != model_sha:
        raise ComponentBasisError("refusal bank model identity mismatch")
    _sha(bank.self_sha256, "refusal bank self_sha256")
    if (
        not destination_layers
        or destination_layers != tuple(sorted(set(destination_layers)))
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in destination_layers
        )
    ):
        raise ComponentBasisError(
            "refusal destination layers must be nonempty sorted unique integers"
        )
    if source_scope not in {"global", "per_layer"}:
        raise ComponentBasisError("refusal source scope is unsupported")
    if source_scope == "global":
        if (
            isinstance(source_layer, bool)
            or not isinstance(source_layer, int)
            or source_layer < 0
        ):
            raise ComponentBasisError("global refusal scope requires one source layer")
        source_layers = (source_layer,) * len(destination_layers)
    else:
        if source_layer is not None:
            raise ComponentBasisError(
                "per-layer refusal scope must not provide an ambiguous source layer"
            )
        source_layers = destination_layers
    receipts = {receipt.source_layer: receipt for receipt in bank.per_layer_receipts}
    if len(receipts) != len(bank.per_layer_receipts):
        raise ComponentBasisError("refusal bank source layers must be unique")
    root = Path(artifact_root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ComponentBasisError("refusal artifact root must be a regular directory")
    loaded: dict[int, CompiledBasis] = {}
    for layer in sorted(set(source_layers)):
        receipt = receipts.get(layer)
        if receipt is None:
            raise ComponentBasisError(f"refusal bank lacks source layer {layer}")
        if not receipt.finite or not receipt.unit_norm:
            raise ComponentBasisError("refusal vector receipt is not qualified")
        receipt_unsigned = asdict(receipt)
        receipt_claimed = receipt_unsigned.pop("self_sha256")
        if (
            receipt.format != LAYER_RECEIPT_FORMAT
            or _hash(receipt_unsigned) != receipt_claimed
        ):
            raise ComponentBasisError("refusal layer receipt identity mismatch")
        relative = Path(receipt.vector_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ComponentBasisError("refusal vector path escapes its artifact root")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ComponentBasisError(
                "refusal vector path escapes its artifact root"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise ComponentBasisError("refusal vector must be a regular file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.vector_file_sha256:
            raise ComponentBasisError("refusal vector file identity mismatch")
        try:
            raw = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ComponentBasisError(
                "refusal vector artifact is unreadable"
            ) from error
        vector = np.asarray(raw, dtype=np.float64)
        if vector.shape != (receipt.width,) or not np.isfinite(vector).all():
            raise ComponentBasisError("refusal vector width or finiteness differs")
        if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-10):
            raise ComponentBasisError("refusal vector is not unit normalized")
        if _vector_sha(vector) != receipt.vector_sha256:
            raise ComponentBasisError("refusal vector content identity mismatch")
        matrix = _readonly_matrix(vector[:, None])
        loaded[layer] = CompiledBasis(
            direction_ids=(receipt.receipt_id,),
            method="qr",
            requested_rank=1,
            matrix=matrix,
            basis_sha256=vector_sha256(vector),
        )
    by_layer = tuple(
        (destination, loaded[source])
        for destination, source in zip(destination_layers, source_layers, strict=True)
    )
    if source_scope == "global":
        lineage = tuple(zip(destination_layers, source_layers, strict=True))
        destination_hashes = _destination_basis_hashes(
            bank.self_sha256, model_sha, lineage, by_layer
        )
        result = CompiledBasisSet(
            manifest_sha256=bank.self_sha256,
            method="qr",
            requested_rank=1,
            by_layer=by_layer,
            basis_set_sha256=_basis_set_hash(
                bank.self_sha256,
                "qr",
                1,
                by_layer,
                model_sha256=model_sha,
                source_by_destination=lineage,
                destination_basis_sha256s=destination_hashes,
            ),
            model_sha256=model_sha,
            source_by_destination=lineage,
            destination_basis_sha256s=destination_hashes,
        )
    else:
        result = CompiledBasisSet(
            manifest_sha256=bank.self_sha256,
            method="qr",
            requested_rank=1,
            by_layer=by_layer,
            basis_set_sha256=_basis_set_hash(bank.self_sha256, "qr", 1, by_layer),
        )
    try:
        result.verify()
    except Exception as error:
        raise ComponentBasisError(
            "compiled refusal basis failed verification"
        ) from error
    return result


def compile_component_writer_edit(
    *,
    recipe_id: str,
    basis_set: CombinedComponentBasisSet,
    strengths: ComponentStrengthPlan,
) -> CompiledWriterEdit:
    """Flatten explicit component coefficients into the exact writer-edit diagonal."""

    basis_set.verify()
    if (
        not isinstance(recipe_id, str)
        or not recipe_id
        or recipe_id.strip() != recipe_id
    ):
        raise ComponentBasisError("recipe_id must be a nonempty trimmed string")
    expected_labels = tuple(label for label, _, _ in basis_set.source_components)
    supplied_labels = tuple(component.label for component in strengths.components)
    if supplied_labels != expected_labels:
        raise ComponentBasisError(
            "strength components must exactly match basis components and order"
        )
    component_plans = {component.label: component for component in strengths.components}
    compiled_layers: list[CompiledLayerWriterEdit] = []
    for layer_index, layer in basis_set.by_layer:
        attention: list[float] = []
        mlp: list[float] = []
        for component_slice in layer.components:
            plan = component_plans[component_slice.label]
            matches = tuple(
                item for item in plan.by_layer if item.layer_index == layer_index
            )
            if len(matches) != 1 or len(plan.by_layer) != len(basis_set.by_layer):
                raise ComponentBasisError(
                    "component strength layers must exactly match basis layers"
                )
            item = matches[0]
            if (
                len(item.attention) != component_slice.rank
                or len(item.mlp) != component_slice.rank
            ):
                raise ComponentBasisError(
                    "component strength rank differs from labeled basis slice"
                )
            attention.extend(item.attention)
            mlp.extend(item.mlp)
        compiled_layers.append(
            CompiledLayerWriterEdit(
                layer_index=layer_index,
                basis=torch.from_numpy(np.array(layer.matrix, copy=True)),
                attention_strength=tuple(attention),
                mlp_strength=tuple(mlp),
            )
        )
    return CompiledWriterEdit(
        recipe_id=recipe_id,
        model_sha256=basis_set.model_sha256,
        layers=tuple(compiled_layers),
    )


def _validate_labels(labels: tuple[str, ...]) -> None:
    if not labels:
        raise ComponentBasisError("component sequence must not be empty")
    if len(set(labels)) != len(labels):
        raise ComponentBasisError("component labels must be unique")
    if any(label not in _LABEL_ORDER for label in labels):
        raise ComponentBasisError("component label is unsupported")
    if (
        sum(
            label in {"refusal_raw", "truth_orthogonalized_refusal"} for label in labels
        )
        > 1
    ):
        raise ComponentBasisError("refusal component is ambiguous")
    orders = tuple(_LABEL_ORDER[label] for label in labels)
    if orders != tuple(sorted(orders)):
        raise ComponentBasisError("components must be supplied in canonical order")


def _output_label(label: str, orthogonalized: bool) -> ComponentLabel:
    if label == "refusal_raw" and orthogonalized:
        return "truth_orthogonalized_refusal"
    return label  # type: ignore[return-value]


def _strength_tuple(raw: tuple[float, ...], label: str) -> tuple[float, ...]:
    if not isinstance(raw, tuple) or not raw:
        raise ComponentBasisError(
            f"{label} strengths must be an explicit nonempty tuple"
        )
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ComponentBasisError(
                f"{label} strengths must be finite numbers in [0, 2]"
            )
        value = float(item)
        if not math.isfinite(value) or not 0.0 <= value <= 2.0:
            raise ComponentBasisError(
                f"{label} strengths must be finite numbers in [0, 2]"
            )
        values.append(value)
    return tuple(values)


def _orthogonalize_refusal(
    layer_index: int, truth: np.ndarray, refusal: np.ndarray
) -> tuple[np.ndarray, LayerOrthogonalizationReceipt]:
    projection = truth @ (truth.T @ refusal)
    residual = refusal - projection
    q, r = np.linalg.qr(residual, mode="reduced")
    if q.shape[1] != refusal.shape[1] or np.any(np.abs(np.diag(r)) < 1e-10):
        raise ComponentBasisError(
            "refusal basis loses rank after truth orthogonalization"
        )
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    output = _readonly_matrix(q * signs)
    truth_sha = _matrix_sha(truth)
    refusal_raw_sha = _matrix_sha(refusal)
    refusal_output_sha = _matrix_sha(output)
    source_vectors = tuple(_vector_sha(refusal[:, i]) for i in range(refusal.shape[1]))
    output_vectors = tuple(_vector_sha(output[:, i]) for i in range(output.shape[1]))
    removed_norm = float(np.linalg.norm(projection, ord="fro"))
    unsigned = {
        "layer_index": layer_index,
        "truth_matrix_sha256": truth_sha,
        "refusal_raw_matrix_sha256": refusal_raw_sha,
        "refusal_output_matrix_sha256": refusal_output_sha,
        "source_vector_sha256s": source_vectors,
        "output_vector_sha256s": output_vectors,
        "removed_projection_frobenius_norm": removed_norm,
    }
    return output, LayerOrthogonalizationReceipt(
        layer_index=layer_index,
        truth_matrix_sha256=truth_sha,
        refusal_raw_matrix_sha256=refusal_raw_sha,
        refusal_output_matrix_sha256=refusal_output_sha,
        source_vector_sha256s=source_vectors,
        output_vector_sha256s=output_vectors,
        removed_projection_frobenius_norm=removed_norm,
        self_sha256=_hash(unsigned),
    )


def _orthogonalization_receipt(
    layers: tuple[LayerOrthogonalizationReceipt, ...],
) -> ComponentOrthogonalizationReceipt | None:
    if not layers:
        return None
    unsigned = {
        "format": ORTHOGONALIZATION_FORMAT,
        "layers": [
            layer.unsigned() | {"self_sha256": layer.self_sha256} for layer in layers
        ],
    }
    return ComponentOrthogonalizationReceipt(
        format=ORTHOGONALIZATION_FORMAT, layers=layers, self_sha256=_hash(unsigned)
    )


def _slice_payload(item: ComponentSlice) -> dict[str, Any]:
    return {
        "label": item.label,
        "start": item.start,
        "stop": item.stop,
        "source_sha256": item.source_sha256,
        "source_basis_set_sha256": item.source_basis_set_sha256,
        "source_basis_sha256": item.source_basis_sha256,
        "source_matrix_sha256": item.source_matrix_sha256,
        "source_vector_sha256s": list(item.source_vector_sha256s),
        "output_vector_sha256s": list(item.output_vector_sha256s),
        "component_sha256": item.component_sha256,
    }


def _layer_payload(layer: CombinedLayerComponentBasis) -> dict[str, Any]:
    return {
        "layer_index": layer.layer_index,
        "width": int(layer.matrix.shape[0]),
        "rank": int(layer.matrix.shape[1]),
        "matrix_sha256": layer.matrix_sha256,
        "components": [_slice_payload(item) for item in layer.components],
    }


def _verify_layer(
    layer: CombinedLayerComponentBasis, expected_labels: tuple[str, ...]
) -> None:
    if layer.matrix.ndim != 2 or min(layer.matrix.shape) < 1:
        raise ComponentBasisError("combined component matrix must be nonempty rank two")
    if layer.matrix.flags.writeable:
        raise ComponentBasisError("combined component matrix must be immutable")
    if not np.isfinite(layer.matrix).all():
        raise ComponentBasisError("combined component matrix must be finite")
    if _matrix_sha(layer.matrix) != layer.matrix_sha256:
        raise ComponentBasisError("combined component matrix identity mismatch")
    labels = tuple(item.label for item in layer.components)
    if labels != expected_labels:
        raise ComponentBasisError("component slice labels or order differ")
    offset = 0
    for item in layer.components:
        if item.start != offset or item.stop <= item.start:
            raise ComponentBasisError("component rank slices overlap or contain gaps")
        for digest in (
            item.source_sha256,
            item.source_basis_set_sha256,
            item.source_basis_sha256,
            item.source_matrix_sha256,
            item.component_sha256,
            *item.source_vector_sha256s,
            *item.output_vector_sha256s,
        ):
            _sha(digest, "component slice hash")
        if (
            len(item.source_vector_sha256s) != item.rank
            or len(item.output_vector_sha256s) != item.rank
        ):
            raise ComponentBasisError("component vector hash count differs from rank")
        unsigned = _slice_payload(item)
        claimed = unsigned.pop("component_sha256")
        if _hash(unsigned) != claimed:
            raise ComponentBasisError("component slice identity mismatch")
        observed = tuple(
            _vector_sha(layer.matrix[:, i]) for i in range(item.start, item.stop)
        )
        if observed != item.output_vector_sha256s:
            raise ComponentBasisError("component output vector identity mismatch")
        offset = item.stop
    if offset != layer.matrix.shape[1]:
        raise ComponentBasisError("component slices do not cover combined rank")
    gram = layer.matrix.T @ layer.matrix
    if not np.allclose(gram, np.eye(gram.shape[0]), rtol=0.0, atol=1e-10):
        raise ComponentBasisError("combined component basis is not orthonormal")


def _verify_orthogonalization(
    receipt: ComponentOrthogonalizationReceipt | None,
    by_layer: tuple[tuple[int, CombinedLayerComponentBasis], ...],
) -> None:
    if receipt is None:
        return
    if receipt.format != ORTHOGONALIZATION_FORMAT:
        raise ComponentBasisError("orthogonalization format is unsupported")
    if tuple(item.layer_index for item in receipt.layers) != tuple(
        index for index, _ in by_layer
    ):
        raise ComponentBasisError("orthogonalization receipt layer coverage differs")
    layers_by_index = dict(by_layer)
    for item in receipt.layers:
        if (
            not math.isfinite(item.removed_projection_frobenius_norm)
            or item.removed_projection_frobenius_norm < 0
        ):
            raise ComponentBasisError("orthogonalization projection norm is invalid")
        if _hash(item.unsigned()) != item.self_sha256:
            raise ComponentBasisError("orthogonalization layer identity mismatch")
        layer = layers_by_index[item.layer_index]
        slices = {component.label: component for component in layer.components}
        truth = slices.get("truth")
        refusal = slices.get("truth_orthogonalized_refusal")
        if truth is None or refusal is None:
            raise ComponentBasisError(
                "orthogonalization receipt requires truth and orthogonalized refusal slices"
            )
        truth_matrix = layer.matrix[:, truth.start : truth.stop]
        refusal_matrix = layer.matrix[:, refusal.start : refusal.stop]
        if (
            item.truth_matrix_sha256 != _matrix_sha(truth_matrix)
            or item.refusal_raw_matrix_sha256 != refusal.source_matrix_sha256
            or item.refusal_output_matrix_sha256 != _matrix_sha(refusal_matrix)
            or item.source_vector_sha256s != refusal.source_vector_sha256s
            or item.output_vector_sha256s != refusal.output_vector_sha256s
        ):
            raise ComponentBasisError(
                "orthogonalization receipt differs from component rank slices"
            )
    unsigned = {
        "format": receipt.format,
        "layers": [
            item.unsigned() | {"self_sha256": item.self_sha256}
            for item in receipt.layers
        ],
    }
    if _hash(unsigned) != receipt.self_sha256:
        raise ComponentBasisError("orthogonalization receipt identity mismatch")


__all__ = [
    "CombinedComponentBasisSet",
    "ComponentBasisError",
    "ComponentBasisInput",
    "ComponentLayerStrength",
    "ComponentRankStrengths",
    "ComponentStrengthPlan",
    "compile_component_basis_set",
    "compile_component_writer_edit",
    "compile_refusal_basis_set",
]
