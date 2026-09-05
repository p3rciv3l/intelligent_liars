"""Reversible persistent writer edits for compiled truth-editing recipes.

The public seam is intentionally small: a caller supplies a Qwen-shaped model
and a fully compiled edit, then receives a lease which owns exact restoration.
Direction loading, basis construction, recipe search, model loading, and
checkpoint I/O belong outside this module.

For an orthonormal basis ``U`` and writer-specific diagonal strengths
``Lambda``, this module applies the rank-k edit

``W' = W - U Lambda U^T W``.

An attention and an MLP writer may use different strengths at every layer and
for every selected basis coordinate.  A coefficient of 0 is identity, 1
removes the coordinate, and 2 reflects it.
"""

from __future__ import annotations

import math
import hashlib
import json
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch


StrengthSpec = float | tuple[float, ...]

_ACTIVE_MODELS: weakref.WeakSet[Any] = weakref.WeakSet()
_POISONED_MODELS: weakref.WeakSet[Any] = weakref.WeakSet()
_ACTIVE_MODELS_LOCK = threading.RLock()


class WriterEditError(ValueError):
    """A compiled writer edit cannot be applied exactly and safely."""


@dataclass(frozen=True)
class CompiledLayerWriterEdit:
    """One layer's already-constructed basis and writer strengths.

    ``basis`` has shape ``[hidden_width, rank]`` and must have orthonormal
    columns.  A strength may be one scalar (isotropic Lambda) or one coefficient
    per basis column (general diagonal Lambda); every coefficient lies in
    ``[0, 2]``.
    """

    layer_index: int
    basis: torch.Tensor
    attention_strength: StrengthSpec
    mlp_strength: StrengthSpec


@dataclass(frozen=True)
class CompiledWriterEdit:
    """A model-bound persistent edit ready to apply, with no artifact I/O."""

    recipe_id: str
    model_sha256: str
    layers: tuple[CompiledLayerWriterEdit, ...]


@dataclass(frozen=True)
class _WriterTarget:
    name: str
    layer_index: int
    site: str
    weight: torch.Tensor
    basis: torch.Tensor
    strengths: tuple[float, ...]


def _qwen_decoder_layers(model: Any) -> Sequence[Any]:
    try:
        layers = model.model.language_model.layers
    except AttributeError as error:
        raise WriterEditError(
            "expected Qwen decoder layers at model.model.language_model.layers"
        ) from error
    if isinstance(layers, (str, bytes)) or not (
        hasattr(layers, "__len__")
        and hasattr(layers, "__getitem__")
        and hasattr(layers, "__iter__")
    ):
        raise WriterEditError("Qwen decoder layers must be an ordered sequence")
    if not layers:
        raise WriterEditError("Qwen decoder layers must not be empty")
    return layers


def _writer_weight(layer: Any, *, layer_index: int, site: str) -> torch.Tensor:
    try:
        module = layer.self_attn.o_proj if site == "attention" else layer.mlp.down_proj
        weight = module.weight
    except AttributeError as error:
        suffix = "self_attn.o_proj" if site == "attention" else "mlp.down_proj"
        raise WriterEditError(
            f"Qwen decoder layer {layer_index} is missing {suffix}.weight"
        ) from error
    if not isinstance(weight, torch.Tensor):
        raise WriterEditError(
            f"Qwen decoder layer {layer_index} {site} writer weight must be a tensor"
        )
    if weight.ndim != 2 or weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise WriterEditError(
            f"Qwen decoder layer {layer_index} {site} writer weight must be a nonempty matrix"
        )
    if not torch.is_floating_point(weight):
        raise WriterEditError(
            f"Qwen decoder layer {layer_index} {site} writer weight must be floating point"
        )
    return weight


def _validate_strengths(
    value: StrengthSpec, *, rank: int, name: str
) -> tuple[float, ...]:
    if isinstance(value, bool):
        raise WriterEditError(f"{name} must contain numbers between 0 and 2")
    if isinstance(value, (int, float)):
        raw = (float(value),) * rank
    elif isinstance(value, tuple) and len(value) == rank:
        raw = value
    else:
        raise WriterEditError(
            f"{name} must be one number or exactly {rank} per-basis coefficients"
        )
    results: list[float] = []
    for coefficient in raw:
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise WriterEditError(f"{name} must contain numbers between 0 and 2")
        result = float(coefficient)
        if not math.isfinite(result) or not 0.0 <= result <= 2.0:
            raise WriterEditError(f"{name} must be finite and between 0 and 2")
        results.append(result)
    return tuple(results)


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WriterEditError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_model_binding(
    compiled_recipe: CompiledWriterEdit, verified_model_sha256: str
) -> None:
    verified = _sha256(verified_model_sha256, "verified model identity")
    if compiled_recipe.model_sha256 != verified:
        raise WriterEditError(
            "compiled recipe model identity does not match the verified loaded model identity"
        )


def _validated_basis(basis: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(basis, torch.Tensor):
        raise WriterEditError(f"{label} basis must be a tensor")
    if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
        raise WriterEditError(
            f"{label} basis must have nonempty [hidden_width, rank] shape"
        )
    if basis.shape[1] > basis.shape[0]:
        raise WriterEditError(f"{label} basis rank exceeds its hidden width")
    if not torch.is_floating_point(basis):
        raise WriterEditError(f"{label} basis must be floating point")
    detached = basis.detach()
    if not torch.isfinite(detached).all().item():
        raise WriterEditError(f"{label} basis must be finite")
    # Qualification is intentionally done in float64.  A loose-enough tolerance
    # accepts bases stored as float32 while rejecting materially overlapping
    # columns; runtime computation still uses the writer's safe compute dtype.
    checking = detached.to(device="cpu", dtype=torch.float64)
    gram = checking.transpose(0, 1) @ checking
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    if not torch.allclose(gram, identity, rtol=1e-5, atol=1e-6):
        raise WriterEditError(f"{label} basis columns must be orthonormal")
    # Clone so mutation of the caller's tensor cannot change an active plan.
    return detached.clone()


def _validated_targets(
    model: Any, compiled_recipe: CompiledWriterEdit
) -> tuple[_WriterTarget, ...]:
    if not isinstance(compiled_recipe, CompiledWriterEdit):
        raise WriterEditError("compiled_recipe must be a CompiledWriterEdit")
    if (
        not isinstance(compiled_recipe.recipe_id, str)
        or not compiled_recipe.recipe_id
        or compiled_recipe.recipe_id.strip() != compiled_recipe.recipe_id
    ):
        raise WriterEditError("compiled recipe_id must be a nonempty trimmed string")
    _sha256(compiled_recipe.model_sha256, "compiled model_sha256")
    if not compiled_recipe.layers:
        raise WriterEditError("compiled writer edit must contain at least one layer")

    layer_indices = tuple(edit.layer_index for edit in compiled_recipe.layers)
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in layer_indices):
        raise WriterEditError("compiled layer indices must be non-negative integers")
    if len(set(layer_indices)) != len(layer_indices):
        raise WriterEditError("compiled layer indices must be unique")

    layers = _qwen_decoder_layers(model)
    # Discover and validate both writer sites across every decoder layer before
    # touching a single parameter.  This prevents a partially compatible Qwen
    # wrapper from producing a partly edited model.
    discovered: list[tuple[torch.Tensor, torch.Tensor]] = []
    hidden_width: int | None = None
    seen_storages: set[int] = set()
    for layer_index, layer in enumerate(layers):
        attention = _writer_weight(layer, layer_index=layer_index, site="attention")
        mlp = _writer_weight(layer, layer_index=layer_index, site="mlp")
        for weight in (attention, mlp):
            try:
                storage_identity = weight.untyped_storage().data_ptr()
            except (AttributeError, RuntimeError) as error:
                raise WriterEditError(
                    "Qwen writer parameters must expose concrete untyped storage"
                ) from error
            if storage_identity in seen_storages:
                raise WriterEditError("Qwen writer parameters must not share storage")
            seen_storages.add(storage_identity)
            if hidden_width is None:
                hidden_width = int(weight.shape[0])
            elif weight.shape[0] != hidden_width:
                raise WriterEditError("Qwen writer output hidden widths must match")
        discovered.append((attention, mlp))

    invalid_layers = sorted(index for index in layer_indices if index >= len(layers))
    if invalid_layers:
        raise WriterEditError(
            f"compiled layers {invalid_layers} are outside a model with {len(layers)} layers"
        )
    assert hidden_width is not None  # guarded by the nonempty layer sequence

    targets: list[_WriterTarget] = []
    for edit in compiled_recipe.layers:
        label = f"compiled layer {edit.layer_index}"
        basis = _validated_basis(edit.basis, label=label)
        if basis.shape[0] != hidden_width:
            raise WriterEditError(
                f"{label} basis hidden width {basis.shape[0]} does not match model hidden width {hidden_width}"
            )
        attention_strengths = _validate_strengths(
            edit.attention_strength,
            rank=basis.shape[1],
            name=f"{label} attention_strength",
        )
        mlp_strengths = _validate_strengths(
            edit.mlp_strength,
            rank=basis.shape[1],
            name=f"{label} mlp_strength",
        )
        attention_weight, mlp_weight = discovered[edit.layer_index]
        targets.extend(
            (
                _WriterTarget(
                    name=(
                        f"model.language_model.layers.{edit.layer_index}."
                        "self_attn.o_proj.weight"
                    ),
                    layer_index=edit.layer_index,
                    site="attention",
                    weight=attention_weight,
                    basis=basis,
                    strengths=attention_strengths,
                ),
                _WriterTarget(
                    name=(
                        f"model.language_model.layers.{edit.layer_index}."
                        "mlp.down_proj.weight"
                    ),
                    layer_index=edit.layer_index,
                    site="mlp",
                    weight=mlp_weight,
                    basis=basis,
                    strengths=mlp_strengths,
                ),
            )
        )
    return tuple(targets)


def _edited_weight(target: _WriterTarget) -> torch.Tensor:
    weight = target.weight.detach()
    if not torch.isfinite(weight).all().item():
        raise WriterEditError(f"writer weight must be finite for {target.name}")
    if all(strength == 0.0 for strength in target.strengths):
        return weight.clone()
    compute_dtype = (
        torch.float32
        if weight.dtype in {torch.float16, torch.bfloat16}
        else weight.dtype
    )
    working = weight.to(dtype=compute_dtype)
    basis = target.basis.to(device=weight.device, dtype=compute_dtype)
    coordinates = basis.transpose(0, 1) @ working
    strengths = torch.tensor(
        target.strengths, device=weight.device, dtype=compute_dtype
    ).unsqueeze(1)
    edited = working - basis @ (coordinates * strengths)
    result = edited.to(dtype=weight.dtype)
    if not torch.isfinite(result).all().item():
        raise WriterEditError(f"writer edit produced non-finite values for {target.name}")
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _projection_receipt(
    target: _WriterTarget,
    edited: torch.Tensor,
    *,
    recipe_id: str,
) -> dict[str, Any]:
    weight = target.weight.detach()
    compute_dtype = (
        torch.float32 if weight.dtype in {torch.float16, torch.bfloat16} else weight.dtype
    )
    basis = target.basis.to(device=weight.device, dtype=compute_dtype)
    before = basis.transpose(0, 1) @ weight.to(dtype=compute_dtype)
    after = basis.transpose(0, 1) @ edited.to(dtype=compute_dtype)
    retained = torch.tensor(
        tuple(1.0 - value for value in target.strengths),
        device=weight.device,
        dtype=compute_dtype,
    ).unsqueeze(1)
    expected_after = before * retained
    pre_norm = float(torch.linalg.vector_norm(before).item())
    post_norm = float(torch.linalg.vector_norm(after).item())
    scale = max(pre_norm, torch.finfo(compute_dtype).eps)
    error_ratio = float(torch.linalg.vector_norm(after - expected_after).item()) / scale
    residual_ratio = post_norm / scale
    delta_norm = float(
        torch.linalg.vector_norm(
            edited.to(dtype=compute_dtype) - weight.to(dtype=compute_dtype)
        ).item()
    )
    tolerance = 5e-3
    if error_ratio > tolerance:
        raise WriterEditError(
            f"installed projection verification failed for {target.name}: "
            f"error ratio {error_ratio:.8g} exceeds {tolerance}"
        )
    if all(value == 0.0 for value in target.strengths) and delta_norm != 0.0:
        raise WriterEditError(f"identity edit changed writer weight for {target.name}")
    receipt: dict[str, Any] = {
        "layer_index": target.layer_index,
        "writer_site": target.site,
        "parameter_name": target.name,
        "basis_sha256": _tensor_sha256(target.basis),
        "basis_rank": int(target.basis.shape[1]),
        "requested_strengths": list(target.strengths),
        "coordinate_retention_factors": [1.0 - value for value in target.strengths],
        "pre_edit_projection_norm": pre_norm,
        "post_edit_projection_norm": post_norm,
        "normalized_residual_ratio": residual_ratio,
        "projection_error_ratio": error_ratio,
        "weight_delta_norm": delta_norm,
        "exact_restoration_verified": False,
    }
    binding = {
        "format": "truth_editing_projection_binding_v1",
        "recipe_id": recipe_id,
        **{key: value for key, value in receipt.items() if key != "exact_restoration_verified"},
    }
    receipt["edited_weight_binding_sha256"] = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return receipt


def materialized_writer_weights(
    model: Any,
    compiled_recipe: CompiledWriterEdit,
    *,
    verified_model_sha256: str,
) -> dict[str, torch.Tensor]:
    """Return edited writer tensors without mutating ``model``.

    The returned mapping uses Hugging Face state-dict names and is the parity
    seam for checkpoint materializers: copying these tensors into a clone must
    produce the same writer values as an active overlay lease.
    """

    _validate_model_binding(compiled_recipe, verified_model_sha256)
    with _ACTIVE_MODELS_LOCK:
        if model in _ACTIVE_MODELS:
            raise WriterEditError("model already has an active writer edit")
        if model in _POISONED_MODELS:
            raise WriterEditError("model is poisoned after an incomplete writer restoration")
    targets = _validated_targets(model, compiled_recipe)
    return {target.name: _edited_weight(target) for target in targets}


class EditLease:
    """Ownership token for one active reversible writer edit."""

    def __init__(
        self,
        *,
        runtime: WriterEditRuntime,
        model: Any,
        originals: Mapping[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        projection_evidence: Sequence[dict[str, Any]],
    ) -> None:
        self._runtime = runtime
        self._model = model
        self._originals = dict(originals)
        self._projection_evidence = list(projection_evidence)
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def edited_parameters(self) -> tuple[str, ...]:
        return tuple(self._originals)

    @property
    def projection_evidence(self) -> tuple[Mapping[str, Any], ...]:
        """Numerical edit receipts, updated after exact restoration succeeds."""

        return tuple(dict(item) for item in self._projection_evidence)

    def close(self) -> None:
        """Restore original writer bytes exactly; repeated closes are harmless."""

        if not self._active:
            return
        restore_error: Exception | None = None
        overlay_diverged = False
        with torch.no_grad():
            for name, (weight, original, expected_overlay) in self._originals.items():
                try:
                    try:
                        installed = self._model.get_parameter(name)
                    except (AttributeError, KeyError) as error:
                        raise WriterEditError(
                            f"writer parameter {name} was removed during its active lease"
                        ) from error
                    if installed is not weight:
                        raise WriterEditError(
                            f"writer parameter {name} was replaced during its active lease"
                        )
                    if not torch.equal(weight, expected_overlay):
                        overlay_diverged = True
                    weight.copy_(original)
                    if not torch.equal(weight, original):
                        raise WriterEditError(
                            f"writer parameter {name} did not restore exactly"
                        )
                except Exception as error:  # pragma: no cover - device failure
                    restore_error = restore_error or error
        self._active = False
        if restore_error is not None:
            with _ACTIVE_MODELS_LOCK:
                _POISONED_MODELS.add(self._model)
        self._runtime._release(self)
        if restore_error is not None:
            raise WriterEditError("failed to restore one or more writer parameters") from restore_error
        if overlay_diverged:
            raise WriterEditError(
                "writer parameters changed outside the active lease; originals were restored"
            )
        for item in self._projection_evidence:
            item["exact_restoration_verified"] = True

    def __enter__(self) -> EditLease:
        if not self._active:
            raise WriterEditError("cannot enter a closed writer edit lease")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class WriterEditRuntime:
    """Apply one compiled persistent edit at a time and own its restoration.

    The lease grants exclusive mutation ownership, not safe concurrent
    inference: callers must not read or run the model while another thread is
    entering or closing a lease.  A second runtime cannot lease the same model,
    and changes to leased writer parameters are detected during restoration.
    """

    def __init__(self, *, verified_model_sha256: str) -> None:
        self._verified_model_sha256 = _sha256(
            verified_model_sha256, "verified model identity"
        )
        self._active_lease: EditLease | None = None
        self._active_model: Any | None = None

    def activate(self, model: Any, compiled_recipe: CompiledWriterEdit) -> EditLease:
        """Apply ``compiled_recipe`` and return its already-active restore lease."""

        if self._active_lease is not None:
            raise WriterEditError("writer edit runtime already has an active lease")
        _validate_model_binding(compiled_recipe, self._verified_model_sha256)
        targets = _validated_targets(model, compiled_recipe)
        with _ACTIVE_MODELS_LOCK:
            try:
                if model in _ACTIVE_MODELS:
                    raise WriterEditError("model already has an active writer edit")
                if model in _POISONED_MODELS:
                    raise WriterEditError(
                        "model is poisoned after an incomplete writer restoration"
                    )
                _ACTIVE_MODELS.add(model)
            except TypeError as error:
                raise WriterEditError(
                    "model must support weak identity tracking for exclusive edit leases"
                ) from error
        try:
            mutation_targets = tuple(
                target
                for target in targets
                if any(strength != 0.0 for strength in target.strengths)
            )
            edited = {
                target.name: _edited_weight(target) for target in mutation_targets
            }
            originals = {
                target.name: (
                    target.weight,
                    target.weight.detach().clone(),
                    edited[target.name].detach().clone(),
                )
                for target in mutation_targets
            }
            projection_evidence = tuple(
                _projection_receipt(
                    target,
                    edited.get(target.name, target.weight.detach().clone()),
                    recipe_id=compiled_recipe.recipe_id,
                )
                for target in targets
            )
            with torch.no_grad():
                for target in mutation_targets:
                    target.weight.copy_(edited[target.name])
                    if not torch.equal(target.weight, edited[target.name]):
                        raise WriterEditError(
                            f"writer projection was not installed exactly for {target.name}"
                        )
        except Exception as error:
            rollback_error: Exception | None = None
            if "originals" in locals():
                with torch.no_grad():
                    for weight, original, _expected in originals.values():
                        try:
                            weight.copy_(original)
                            if not torch.equal(weight, original):
                                raise WriterEditError(
                                    "writer parameter did not roll back exactly"
                                )
                        except Exception as restore_error:
                            rollback_error = rollback_error or restore_error
            with _ACTIVE_MODELS_LOCK:
                _ACTIVE_MODELS.discard(model)
                if rollback_error is not None:
                    _POISONED_MODELS.add(model)
            if rollback_error is not None:
                raise WriterEditError(
                    "failed to apply and completely roll back writer edit transaction; model is poisoned"
                ) from rollback_error
            raise WriterEditError(
                f"failed to apply writer edit transaction: {type(error).__name__}: {error}"
            ) from error

        lease = EditLease(
            runtime=self,
            model=model,
            originals=originals,
            projection_evidence=projection_evidence,
        )
        self._active_lease = lease
        self._active_model = model
        return lease

    def _release(self, lease: EditLease) -> None:
        if self._active_lease is lease:
            with _ACTIVE_MODELS_LOCK:
                if self._active_model is not None:
                    _ACTIVE_MODELS.discard(self._active_model)
            self._active_lease = None
            self._active_model = None


def require_unedited_writer_model(model: Any) -> None:
    """Fail closed unless the model has no active or poisoned writer edit."""

    with _ACTIVE_MODELS_LOCK:
        try:
            if model in _ACTIVE_MODELS:
                raise WriterEditError("model has an active writer edit")
            if model in _POISONED_MODELS:
                raise WriterEditError("model is poisoned after writer restoration")
            # Prove that the model supports the same weak identity tracking used
            # by every writer-edit lease; an untrackable object is not safe base
            # evidence.
            probe = weakref.ref(model)
            del probe
        except TypeError as error:
            raise WriterEditError(
                "model does not support writer-edit identity tracking"
            ) from error


__all__ = [
    "CompiledLayerWriterEdit",
    "CompiledWriterEdit",
    "EditLease",
    "WriterEditError",
    "WriterEditRuntime",
    "StrengthSpec",
    "materialized_writer_weights",
    "require_unedited_writer_model",
]
