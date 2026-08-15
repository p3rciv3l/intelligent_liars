"""Pure, offline activation edits around a fixed linear-probe coordinate.

The operators in this module only transform in-memory tensors. They do not
load models, persist modified weights, generate outputs, or execute tools.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True)
class LinearProbe:
    """A fixed affine probe whose positive scores denote deception."""

    coef: Tensor
    intercept: float = 0.0

    @classmethod
    def from_values(
        cls,
        *,
        coef: Sequence[float] | Tensor,
        intercept: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> LinearProbe:
        """Build a probe from in-memory values read from a saved probe record."""

        return cls(
            coef=torch.as_tensor(coef, dtype=dtype, device=device),
            intercept=intercept,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.coef, Tensor):
            raise TypeError("probe coef must be a torch.Tensor")
        if self.coef.ndim != 1:
            raise ValueError("probe coef must be one-dimensional")
        if not self.coef.is_floating_point():
            raise TypeError("probe coef must use a floating-point dtype")
        if not bool(torch.isfinite(self.coef).all()):
            raise ValueError("probe coef must contain only finite values")
        if self.coef.numel() == 0 or float(torch.linalg.vector_norm(self.coef)) == 0.0:
            raise ValueError("probe coef must be non-empty and non-zero")
        if isinstance(self.intercept, bool) or not isinstance(self.intercept, (int, float)):
            raise TypeError("probe intercept must be a real scalar")
        if not math.isfinite(float(self.intercept)):
            raise ValueError("probe intercept must be finite")

    def score(self, activations: Tensor) -> Tensor:
        """Return ``w^T h + b`` over the activation tensor's last axis."""

        _validate_activations(activations, self)
        return activations @ self.coef + float(self.intercept)


@runtime_checkable
class ActivationMapping(Protocol):
    """Future seam for activation mappings, including later learned methods.

    Current implementations are fixed analytic controls only. Learned mappings
    can implement this protocol in a future research phase without changing the
    generation backend's callable boundary.
    """

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        """Return transformed activations without mutating the input."""


@dataclass(frozen=True)
class Identity:
    """Exact no-op control; this is the only behavior when selected explicitly."""

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        _validate_activations(activations, probe)
        return activations


@dataclass(frozen=True)
class ScalarAddition:
    """Add a scalar multiple of the stored probe coefficient vector."""

    multiplier: float

    def __post_init__(self) -> None:
        _validate_finite_scalar(self.multiplier, name="multiplier")

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        _validate_activations(activations, probe)
        return activations + float(self.multiplier) * probe.coef


@dataclass(frozen=True)
class RandomDirectionControl:
    """Add a seeded random vector with the probe coefficient's norm."""

    multiplier: float
    seed: int

    def __post_init__(self) -> None:
        _validate_finite_scalar(self.multiplier, name="multiplier")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be between 0 and 2**63 - 1")

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        _validate_activations(activations, probe)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        direction = torch.randn(
            probe.coef.shape,
            dtype=torch.float64,
            device="cpu",
            generator=generator,
        ).to(device=probe.coef.device, dtype=probe.coef.dtype)
        direction = direction / torch.linalg.vector_norm(direction)
        direction = direction * torch.linalg.vector_norm(probe.coef)
        return activations + float(self.multiplier) * direction


@dataclass(frozen=True)
class CoordinateRemoval:
    """Project activations onto the probe's affine decision boundary."""

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        scores = probe.score(activations)
        norm_squared = torch.dot(probe.coef, probe.coef)
        return activations - (scores / norm_squared).unsqueeze(-1) * probe.coef


@dataclass(frozen=True)
class Reflection:
    """Map ``z`` to ``-scale * z``, optionally only where ``z < 0``."""

    scale: float = 1.0
    one_sided: bool = False

    def __post_init__(self) -> None:
        _validate_finite_scalar(self.scale, name="scale")
        if self.scale < 0.0:
            raise ValueError("scale must be non-negative")
        if not isinstance(self.one_sided, bool):
            raise TypeError("one_sided must be a bool")

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        scores = probe.score(activations)
        target_scores = -float(self.scale) * scores
        if self.one_sided:
            target_scores = torch.where(scores < 0.0, target_scores, scores)
        return _move_to_target_scores(activations, probe, scores, target_scores)


@runtime_checkable
class ScoreRemapping(Protocol):
    """Maps affine probe scores while leaving the edit geometry fixed."""

    def __call__(self, scores: Tensor) -> Tensor:
        """Return one finite target score for each input score."""


@dataclass(frozen=True)
class BoundedCoordinateRemap:
    """Apply a scalar score remapping with an L2 displacement cap per state."""

    remap_scores: ScoreRemapping
    max_displacement: float

    def __post_init__(self) -> None:
        if not callable(self.remap_scores):
            raise TypeError("remap_scores must be callable")
        _validate_non_negative_finite_scalar(
            self.max_displacement,
            name="max_displacement",
        )

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        scores = probe.score(activations)
        target_scores = self.remap_scores(scores)
        _validate_target_scores(target_scores, scores)
        coef_norm = torch.linalg.vector_norm(probe.coef)
        max_score_delta = float(self.max_displacement) * coef_norm
        raw_score_delta = target_scores - scores
        bounded_score_delta = torch.minimum(
            torch.maximum(raw_score_delta, -max_score_delta),
            max_score_delta,
        )
        return _move_to_target_scores(
            activations,
            probe,
            scores,
            scores + bounded_score_delta,
        )


@dataclass(frozen=True)
class DeceptiveMarginClamp:
    """Move scores below a deceptive target toward it, subject to an L2 cap."""

    target_score: float
    max_displacement: float

    def __post_init__(self) -> None:
        _validate_finite_scalar(self.target_score, name="target_score")
        _validate_non_negative_finite_scalar(
            self.max_displacement,
            name="max_displacement",
        )

    def __call__(self, activations: Tensor, probe: LinearProbe) -> Tensor:
        return BoundedCoordinateRemap(
            remap_scores=lambda scores: torch.clamp_min(scores, float(self.target_score)),
            max_displacement=self.max_displacement,
        )(activations, probe)


def _validate_activations(activations: Tensor, probe: LinearProbe) -> None:
    if not isinstance(activations, Tensor):
        raise TypeError("activations must be a torch.Tensor")
    if activations.ndim < 1:
        raise ValueError("activations must have at least one dimension")
    if not activations.is_floating_point():
        raise TypeError("activations must use a floating-point dtype")
    if activations.shape[-1] != probe.coef.numel():
        raise ValueError(
            "activation hidden dimension must match probe coef length "
            f"({activations.shape[-1]} != {probe.coef.numel()})"
        )
    if activations.device != probe.coef.device:
        raise ValueError("activations and probe coef must be on the same device")
    if activations.dtype != probe.coef.dtype:
        raise TypeError("activations and probe coef must use the same dtype")
    if not bool(torch.isfinite(activations).all()):
        raise ValueError("activations must contain only finite values")


def _validate_finite_scalar(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_non_negative_finite_scalar(value: float, *, name: str) -> None:
    _validate_finite_scalar(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_target_scores(target_scores: Tensor, current_scores: Tensor) -> None:
    if not isinstance(target_scores, Tensor):
        raise TypeError("remap_scores must return a torch.Tensor")
    if target_scores.shape != current_scores.shape:
        raise ValueError("remap_scores must preserve the score tensor shape")
    if target_scores.device != current_scores.device:
        raise ValueError("remapped scores must remain on the activation device")
    if target_scores.dtype != current_scores.dtype:
        raise TypeError("remapped scores must preserve the activation dtype")
    if not bool(torch.isfinite(target_scores).all()):
        raise ValueError("remapped scores must contain only finite values")


def _move_to_target_scores(
    activations: Tensor,
    probe: LinearProbe,
    current_scores: Tensor,
    target_scores: Tensor,
) -> Tensor:
    norm_squared = torch.dot(probe.coef, probe.coef)
    score_delta = target_scores - current_scores
    return activations + (score_delta / norm_squared).unsqueeze(-1) * probe.coef
