"""Truth-editing broad-stage coverage contract.

This module is the coverage seam shared by deterministic anchors, Optuna's
concentration gate, and final study reporting.  It classifies the *effective*
edit represented by a proposal: a refusal-only proposal does not count as a
truth attention/MLP configuration merely because its legacy ``writer_policy``
field says ``both``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


WRITER_CONFIGURATIONS = ("disabled", "attention", "mlp", "both")
REFUSAL_SETTINGS = ("disabled", "global", "per_layer")
REFUSAL_WRITER_POLICIES = ("attention", "mlp", "both")
KERNEL_CENTER_REGIONS = ("early", "middle", "late")
KERNEL_HALF_WIDTH_MODES = ("disabled", "local", "broad")
KERNEL_SHAPES = ("flat", "tapered")
REFUSAL_STRENGTH_REGIONS = ("disabled", "projection", "reflection")


class _Proposal(Protocol):
    attention_enabled: bool | None
    mlp_enabled: bool | None
    attention_kernel_center: float | None
    attention_kernel_half_width: float | None
    attention_edge_strength: float | None
    attention_peak_strength: float | None
    mlp_kernel_center: float | None
    mlp_kernel_half_width: float | None
    mlp_edge_strength: float | None
    mlp_peak_strength: float | None
    writer_layers: tuple[int, ...]
    refusal_enabled: bool
    refusal_direction_scope: str
    refusal_writer_policy: str
    refusal_strength: float


def writer_configuration(proposal: _Proposal) -> str:
    """Return the truth-writer configuration that is actually active."""

    attention = (
        proposal.attention_enabled is True
        and proposal.attention_peak_strength is not None
        and proposal.attention_peak_strength > 0.0
    )
    mlp = (
        proposal.mlp_enabled is True
        and proposal.mlp_peak_strength is not None
        and proposal.mlp_peak_strength > 0.0
    )
    if attention and mlp:
        return "both"
    if attention:
        return "attention"
    if mlp:
        return "mlp"
    return "disabled"


def refusal_setting(proposal: _Proposal) -> str:
    """Return disabled/global/per-layer refusal contribution semantics."""

    if not proposal.refusal_enabled or proposal.refusal_strength == 0.0:
        return "disabled"
    if proposal.refusal_direction_scope not in {"global", "per_layer"}:
        raise ValueError("enabled refusal proposal has an invalid direction scope")
    return proposal.refusal_direction_scope


def _active_kernel(proposal: _Proposal) -> tuple[float, float, float, float] | None:
    for enabled, center, width, edge, peak in (
        (
            proposal.attention_enabled,
            proposal.attention_kernel_center,
            proposal.attention_kernel_half_width,
            proposal.attention_edge_strength,
            proposal.attention_peak_strength,
        ),
        (
            proposal.mlp_enabled,
            proposal.mlp_kernel_center,
            proposal.mlp_kernel_half_width,
            proposal.mlp_edge_strength,
            proposal.mlp_peak_strength,
        ),
    ):
        if enabled is True and peak is not None and peak > 0.0:
            assert center is not None and width is not None and edge is not None
            return center, width, edge, peak
    return None


def kernel_center_region(proposal: _Proposal) -> str | None:
    kernel = _active_kernel(proposal)
    if kernel is None:
        return None
    center = kernel[0]
    low, high = min(proposal.writer_layers), max(proposal.writer_layers)
    if high == low:
        return "middle"
    relative = (center - low) / (high - low)
    if relative < 1.0 / 3.0:
        return "early"
    if relative > 2.0 / 3.0:
        return "late"
    return "middle"


def kernel_half_width_mode(proposal: _Proposal) -> str:
    kernel = _active_kernel(proposal)
    if kernel is None:
        return "disabled"
    span = max(proposal.writer_layers) - min(proposal.writer_layers)
    return "broad" if span > 0 and kernel[1] > span / 2.0 else "local"


def kernel_shape(proposal: _Proposal) -> str | None:
    kernel = _active_kernel(proposal)
    if kernel is None:
        return None
    return "flat" if kernel[2] == kernel[3] else "tapered"


def refusal_strength_region(proposal: _Proposal) -> str:
    strength = proposal.refusal_strength if proposal.refusal_enabled else 0.0
    if strength == 0.0:
        return "disabled"
    if strength <= 1.0:
        return "projection"
    return "reflection"


def active_edit_arm(proposal: Any) -> str | None:
    """Return the arm only when every contribution it names is nonzero."""

    truth_active = writer_configuration(proposal) != "disabled"
    refusal_active = refusal_strength_region(proposal) != "disabled"
    if proposal.edit_arm == "truth_only" and truth_active and not refusal_active:
        return "truth_only"
    if proposal.edit_arm == "refusal_only" and refusal_active and not truth_active:
        return "refusal_only"
    if proposal.edit_arm == "joint" and truth_active and refusal_active:
        return "joint"
    return None


@dataclass(frozen=True)
class BroadCoverageContract:
    """Exact required values for the deliberate stage before concentration."""

    families: frozenset[str]
    source_layers: frozenset[int]
    writer_regions: frozenset[str]
    writer_policies: frozenset[str]
    basis_methods: frozenset[str]
    strength_regions: frozenset[str]
    basis_scopes: frozenset[str]
    direction_scopes: frozenset[str]
    normalization_modes: frozenset[str]
    edit_arms: frozenset[str]
    active_edit_arms: frozenset[str] = frozenset(
        ("truth_only", "refusal_only", "joint")
    )
    writer_configurations: frozenset[str] = frozenset(WRITER_CONFIGURATIONS)
    refusal_settings: frozenset[str] = frozenset(REFUSAL_SETTINGS)
    refusal_writer_policies: frozenset[str] = frozenset(REFUSAL_WRITER_POLICIES)
    kernel_center_regions: frozenset[str] = frozenset(KERNEL_CENTER_REGIONS)
    kernel_half_width_modes: frozenset[str] = frozenset(KERNEL_HALF_WIDTH_MODES)
    kernel_shapes: frozenset[str] = frozenset(KERNEL_SHAPES)
    refusal_strength_regions: frozenset[str] = frozenset(REFUSAL_STRENGTH_REGIONS)

    def is_complete(self, ledger: Any) -> bool:
        """Fail closed unless every declared broad axis matches exactly."""

        return all(
            set(getattr(ledger, name, ())) == required
            for name, required in (
                ("families", self.families),
                ("source_layers", self.source_layers),
                ("writer_regions", self.writer_regions),
                ("writer_policies", self.writer_policies),
                ("basis_methods", self.basis_methods),
                ("strength_regions", self.strength_regions),
                ("basis_scopes", self.basis_scopes),
                ("direction_scopes", self.direction_scopes),
                ("normalization_modes", self.normalization_modes),
                ("edit_arms", self.edit_arms),
                ("active_edit_arms", self.active_edit_arms),
                ("writer_configurations", self.writer_configurations),
                ("refusal_settings", self.refusal_settings),
                ("refusal_writer_policies", self.refusal_writer_policies),
                ("kernel_center_regions", self.kernel_center_regions),
                ("kernel_half_width_modes", self.kernel_half_width_modes),
                ("kernel_shapes", self.kernel_shapes),
                ("refusal_strength_regions", self.refusal_strength_regions),
            )
        )
