"""Strict offline contracts for frozen truth-editing directions and recipes.

The module is deliberately independent of torch, model loading, optimizers, and
judge clients.  Parsing is the fail-closed seam: accepted values have exact
fields, immutable typed representations, finite numerics, and canonical
self-identities.  Compatibility is checked separately so manifests and recipes
can be parsed, cached, and audited independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias


DIRECTION_BANK_MANIFEST_FORMAT: Literal["truth_editing_direction_bank_manifest_v1"] = (
    "truth_editing_direction_bank_manifest_v1"
)
INTERVENTION_RECIPE_FORMAT: Literal["truth_editing_intervention_recipe_v2"] = (
    "truth_editing_intervention_recipe_v2"
)

_HEX = frozenset("0123456789abcdef")
_DIRECTION_KINDS = frozenset(
    {"truth", "refusal", "orthogonal_control", "shuffled_control"}
)
_DIRECTION_FAMILIES = frozenset(
    {"general", "intermediate", "domain_specific", "mixed", "refusal", "control"}
)
_CONDITION_KINDS = frozenset(
    {
        "base",
        "truth_only",
        "refusal_only",
        "joint",
        "orthogonal_control",
        "shuffled_control",
    }
)
_ACTIVATION_TRANSFORMS = frozenset(
    {
        "projection_reflection",
        "affine_projection",
        "partial_reflection",
        "one_sided_reflection",
        "bounded_remap",
        "bounded_margin_clamp",
    }
)
_TOKEN_SCOPES = frozenset(
    {
        "teacher_forced_masked",
        "prefill_last",
        "cached_generation",
        "prefill_last_and_cached_generation",
        "selected_prompt_positions",
        "selected_prompt_positions_and_cached_generation",
    }
)


class TruthEditingContractError(ValueError):
    """A truth-editing manifest or recipe cannot be verified exactly."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON or fail on non-JSON/non-finite values."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TruthEditingContractError("value is not canonical JSON") from error
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TruthEditingContractError(f"{name} must be an object")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise TruthEditingContractError(
            f"{name} fields differ; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TruthEditingContractError(f"{name} must be a nonempty trimmed string")
    return value


def _enum(value: Any, allowed: frozenset[str], name: str) -> str:
    result = _text(value, name)
    if result not in allowed:
        raise TruthEditingContractError(
            f"{name} must be one of {sorted(allowed)}, got {result!r}"
        )
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise TruthEditingContractError(f"{name} must be a lowercase SHA-256")
    return value


def _git_revision(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in _HEX for character in value)
    ):
        raise TruthEditingContractError(
            f"{name} must be a lowercase 40-character Git revision"
        )
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TruthEditingContractError(f"{name} must be boolean")
    return value


def _integer(
    value: Any, name: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TruthEditingContractError(
            f"{name} must be an integer of at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise TruthEditingContractError(f"{name} must be at most {maximum}")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, float):
        raise TruthEditingContractError(
            f"{name} must be a JSON float, not a boolean or integer"
        )
    result = value
    if not math.isfinite(result):
        raise TruthEditingContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        if maximum is not None:
            raise TruthEditingContractError(
                f"{name} must be between {minimum:g} and {maximum:g}"
            )
        raise TruthEditingContractError(f"{name} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        if minimum is not None:
            raise TruthEditingContractError(
                f"{name} must be between {minimum:g} and {maximum:g}"
            )
        raise TruthEditingContractError(f"{name} must be at most {maximum:g}")
    return result


def _optional_number(value: Any, name: str) -> float | None:
    return None if value is None else _number(value, name)


def _string_tuple(
    value: Any, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TruthEditingContractError(f"{name} must be an array of strings")
    result = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise TruthEditingContractError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise TruthEditingContractError(f"{name} entries must be unique")
    return result


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TruthEditingContractError(f"{name} must be an array of integers")
    result = tuple(
        _integer(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if not result:
        raise TruthEditingContractError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise TruthEditingContractError(f"{name} entries must be unique")
    return result


def _self_hash(raw: Mapping[str, Any], name: str) -> str:
    claimed = _sha256(raw["self_sha256"], f"{name}.self_sha256")
    unsigned = dict(raw)
    del unsigned["self_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise TruthEditingContractError(f"{name} self hash mismatch")
    return claimed


@dataclass(frozen=True)
class DirectionModelIdentity:
    repository: str
    revision: str
    model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    decoder_layer_count: int
    hidden_width: int


@dataclass(frozen=True)
class DirectionArtifactIdentity:
    path: str
    file_sha256: str
    vector_sha256: str


@dataclass(frozen=True)
class DirectionConstruction:
    basis_method: Literal["raw", "qr", "svd"]
    pooling: str
    token_position: str
    normalization: Literal["unit_l2", "orthonormal"]
    sign_convention: str
    intercept: float


@dataclass(frozen=True)
class ControlProvenance:
    seed: int
    parent_direction_ids: tuple[str, ...]
    match_policy: Literal["equal_rank_equal_norm"]


@dataclass(frozen=True)
class DirectionProvenance:
    dataset: str
    dataset_revision: str
    split: str
    ordered_row_ids_sha256: str
    source_code_revision: str


@dataclass(frozen=True)
class LeakageQualification:
    evaluation_disjoint: bool
    heldout_family_disjoint: bool
    sealed_audit_accessed: bool
    audit_receipt_sha256: str


@dataclass(frozen=True)
class DirectionQualification:
    status: str
    receipt_sha256: str
    finite: bool
    unit_norm: bool
    qualified_rank: int


@dataclass(frozen=True)
class DirectionEntry:
    direction_id: str
    kind: Literal["truth", "refusal", "orthogonal_control", "shuffled_control"]
    family: str
    basis_variant: Literal["raw", "truth_orthogonalized", "joint"]
    domains: tuple[str, ...]
    source_layer: int
    width: int
    rank: int
    artifact: DirectionArtifactIdentity
    construction: DirectionConstruction
    control_provenance: ControlProvenance | None
    provenance: DirectionProvenance
    leakage: LeakageQualification
    qualification: DirectionQualification


@dataclass(frozen=True)
class DirectionBankManifest:
    format: Literal["truth_editing_direction_bank_manifest_v1"]
    manifest_id: str
    model: DirectionModelIdentity
    directions: tuple[DirectionEntry, ...]
    self_sha256: str

    @property
    def direction_ids(self) -> tuple[str, ...]:
        return tuple(direction.direction_id for direction in self.directions)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


def _parse_model(value: Any) -> DirectionModelIdentity:
    raw = _mapping(value, "manifest.model")
    _exact(
        raw,
        {
            "repository",
            "revision",
            "model_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "decoder_layer_count",
            "hidden_width",
        },
        "manifest.model",
    )
    return DirectionModelIdentity(
        repository=_text(raw["repository"], "manifest.model.repository"),
        revision=_git_revision(raw["revision"], "manifest.model.revision"),
        model_sha256=_sha256(raw["model_sha256"], "manifest.model.model_sha256"),
        tokenizer_sha256=_sha256(
            raw["tokenizer_sha256"], "manifest.model.tokenizer_sha256"
        ),
        chat_template_sha256=_sha256(
            raw["chat_template_sha256"], "manifest.model.chat_template_sha256"
        ),
        decoder_layer_count=_integer(
            raw["decoder_layer_count"], "manifest.model.decoder_layer_count", minimum=1
        ),
        hidden_width=_integer(
            raw["hidden_width"], "manifest.model.hidden_width", minimum=1
        ),
    )


def _parse_direction(
    value: Any, index: int, model: DirectionModelIdentity
) -> DirectionEntry:
    name = f"manifest.directions[{index}]"
    raw = _mapping(value, name)
    _exact(
        raw,
        {
            "direction_id",
            "kind",
            "family",
            "basis_variant",
            "domains",
            "source_layer",
            "width",
            "rank",
            "artifact",
            "construction",
            "control_provenance",
            "provenance",
            "leakage",
            "qualification",
        },
        name,
    )
    direction_kind = _enum(raw["kind"], _DIRECTION_KINDS, f"{name}.kind")
    direction_family = _enum(raw["family"], _DIRECTION_FAMILIES, f"{name}.family")
    expected_families = {
        "truth": {"general", "intermediate", "domain_specific", "mixed"},
        "refusal": {"refusal"},
        "orthogonal_control": {"control"},
        "shuffled_control": {"control"},
    }
    if direction_family not in expected_families[direction_kind]:
        raise TruthEditingContractError(
            f"{name}.family is incompatible with direction kind {direction_kind}"
        )
    basis_variant = _enum(
        raw["basis_variant"],
        frozenset({"raw", "truth_orthogonalized", "joint"}),
        f"{name}.basis_variant",
    )
    if direction_kind != "refusal" and basis_variant != "raw":
        raise TruthEditingContractError(
            f"{name}.basis_variant must be raw for {direction_kind} directions"
        )

    artifact_raw = _mapping(raw["artifact"], f"{name}.artifact")
    _exact(artifact_raw, {"path", "file_sha256", "vector_sha256"}, f"{name}.artifact")
    artifact = DirectionArtifactIdentity(
        path=_text(artifact_raw["path"], f"{name}.artifact.path"),
        file_sha256=_sha256(
            artifact_raw["file_sha256"], f"{name}.artifact.file_sha256"
        ),
        vector_sha256=_sha256(
            artifact_raw["vector_sha256"], f"{name}.artifact.vector_sha256"
        ),
    )

    construction_raw = _mapping(raw["construction"], f"{name}.construction")
    _exact(
        construction_raw,
        {
            "basis_method",
            "pooling",
            "token_position",
            "normalization",
            "sign_convention",
            "intercept",
        },
        f"{name}.construction",
    )
    construction = DirectionConstruction(
        basis_method=_enum(
            construction_raw["basis_method"],
            frozenset({"raw", "qr", "svd"}),
            f"{name}.construction.basis_method",
        ),  # type: ignore[arg-type]
        pooling=_text(construction_raw["pooling"], f"{name}.construction.pooling"),
        token_position=_text(
            construction_raw["token_position"], f"{name}.construction.token_position"
        ),
        normalization=_enum(
            construction_raw["normalization"],
            frozenset({"unit_l2", "orthonormal"}),
            f"{name}.construction.normalization",
        ),  # type: ignore[arg-type]
        sign_convention=_text(
            construction_raw["sign_convention"], f"{name}.construction.sign_convention"
        ),
        intercept=_number(
            construction_raw["intercept"], f"{name}.construction.intercept"
        ),
    )
    allowed_sign_conventions = {
        "truth": {
            "sklearn_logistic_coef_positive_points_honest_to_deceptive",
            "positive_points_honest_to_deceptive",
        },
        "refusal": {"bad_minus_good_first_generated_token_residual_mean"},
        "orthogonal_control": {"seeded_orthogonal_control"},
        "shuffled_control": {"shuffled_label_control"},
    }
    if construction.sign_convention not in allowed_sign_conventions[direction_kind]:
        raise TruthEditingContractError(
            f"{name}.construction.sign_convention is invalid for {direction_kind}"
        )

    control_raw = raw["control_provenance"]
    if direction_kind in {"orthogonal_control", "shuffled_control"}:
        control_mapping = _mapping(control_raw, f"{name}.control_provenance")
        _exact(
            control_mapping,
            {"seed", "parent_direction_ids", "match_policy"},
            f"{name}.control_provenance",
        )
        match_policy = _enum(
            control_mapping["match_policy"],
            frozenset({"equal_rank_equal_norm"}),
            f"{name}.control_provenance.match_policy",
        )
        control_provenance = ControlProvenance(
            seed=_integer(control_mapping["seed"], f"{name}.control_provenance.seed"),
            parent_direction_ids=_string_tuple(
                control_mapping["parent_direction_ids"],
                f"{name}.control_provenance.parent_direction_ids",
            ),
            match_policy=match_policy,  # type: ignore[arg-type]
        )
    else:
        if control_raw is not None:
            raise TruthEditingContractError(
                f"{name}.control_provenance must be null for non-control directions"
            )
        control_provenance = None

    provenance_raw = _mapping(raw["provenance"], f"{name}.provenance")
    _exact(
        provenance_raw,
        {
            "dataset",
            "dataset_revision",
            "split",
            "ordered_row_ids_sha256",
            "source_code_revision",
        },
        f"{name}.provenance",
    )
    provenance = DirectionProvenance(
        dataset=_text(provenance_raw["dataset"], f"{name}.provenance.dataset"),
        dataset_revision=_text(
            provenance_raw["dataset_revision"], f"{name}.provenance.dataset_revision"
        ),
        split=_text(provenance_raw["split"], f"{name}.provenance.split"),
        ordered_row_ids_sha256=_sha256(
            provenance_raw["ordered_row_ids_sha256"],
            f"{name}.provenance.ordered_row_ids_sha256",
        ),
        source_code_revision=_git_revision(
            provenance_raw["source_code_revision"],
            f"{name}.provenance.source_code_revision",
        ),
    )

    leakage_raw = _mapping(raw["leakage"], f"{name}.leakage")
    _exact(
        leakage_raw,
        {
            "evaluation_disjoint",
            "heldout_family_disjoint",
            "sealed_audit_accessed",
            "audit_receipt_sha256",
        },
        f"{name}.leakage",
    )
    leakage = LeakageQualification(
        evaluation_disjoint=_boolean(
            leakage_raw["evaluation_disjoint"], f"{name}.leakage.evaluation_disjoint"
        ),
        heldout_family_disjoint=_boolean(
            leakage_raw["heldout_family_disjoint"],
            f"{name}.leakage.heldout_family_disjoint",
        ),
        sealed_audit_accessed=_boolean(
            leakage_raw["sealed_audit_accessed"],
            f"{name}.leakage.sealed_audit_accessed",
        ),
        audit_receipt_sha256=_sha256(
            leakage_raw["audit_receipt_sha256"], f"{name}.leakage.audit_receipt_sha256"
        ),
    )
    if leakage.sealed_audit_accessed:
        raise TruthEditingContractError(
            f"{name} must not have accessed the sealed audit"
        )

    qualification_raw = _mapping(raw["qualification"], f"{name}.qualification")
    _exact(
        qualification_raw,
        {"status", "receipt_sha256", "finite", "unit_norm", "qualified_rank"},
        f"{name}.qualification",
    )
    qualification = DirectionQualification(
        status=_text(qualification_raw["status"], f"{name}.qualification.status"),
        receipt_sha256=_sha256(
            qualification_raw["receipt_sha256"], f"{name}.qualification.receipt_sha256"
        ),
        finite=_boolean(qualification_raw["finite"], f"{name}.qualification.finite"),
        unit_norm=_boolean(
            qualification_raw["unit_norm"], f"{name}.qualification.unit_norm"
        ),
        qualified_rank=_integer(
            qualification_raw["qualified_rank"],
            f"{name}.qualification.qualified_rank",
            minimum=1,
        ),
    )
    allowed_statuses = (
        {"qualified_control"}
        if direction_kind in {"orthogonal_control", "shuffled_control"}
        else {"qualified", "candidate", "diagnostic_only"}
    )
    if qualification.status not in allowed_statuses:
        raise TruthEditingContractError(
            f"{name}.qualification.status must be one of {sorted(allowed_statuses)}"
        )
    if not qualification.finite or not qualification.unit_norm:
        raise TruthEditingContractError(f"{name} failed finite/unit-norm qualification")
    if qualification.status in {"qualified", "qualified_control"} and (
        not leakage.evaluation_disjoint or not leakage.heldout_family_disjoint
    ):
        raise TruthEditingContractError(
            f"{name} failed source/evaluation leakage gates"
        )

    source_layer = _integer(raw["source_layer"], f"{name}.source_layer")
    width = _integer(raw["width"], f"{name}.width", minimum=1)
    rank = _integer(raw["rank"], f"{name}.rank", minimum=1)
    if source_layer >= model.decoder_layer_count:
        raise TruthEditingContractError(
            f"{name}.source_layer exceeds model layer range"
        )
    if width != model.hidden_width:
        raise TruthEditingContractError(f"{name}.width differs from model hidden_width")
    if rank > qualification.qualified_rank or rank > width:
        raise TruthEditingContractError(
            f"{name}.rank exceeds its qualified rank or width"
        )
    return DirectionEntry(
        direction_id=_text(raw["direction_id"], f"{name}.direction_id"),
        kind=direction_kind,  # type: ignore[arg-type]
        family=direction_family,
        basis_variant=basis_variant,  # type: ignore[arg-type]
        domains=_string_tuple(raw["domains"], f"{name}.domains"),
        source_layer=source_layer,
        width=width,
        rank=rank,
        artifact=artifact,
        construction=construction,
        control_provenance=control_provenance,
        provenance=provenance,
        leakage=leakage,
        qualification=qualification,
    )


def parse_direction_bank_manifest(value: Any) -> DirectionBankManifest:
    raw = _mapping(value, "manifest")
    _exact(
        raw, {"format", "manifest_id", "model", "directions", "self_sha256"}, "manifest"
    )
    if raw["format"] != DIRECTION_BANK_MANIFEST_FORMAT:
        raise TruthEditingContractError("manifest format is unsupported")
    model = _parse_model(raw["model"])
    directions_raw = raw["directions"]
    if isinstance(directions_raw, (str, bytes)) or not isinstance(
        directions_raw, Sequence
    ):
        raise TruthEditingContractError("manifest.directions must be an array")
    if not directions_raw:
        raise TruthEditingContractError("manifest.directions must not be empty")
    directions = tuple(
        _parse_direction(direction, index, model)
        for index, direction in enumerate(directions_raw)
    )
    direction_ids = [direction.direction_id for direction in directions]
    if len(set(direction_ids)) != len(direction_ids):
        raise TruthEditingContractError("manifest direction IDs must be unique")
    vector_hashes = [direction.artifact.vector_sha256 for direction in directions]
    if len(set(vector_hashes)) != len(vector_hashes):
        raise TruthEditingContractError("manifest vector identities must be unique")
    by_id = {direction.direction_id: direction for direction in directions}
    for direction in directions:
        control = direction.control_provenance
        if control is None:
            continue
        missing_parents = sorted(set(control.parent_direction_ids) - set(by_id))
        if missing_parents:
            raise TruthEditingContractError(
                f"control {direction.direction_id} has absent parent direction IDs "
                f"{missing_parents}"
            )
        parents = tuple(by_id[parent_id] for parent_id in control.parent_direction_ids)
        if any(
            parent.kind in {"orthogonal_control", "shuffled_control"}
            for parent in parents
        ):
            raise TruthEditingContractError(
                f"control {direction.direction_id} cannot use another control as parent direction"
            )
        if direction.rank != sum(parent.rank for parent in parents):
            raise TruthEditingContractError(
                f"control {direction.direction_id} does not match parent direction rank"
            )
        if any(
            parent.width != direction.width
            or parent.source_layer != direction.source_layer
            for parent in parents
        ):
            raise TruthEditingContractError(
                f"control {direction.direction_id} does not match parent layer and width"
            )
    return DirectionBankManifest(
        format=DIRECTION_BANK_MANIFEST_FORMAT,
        manifest_id=_text(raw["manifest_id"], "manifest.manifest_id"),
        model=model,
        directions=directions,
        self_sha256=_self_hash(raw, "manifest"),
    )


@dataclass(frozen=True)
class DirectionSelection:
    direction_ids: tuple[str, ...]
    basis_method: Literal["qr", "svd"]
    requested_rank: int
    basis_sha256: str
    truth_direction_scope: Literal["global", "per_layer"]
    truth_direction_index: int


@dataclass(frozen=True)
class RefusalEdit:
    enabled: bool
    direction_ids: tuple[str, ...]
    direction_scope: Literal["global", "per_layer"]
    direction_index: int
    basis_method: Literal["qr", "svd"]
    requested_rank: int
    basis_sha256: str
    basis_variant: Literal["raw", "truth_orthogonalized", "joint"]
    strength: float
    writer_policy: Literal["attention", "mlp", "both"]


@dataclass(frozen=True)
class WriterKernel:
    enabled: bool
    kernel_center: float
    kernel_half_width: float
    edge_strength: float
    peak_strength: float


@dataclass(frozen=True)
class PersistentWeightBackend:
    type: Literal["persistent_weight"]
    transform_family: Literal["projection_reflection"]
    normalization_mode: Literal["exact", "norm_preserving"]
    attention: WriterKernel
    mlp: WriterKernel


@dataclass(frozen=True)
class ActivationParameters:
    strength: float
    target_probe_score: float | None
    reflection_coefficient: float | None
    one_sided_boundary: float | None
    affected_side: Literal["below", "above"] | None
    input_interval: tuple[float, float] | None
    output_interval: tuple[float, float] | None
    invert: bool | None
    margin_lower: float | None
    margin_upper: float | None


@dataclass(frozen=True)
class ActivationHookBackend:
    type: Literal["activation_hook"]
    transform_family: str
    token_scope: str
    source_layers: tuple[int, ...]
    prompt_positions: tuple[int, ...]
    generation_step_persistence: bool
    parameters: ActivationParameters


RecipeBackend: TypeAlias = PersistentWeightBackend | ActivationHookBackend
ConditionKind: TypeAlias = Literal[
    "base",
    "truth_only",
    "refusal_only",
    "joint",
    "orthogonal_control",
    "shuffled_control",
]


@dataclass(frozen=True)
class InterventionRecipe:
    format: Literal["truth_editing_intervention_recipe_v2"]
    recipe_id: str
    model_sha256: str
    direction_manifest_sha256: str
    direction_selection: DirectionSelection
    condition_kind: ConditionKind
    refusal: RefusalEdit
    backend: RecipeBackend
    self_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(asdict(self)))


def _parse_selection(value: Any) -> DirectionSelection:
    raw = _mapping(value, "recipe.direction_selection")
    _exact(
        raw,
        {
            "direction_ids",
            "basis_method",
            "requested_rank",
            "basis_sha256",
            "truth_direction_scope",
            "truth_direction_index",
        },
        "recipe.direction_selection",
    )
    return DirectionSelection(
        direction_ids=_string_tuple(
            raw["direction_ids"],
            "recipe.direction_selection.direction_ids",
            allow_empty=True,
        ),
        basis_method=_enum(
            raw["basis_method"],
            frozenset({"qr", "svd"}),
            "recipe.direction_selection.basis_method",
        ),  # type: ignore[arg-type]
        requested_rank=_integer(
            raw["requested_rank"], "recipe.direction_selection.requested_rank"
        ),
        basis_sha256=_sha256(
            raw["basis_sha256"], "recipe.direction_selection.basis_sha256"
        ),
        truth_direction_scope=_enum(
            raw["truth_direction_scope"],
            frozenset({"global", "per_layer"}),
            "recipe.direction_selection.truth_direction_scope",
        ),  # type: ignore[arg-type]
        truth_direction_index=_integer(
            raw["truth_direction_index"],
            "recipe.direction_selection.truth_direction_index",
        ),
    )


def _parse_refusal(value: Any) -> RefusalEdit:
    raw = _mapping(value, "recipe.refusal")
    _exact(
        raw,
        {
            "enabled",
            "direction_ids",
            "direction_scope",
            "direction_index",
            "basis_method",
            "requested_rank",
            "basis_sha256",
            "basis_variant",
            "strength",
            "writer_policy",
        },
        "recipe.refusal",
    )
    enabled = _boolean(raw["enabled"], "recipe.refusal.enabled")
    direction_ids = _string_tuple(
        raw["direction_ids"], "recipe.refusal.direction_ids", allow_empty=True
    )
    strength = _number(raw["strength"], "recipe.refusal.strength", minimum=0, maximum=2)
    requested_rank = _integer(raw["requested_rank"], "recipe.refusal.requested_rank")
    basis_variant = _enum(
        raw["basis_variant"],
        frozenset({"raw", "truth_orthogonalized", "joint"}),
        "recipe.refusal.basis_variant",
    )
    if enabled and (not direction_ids or strength == 0 or requested_rank == 0):
        raise TruthEditingContractError(
            "enabled refusal editing requires direction IDs, positive requested_rank, "
            "and positive strength"
        )
    if not enabled and (direction_ids or strength != 0 or requested_rank != 0):
        raise TruthEditingContractError(
            "disabled refusal editing requires empty direction IDs, zero rank, "
            "and exact zero strength"
        )
    if not enabled and basis_variant != "raw":
        raise TruthEditingContractError(
            "disabled refusal editing requires raw basis_variant"
        )
    return RefusalEdit(
        enabled=enabled,
        direction_ids=direction_ids,
        direction_scope=_enum(
            raw["direction_scope"],
            frozenset({"global", "per_layer"}),
            "recipe.refusal.direction_scope",
        ),  # type: ignore[arg-type]
        direction_index=_integer(
            raw["direction_index"], "recipe.refusal.direction_index"
        ),
        basis_method=_enum(
            raw["basis_method"],
            frozenset({"qr", "svd"}),
            "recipe.refusal.basis_method",
        ),  # type: ignore[arg-type]
        requested_rank=requested_rank,
        basis_sha256=_sha256(raw["basis_sha256"], "recipe.refusal.basis_sha256"),
        basis_variant=basis_variant,  # type: ignore[arg-type]
        strength=strength,
        writer_policy=_enum(
            raw["writer_policy"],
            frozenset({"attention", "mlp", "both"}),
            "recipe.refusal.writer_policy",
        ),  # type: ignore[arg-type]
    )


def _parse_kernel(value: Any, name: str) -> WriterKernel:
    raw = _mapping(value, name)
    _exact(
        raw,
        {
            "enabled",
            "kernel_center",
            "kernel_half_width",
            "edge_strength",
            "peak_strength",
        },
        name,
    )
    enabled = _boolean(raw["enabled"], f"{name}.enabled")
    center = _number(raw["kernel_center"], f"{name}.kernel_center", minimum=0)
    half_width = _number(
        raw["kernel_half_width"], f"{name}.kernel_half_width", minimum=0
    )
    edge = _number(raw["edge_strength"], f"{name}.edge_strength", minimum=0, maximum=2)
    peak = _number(raw["peak_strength"], f"{name}.peak_strength", minimum=0, maximum=2)
    if edge > peak:
        raise TruthEditingContractError(
            f"{name}.edge_strength must not exceed peak_strength"
        )
    if not enabled and (edge != 0 or peak != 0):
        raise TruthEditingContractError(
            f"disabled writer {name} requires exact zero strengths"
        )
    return WriterKernel(enabled, center, half_width, edge, peak)


def _interval(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TruthEditingContractError(f"{name} must be null or a two-number interval")
    result = (_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]"))
    if result[0] >= result[1]:
        raise TruthEditingContractError(f"{name} lower bound must be below upper bound")
    return result


def _parse_activation_parameters(value: Any, family: str) -> ActivationParameters:
    raw = _mapping(value, "recipe.backend.parameters")
    fields = {
        "strength",
        "target_probe_score",
        "reflection_coefficient",
        "one_sided_boundary",
        "affected_side",
        "input_interval",
        "output_interval",
        "invert",
        "margin_lower",
        "margin_upper",
    }
    _exact(raw, fields, "recipe.backend.parameters")
    strength = _number(
        raw["strength"], "recipe.backend.parameters.strength", minimum=0, maximum=2
    )
    target = _optional_number(
        raw["target_probe_score"], "recipe.backend.parameters.target_probe_score"
    )
    coefficient = _optional_number(
        raw["reflection_coefficient"],
        "recipe.backend.parameters.reflection_coefficient",
    )
    boundary = _optional_number(
        raw["one_sided_boundary"], "recipe.backend.parameters.one_sided_boundary"
    )
    side = raw["affected_side"]
    if side is not None:
        side = _enum(
            side,
            frozenset({"below", "above"}),
            "recipe.backend.parameters.affected_side",
        )
    input_interval = _interval(
        raw["input_interval"], "recipe.backend.parameters.input_interval"
    )
    output_interval = _interval(
        raw["output_interval"], "recipe.backend.parameters.output_interval"
    )
    invert = raw["invert"]
    if invert is not None:
        invert = _boolean(invert, "recipe.backend.parameters.invert")
    lower = _optional_number(
        raw["margin_lower"], "recipe.backend.parameters.margin_lower"
    )
    upper = _optional_number(
        raw["margin_upper"], "recipe.backend.parameters.margin_upper"
    )

    optional = {
        "target_probe_score": target,
        "reflection_coefficient": coefficient,
        "one_sided_boundary": boundary,
        "affected_side": side,
        "input_interval": input_interval,
        "output_interval": output_interval,
        "invert": invert,
        "margin_lower": lower,
        "margin_upper": upper,
    }
    required_by_family = {
        "projection_reflection": set(),
        "affine_projection": {"target_probe_score"},
        "partial_reflection": {"reflection_coefficient"},
        "one_sided_reflection": {
            "reflection_coefficient",
            "one_sided_boundary",
            "affected_side",
        },
        "bounded_remap": {"input_interval", "output_interval", "invert"},
        "bounded_margin_clamp": {"margin_lower", "margin_upper"},
    }
    required = required_by_family[family]
    missing = sorted(field for field in required if optional[field] is None)
    unexpected = sorted(
        field
        for field, item in optional.items()
        if item is not None and field not in required
    )
    if missing:
        raise TruthEditingContractError(
            f"recipe backend {family} requires {', '.join(missing)}"
        )
    if unexpected:
        raise TruthEditingContractError(
            f"recipe backend {family} forbids {', '.join(unexpected)}"
        )
    if coefficient is not None and not 0 <= coefficient <= 2:
        raise TruthEditingContractError(
            "reflection_coefficient must be between 0 and 2"
        )
    if lower is not None and upper is not None and lower >= upper:
        raise TruthEditingContractError("margin_lower must be below margin_upper")
    return ActivationParameters(
        strength=strength,
        target_probe_score=target,
        reflection_coefficient=coefficient,
        one_sided_boundary=boundary,
        affected_side=side,  # type: ignore[arg-type]
        input_interval=input_interval,
        output_interval=output_interval,
        invert=invert,
        margin_lower=lower,
        margin_upper=upper,
    )


def _parse_backend(value: Any) -> RecipeBackend:
    raw = _mapping(value, "recipe.backend")
    backend_type = raw.get("type")
    if backend_type == "persistent_weight":
        _exact(
            raw,
            {"type", "transform_family", "normalization_mode", "attention", "mlp"},
            "recipe.backend",
        )
        if raw["transform_family"] != "projection_reflection":
            raise TruthEditingContractError(
                "persistent_weight supports only projection_reflection"
            )
        return PersistentWeightBackend(
            type="persistent_weight",
            transform_family="projection_reflection",
            normalization_mode=_enum(
                raw["normalization_mode"],
                frozenset({"exact", "norm_preserving"}),
                "recipe.backend.normalization_mode",
            ),  # type: ignore[arg-type]
            attention=_parse_kernel(raw["attention"], "recipe.backend.attention"),
            mlp=_parse_kernel(raw["mlp"], "recipe.backend.mlp"),
        )
    if backend_type == "activation_hook":
        _exact(
            raw,
            {
                "type",
                "transform_family",
                "token_scope",
                "source_layers",
                "prompt_positions",
                "generation_step_persistence",
                "parameters",
            },
            "recipe.backend",
        )
        family = _enum(
            raw["transform_family"],
            _ACTIVATION_TRANSFORMS,
            "recipe.backend.transform_family",
        )
        token_scope = _enum(
            raw["token_scope"], _TOKEN_SCOPES, "recipe.backend.token_scope"
        )
        persistence = _boolean(
            raw["generation_step_persistence"],
            "recipe.backend.generation_step_persistence",
        )
        has_generation = token_scope in {
            "cached_generation",
            "prefill_last_and_cached_generation",
            "selected_prompt_positions_and_cached_generation",
        }
        if persistence != has_generation:
            raise TruthEditingContractError(
                "generation_step_persistence must exactly match a cached-generation token scope"
            )
        prompt_positions_raw = raw["prompt_positions"]
        if isinstance(prompt_positions_raw, (str, bytes)) or not isinstance(
            prompt_positions_raw, Sequence
        ):
            raise TruthEditingContractError(
                "recipe.backend.prompt_positions must be an array of integers"
            )
        prompt_positions = tuple(
            _integer(position, f"recipe.backend.prompt_positions[{index}]")
            for index, position in enumerate(prompt_positions_raw)
        )
        if tuple(sorted(set(prompt_positions))) != prompt_positions:
            raise TruthEditingContractError(
                "recipe.backend.prompt_positions must be sorted unique integers"
            )
        selects_prompt_positions = token_scope in {
            "selected_prompt_positions",
            "selected_prompt_positions_and_cached_generation",
        }
        if selects_prompt_positions and not prompt_positions:
            raise TruthEditingContractError(
                f"token scope {token_scope} requires prompt_positions"
            )
        if not selects_prompt_positions and prompt_positions:
            raise TruthEditingContractError(
                f"prompt_positions must be empty for token scope {token_scope}"
            )
        return ActivationHookBackend(
            type="activation_hook",
            transform_family=family,
            token_scope=token_scope,
            source_layers=_integer_tuple(
                raw["source_layers"], "recipe.backend.source_layers"
            ),
            prompt_positions=prompt_positions,
            generation_step_persistence=persistence,
            parameters=_parse_activation_parameters(raw["parameters"], family),
        )
    raise TruthEditingContractError(
        "recipe.backend.type must discriminate persistent_weight or activation_hook"
    )


def parse_intervention_recipe(value: Any) -> InterventionRecipe:
    raw = _mapping(value, "recipe")
    _exact(
        raw,
        {
            "format",
            "recipe_id",
            "model_sha256",
            "direction_manifest_sha256",
            "direction_selection",
            "condition_kind",
            "refusal",
            "backend",
            "self_sha256",
        },
        "recipe",
    )
    if raw["format"] != INTERVENTION_RECIPE_FORMAT:
        raise TruthEditingContractError("recipe format is unsupported")
    selection = _parse_selection(raw["direction_selection"])
    condition_kind = _enum(
        raw["condition_kind"], _CONDITION_KINDS, "recipe.condition_kind"
    )
    refusal = _parse_refusal(raw["refusal"])
    backend = _parse_backend(raw["backend"])
    has_truth_selection = bool(selection.direction_ids)
    if has_truth_selection != (selection.requested_rank > 0):
        raise TruthEditingContractError(
            "direction selection IDs and positive requested_rank must appear together"
        )
    if condition_kind == "base":
        if selection.direction_ids or selection.requested_rank != 0 or refusal.enabled:
            raise TruthEditingContractError(
                "base recipes must select no directions or refusal edit"
            )
        if isinstance(backend, PersistentWeightBackend) and (
            backend.attention.enabled or backend.mlp.enabled
        ):
            raise TruthEditingContractError(
                "base recipes must disable persistent writers"
            )
        if (
            isinstance(backend, ActivationHookBackend)
            and backend.parameters.strength != 0
        ):
            raise TruthEditingContractError(
                "base recipes must use exact zero activation strength"
            )
    elif condition_kind == "refusal_only":
        if has_truth_selection:
            raise TruthEditingContractError(
                "refusal_only recipes require no truth direction selection"
            )
        if not refusal.enabled:
            raise TruthEditingContractError(
                "refusal_only recipes require refusal editing"
            )
    elif condition_kind == "joint":
        if not has_truth_selection:
            raise TruthEditingContractError("joint recipes require truth directions")
        if not refusal.enabled:
            raise TruthEditingContractError("joint recipes require refusal editing")
    else:
        if not has_truth_selection:
            raise TruthEditingContractError(
                f"{condition_kind} recipes require selected directions"
            )
        if refusal.enabled:
            raise TruthEditingContractError(
                f"{condition_kind} recipes must disable refusal editing"
            )
    if (
        refusal.basis_variant in {"truth_orthogonalized", "joint"}
        and condition_kind != "joint"
    ):
        raise TruthEditingContractError(
            f"refusal basis_variant {refusal.basis_variant} requires a joint condition"
        )
    return InterventionRecipe(
        format=INTERVENTION_RECIPE_FORMAT,
        recipe_id=_text(raw["recipe_id"], "recipe.recipe_id"),
        model_sha256=_sha256(raw["model_sha256"], "recipe.model_sha256"),
        direction_manifest_sha256=_sha256(
            raw["direction_manifest_sha256"], "recipe.direction_manifest_sha256"
        ),
        direction_selection=selection,
        condition_kind=condition_kind,  # type: ignore[arg-type]
        refusal=refusal,
        backend=backend,
        self_sha256=_self_hash(raw, "recipe"),
    )


def _active_kernel_layers(kernel: WriterKernel, decoder_layer_count: int) -> set[int]:
    if not kernel.enabled:
        return set()
    active: set[int] = set()
    for layer in range(decoder_layer_count):
        distance = abs(layer - kernel.kernel_center)
        if kernel.kernel_half_width == 0:
            strength = kernel.peak_strength if distance == 0 else 0.0
        elif distance <= kernel.kernel_half_width:
            strength = kernel.peak_strength + (distance / kernel.kernel_half_width) * (
                kernel.edge_strength - kernel.peak_strength
            )
        else:
            strength = 0.0
        if strength > 0:
            active.add(layer)
    return active


def _validate_basis_rank(
    *,
    label: str,
    method: str,
    requested_rank: int,
    directions: Sequence[DirectionEntry],
) -> None:
    declared_rank = sum(direction.rank for direction in directions)
    qualified_rank = sum(
        direction.qualification.qualified_rank for direction in directions
    )
    if method == "qr" and requested_rank != declared_rank:
        raise TruthEditingContractError(
            f"{label} QR requested_rank must equal the exact selected span "
            "declared direction rank"
        )
    if requested_rank > declared_rank:
        raise TruthEditingContractError(
            f"{label} requested_rank exceeds aggregate declared direction rank"
        )
    if requested_rank > qualified_rank:
        raise TruthEditingContractError(
            f"{label} requested_rank exceeds selected qualified rank"
        )


def validate_recipe_compatibility(
    recipe: InterventionRecipe, manifest: DirectionBankManifest
) -> None:
    """Fail unless a parsed recipe can be materialized from a parsed manifest."""

    if recipe.direction_manifest_sha256 != manifest.self_sha256:
        raise TruthEditingContractError("recipe direction_manifest_sha256 mismatch")
    if recipe.model_sha256 != manifest.model.model_sha256:
        raise TruthEditingContractError("recipe model_sha256 mismatch")

    by_id = {direction.direction_id: direction for direction in manifest.directions}
    selected_ids = recipe.direction_selection.direction_ids
    missing = sorted(set(selected_ids + recipe.refusal.direction_ids) - set(by_id))
    if missing:
        raise TruthEditingContractError(
            f"recipe direction IDs absent from manifest: {missing}"
        )

    selected = tuple(by_id[direction_id] for direction_id in selected_ids)
    refusal = tuple(
        by_id[direction_id] for direction_id in recipe.refusal.direction_ids
    )
    nonqualified = sorted(
        direction.direction_id
        for direction in selected + refusal
        if direction.qualification.status not in {"qualified", "qualified_control"}
    )
    if nonqualified:
        raise TruthEditingContractError(
            f"recipe directions are not optimizer-qualified: {nonqualified}"
        )
    if len(selected) == 1 and (
        recipe.direction_selection.basis_sha256 != selected[0].artifact.vector_sha256
    ):
        raise TruthEditingContractError(
            "single-direction truth basis_sha256 must match the stored artifact "
            "vector_sha256"
        )
    if any(direction.kind != "refusal" for direction in refusal):
        raise TruthEditingContractError(
            "refusal direction IDs must select refusal directions"
        )
    mismatched_refusal_variants = sorted(
        {
            direction.basis_variant
            for direction in refusal
            if direction.basis_variant != recipe.refusal.basis_variant
        }
    )
    if mismatched_refusal_variants:
        raise TruthEditingContractError(
            "recipe refusal basis_variant does not match manifest-qualified variants "
            f"{mismatched_refusal_variants}"
        )
    if len(refusal) == 1 and (
        recipe.refusal.basis_sha256 != refusal[0].artifact.vector_sha256
    ):
        raise TruthEditingContractError(
            "single-direction refusal basis_sha256 must match the stored artifact "
            "vector_sha256"
        )
    if recipe.condition_kind == "orthogonal_control" and any(
        direction.kind != "orthogonal_control" for direction in selected
    ):
        raise TruthEditingContractError(
            "orthogonal_control recipe must select only orthogonal_control directions"
        )
    if recipe.condition_kind == "shuffled_control" and any(
        direction.kind != "shuffled_control" for direction in selected
    ):
        raise TruthEditingContractError(
            "shuffled_control recipe must select only shuffled_control directions"
        )
    if recipe.condition_kind in {"truth_only", "joint"} and any(
        direction.kind != "truth" for direction in selected
    ):
        raise TruthEditingContractError(
            "truth recipe must select only truth directions"
        )

    if selected and len({direction.width for direction in selected}) != 1:
        raise TruthEditingContractError("selected directions have incompatible widths")
    if (
        recipe.direction_selection.truth_direction_index
        >= manifest.model.decoder_layer_count
    ):
        raise TruthEditingContractError(
            "truth_direction_index exceeds model layer range"
        )
    selected_layers = {direction.source_layer for direction in selected}
    if (
        selected
        and recipe.direction_selection.truth_direction_scope == "global"
        and selected_layers != {recipe.direction_selection.truth_direction_index}
    ):
        raise TruthEditingContractError(
            "global truth_direction_index must match every selected source layer"
        )
    if recipe.refusal.enabled:
        if recipe.refusal.direction_index >= manifest.model.decoder_layer_count:
            raise TruthEditingContractError(
                "refusal direction_index exceeds model layer range"
            )
        refusal_layers = {direction.source_layer for direction in refusal}
        if recipe.refusal.direction_scope == "global" and refusal_layers != {
            recipe.refusal.direction_index
        }:
            raise TruthEditingContractError(
                "global refusal direction_index must match every selected source layer"
            )

    if isinstance(recipe.backend, PersistentWeightBackend):
        maximum_layer = manifest.model.decoder_layer_count - 1
        for name, kernel in (
            ("attention", recipe.backend.attention),
            ("mlp", recipe.backend.mlp),
        ):
            if kernel.kernel_center > maximum_layer:
                raise TruthEditingContractError(
                    f"{name} kernel_center exceeds model layer range"
                )
            if kernel.kernel_half_width > maximum_layer:
                raise TruthEditingContractError(
                    f"{name} kernel_half_width exceeds model layer range"
                )
        if recipe.condition_kind != "base" and not (
            recipe.backend.attention.enabled or recipe.backend.mlp.enabled
        ):
            raise TruthEditingContractError(
                "non-base persistent recipe has no enabled writer"
            )
        policy = recipe.refusal.writer_policy
        if recipe.refusal.enabled and (
            (policy in {"attention", "both"} and not recipe.backend.attention.enabled)
            or (policy in {"mlp", "both"} and not recipe.backend.mlp.enabled)
        ):
            raise TruthEditingContractError(
                "refusal writer_policy targets a disabled writer"
            )
        required_truth_layers = _active_kernel_layers(
            recipe.backend.attention, manifest.model.decoder_layer_count
        ) | _active_kernel_layers(
            recipe.backend.mlp, manifest.model.decoder_layer_count
        )
    else:
        invalid_layers = [
            layer
            for layer in recipe.backend.source_layers
            if layer >= manifest.model.decoder_layer_count
        ]
        if invalid_layers:
            raise TruthEditingContractError(
                f"activation source layers exceed model range: {invalid_layers}"
            )
        required_truth_layers = set(recipe.backend.source_layers)

    if recipe.direction_selection.truth_direction_scope == "global":
        _validate_basis_rank(
            label="truth basis",
            method=recipe.direction_selection.basis_method,
            requested_rank=recipe.direction_selection.requested_rank,
            directions=selected,
        )
    elif selected:
        missing_truth_layers = sorted(required_truth_layers - selected_layers)
        if missing_truth_layers:
            raise TruthEditingContractError(
                "per_layer truth directions missing active writer/hook layers "
                f"{missing_truth_layers}"
            )
        for layer in sorted(required_truth_layers):
            _validate_basis_rank(
                label=f"truth basis layer {layer}",
                method=recipe.direction_selection.basis_method,
                requested_rank=recipe.direction_selection.requested_rank,
                directions=tuple(
                    direction
                    for direction in selected
                    if direction.source_layer == layer
                ),
            )

    if recipe.refusal.enabled and recipe.refusal.direction_scope == "global":
        _validate_basis_rank(
            label="refusal basis",
            method=recipe.refusal.basis_method,
            requested_rank=recipe.refusal.requested_rank,
            directions=refusal,
        )

    if recipe.refusal.enabled and recipe.refusal.direction_scope == "per_layer":
        required_refusal_layers: set[int]
        if isinstance(recipe.backend, ActivationHookBackend):
            required_refusal_layers = set(recipe.backend.source_layers)
        else:
            required_refusal_layers = set()
            writer_kernels = {
                "attention": recipe.backend.attention,
                "mlp": recipe.backend.mlp,
            }
            selected_writers = (
                {"attention", "mlp"}
                if recipe.refusal.writer_policy == "both"
                else {recipe.refusal.writer_policy}
            )
            for writer_name in selected_writers:
                kernel = writer_kernels[writer_name]
                required_refusal_layers.update(
                    _active_kernel_layers(kernel, manifest.model.decoder_layer_count)
                )
        missing_refusal_layers = sorted(required_refusal_layers - refusal_layers)
        if missing_refusal_layers:
            raise TruthEditingContractError(
                "per_layer refusal directions missing writer layers "
                f"{missing_refusal_layers}"
            )
        for layer in sorted(required_refusal_layers):
            _validate_basis_rank(
                label=f"refusal basis layer {layer}",
                method=recipe.refusal.basis_method,
                requested_rank=recipe.refusal.requested_rank,
                directions=tuple(
                    direction
                    for direction in refusal
                    if direction.source_layer == layer
                ),
            )


__all__ = [
    "DIRECTION_BANK_MANIFEST_FORMAT",
    "INTERVENTION_RECIPE_FORMAT",
    "ActivationHookBackend",
    "DirectionBankManifest",
    "InterventionRecipe",
    "PersistentWeightBackend",
    "TruthEditingContractError",
    "canonical_json_bytes",
    "canonical_sha256",
    "parse_direction_bank_manifest",
    "parse_intervention_recipe",
    "validate_recipe_compatibility",
]
