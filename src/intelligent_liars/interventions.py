from __future__ import annotations

import json
import math
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Sequence

import torch


INTERVENTION_BUNDLE_FORMAT = "qwen_truth_intervention_v2"
LEGACY_INTERVENTION_BUNDLE_FORMAT = "qwen_truth_intervention_v1"
EXPECTED_DIRECTION_SIGN = "sklearn_logistic_coef_positive_points_honest_to_deceptive"


class InterventionMethod(StrEnum):
    SCALAR_ADDITION = "scalar_addition"
    AFFINE_PROJECTION = "affine_projection"
    FULL_REFLECTION = "full_reflection"
    PARTIAL_REFLECTION = "partial_reflection"
    ONE_SIDED_REFLECTION = "one_sided_reflection"
    BOUNDED_REMAP = "bounded_remap"
    BOUNDED_MARGIN_CLAMP = "bounded_margin_clamp"


DECEPTION_DIRECTED_SUITE_METHODS = {
    "directed_scalar_add_deceptive": InterventionMethod.SCALAR_ADDITION,
    "directed_affine_project_deceptive": InterventionMethod.AFFINE_PROJECTION,
    "directed_full_reflection": InterventionMethod.FULL_REFLECTION,
    "directed_partial_reflection": InterventionMethod.PARTIAL_REFLECTION,
    "honest_boundary_ablation": InterventionMethod.ONE_SIDED_REFLECTION,
    "bounded_directed_inversion": InterventionMethod.BOUNDED_REMAP,
    "bounded_deceptive_margin": InterventionMethod.BOUNDED_MARGIN_CLAMP,
}
SEEDED_ORTHOGONAL_CONTROL_VARIANT = "seeded_orthogonal_full_reflection"


class TokenScope(StrEnum):
    ALL = "all"
    LAST_TOKEN = "last_token"


class DirectionMode(StrEnum):
    PROBE = "probe"
    SEEDED_ORTHOGONAL_CONTROL = "seeded_orthogonal_control"
    MATCHED_RANDOM = "seeded_orthogonal_control"


class ScoreDirectionality(StrEnum):
    SYMMETRIC = "symmetric"
    TOWARD_DECEPTIVE = "toward_deceptive"


@dataclass(frozen=True)
class ProbeDirection:
    vector: tuple[float, ...]
    intercept: float
    layer: int
    task: str
    sign_convention: str = EXPECTED_DIRECTION_SIGN
    source_path: str | None = None
    original_intercept: float | None = None

    def tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.vector, dtype=torch.float64, device=device)


@dataclass(frozen=True)
class InterventionSpec:
    method: InterventionMethod
    layers: tuple[int, ...]
    token_scope: TokenScope = TokenScope.LAST_TOKEN
    direction_mode: DirectionMode = DirectionMode.PROBE
    score_directionality: ScoreDirectionality = ScoreDirectionality.SYMMETRIC
    control_seed: int | None = None
    score_delta: float = 0.0
    projection_target: float = 0.0
    reflection_strength: float = 1.0
    selected_side: Literal["honest", "deceptive"] = "honest"
    remap_input_min: float = -1.0
    remap_input_max: float = 1.0
    remap_output_min: float = 1.0
    remap_output_max: float = -1.0
    margin: float = 1.0
    max_score_delta: float | None = None


@dataclass(frozen=True)
class InterventionBundle:
    direction: ProbeDirection
    spec: InterventionSpec

    def effective_direction(self) -> ProbeDirection:
        if self.spec.direction_mode == DirectionMode.PROBE:
            return self.direction
        if self.spec.control_seed is None:
            raise ValueError(
                "seeded orthogonal control direction requires control_seed"
            )
        return replace(
            self.direction,
            vector=tuple(
                seeded_orthogonal_direction(
                    self.direction.tensor(), seed=self.spec.control_seed
                ).tolist()
            ),
            intercept=0.0,
            task=f"seeded_orthogonal_control_for:{self.direction.task}",
        )


def canonical_intervention_suite_specs(
    *,
    layers: tuple[int, ...],
    control_seed: int,
    deceptive_margin: float,
    score_movement_budget: float,
) -> dict[str, InterventionSpec]:
    if deceptive_margin <= 0 or score_movement_budget <= 0:
        raise ValueError("Suite margin and movement budget must be positive")
    return {
        "directed_scalar_add_deceptive": InterventionSpec(
            method=InterventionMethod.SCALAR_ADDITION,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            score_delta=score_movement_budget,
        ),
        "directed_affine_project_deceptive": InterventionSpec(
            method=InterventionMethod.AFFINE_PROJECTION,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            projection_target=deceptive_margin,
        ),
        "directed_full_reflection": InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
        ),
        "directed_partial_reflection": InterventionSpec(
            method=InterventionMethod.PARTIAL_REFLECTION,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            reflection_strength=0.75,
        ),
        "honest_boundary_ablation": InterventionSpec(
            method=InterventionMethod.ONE_SIDED_REFLECTION,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            reflection_strength=0.5,
            selected_side="honest",
        ),
        "bounded_directed_inversion": InterventionSpec(
            method=InterventionMethod.BOUNDED_REMAP,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            remap_input_min=-score_movement_budget,
            remap_input_max=score_movement_budget,
            remap_output_min=score_movement_budget,
            remap_output_max=-score_movement_budget,
            max_score_delta=2.0 * score_movement_budget,
        ),
        "bounded_deceptive_margin": InterventionSpec(
            method=InterventionMethod.BOUNDED_MARGIN_CLAMP,
            layers=layers,
            score_directionality=ScoreDirectionality.TOWARD_DECEPTIVE,
            selected_side="deceptive",
            margin=deceptive_margin,
            max_score_delta=score_movement_budget,
        ),
        SEEDED_ORTHOGONAL_CONTROL_VARIANT: InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=layers,
            direction_mode=DirectionMode.SEEDED_ORTHOGONAL_CONTROL,
            control_seed=control_seed,
        ),
    }


def load_probe_direction(
    path: Path,
    *,
    layer: int,
    task: str = "general_domain",
) -> ProbeDirection:
    payload = json.loads(path.read_text())
    if payload.get("direction_sign_convention") != EXPECTED_DIRECTION_SIGN:
        raise ValueError(
            "Unsupported probe direction sign convention: "
            f"{payload.get('direction_sign_convention')!r}"
        )
    if task == "general_domain":
        candidates = payload.get("general_domain", {}).get("directions", [])
    else:
        candidates = payload.get("directions", [])
    matches = [
        row
        for row in candidates
        if int(row.get("layer", -1)) == layer and row.get("task") == task
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one direction for task={task!r}, layer={layer}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    if row.get("direction_sign_convention") != EXPECTED_DIRECTION_SIGN:
        raise ValueError(
            "Selected direction has unsupported sign convention: "
            f"{row.get('direction_sign_convention')!r}"
        )
    vector = tuple(float(value) for value in row["direction_vector"])
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Probe direction must be a non-empty finite vector")
    if math.isclose(sum(value * value for value in vector), 0.0):
        raise ValueError("Probe direction must be non-zero")
    direction = ProbeDirection(
        vector=vector,
        intercept=float(row["intercept"]),
        layer=layer,
        task=task,
        sign_convention=str(row["direction_sign_convention"]),
        source_path=str(path.resolve()),
    )
    validate_probe_direction(direction)
    return direction


def save_intervention_bundle(
    bundle: InterventionBundle, path: Path, *, overwrite: bool = False
) -> None:
    validate_probe_direction(bundle.direction)
    validate_intervention_spec(bundle.spec)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Intervention bundle already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": INTERVENTION_BUNDLE_FORMAT,
        "direction": asdict(bundle.direction),
        "spec": {
            **asdict(bundle.spec),
            "method": bundle.spec.method.value,
            "token_scope": bundle.spec.token_scope.value,
            "direction_mode": bundle.spec.direction_mode.value,
            "layers": list(bundle.spec.layers),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_intervention_bundle(path: Path) -> InterventionBundle:
    payload = json.loads(path.read_text())
    if payload.get("format") not in {
        INTERVENTION_BUNDLE_FORMAT,
        LEGACY_INTERVENTION_BUNDLE_FORMAT,
    }:
        raise ValueError(
            f"Unsupported intervention bundle format: {payload.get('format')!r}"
        )
    raw_direction = dict(payload["direction"])
    raw_direction["vector"] = tuple(float(value) for value in raw_direction["vector"])
    direction = ProbeDirection(**raw_direction)
    raw_spec = dict(payload["spec"])
    raw_spec["method"] = InterventionMethod(raw_spec["method"])
    raw_spec["token_scope"] = TokenScope(raw_spec["token_scope"])
    if raw_spec.get("direction_mode") == "matched_random":
        raw_spec["direction_mode"] = DirectionMode.SEEDED_ORTHOGONAL_CONTROL
    else:
        raw_spec["direction_mode"] = DirectionMode(raw_spec["direction_mode"])
    if "control_seed" not in raw_spec:
        raw_spec["control_seed"] = raw_spec.pop("random_seed", None)
    else:
        raw_spec.pop("random_seed", None)
    raw_spec["score_directionality"] = ScoreDirectionality(
        raw_spec.get("score_directionality", ScoreDirectionality.SYMMETRIC)
    )
    raw_spec["layers"] = tuple(int(layer) for layer in raw_spec["layers"])
    spec = InterventionSpec(**raw_spec)
    validate_probe_direction(direction)
    validate_intervention_spec(spec)
    return InterventionBundle(direction=direction, spec=spec)


def validate_probe_direction(direction: ProbeDirection) -> None:
    if direction.sign_convention != EXPECTED_DIRECTION_SIGN:
        raise ValueError(
            f"Unsupported probe direction sign convention: {direction.sign_convention!r}"
        )
    if not direction.vector or not all(
        math.isfinite(value) for value in direction.vector
    ):
        raise ValueError("Probe direction must be a non-empty finite vector")
    if math.isclose(sum(value * value for value in direction.vector), 0.0):
        raise ValueError("Probe direction must be non-zero")
    if not math.isfinite(direction.intercept):
        raise ValueError("Probe direction intercept must be finite")
    if direction.original_intercept is not None and not math.isfinite(
        direction.original_intercept
    ):
        raise ValueError(
            "Probe direction original_intercept must be finite when provided"
        )
    if direction.layer < 0:
        raise ValueError("Probe direction layer must be non-negative")


def validate_intervention_spec(spec: InterventionSpec) -> None:
    if not spec.layers:
        raise ValueError("At least one intervention layer is required")
    if any(layer < 0 for layer in spec.layers):
        raise ValueError("Intervention layers must be non-negative")
    if len(set(spec.layers)) != len(spec.layers):
        raise ValueError("Intervention layers must be unique")
    numeric_values = {
        "score_delta": spec.score_delta,
        "projection_target": spec.projection_target,
        "reflection_strength": spec.reflection_strength,
        "remap_input_min": spec.remap_input_min,
        "remap_input_max": spec.remap_input_max,
        "remap_output_min": spec.remap_output_min,
        "remap_output_max": spec.remap_output_max,
        "margin": spec.margin,
    }
    if spec.max_score_delta is not None:
        numeric_values["max_score_delta"] = spec.max_score_delta
    non_finite = [
        name for name, value in numeric_values.items() if not math.isfinite(value)
    ]
    if non_finite:
        raise ValueError(f"Intervention numeric settings must be finite: {non_finite}")
    if not 0.0 <= spec.reflection_strength <= 1.0:
        raise ValueError("reflection_strength must be between 0 and 1")
    if spec.remap_input_min >= spec.remap_input_max:
        raise ValueError("remap_input_min must be smaller than remap_input_max")
    if spec.margin < 0:
        raise ValueError("margin must be non-negative")
    if spec.selected_side not in {"honest", "deceptive"}:
        raise ValueError("selected_side must be 'honest' or 'deceptive'")
    if spec.max_score_delta is not None and spec.max_score_delta <= 0:
        raise ValueError("max_score_delta must be positive when provided")
    if (
        spec.method == InterventionMethod.BOUNDED_MARGIN_CLAMP
        and spec.max_score_delta is None
    ):
        raise ValueError("bounded_margin_clamp requires max_score_delta")
    if (
        spec.direction_mode == DirectionMode.SEEDED_ORTHOGONAL_CONTROL
        and spec.control_seed is None
    ):
        raise ValueError("seeded orthogonal control direction requires control_seed")
    if (
        spec.direction_mode == DirectionMode.SEEDED_ORTHOGONAL_CONTROL
        and spec.score_directionality != ScoreDirectionality.SYMMETRIC
    ):
        raise ValueError(
            "seeded orthogonal controls require symmetric score directionality"
        )
    if spec.score_directionality == ScoreDirectionality.TOWARD_DECEPTIVE:
        if spec.method == InterventionMethod.SCALAR_ADDITION and spec.score_delta <= 0:
            raise ValueError("directed scalar addition requires a positive score_delta")
        if (
            spec.method == InterventionMethod.AFFINE_PROJECTION
            and spec.projection_target <= 0
        ):
            raise ValueError(
                "directed affine projection requires a positive projection_target"
            )
        if (
            spec.method == InterventionMethod.ONE_SIDED_REFLECTION
            and spec.selected_side != "honest"
        ):
            raise ValueError(
                "deception-directed one-sided reflection must select the honest side"
            )
        if spec.method == InterventionMethod.BOUNDED_MARGIN_CLAMP and (
            spec.selected_side != "deceptive" or spec.margin <= 0
        ):
            raise ValueError(
                "deception-directed margin clamp requires a positive deceptive margin"
            )


def seeded_orthogonal_direction(
    direction: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    if direction.ndim != 1 or direction.numel() < 2:
        raise ValueError("A seeded orthogonal control requires at least two dimensions")
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("Direction must have a finite non-zero norm")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(direction.shape, generator=generator, dtype=torch.float64)
    source = direction.detach().to(device="cpu", dtype=torch.float64)
    random = random - torch.dot(random, source) / torch.dot(source, source) * source
    random_norm = torch.linalg.vector_norm(random)
    if random_norm.item() == 0.0:
        raise ValueError("Failed to construct a seeded orthogonal control direction")
    return (random / random_norm * norm.to(device="cpu", dtype=torch.float64)).to(
        device=direction.device,
        dtype=direction.dtype,
    )


def matched_random_direction(direction: torch.Tensor, *, seed: int) -> torch.Tensor:
    return seeded_orthogonal_direction(direction, seed=seed)


def transform_activations(
    activations: torch.Tensor,
    direction: ProbeDirection,
    spec: InterventionSpec,
) -> torch.Tensor:
    validate_probe_direction(direction)
    validate_intervention_spec(spec)
    if activations.ndim < 1:
        raise ValueError("Activations must have at least one dimension")
    if activations.shape[-1] == 0:
        raise ValueError("Activations must have a non-empty hidden dimension")
    if not torch.is_floating_point(activations):
        raise ValueError("Activations must have a floating-point dtype")
    if not torch.isfinite(activations).all().item():
        raise ValueError("Activations must contain only finite values")
    output_dtype = activations.dtype
    working = activations.float()
    vector = direction.tensor(device=activations.device).float()
    if vector.shape != (activations.shape[-1],):
        raise ValueError(
            f"Direction has shape {tuple(vector.shape)}, expected {(activations.shape[-1],)}"
        )
    denominator = torch.dot(vector, vector)
    if (
        not torch.isfinite(vector).all().item()
        or not torch.isfinite(denominator).item()
    ):
        raise ValueError("Probe direction must remain finite in the compute dtype")
    if denominator.item() == 0.0:
        raise ValueError("Probe direction must remain non-zero in the compute dtype")
    scores = torch.einsum("...d,d->...", working, vector) + direction.intercept

    if spec.method == InterventionMethod.SCALAR_ADDITION:
        target_scores = scores + spec.score_delta
    elif spec.method == InterventionMethod.AFFINE_PROJECTION:
        target_scores = torch.full_like(scores, spec.projection_target)
    elif spec.method == InterventionMethod.FULL_REFLECTION:
        target_scores = -scores
    elif spec.method == InterventionMethod.PARTIAL_REFLECTION:
        target_scores = (1.0 - 2.0 * spec.reflection_strength) * scores
    elif spec.method == InterventionMethod.ONE_SIDED_REFLECTION:
        reflected = (1.0 - 2.0 * spec.reflection_strength) * scores
        selected = scores < 0 if spec.selected_side == "honest" else scores > 0
        target_scores = torch.where(selected, reflected, scores)
    elif spec.method == InterventionMethod.BOUNDED_REMAP:
        unit = (
            scores.clamp(spec.remap_input_min, spec.remap_input_max)
            - spec.remap_input_min
        ) / (spec.remap_input_max - spec.remap_input_min)
        target_scores = spec.remap_output_min + unit * (
            spec.remap_output_max - spec.remap_output_min
        )
    elif spec.method == InterventionMethod.BOUNDED_MARGIN_CLAMP:
        if spec.selected_side == "deceptive":
            target_scores = torch.maximum(scores, torch.full_like(scores, spec.margin))
        else:
            target_scores = torch.minimum(scores, torch.full_like(scores, -spec.margin))
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"Unsupported intervention method: {spec.method}")

    if spec.score_directionality == ScoreDirectionality.TOWARD_DECEPTIVE:
        target_scores = torch.where(
            scores < 0,
            torch.maximum(scores, target_scores),
            scores,
        )

    score_delta = target_scores - scores
    if spec.max_score_delta is not None:
        score_delta = score_delta.clamp(-spec.max_score_delta, spec.max_score_delta)
    transformed = working + score_delta.unsqueeze(-1) / denominator * vector
    if not torch.isfinite(transformed).all().item():
        raise ValueError("Intervention produced non-finite output")
    output = transformed.to(dtype=output_dtype)
    if not torch.isfinite(output).all().item():
        raise ValueError(
            "Intervention produced non-finite output after dtype conversion"
        )
    return output


def qwen_language_layers(model: Any) -> Sequence[Any]:
    try:
        return model.model.language_model.layers
    except AttributeError as error:
        raise TypeError(
            "Expected Qwen3-VL decoder layers at model.model.language_model.layers"
        ) from error


class RuntimeIntervention(AbstractContextManager["RuntimeIntervention"]):
    def __init__(self, model: Any, bundle: InterventionBundle) -> None:
        self.model = model
        self.bundle = bundle
        self.handles: list[Any] = []

    def __enter__(self) -> RuntimeIntervention:
        layers = qwen_language_layers(self.model)
        direction = self.bundle.effective_direction()
        invalid_layers = [
            layer for layer in self.bundle.spec.layers if layer >= len(layers)
        ]
        if invalid_layers:
            raise ValueError(
                f"Intervention layers {invalid_layers} are outside model with {len(layers)} layers"
            )
        for layer_idx in self.bundle.spec.layers:

            def hook(_module: Any, _args: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                if self.bundle.spec.token_scope == TokenScope.LAST_TOKEN:
                    transformed = hidden.clone()
                    transformed[..., -1:, :] = transform_activations(
                        hidden[..., -1:, :], direction, self.bundle.spec
                    )
                else:
                    transformed = transform_activations(
                        hidden, direction, self.bundle.spec
                    )
                if isinstance(output, tuple):
                    return (transformed, *output[1:])
                return transformed

            self.handles.append(layers[layer_idx].register_forward_hook(hook))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
