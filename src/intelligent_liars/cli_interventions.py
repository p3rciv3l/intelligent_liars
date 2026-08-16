from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import typer

from intelligent_liars.cli_common import (
    app,
    console,
    _project_path,
    _resolve_project_root,
)
from intelligent_liars.interventions import (
    DirectionMode,
    InterventionBundle,
    InterventionMethod,
    InterventionSpec,
    TokenScope,
    load_intervention_bundle,
    load_probe_direction,
    save_intervention_bundle,
    save_writer_edited_model,
    validate_materializable_bundle,
    validate_intervention_spec,
)
from intelligent_liars.models import load_model_and_processor, model_config_from_env


@app.command("build-intervention")
def build_intervention(
    probe_path: Path = typer.Option(
        ..., "--probe", help="Probe-result JSON containing the direction."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Portable intervention JSON to write."
    ),
    layer: int = typer.Option(
        ..., "--layer", min=0, help="Probe direction layer to load."
    ),
    intervention_layers: list[int] | None = typer.Option(
        None,
        "--intervention-layer",
        help="Decoder layer to intervene on. Repeatable; defaults to the probe layer.",
    ),
    task: str = typer.Option(
        "general_domain", help="Direction task in the probe artifact."
    ),
    method: InterventionMethod = typer.Option(..., case_sensitive=False),
    token_scope: TokenScope = typer.Option(TokenScope.LAST_TOKEN, case_sensitive=False),
    matched_random_seed: int | None = typer.Option(
        None,
        help="Replace the probe with a seeded orthogonal direction of matching norm.",
    ),
    score_delta: float = typer.Option(
        0.0, help="Probe-score displacement for scalar addition."
    ),
    projection_target: float = typer.Option(
        0.0, help="Target probe score for affine projection."
    ),
    reflection_strength: float = typer.Option(1.0, min=0.0, max=1.0),
    selected_side: Literal["honest", "deceptive"] = typer.Option("honest"),
    remap_input_min: float = typer.Option(-1.0),
    remap_input_max: float = typer.Option(1.0),
    remap_output_min: float = typer.Option(1.0),
    remap_output_max: float = typer.Option(-1.0),
    margin: float = typer.Option(1.0, min=0.0),
    max_score_delta: float | None = typer.Option(
        None,
        min=1e-12,
        help="Optional absolute bound on the probe-score adjustment.",
    ),
    project_root: Path | None = typer.Option(None),
    overwrite: bool = typer.Option(False, "--overwrite"),
    homogeneous_writer_edit: bool = typer.Option(
        False,
        "--homogeneous-writer-edit",
        help=(
            "Build an all-token, zero-intercept bundle for an experimental persistent "
            "rank-one writer edit. Records the dropped probe intercept."
        ),
    ),
) -> None:
    """Build a portable runtime or model-edit intervention artifact."""

    project_root = _resolve_project_root(project_root)
    resolved_probe = _project_path(project_root, probe_path)
    resolved_output = _project_path(project_root, output_path)
    direction = load_probe_direction(resolved_probe, layer=layer, task=task)
    if homogeneous_writer_edit:
        if max_score_delta is not None:
            raise ValueError(
                "A homogeneous writer edit cannot preserve max_score_delta"
            )
        direction = replace(
            direction,
            intercept=0.0,
            original_intercept=direction.intercept,
        )
        token_scope = TokenScope.ALL
    spec = InterventionSpec(
        method=method,
        layers=tuple(intervention_layers or [layer]),
        token_scope=token_scope,
        direction_mode=(
            DirectionMode.MATCHED_RANDOM
            if matched_random_seed is not None
            else DirectionMode.PROBE
        ),
        random_seed=matched_random_seed,
        score_delta=score_delta,
        projection_target=projection_target,
        reflection_strength=reflection_strength,
        selected_side=selected_side,
        remap_input_min=remap_input_min,
        remap_input_max=remap_input_max,
        remap_output_min=remap_output_min,
        remap_output_max=remap_output_max,
        margin=margin,
        max_score_delta=max_score_delta,
    )
    validate_intervention_spec(spec)
    bundle = InterventionBundle(direction=direction, spec=spec)
    if homogeneous_writer_edit:
        validate_materializable_bundle(bundle)
    save_intervention_bundle(
        bundle,
        resolved_output,
        overwrite=overwrite,
    )
    console.print(
        "[green]Built intervention[/green] "
        f"method={method.value} layers={list(spec.layers)} path={resolved_output}"
    )


@app.command("build-intervention-suite")
def build_intervention_suite(
    probe_path: Path = typer.Option(..., "--probe"),
    output_dir: Path = typer.Option(..., "--output", "-o"),
    layer: int = typer.Option(..., "--layer", min=0),
    intervention_layers: list[int] | None = typer.Option(None, "--intervention-layer"),
    task: str = typer.Option("general_domain"),
    random_seed: int = typer.Option(0),
    project_root: Path | None = typer.Option(None),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Build the standard intervention matrix and one matched-random control."""

    project_root = _resolve_project_root(project_root)
    direction = load_probe_direction(
        _project_path(project_root, probe_path), layer=layer, task=task
    )
    layers = tuple(intervention_layers or [layer])
    specs = {
        "scalar_add_deceptive": InterventionSpec(
            method=InterventionMethod.SCALAR_ADDITION,
            layers=layers,
            score_delta=2.0,
        ),
        "affine_project_deceptive": InterventionSpec(
            method=InterventionMethod.AFFINE_PROJECTION,
            layers=layers,
            projection_target=1.0,
        ),
        "full_reflection": InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=layers,
        ),
        "partial_reflection": InterventionSpec(
            method=InterventionMethod.PARTIAL_REFLECTION,
            layers=layers,
            reflection_strength=0.75,
        ),
        "one_sided_reflection": InterventionSpec(
            method=InterventionMethod.ONE_SIDED_REFLECTION,
            layers=layers,
            selected_side="honest",
        ),
        "bounded_inversion": InterventionSpec(
            method=InterventionMethod.BOUNDED_REMAP,
            layers=layers,
            remap_input_min=-2.0,
            remap_input_max=2.0,
            remap_output_min=2.0,
            remap_output_max=-2.0,
            max_score_delta=4.0,
        ),
        "bounded_deceptive_margin": InterventionSpec(
            method=InterventionMethod.BOUNDED_MARGIN_CLAMP,
            layers=layers,
            selected_side="deceptive",
            margin=1.0,
            max_score_delta=2.0,
        ),
        "matched_random_full_reflection": InterventionSpec(
            method=InterventionMethod.FULL_REFLECTION,
            layers=layers,
            direction_mode=DirectionMode.MATCHED_RANDOM,
            random_seed=random_seed,
        ),
    }
    resolved_output = _project_path(project_root, output_dir)
    written: list[Path] = []
    for name, spec in specs.items():
        path = resolved_output / f"{name}.json"
        save_intervention_bundle(
            InterventionBundle(direction=direction, spec=spec),
            path,
            overwrite=overwrite,
        )
        written.append(path)
    console.print(
        "[green]Built intervention suite[/green] "
        f"variants={len(written)} path={resolved_output}"
    )


@app.command("materialize-writer-edit")
def materialize_writer_edit(
    intervention_path: Path = typer.Option(
        ...,
        "--intervention",
        "-i",
        help="Intervention JSON produced by build-intervention.",
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="New Hugging Face model directory to create."
    ),
    cache_dir: Path | None = typer.Option(
        None, help="Optional Hugging Face cache directory."
    ),
    project_root: Path | None = typer.Option(None),
) -> None:
    """Apply an experimental rank-one writer edit to a new Qwen checkpoint."""

    project_root = _resolve_project_root(project_root)
    resolved_intervention = _project_path(project_root, intervention_path)
    resolved_output = _project_path(project_root, output_dir)
    bundle = load_intervention_bundle(resolved_intervention)
    validate_materializable_bundle(bundle)
    if resolved_output.exists():
        raise FileExistsError(
            f"Output model directory already exists: {resolved_output}"
        )
    model_bundle = load_model_and_processor(
        model_config_from_env(
            cache_dir=str(_project_path(project_root, cache_dir))
            if cache_dir is not None
            else None
        )
    )
    edited = save_writer_edited_model(
        bundle=bundle,
        model_bundle=model_bundle,
        output_dir=resolved_output,
    )
    console.print(
        "[green]Materialized writer edit[/green] "
        f"edited={len(edited)} path={resolved_output}"
    )
