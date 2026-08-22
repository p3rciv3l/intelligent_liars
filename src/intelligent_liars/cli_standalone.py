from __future__ import annotations

import gc
import re
from dataclasses import replace
from pathlib import Path

import typer

from intelligent_liars.cli_common import (
    app,
    console,
    _project_path,
    _resolve_project_root,
)
from intelligent_liars.models import load_model_and_processor, model_config_from_env
from intelligent_liars.models import DEFAULT_MODEL_ID
from intelligent_liars.rollouts import (
    DEFAULT_ROLLOUT_GENERATION_SETTINGS,
    GenerationSettings,
)
from intelligent_liars.standalone_models import (
    DEFAULT_TINYLORA_TARGETS,
    TinyLoRATrainingConfig,
    assert_fleet_claim_owned,
    claim_fleet_variant,
    create_fleet_plan,
    finish_fleet_claim,
    fleet_status,
    generate_teacher_dataset,
    load_fleet_plan,
    recover_running_fleet_variants,
    record_base_control,
    save_fleet_plan,
    train_standalone_model,
    verify_standalone_model,
)


def _generation(
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int,
) -> GenerationSettings:
    return GenerationSettings(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        flush_every=1,
    )


def _training(
    *,
    svd_rank: int,
    projection_dim: int,
    projection_seed: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    seed: int,
    preservation_weight: float,
    target_modules: list[str] | None,
    train_layers: list[int] | None,
) -> TinyLoRATrainingConfig:
    return TinyLoRATrainingConfig(
        svd_rank=svd_rank,
        projection_dim=projection_dim,
        projection_seed=projection_seed,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_length=max_length,
        seed=seed,
        preservation_weight=preservation_weight,
        target_modules=tuple(target_modules or DEFAULT_TINYLORA_TARGETS),
        train_layers=tuple(train_layers) if train_layers else None,
    )


@app.command("generate-intervention-teacher")
def generate_intervention_teacher(
    prompt_path: Path = typer.Option(..., "--prompts", help="JSON prompt records."),
    output_path: Path = typer.Option(..., "--output", "-o"),
    intervention_path: Path | None = typer.Option(
        None,
        "--intervention",
        help="Intervention bundle. Omit to generate an unmodified preservation teacher.",
    ),
    project_root: Path | None = typer.Option(None),
    cache_dir: Path | None = typer.Option(None),
    batch_size: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.batch_size, min=1
    ),
    max_new_tokens: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.max_new_tokens, min=1
    ),
    do_sample: bool = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.do_sample),
    temperature: float = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.temperature),
    top_p: float = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_p),
    top_k: int | None = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_k),
    seed: int = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.seed),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    base_revision: str | None = typer.Option(None, "--base-revision"),
) -> None:
    """Generate durable teacher completions with an intervention applied."""

    project_root = _resolve_project_root(project_root)
    model_bundle = load_model_and_processor(
        replace(
            model_config_from_env(
                cache_dir=(
                    str(_project_path(project_root, cache_dir))
                    if cache_dir is not None
                    else None
                )
            ),
            revision=base_revision,
        )
    )
    summary = generate_teacher_dataset(
        model_bundle=model_bundle,
        prompt_path=_project_path(project_root, prompt_path),
        output_path=_project_path(project_root, output_path),
        intervention_path=(
            _project_path(project_root, intervention_path)
            if intervention_path is not None
            else None
        ),
        generation=_generation(
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        ),
        resume=resume,
    )
    console.print(
        "[green]Generated intervention teacher[/green] "
        f"records={summary.records} path={summary.output_path}"
    )


@app.command("distill-intervention-model")
def distill_intervention_model(
    teacher_path: Path = typer.Option(..., "--teacher"),
    output_dir: Path = typer.Option(..., "--output", "-o"),
    preservation_teacher_path: Path = typer.Option(..., "--preservation-teacher"),
    project_root: Path | None = typer.Option(None),
    cache_dir: Path | None = typer.Option(None),
    tinylora_basis_path: Path | None = typer.Option(None, "--tinylora-basis"),
    svd_rank: int = typer.Option(2, "--svd-rank", min=1),
    projection_dim: int = typer.Option(13, "--projection-dim", min=1),
    projection_seed: int = typer.Option(42),
    dropout: float = typer.Option(0.0, min=0.0, max=0.999999),
    learning_rate: float = typer.Option(2e-4, min=1e-12),
    epochs: int = typer.Option(1, min=1),
    batch_size: int = typer.Option(1, min=1),
    gradient_accumulation_steps: int = typer.Option(8, min=1),
    max_length: int = typer.Option(4096, min=1),
    seed: int = typer.Option(0),
    preservation_weight: float = typer.Option(1.0, min=1e-12),
    target_modules: list[str] | None = typer.Option(None, "--target-module"),
    train_layers: list[int] | None = typer.Option(None, "--train-layer", min=0),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    base_revision: str | None = typer.Option(None, "--base-revision"),
) -> None:
    """Distill into TinyLoRA, merge it, and save stock Qwen weights."""

    project_root = _resolve_project_root(project_root)
    load_config = replace(
        model_config_from_env(
            cache_dir=(
                str(_project_path(project_root, cache_dir))
                if cache_dir is not None
                else None
            )
        ),
        attention_implementation="sdpa",
        device_map={"": 0},
        revision=base_revision,
    )
    model_bundle = load_model_and_processor(load_config)
    summary = train_standalone_model(
        model_bundle=model_bundle,
        teacher_path=_project_path(project_root, teacher_path),
        preservation_teacher_path=_project_path(
            project_root, preservation_teacher_path
        ),
        output_dir=_project_path(project_root, output_dir),
        config=_training(
            svd_rank=svd_rank,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            seed=seed,
            preservation_weight=preservation_weight,
            target_modules=target_modules,
            train_layers=train_layers,
        ),
        resume=resume,
        tinylora_basis_path=(
            _project_path(project_root, tinylora_basis_path)
            if tinylora_basis_path is not None
            else None
        ),
    )
    console.print(
        "[green]Created standalone intervention model[/green] "
        f"steps={summary.optimizer_steps} modules={len(summary.merged_modules)} "
        f"trainable_scalars={summary.trainable_scalars} "
        f"path={summary.output_dir}"
    )


@app.command("plan-intervention-model-fleet")
def plan_intervention_model_fleet(
    intervention_paths: list[Path] = typer.Option(..., "--intervention"),
    prompt_path: Path = typer.Option(..., "--prompts"),
    output_root: Path = typer.Option(..., "--output-root"),
    plan_path: Path = typer.Option(..., "--plan", "-o"),
    preservation_teacher_path: Path = typer.Option(..., "--preservation-teacher"),
    project_root: Path | None = typer.Option(None),
    generation_batch_size: int = typer.Option(1, min=1),
    max_new_tokens: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.max_new_tokens, min=1
    ),
    do_sample: bool = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.do_sample),
    temperature: float = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.temperature),
    top_p: float = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_p),
    top_k: int | None = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_k),
    seed: int = typer.Option(0),
    svd_rank: int = typer.Option(2, "--svd-rank", min=1),
    projection_dim: int = typer.Option(13, "--projection-dim", min=1),
    projection_seed: int = typer.Option(42),
    dropout: float = typer.Option(0.0, min=0.0, max=0.999999),
    learning_rate: float = typer.Option(2e-4, min=1e-12),
    epochs: int = typer.Option(1, min=1),
    train_batch_size: int = typer.Option(1, min=1),
    gradient_accumulation_steps: int = typer.Option(8, min=1),
    max_length: int = typer.Option(4096, min=1),
    preservation_weight: float = typer.Option(1.0, min=1e-12),
    target_modules: list[str] | None = typer.Option(None, "--target-module"),
    train_layers: list[int] | None = typer.Option(None, "--train-layer", min=0),
    overwrite: bool = typer.Option(False, "--overwrite"),
    base_revision: str | None = typer.Option(None, "--base-revision"),
) -> None:
    """Write an immutable plan for one standalone model per intervention bundle."""

    project_root = _resolve_project_root(project_root)
    if base_revision is None or re.fullmatch(r"[0-9a-f]{40}", base_revision) is None:
        from huggingface_hub import HfApi

        base_revision = (
            HfApi()
            .model_info(
                DEFAULT_MODEL_ID,
                revision=base_revision,
            )
            .sha
        )
    if not base_revision:
        raise ValueError("Could not resolve an immutable Qwen base revision")
    plan = create_fleet_plan(
        intervention_paths=[
            _project_path(project_root, path) for path in intervention_paths
        ],
        prompt_path=_project_path(project_root, prompt_path),
        preservation_teacher_path=(
            _project_path(project_root, preservation_teacher_path)
            if preservation_teacher_path is not None
            else None
        ),
        output_root=_project_path(project_root, output_root),
        generation=_generation(
            batch_size=generation_batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        ),
        training=_training(
            svd_rank=svd_rank,
            projection_dim=projection_dim,
            projection_seed=projection_seed,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_length=max_length,
            seed=seed,
            preservation_weight=preservation_weight,
            target_modules=target_modules,
            train_layers=train_layers,
        ),
        base_revision=base_revision,
    )
    resolved_plan = _project_path(project_root, plan_path)
    save_fleet_plan(plan, resolved_plan, overwrite=overwrite)
    console.print(
        "[green]Planned standalone model fleet[/green] "
        f"variants={len(plan.variants)} path={resolved_plan}"
    )


@app.command("run-intervention-model-job")
def run_intervention_model_job(
    plan_path: Path = typer.Option(..., "--plan"),
    variant_name: str | None = typer.Option(None, "--variant"),
    project_root: Path | None = typer.Option(None),
    cache_dir: Path | None = typer.Option(None),
    drain: bool = typer.Option(False, "--drain"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    recover_running: bool = typer.Option(False, "--recover-running"),
    max_attempts: int = typer.Option(2, "--max-attempts", min=1),
) -> None:
    """Atomically claim and run fleet variants from a shared output directory."""

    project_root = _resolve_project_root(project_root)
    plan = load_fleet_plan(_project_path(project_root, plan_path))
    if recover_running:
        recover_running_fleet_variants(plan)
        retry_failed = True
    cache = (
        str(_project_path(project_root, cache_dir)) if cache_dir is not None else None
    )
    requested_name = variant_name
    completed = 0
    while True:
        claim = claim_fleet_variant(
            plan,
            name=requested_name,
            retry_failed=retry_failed,
            max_attempts=max_attempts,
        )
        if claim is None:
            break
        try:
            artifacts = _run_intervention_variant(
                plan=plan,
                claim=claim,
                cache=cache,
            )
        except BaseException as error:
            finish_fleet_claim(plan, claim, error=error)
            if not drain:
                raise
            requested_name = None
            retry_failed = True
            continue
        finish_fleet_claim(plan, claim, artifacts=artifacts)
        completed += 1
        if not drain:
            break
        requested_name = None
    console.print(f"[green]Completed intervention model jobs[/green] count={completed}")


def _run_intervention_variant(
    *,
    plan,
    claim,
    cache: str | None,
) -> dict[str, str]:
    variant = claim.variant
    teacher_output = (
        Path(variant.teacher_output).parent / variant.name / f"{claim.claim_token}.json"
    )
    model_output = Path(variant.model_output) / "attempts" / claim.claim_token
    teacher_bundle = load_model_and_processor(
        replace(
            model_config_from_env(cache_dir=cache),
            revision=plan.base_revision,
        )
    )
    record_base_control(plan, teacher_bundle)
    generate_teacher_dataset(
        model_bundle=teacher_bundle,
        prompt_path=Path(plan.prompt_path),
        output_path=teacher_output,
        intervention_path=Path(variant.intervention_path),
        generation=plan.generation,
        resume=True,
    )
    del teacher_bundle
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    training_bundle = load_model_and_processor(
        replace(
            model_config_from_env(cache_dir=cache),
            attention_implementation="sdpa",
            device_map={"": 0},
            revision=plan.base_revision,
        )
    )
    train_standalone_model(
        model_bundle=training_bundle,
        teacher_path=teacher_output,
        preservation_teacher_path=Path(plan.preservation_teacher_path),
        output_dir=model_output,
        config=plan.training,
        resume=True,
        fleet_plan_id=plan.plan_id,
        tinylora_basis_path=(Path(plan.output_root) / "controls" / "tinylora_basis.pt"),
    )
    del training_bundle
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    verify_standalone_model(model_dir=model_output)
    assert_fleet_claim_owned(plan, claim)
    return {
        "teacher_output": str(teacher_output.resolve()),
        "model_output": str(model_output.resolve()),
    }


@app.command("intervention-model-fleet-status")
def intervention_model_fleet_status(
    plan_path: Path = typer.Option(..., "--plan"),
    project_root: Path | None = typer.Option(None),
) -> None:
    """Print pending, running, failed, and completed standalone variants."""

    project_root = _resolve_project_root(project_root)
    plan = load_fleet_plan(_project_path(project_root, plan_path))
    console.print_json(data=fleet_status(plan))


@app.command("verify-standalone-intervention-model")
def verify_standalone_intervention_model(
    model_dir: Path = typer.Option(..., "--model"),
    prompt: str = typer.Option("Answer with exactly one word: ready"),
    project_root: Path | None = typer.Option(None),
) -> None:
    """Reload a merged checkpoint with stock Qwen and run a greedy smoke test."""

    project_root = _resolve_project_root(project_root)
    result = verify_standalone_model(
        model_dir=_project_path(project_root, model_dir),
        prompt=prompt,
    )
    console.print(
        "[green]Verified standalone intervention model[/green] "
        f"completion={result['completion']!r}"
    )
