from __future__ import annotations

from pathlib import Path

import typer

from intelligent_liars.insider_trading import LabelMode
from intelligent_liars.judging import (
    DEFAULT_INSIDER_GRADING_FLUSH_EVERY,
    DEFAULT_JUDGE_CONFIG,
)
from intelligent_liars.rollouts import DEFAULT_INSIDER_GENERATION_SETTINGS, MODEL_SLUG

from intelligent_liars.cli_common import (
    app,
    console,
    _bind_judge_config_to_project_root,
    _generation_settings,
    _project_path,
    _resolve_project_root,
    _validate_insider_prompt_glob,
    _judge_config_from_options,
)


def _cli():
    from intelligent_liars import cli

    return cli


@app.command("generate-insider-trading")
def generate_insider_trading(
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/insider_trading/prompts. Defaults to the nearest checkout root.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Output JSON path. Defaults to data/insider_trading/{model_slug}-generations.json.",
    ),
    prompt_glob: str = typer.Option(
        "data/insider_trading/prompts/**/*.yaml",
        help="Glob, relative to project root, for insider-trading prompt YAML files.",
    ),
    samples_per_prompt: int = typer.Option(
        1, min=1, help="Number of sampled transcripts per prompt file."
    ),
    batch_size: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.batch_size,
        min=1,
        help="Generation batch size. Insider flow uses sequential turns.",
    ),
    max_new_tokens: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.max_new_tokens,
        min=1,
        help="Maximum new tokens per assistant turn.",
    ),
    do_sample: bool = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.do_sample,
        help="Use sampling. Recommended for multiple transcripts per prompt.",
    ),
    temperature: float = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.temperature,
        min=0.0,
        help="Sampling temperature.",
    ),
    top_p: float = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.top_p,
        min=0.0,
        max=1.0,
        help="Sampling nucleus top-p. Used only with --do-sample.",
    ),
    top_k: int | None = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.top_k,
        min=0,
        help="Sampling top-k. Used only with --do-sample.",
    ),
    seed: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.seed,
        help="Random seed for reproducible sampling.",
    ),
    flush_every: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.flush_every,
        min=1,
        help="Write JSON after this many new transcripts.",
    ),
    label_mode: str = typer.Option(
        "unknown",
        help='Use "unknown" for ungraded data or "heuristic" only for smoke/debug labels.',
    ),
    overwrite: bool = typer.Option(
        False, help="Replace an existing output file instead of resuming."
    ),
) -> None:
    """Generate Qwen insider-trading transcripts from local prompt YAML files.

    Args:
        project_root: Repository root containing insider-trading prompt YAMLs.
        output_path: Optional transcript JSON output path.
        prompt_glob: Project-relative glob for YAML prompt configs.
        samples_per_prompt: Number of transcript samples per YAML config.
        batch_size: Generation batch size; insider flow mostly generates one
            sequential assistant turn at a time.
        max_new_tokens: Maximum tokens for each assistant turn.
        do_sample: Enable stochastic decoding. Defaults true for transcript
            diversity.
        temperature: Sampling temperature used only with `do_sample`.
        top_p: Nucleus sampling cutoff used only with `do_sample`.
        top_k: Top-k sampling cutoff used only with `do_sample`.
        seed: Random seed for sampling.
        flush_every: Number of new transcripts between JSON checkpoints.
        label_mode: `unknown` for ungraded data or `heuristic` for debug labels.
        overwrite: Replace existing transcript file instead of resuming.

    References:
        Prompt configs live under `data/insider_trading/prompts/**/*.yaml`.
        Generation implementation lives in `insider_trading.py`.
    """

    if label_mode == "unknown":
        validated_label_mode: LabelMode = "unknown"
    elif label_mode == "heuristic":
        validated_label_mode = "heuristic"
    else:
        raise typer.BadParameter('label-mode must be "unknown" or "heuristic".')

    _cli().load_dotenv()
    project_root = _resolve_project_root(project_root)
    _validate_insider_prompt_glob(project_root, prompt_glob)
    settings = _generation_settings(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        flush_every=flush_every,
    )
    _cli().seed_everything(seed)

    model_config = _cli().model_config_from_env()
    model_id = _cli().resolve_model_id(model_config.model_name)
    model_slug = _cli().qwen_model_slug(model_id)
    output_path = _project_path(
        project_root,
        output_path
        or Path("data") / "insider_trading" / f"{model_slug}-generations.json",
    )
    _cli().validate_insider_trading_generation_resume(
        project_root=project_root,
        output_path=output_path,
        prompt_glob=prompt_glob,
        model_id=model_id,
        settings=settings,
        samples_per_prompt=samples_per_prompt,
        label_mode=validated_label_mode,
        overwrite=overwrite,
    )
    bundle = _cli().load_model_and_processor(model_config)

    summary = _cli().generate_insider_trading_transcripts(
        bundle=bundle,
        project_root=project_root,
        output_path=output_path,
        prompt_glob=prompt_glob,
        settings=settings,
        samples_per_prompt=samples_per_prompt,
        label_mode=validated_label_mode,
        overwrite=overwrite,
    )
    console.print(
        "[green]Generated insider-trading transcripts[/green] "
        f"new={summary.generated_examples} "
        f"skipped={summary.skipped_existing} "
        f"path={summary.output_path}"
    )


@app.command("grade-insider-trading")
def grade_insider_trading(
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/insider_trading grading prompts. Defaults to the nearest checkout root.",
    ),
    input_path: Path | None = typer.Option(
        None,
        help="Transcript JSON path. Defaults to data/insider_trading/qwen3-vl-8b-thinking-generations.json.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Output path. Defaults to in-place grading of input-path.",
    ),
    provider: str = typer.Option(
        DEFAULT_JUDGE_CONFIG.provider,
        help='Judge transport. Currently only "openrouter" is supported.',
    ),
    judge_model: str | None = typer.Option(
        DEFAULT_JUDGE_CONFIG.model,
        help="OpenRouter model alias from model_deployments.yaml or raw OpenRouter model ID.",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_JUDGE_CONFIG.max_tokens,
        min=1,
        help="Maximum judge output tokens.",
    ),
    max_retries: int = typer.Option(
        DEFAULT_JUDGE_CONFIG.max_retries,
        min=1,
        help="Judge API/parse retries.",
    ),
    timeout: float = typer.Option(
        DEFAULT_JUDGE_CONFIG.timeout,
        min=1.0,
        help="OpenRouter request timeout in seconds.",
    ),
    structured_outputs: bool = typer.Option(
        DEFAULT_JUDGE_CONFIG.structured_outputs,
        help="Request JSON schema outputs for LLM judges.",
    ),
    require_structured_outputs: bool = typer.Option(
        DEFAULT_JUDGE_CONFIG.require_structured_outputs,
        help="Require routed OpenRouter providers to support structured output parameters.",
    ),
    start: int = typer.Option(0, min=0, help="Start transcript index."),
    limit: int | None = typer.Option(
        None, min=1, help="Limit transcript records to grade."
    ),
    flush_every: int = typer.Option(
        DEFAULT_INSIDER_GRADING_FLUSH_EVERY,
        min=1,
        help="Write JSON after this many graded transcripts.",
    ),
    overwrite: bool = typer.Option(
        False, help="Re-grade transcripts that already have labels."
    ),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling an API.",
    ),
) -> None:
    """Grade generated insider-trading transcripts with task-specific judges.

    Args:
        project_root: Repository root containing insider-trading grading prompts.
        input_path: Transcript JSON to grade. Defaults to the Qwen generation
            output path.
        output_path: Optional separate graded output path. If omitted, grading
            is in-place.
        provider: Judge provider; currently only `openrouter`.
        judge_model: OpenRouter judge alias or raw model id.
        max_tokens: Maximum judge output tokens.
        max_retries: Judge retry count.
        timeout: Judge request timeout in seconds.
        structured_outputs: Request JSON-schema judge output where supported.
        require_structured_outputs: Require routed providers to support schema
            output parameters.
        start: First transcript index to grade.
        limit: Optional number of transcripts to grade.
        flush_every: Number of graded transcripts between writes.
        overwrite: Regrade transcripts that already have labels.
        mock_response: Literal judge response for tests/smoke runs.

    References:
        Judge prompt files live in `data/insider_trading/`.
    """

    _cli().load_dotenv()
    project_root = _resolve_project_root(project_root)
    input_path = _project_path(
        project_root,
        input_path
        or Path("data") / "insider_trading" / f"{MODEL_SLUG}-generations.json",
    )
    output_path = (
        _project_path(project_root, output_path) if output_path is not None else None
    )

    judge_config = _judge_config_from_options(
        provider=provider,
        judge_model=judge_model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        structured_outputs=structured_outputs,
        require_structured_outputs=require_structured_outputs,
        mock_response=mock_response,
    )
    judge_config = _bind_judge_config_to_project_root(judge_config, project_root)
    _cli().preflight_judge_config(judge_config)
    summary = _cli().grade_insider_trading_transcripts_with_judge(
        input_path,
        project_root=project_root,
        config=judge_config,
        output_path=output_path,
        overwrite=overwrite,
        start=start,
        limit=limit,
        flush_every=flush_every,
    )
    console.print(
        "[green]Graded insider-trading transcripts[/green] "
        f"graded={summary.graded_items} "
        f"skipped={summary.skipped_items} "
        f"path={summary.output_path}"
    )


@app.command("run-insider-trading-sweep")
def run_insider_trading_sweep(
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/insider_trading. Defaults to the nearest checkout root.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Transcript JSON path. Defaults to data/insider_trading/{model_slug}-generations.json.",
    ),
    prompt_glob: str = typer.Option(
        "data/insider_trading/prompts/**/*.yaml",
        help="Glob, relative to project root, for insider-trading prompt YAML files.",
    ),
    samples_per_prompt: int = typer.Option(
        1, min=1, help="Number of sampled transcripts per prompt file."
    ),
    batch_size: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.batch_size, min=1
    ),
    max_new_tokens: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.max_new_tokens, min=1
    ),
    do_sample: bool = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.do_sample),
    temperature: float = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.temperature, min=0.0
    ),
    top_p: float = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.top_p, min=0.0, max=1.0
    ),
    top_k: int | None = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.top_k, min=0),
    seed: int = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.seed),
    generation_flush_every: int = typer.Option(
        DEFAULT_INSIDER_GENERATION_SETTINGS.flush_every, min=1
    ),
    grading_start: int = typer.Option(
        0, min=0, help="Start transcript index for grading."
    ),
    grading_limit: int | None = typer.Option(
        None, min=1, help="Limit transcripts to grade after generation."
    ),
    overwrite_generation: bool = typer.Option(
        False, help="Regenerate transcript file instead of resuming."
    ),
    overwrite_grading: bool = typer.Option(
        False, help="Re-grade already populated transcript labels."
    ),
    provider: str = typer.Option(DEFAULT_JUDGE_CONFIG.provider),
    judge_model: str | None = typer.Option(DEFAULT_JUDGE_CONFIG.model),
    max_tokens: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_tokens, min=1),
    max_retries: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_retries, min=1),
    timeout: float = typer.Option(DEFAULT_JUDGE_CONFIG.timeout, min=1.0),
    structured_outputs: bool = typer.Option(DEFAULT_JUDGE_CONFIG.structured_outputs),
    require_structured_outputs: bool = typer.Option(
        DEFAULT_JUDGE_CONFIG.require_structured_outputs
    ),
    grading_flush_every: int = typer.Option(DEFAULT_INSIDER_GRADING_FLUSH_EVERY, min=1),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling OpenRouter.",
    ),
) -> None:
    """Generate Qwen insider-trading transcripts and judge them in one sweep."""

    _cli().load_dotenv()
    project_root = _resolve_project_root(project_root)
    _validate_insider_prompt_glob(project_root, prompt_glob)
    settings = _generation_settings(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        flush_every=generation_flush_every,
    )
    judge_config = _judge_config_from_options(
        provider=provider,
        judge_model=judge_model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        structured_outputs=structured_outputs,
        require_structured_outputs=require_structured_outputs,
        mock_response=mock_response,
    )
    judge_config = _bind_judge_config_to_project_root(judge_config, project_root)
    preflight = _cli().preflight_judge_config(judge_config)
    console.print(
        "[green]OpenRouter judge preflight passed[/green] "
        f"alias={preflight.alias} resolved_model={preflight.resolved_model}"
    )

    _cli().seed_everything(seed)
    model_config = _cli().model_config_from_env()
    model_id = _cli().resolve_model_id(model_config.model_name)
    model_slug = _cli().qwen_model_slug(model_id)
    output_path = _project_path(
        project_root,
        output_path
        or Path("data") / "insider_trading" / f"{model_slug}-generations.json",
    )
    _cli().validate_insider_trading_generation_resume(
        project_root=project_root,
        output_path=output_path,
        prompt_glob=prompt_glob,
        model_id=model_id,
        settings=settings,
        samples_per_prompt=samples_per_prompt,
        label_mode="unknown",
        overwrite=overwrite_generation,
    )
    bundle = _cli().load_model_and_processor(model_config)

    generation_summary = _cli().generate_insider_trading_transcripts(
        bundle=bundle,
        project_root=project_root,
        output_path=output_path,
        prompt_glob=prompt_glob,
        settings=settings,
        samples_per_prompt=samples_per_prompt,
        label_mode="unknown",
        overwrite=overwrite_generation,
    )
    console.print(
        "[green]Generated insider-trading transcripts[/green] "
        f"new={generation_summary.generated_examples} "
        f"skipped={generation_summary.skipped_existing} "
        f"path={generation_summary.output_path}"
    )

    grading_summary = _cli().grade_insider_trading_transcripts_with_judge(
        output_path,
        project_root=project_root,
        config=judge_config,
        overwrite=overwrite_grading,
        start=grading_start,
        limit=grading_limit,
        flush_every=grading_flush_every,
    )
    console.print(
        "[green]Graded insider-trading transcripts[/green] "
        f"graded={grading_summary.graded_items} "
        f"skipped={grading_summary.skipped_items} "
        f"path={grading_summary.output_path}"
    )
