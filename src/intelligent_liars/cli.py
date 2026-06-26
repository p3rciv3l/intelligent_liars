from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from intelligent_liars.activations import (
    ActivationDataset,
    ActivationExtractionSettings,
    default_activation_output_path,
    extract_dataset_activations,
    extract_rollout_activations,
    merge_activation_hdf5_shards,
    parse_layer_spec,
)
from intelligent_liars.activation_backends import ActivationBackend, TransformersHookBackend
from intelligent_liars.environment import check_environment
from intelligent_liars.insider_trading import (
    LabelMode,
    generate_insider_trading_transcripts,
    validate_insider_trading_generation_resume,
)
from intelligent_liars.judging import (
    DEFAULT_INSIDER_GRADING_FLUSH_EVERY,
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY,
    DEFAULT_ROLLOUT_GRADING_MAX_WORKERS,
    JudgeConfig,
    grade_insider_trading_transcripts as grade_insider_trading_transcripts_with_judge,
    grade_rollout_file,
    infer_rollout_task,
    preflight_judge_config,
)
from intelligent_liars.models import (
    ModelBundle,
    load_model_and_processor,
    model_config_from_env,
    resolve_model_id,
)
from intelligent_liars.nnsight_backend import (
    NnsightActivationBackend,
    load_nnsight_bundle,
)
from intelligent_liars.probes import preflight_probe_training, train_probe_directions
from intelligent_liars.rollouts import (
    DEFAULT_INSIDER_GENERATION_SETTINGS,
    DEFAULT_ROLLOUT_PROMPT_SET,
    DEFAULT_ROLLOUT_TASKS,
    DEFAULT_ROLLOUT_GENERATION_SETTINGS,
    GenerationSettings,
    MODEL_SLUG,
    generate_rollout_task,
    load_rollout_prompt_examples,
    merge_rollout_files,
    parse_task_name,
    pending_rollout_source_indices,
    qwen_model_slug,
    seed_everything,
)
from intelligent_liars.sycophancy import (
    STEM_TASKS,
    generate_sycophancy_dataset,
    pair_sycophancy_dataset,
)


app = typer.Typer(no_args_is_help=True)
console = Console()


def _judge_config_from_options(
    *,
    provider: str,
    judge_model: str | None,
    max_tokens: int,
    max_retries: int,
    timeout: float,
    structured_outputs: bool,
    require_structured_outputs: bool,
    mock_response: str | None,
) -> JudgeConfig:
    """Build a `JudgeConfig` from Typer option values.

    Args:
        provider: Judge transport name. The CLI currently supports only
            `openrouter`.
        judge_model: OpenRouter model alias or raw model id.
        max_tokens: Maximum generated tokens requested from the judge.
        max_retries: Retry count for transient judge/API failures.
        timeout: Request timeout in seconds.
        structured_outputs: Whether to request JSON-schema judge responses.
        require_structured_outputs: Whether OpenRouter routing must support the
            structured output parameters.
        mock_response: Optional literal judge response for tests/smoke checks.

    Returns:
        Judge configuration consumed by `judging.py`.

    Raises:
        typer.BadParameter: If a non-supported provider is requested.
    """

    if provider != "openrouter":
        raise typer.BadParameter('provider must be "openrouter". Use --judge-model for OpenRouter aliases or raw model IDs.')
    return JudgeConfig(
        provider="openrouter",
        model=judge_model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        structured_outputs=structured_outputs,
        require_structured_outputs=require_structured_outputs,
        mock_response=mock_response,
    )


def _bind_judge_config_to_project_root(config: JudgeConfig, project_root: Path) -> JudgeConfig:
    return replace(config, model_config_path=project_root / "model_deployments.yaml")


def _project_path(project_root: Path, path: Path) -> Path:
    """Resolve a CLI path relative to the project root when needed.

    Args:
        project_root: Absolute repository root selected by `--project-root`.
        path: User-provided path. Absolute paths are left unchanged.

    Returns:
        Absolute path for local project-relative inputs/outputs.
    """

    return path if path.is_absolute() else project_root / path


def _find_project_root(start: Path | None = None) -> Path:
    """Find the nearest checkout root from a start directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "intelligent_liars").exists():
            return candidate
    return current


def _resolve_project_root(project_root: Path | None) -> Path:
    """Resolve an optional CLI project root with repo-root auto-detection."""

    if project_root is None:
        return _find_project_root()
    resolved = project_root.resolve()
    if (resolved / "pyproject.toml").exists():
        return resolved
    return _find_project_root(resolved)


def _generation_settings(
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int,
    flush_every: int,
) -> GenerationSettings:
    """Collect repeated generation CLI options into one settings object.

    Args:
        batch_size: Number of prompts per `model.generate` batch.
        max_new_tokens: Maximum assistant tokens per completion.
        do_sample: Whether to sample instead of greedy decoding.
        temperature: Sampling temperature used only with `do_sample`.
        top_p: Nucleus sampling cutoff used only with `do_sample`.
        top_k: Top-k sampling cutoff used only with `do_sample`.
        seed: Random seed used by `seed_everything`.
        flush_every: Number of new generations between JSON checkpoints.

    Returns:
        `GenerationSettings` passed to rollout and insider-trading generation.
    """

    return GenerationSettings(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        flush_every=flush_every,
    )


def _rollout_task_needs_llm_judge(task: str) -> bool:
    base, _ = parse_task_name(task)
    return base in {"roleplaying", "insider_trading", "insider_trading_doubledown"}


def _infer_rollout_task_from_path(path: Path) -> str:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Rollout JSON file must contain an object: {path}")
    return infer_rollout_task(path, data)


def _selected_source_indices(examples, *, start: int, limit: int | None) -> set[int]:
    selected = list(examples[start:])
    if limit is not None:
        selected = selected[:limit]
    return {int(example.source_index) for example in selected}


def _selected_example_count(examples, *, start: int, limit: int | None) -> int:
    selected = list(examples[start:])
    if limit is not None:
        selected = selected[:limit]
    return len(selected)


def _source_index_filter(
    *,
    source_indices: list[int] | None,
    source_start: int | None,
    source_limit: int | None,
    record_start: int,
    record_limit: int | None,
) -> set[int] | None:
    selected = set(source_indices or [])
    if any(source_index < 0 for source_index in selected):
        raise typer.BadParameter("--source-index values must be >= 0.")
    if source_start is not None or source_limit is not None:
        if source_limit is None:
            raise typer.BadParameter("--source-limit is required when using --source-start.")
        start = source_start or 0
        selected.update(range(start, start + source_limit))
    if selected and (record_start != 0 or record_limit is not None):
        raise typer.BadParameter("Use either record --start/--limit or source-index filters, not both.")
    return selected or None


def _validate_insider_prompt_glob(project_root: Path, prompt_glob: str) -> None:
    if not sorted(project_root.glob(prompt_glob)):
        raise typer.BadParameter(
            f"No insider-trading prompt YAML files matched {prompt_glob!r} under {project_root}"
        )


@app.command("check-env")
def check_env(
    require_cuda: bool = typer.Option(False, help="Fail if CUDA is unavailable."),
    check_processor: bool = typer.Option(
        False,
        help="Download/load only the hardcoded Qwen3-VL processor.",
    ),
    check_model: bool = typer.Option(
        False,
        help="Download/load the full model. Intended for the remote GPU box.",
    ),
    check_nnsight: bool = typer.Option(
        False,
        help="Load Qwen3-VL through NNsight and trace one text decoder layer.",
    ),
    check_openrouter: bool = typer.Option(
        False,
        help="Validate OpenRouter API key, judge alias resolution, and structured-output support.",
    ),
    nnsight_layer: int = typer.Option(
        0,
        help="Decoder layer index to trace for --check-nnsight.",
    ),
) -> None:
    """Print and optionally validate the local/remote Qwen runtime environment.

    Args:
        require_cuda: Fail if Torch reports no CUDA device.
        check_processor: Load only the Qwen processor/tokenizer.
        check_model: Load the full Qwen model through Transformers.
        check_nnsight: Load and trace one decoder layer through NNsight.
        nnsight_layer: Decoder layer index for the NNsight trace.

    References:
        Delegates implementation to `environment.check_environment`; this CLI
        wrapper exists to keep Typer option definitions in one file.
    """

    check_environment(
        console=console,
        require_cuda=require_cuda,
        check_processor=check_processor,
        check_model=check_model,
        check_nnsight=check_nnsight,
        check_openrouter=check_openrouter,
        nnsight_layer=nnsight_layer,
    )


@app.command("generate-rollouts")
def generate_rollouts(
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Truth Spec rollout task to generate. Repeatable. Defaults to roleplaying and sandbagging.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/. Defaults to the nearest checkout root.",
    ),
    output_dir: Path = typer.Option(
        Path("data/rollouts"),
        help="Directory for Truth Spec-compatible rollout JSON files.",
    ),
    prompt_set: str | None = typer.Option(
        DEFAULT_ROLLOUT_PROMPT_SET,
        "--prompt-set",
        help="Cleaned prompt set id. Use an explicit value when comparing prompt sources.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.batch_size,
        min=1,
        help="Generation batch size.",
    ),
    max_new_tokens: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.max_new_tokens,
        min=1,
        help="Maximum new tokens per completion.",
    ),
    do_sample: bool = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.do_sample,
        help="Use sampling instead of greedy decoding.",
    ),
    temperature: float = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.temperature,
        min=0.0,
        help="Sampling temperature.",
    ),
    top_p: float = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_p,
        min=0.0,
        max=1.0,
        help="Sampling nucleus top-p. Used only with --do-sample.",
    ),
    top_k: int | None = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_k,
        min=0,
        help="Sampling top-k. Used only with --do-sample.",
    ),
    seed: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.seed,
        help="Random seed for reproducible sampling.",
    ),
    start: int = typer.Option(0, min=0, help="Start source index."),
    limit: int | None = typer.Option(None, min=1, help="Limit examples per task."),
    flush_every: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.flush_every,
        min=1,
        help="Write JSON after this many new completions.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing output file instead of resuming."),
) -> None:
    """Generate Qwen rollouts and write resumable Truth Spec-compatible JSON.

    Args:
        tasks: Repeatable task ids. Defaults to `DEFAULT_ROLLOUT_TASKS`.
        project_root: Repository root containing `data/` and prompt sets.
        output_dir: Directory for generated rollout JSON files.
        prompt_set: Cleaned prompt-set id such as `truth-spec-llama-70b`.
        batch_size: Number of prompts per generation batch.
        max_new_tokens: Maximum tokens per generated assistant completion.
        do_sample: Enable stochastic decoding; otherwise generation is greedy.
        temperature: Sampling temperature used only with `do_sample`.
        top_p: Nucleus sampling cutoff used only with `do_sample`.
        top_k: Top-k sampling cutoff used only with `do_sample`.
        seed: Random seed used when sampling.
        start: Source prompt index offset.
        limit: Optional number of prompts per task after `start`.
        flush_every: Number of new completions between JSON checkpoints.
        overwrite: Replace existing rollout files instead of resuming.

    References:
        Inputs come from `data/roleplaying/dataset.yaml` or
        `data/rollout_prompts/*.json`. Outputs are written to
        `data/rollouts/{task}__{qwen_model_slug}.json` by default.
    """

    load_dotenv()
    project_root = _resolve_project_root(project_root)
    output_dir = _project_path(project_root, output_dir)
    task_list = tuple(tasks) if tasks else DEFAULT_ROLLOUT_TASKS

    # One settings object is stored in each rollout record for reproducibility.
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
    seed_everything(seed)

    examples_by_task = {
        task: load_rollout_prompt_examples(
            task,
            project_root=project_root,
            prompt_set=prompt_set,
        )
        for task in task_list
    }

    model_config = model_config_from_env()
    model_id = resolve_model_id(model_config.model_name)
    model_slug = qwen_model_slug(model_id)
    pending_by_task: dict[str, set[int]] = {}
    for task, examples in examples_by_task.items():
        output_path = output_dir / f"{task}__{model_slug}.json"
        pending_by_task[task] = pending_rollout_source_indices(
            task=task,
            output_path=output_path,
            model_id=model_id,
            examples=examples,
            settings=settings,
            overwrite=overwrite,
            start=start,
            limit=limit,
        )
    if not any(pending_by_task.values()):
        for task, examples in examples_by_task.items():
            output_path = output_dir / f"{task}__{model_slug}.json"
            selected_count = _selected_example_count(examples, start=start, limit=limit)
            console.print(
                "[green]Generated rollouts[/green] "
                f"task={task} new=0 skipped={selected_count} path={output_path}"
            )
        return
    bundle = load_model_and_processor(model_config)

    for task in task_list:
        # Fixed prompts are preloaded before model load; Qwen creates fresh completions.
        examples = examples_by_task[task]
        output_path = output_dir / f"{task}__{model_slug}.json"
        if not pending_by_task[task]:
            selected_count = _selected_example_count(examples, start=start, limit=limit)
            console.print(
                "[green]Generated rollouts[/green] "
                f"task={task} new=0 skipped={selected_count} path={output_path}"
            )
            continue
        summary = generate_rollout_task(
            bundle=bundle,
            task=task,
            examples=examples,
            output_path=output_path,
            settings=settings,
            overwrite=overwrite,
            start=start,
            limit=limit,
        )
        console.print(
            "[green]Generated rollouts[/green] "
            f"task={summary.task} "
            f"new={summary.generated_examples} "
            f"skipped={summary.skipped_existing} "
            f"path={summary.output_path}"
        )


@app.command("grade-rollouts")
def grade_rollouts(
    paths: list[Path] | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Rollout JSON path to grade. Repeatable. Defaults to generated Qwen roleplaying/sandbagging files.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/ grading templates. Defaults to the nearest checkout root.",
    ),
    provider: str = typer.Option(
        DEFAULT_JUDGE_CONFIG.provider,
        help='Judge transport. Currently only "openrouter" is supported.',
    ),
    judge_model: str | None = typer.Option(
        DEFAULT_JUDGE_CONFIG.model,
        help="OpenRouter model alias from model_deployments.yaml or raw OpenRouter model ID.",
    ),
    max_workers: int = typer.Option(
        DEFAULT_ROLLOUT_GRADING_MAX_WORKERS,
        min=1,
        help="Concurrent judge calls for LLM-graded rollouts.",
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
    start: int = typer.Option(0, min=0, help="Start rollout index."),
    limit: int | None = typer.Option(None, min=1, help="Limit rollout records to grade per file."),
    source_indices: list[int] | None = typer.Option(
        None,
        "--source-index",
        help="Specific metadata.source_index value to grade. Repeatable; cannot be combined with --start/--limit.",
    ),
    source_start: int | None = typer.Option(
        None,
        min=0,
        help="First metadata.source_index to grade. Requires --source-limit.",
    ),
    source_limit: int | None = typer.Option(
        None,
        min=1,
        help="Number of source indices to grade from --source-start.",
    ),
    flush_every: int = typer.Option(
        DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY,
        min=1,
        help="Write JSON after this many completed judge calls.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional directory for graded copies. Defaults to in-place grading.",
    ),
    overwrite: bool = typer.Option(False, help="Re-grade already populated labels."),
    continue_on_error: bool = typer.Option(
        False,
        help="Keep grading roleplaying files when one judge item fails; failed items are labelled skip.",
    ),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling an API.",
    ),
) -> None:
    """Grade generated Truth Spec-style rollout JSON files.

    Args:
        paths: Optional repeatable rollout paths. Defaults to generated Qwen
            files for `DEFAULT_ROLLOUT_TASKS`.
        project_root: Repository root containing grading templates.
        provider: Judge provider; currently only `openrouter`.
        judge_model: OpenRouter judge alias or raw model id.
        max_workers: Concurrent roleplaying judge calls.
        max_tokens: Maximum judge response tokens.
        max_retries: Judge retry count.
        timeout: Judge request timeout in seconds.
        structured_outputs: Request JSON-schema judge output where supported.
        require_structured_outputs: Require routed providers to support schema
            output parameters.
        start: First rollout index to grade.
        limit: Optional number of records to grade per file.
        source_indices: Explicit source example indices to grade.
        source_start: First source example index to grade.
        source_limit: Number of source example indices to grade.
        flush_every: Number of completed judge calls between writes.
        overwrite: Regrade records that already have labels.
        continue_on_error: Label failed roleplaying judge calls as skip instead
            of aborting the whole file.
        mock_response: Literal judge response for tests/smoke runs.

        References:
        Roleplaying grades use `data/roleplaying/grading_template.txt`.
        Sandbagging grades use the deterministic parser in `judging.py`.
        Insider-trading rollout grades use `data/insider_trading/` judge prompts.
    """

    load_dotenv()
    project_root = _resolve_project_root(project_root)
    output_dir = _project_path(project_root, output_dir) if output_dir is not None else None
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
    # Resolve defaults lazily after project_root is known.
    rollout_paths = [
        _project_path(project_root, path)
        for path in (
            paths
            or [Path("data") / "rollouts" / f"{task}__{MODEL_SLUG}.json" for task in DEFAULT_ROLLOUT_TASKS]
        )
    ]
    rollout_tasks = {path: _infer_rollout_task_from_path(path) for path in rollout_paths}
    if any(_rollout_task_needs_llm_judge(task) for task in rollout_tasks.values()):
        preflight_judge_config(judge_config)
    source_index_filter = _source_index_filter(
        source_indices=source_indices,
        source_start=source_start,
        source_limit=source_limit,
        record_start=start,
        record_limit=limit,
    )

    for path in rollout_paths:
        summary = grade_rollout_file(
            path,
            project_root=project_root,
            config=judge_config,
            task=rollout_tasks[path],
            output_path=(output_dir / path.name) if output_dir is not None else None,
            overwrite=overwrite,
            start=start,
            limit=limit,
            source_indices=source_index_filter,
            flush_every=flush_every,
            max_workers=max_workers,
            continue_on_error=continue_on_error,
        )
        console.print(
            "[green]Graded rollout file[/green] "
            f"task={summary.task} "
            f"graded={summary.graded_items} "
            f"skipped={summary.skipped_items} "
            f"path={summary.output_path}"
        )


@app.command("merge-rollouts")
def merge_rollouts(
    paths: list[Path] = typer.Argument(..., help="Shard rollout JSON files to merge."),
    output_path: Path = typer.Option(..., "--output-path", "-o", help="Final rollout JSON path."),
    no_existing_output: bool = typer.Option(
        False,
        "--no-existing-output",
        help="Do not merge an existing output path before writing the merged file.",
    ),
) -> None:
    """Merge sharded rollout JSON files into one normal rollout file."""

    summary = merge_rollout_files(
        paths,
        output_path=output_path,
        include_existing_output=not no_existing_output,
    )
    console.print(
        "[green]Merged rollout shards[/green] "
        f"task={summary.task} "
        f"examples={summary.merged_examples} "
        f"duplicates={summary.duplicate_examples} "
        f"path={summary.output_path}"
    )


@app.command("run-qwen-sweep")
def run_qwen_sweep(
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Truth Spec rollout task to generate and grade. Defaults to roleplaying and sandbagging.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/. Defaults to the nearest checkout root.",
    ),
    output_dir: Path = typer.Option(
        Path("data/rollouts"),
        help="Directory for generated rollout JSON files.",
    ),
    prompt_set: str | None = typer.Option(
        DEFAULT_ROLLOUT_PROMPT_SET,
        "--prompt-set",
        help="Cleaned prompt set id used for rollout prompts.",
    ),
    batch_size: int = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.batch_size, min=1),
    max_new_tokens: int = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.max_new_tokens, min=1),
    do_sample: bool = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.do_sample),
    temperature: float = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.temperature, min=0.0),
    top_p: float = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_p, min=0.0, max=1.0),
    top_k: int | None = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_k, min=0),
    seed: int = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.seed),
    start: int = typer.Option(0, min=0),
    limit: int | None = typer.Option(None, min=1),
    flush_every: int = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.flush_every, min=1),
    overwrite_generation: bool = typer.Option(False, help="Regenerate rollout files instead of resuming."),
    overwrite_grading: bool = typer.Option(False, help="Re-grade already populated labels."),
    provider: str = typer.Option(DEFAULT_JUDGE_CONFIG.provider),
    judge_model: str | None = typer.Option(DEFAULT_JUDGE_CONFIG.model),
    max_workers: int = typer.Option(DEFAULT_ROLLOUT_GRADING_MAX_WORKERS, min=1),
    max_tokens: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_tokens, min=1),
    max_retries: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_retries, min=1),
    timeout: float = typer.Option(DEFAULT_JUDGE_CONFIG.timeout, min=1.0),
    structured_outputs: bool = typer.Option(DEFAULT_JUDGE_CONFIG.structured_outputs),
    require_structured_outputs: bool = typer.Option(DEFAULT_JUDGE_CONFIG.require_structured_outputs),
    grading_flush_every: int = typer.Option(DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY, min=1),
    continue_on_error: bool = typer.Option(
        False,
        help="Keep roleplaying judging running when one judge item fails.",
    ),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling OpenRouter.",
    ),
) -> None:
    """Generate Qwen rollouts and grade them in one preflighted sweep."""

    load_dotenv()
    project_root = _resolve_project_root(project_root)
    output_dir = _project_path(project_root, output_dir)
    task_list = tuple(tasks) if tasks else DEFAULT_ROLLOUT_TASKS
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
    examples_by_task = {
        task: load_rollout_prompt_examples(
            task,
            project_root=project_root,
            prompt_set=prompt_set,
        )
        for task in task_list
    }
    source_indices_by_task = {
        task: _selected_source_indices(examples, start=start, limit=limit)
        for task, examples in examples_by_task.items()
    }

    if any(_rollout_task_needs_llm_judge(task) for task in task_list):
        preflight = preflight_judge_config(judge_config)
        console.print(
            "[green]OpenRouter judge preflight passed[/green] "
            f"alias={preflight.alias} resolved_model={preflight.resolved_model}"
        )

    seed_everything(seed)
    model_config = model_config_from_env()
    model_id = resolve_model_id(model_config.model_name)
    model_slug = qwen_model_slug(model_id)
    pending_by_task: dict[str, set[int]] = {}
    for task, examples in examples_by_task.items():
        output_path = output_dir / f"{task}__{model_slug}.json"
        pending_by_task[task] = pending_rollout_source_indices(
            task=task,
            output_path=output_path,
            model_id=model_id,
            examples=examples,
            settings=settings,
            overwrite=overwrite_generation,
            start=start,
            limit=limit,
        )
    bundle = load_model_and_processor(model_config) if any(pending_by_task.values()) else None

    generated_paths: list[tuple[str, Path]] = []
    for task in task_list:
        examples = examples_by_task[task]
        output_path = output_dir / f"{task}__{model_slug}.json"
        generated_paths.append((task, output_path))
        if not pending_by_task[task]:
            selected_count = _selected_example_count(examples, start=start, limit=limit)
            console.print(
                "[green]Generated rollouts[/green] "
                f"task={task} new=0 skipped={selected_count} path={output_path}"
            )
            continue
        if bundle is None:
            raise RuntimeError("Internal error: pending rollouts require a loaded Qwen model.")
        summary = generate_rollout_task(
            bundle=bundle,
            task=task,
            examples=examples,
            output_path=output_path,
            settings=settings,
            overwrite=overwrite_generation,
            start=start,
            limit=limit,
        )
        console.print(
            "[green]Generated rollouts[/green] "
            f"task={summary.task} new={summary.generated_examples} "
            f"skipped={summary.skipped_existing} path={summary.output_path}"
        )

    for task, path in generated_paths:
        grading_summary = grade_rollout_file(
            path,
            project_root=project_root,
            config=judge_config,
            overwrite=overwrite_grading,
            start=0,
            limit=None,
            source_indices=source_indices_by_task[task],
            flush_every=grading_flush_every,
            max_workers=max_workers,
            continue_on_error=continue_on_error,
        )
        console.print(
            "[green]Graded rollout file[/green] "
            f"task={grading_summary.task} graded={grading_summary.graded_items} "
            f"skipped={grading_summary.skipped_items} path={grading_summary.output_path}"
        )


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
    samples_per_prompt: int = typer.Option(1, min=1, help="Number of sampled transcripts per prompt file."),
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
    overwrite: bool = typer.Option(False, help="Replace an existing output file instead of resuming."),
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

    load_dotenv()
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
    seed_everything(seed)

    model_config = model_config_from_env()
    model_id = resolve_model_id(model_config.model_name)
    model_slug = qwen_model_slug(model_id)
    output_path = _project_path(
        project_root,
        output_path or Path("data") / "insider_trading" / f"{model_slug}-generations.json",
    )
    validate_insider_trading_generation_resume(
        project_root=project_root,
        output_path=output_path,
        prompt_glob=prompt_glob,
        model_id=model_id,
        settings=settings,
        samples_per_prompt=samples_per_prompt,
        label_mode=validated_label_mode,
        overwrite=overwrite,
    )
    bundle = load_model_and_processor(model_config)

    summary = generate_insider_trading_transcripts(
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
    limit: int | None = typer.Option(None, min=1, help="Limit transcript records to grade."),
    flush_every: int = typer.Option(
        DEFAULT_INSIDER_GRADING_FLUSH_EVERY,
        min=1,
        help="Write JSON after this many graded transcripts.",
    ),
    overwrite: bool = typer.Option(False, help="Re-grade transcripts that already have labels."),
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

    load_dotenv()
    project_root = _resolve_project_root(project_root)
    input_path = _project_path(
        project_root,
        input_path or Path("data") / "insider_trading" / f"{MODEL_SLUG}-generations.json",
    )
    output_path = _project_path(project_root, output_path) if output_path is not None else None

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
    preflight_judge_config(judge_config)
    summary = grade_insider_trading_transcripts_with_judge(
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
    samples_per_prompt: int = typer.Option(1, min=1, help="Number of sampled transcripts per prompt file."),
    batch_size: int = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.batch_size, min=1),
    max_new_tokens: int = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.max_new_tokens, min=1),
    do_sample: bool = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.do_sample),
    temperature: float = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.temperature, min=0.0),
    top_p: float = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.top_p, min=0.0, max=1.0),
    top_k: int | None = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.top_k, min=0),
    seed: int = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.seed),
    generation_flush_every: int = typer.Option(DEFAULT_INSIDER_GENERATION_SETTINGS.flush_every, min=1),
    grading_start: int = typer.Option(0, min=0, help="Start transcript index for grading."),
    grading_limit: int | None = typer.Option(None, min=1, help="Limit transcripts to grade after generation."),
    overwrite_generation: bool = typer.Option(False, help="Regenerate transcript file instead of resuming."),
    overwrite_grading: bool = typer.Option(False, help="Re-grade already populated transcript labels."),
    provider: str = typer.Option(DEFAULT_JUDGE_CONFIG.provider),
    judge_model: str | None = typer.Option(DEFAULT_JUDGE_CONFIG.model),
    max_tokens: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_tokens, min=1),
    max_retries: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_retries, min=1),
    timeout: float = typer.Option(DEFAULT_JUDGE_CONFIG.timeout, min=1.0),
    structured_outputs: bool = typer.Option(DEFAULT_JUDGE_CONFIG.structured_outputs),
    require_structured_outputs: bool = typer.Option(DEFAULT_JUDGE_CONFIG.require_structured_outputs),
    grading_flush_every: int = typer.Option(DEFAULT_INSIDER_GRADING_FLUSH_EVERY, min=1),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling OpenRouter.",
    ),
) -> None:
    """Generate Qwen insider-trading transcripts and judge them in one sweep."""

    load_dotenv()
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
    preflight = preflight_judge_config(judge_config)
    console.print(
        "[green]OpenRouter judge preflight passed[/green] "
        f"alias={preflight.alias} resolved_model={preflight.resolved_model}"
    )

    seed_everything(seed)
    model_config = model_config_from_env()
    model_id = resolve_model_id(model_config.model_name)
    model_slug = qwen_model_slug(model_id)
    output_path = _project_path(
        project_root,
        output_path or Path("data") / "insider_trading" / f"{model_slug}-generations.json",
    )
    validate_insider_trading_generation_resume(
        project_root=project_root,
        output_path=output_path,
        prompt_glob=prompt_glob,
        model_id=model_id,
        settings=settings,
        samples_per_prompt=samples_per_prompt,
        label_mode="unknown",
        overwrite=overwrite_generation,
    )
    bundle = load_model_and_processor(model_config)

    generation_summary = generate_insider_trading_transcripts(
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

    grading_summary = grade_insider_trading_transcripts_with_judge(
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


@app.command("generate-sycophancy")
def generate_sycophancy(
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/sycophancy. Defaults to the nearest checkout root.",
    ),
    output_dir: Path = typer.Option(
        Path("data") / "sycophancy",
        help="Base sycophancy output directory. The generated model slug is appended.",
    ),
    generated_model: str = typer.Option(
        MODEL_SLUG,
        help="Generated model slug used as the sycophancy output subdirectory.",
    ),
    task: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="MMLU STEM task to generate/pair. Repeatable. Defaults to all STEM tasks.",
    ),
    mode: str = typer.Option(
        "all",
        help='Workflow mode: "generate", "pair", or "all". Pair mode does not load Qwen.',
    ),
    limit_per_task: int | None = typer.Option(
        None,
        min=1,
        help="Limit raw MMLU questions per task for smoke tests.",
    ),
    conf_threshold: float = typer.Option(
        0.5,
        min=0.0,
        max=1.0,
        help="Minimum max answer probability for positive and negative pair members.",
    ),
    max_samples_per_task: int = typer.Option(
        200,
        min=2,
        help="Maximum paired examples per task, counting positive plus negative rows.",
    ),
    seed: int = typer.Option(42, help="Pairing random seed."),
    overwrite: bool = typer.Option(False, help="Regenerate existing raw per-task files."),
) -> None:
    """Generate and pair Qwen-specific sycophancy source files."""

    load_dotenv()
    project_root = _resolve_project_root(project_root)
    selected_tasks = tuple(task or STEM_TASKS)
    unknown_tasks = sorted(set(selected_tasks) - set(STEM_TASKS))
    if unknown_tasks:
        raise typer.BadParameter(f"Unsupported MMLU STEM task(s): {', '.join(unknown_tasks)}")
    if mode not in {"generate", "pair", "all"}:
        raise typer.BadParameter('mode must be "generate", "pair", or "all"')

    output_base = _project_path(project_root, output_dir)
    model_output_dir = output_base / generated_model

    if mode in {"generate", "all"}:
        model_config = model_config_from_env()
        model_id = resolve_model_id(model_config.model_name)
        expected_slug = qwen_model_slug(model_id)
        if generated_model != expected_slug:
            raise typer.BadParameter(f"generated-model must match loaded model slug {expected_slug!r}.")
        bundle = load_model_and_processor(model_config)
        generation_summary = generate_sycophancy_dataset(
            bundle=bundle,
            output_dir=model_output_dir,
            tasks=selected_tasks,
            limit_per_task=limit_per_task,
            overwrite=overwrite,
        )
        console.print(
            "[green]Generated sycophancy raw files[/green] "
            f"samples={generation_summary.generated_samples} "
            f"skipped_tasks={generation_summary.skipped_existing_tasks} "
            f"dir={generation_summary.output_dir}"
        )

    if mode in {"pair", "all"}:
        pairing_summary = pair_sycophancy_dataset(
            output_dir=model_output_dir,
            tasks=selected_tasks,
            conf_threshold=conf_threshold,
            max_samples_per_task=max_samples_per_task,
            seed=seed,
            same_answer_only=True,
        )
        console.print(
            "[green]Paired sycophancy files[/green] "
            f"examples={pairing_summary.paired_examples} "
            f"dir={pairing_summary.output_dir}"
        )


@app.command("extract-activations")
def extract_activations(
    paths: list[Path] | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Truth Spec-compatible rollout JSON path. Repeatable. Defaults to generated Qwen rollout files unless --task is provided.",
    ),
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Named activation dataset task, e.g. claims__definitional_gemini_600_full. Repeatable.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root containing data/. Defaults to the nearest checkout root.",
    ),
    generated_model: str = typer.Option(
        MODEL_SLUG,
        help="Generated Qwen model slug used for rollout/transcript filenames.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HDF5 path. Defaults to artifacts/activations/extracted_feats_all_layers_{model_slug}.h5.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        "-l",
        help='Decoder layers to extract: "all", "sparse", comma list, or range.',
    ),
    batch_size: int = typer.Option(1, min=1, help="Extraction batch size."),
    start: int = typer.Option(0, min=0, help="Start rollout index."),
    limit: int | None = typer.Option(None, min=1, help="Limit rollout records per file."),
    max_length: int | None = typer.Option(None, min=1, help="Optional tokenizer truncation length."),
    verify_masks: bool = typer.Option(True, help="Decode masked tokens and verify answer-token alignment."),
    overwrite: bool = typer.Option(False, help="Replace existing task datasets in the HDF5 file."),
    resume: bool = typer.Option(
        False,
        help="Reuse compatible existing task metadata/layers and skip replacing already-written requested layers.",
    ),
    backend_name: str = typer.Option(
        "transformers",
        "--backend",
        help='Activation backend: "transformers" or "nnsight".',
    ),
    capture_logits: bool = typer.Option(
        False,
        help="Also store next-token logits at answer-token scoring positions.",
    ),
    storage_dtype: str = typer.Option(
        "float16",
        "--storage-dtype",
        help='Activation/logit HDF5 storage dtype: "float16" or "float32".',
    ),
    compression: str = typer.Option(
        "lzf",
        "--compression",
        help='Shard and merge compression: "gzip", "lzf", or "none".',
    ),
) -> None:
    """Extract decoder activations from rollouts or named source datasets.

    Args:
        paths: Repeatable rollout JSON paths. Mutually exclusive with `tasks`.
        tasks: Repeatable named activation dataset ids. Mutually exclusive with
            `paths`.
        project_root: Repository root containing data and output directories.
        generated_model: Model slug used to find generated rollout filenames.
        output_path: Optional HDF5 output path.
        layers: Layer spec accepted by `parse_layer_spec`.
        batch_size: Examples per activation extraction batch.
        start: First source example/record index.
        limit: Optional number of records per source.
        max_length: Optional tokenizer truncation length.
        verify_masks: Decode masked tokens and verify answer alignment.
        overwrite: Replace existing task datasets in the HDF5 file.
        resume: Reuse compatible existing layers and skip replacing already-written requested layers.
        backend_name: Activation backend, either `transformers` or `nnsight`.
        capture_logits: Store next-token logits at answer-token positions.
        storage_dtype: Floating dtype used when writing activations to HDF5.

    References:
        Rollout inputs default to `data/rollouts/{task}__{MODEL_SLUG}.json`.
        Named dataset loaders and HDF5 writing live in `activations.py`.
    """

    load_dotenv()
    project_root = _resolve_project_root(project_root)

    if paths and tasks:
        raise typer.BadParameter("Use either --path rollout files or --task named datasets, not both.")

    # Build named datasets before any model load; tests rely on source-data
    # validation failing early when a named task is unavailable.
    named_datasets = [
        ActivationDataset.from_named_task(
            task,
            project_root=project_root,
            generated_model=generated_model,
        )
        for task in tasks or []
    ]
    # When named tasks are provided, rollout paths are intentionally ignored.
    rollout_paths = [
        _project_path(project_root, path)
        for path in (
            []
            if tasks
            else paths
            or [Path("data") / "rollouts" / f"{task}__{MODEL_SLUG}.json" for task in DEFAULT_ROLLOUT_TASKS]
        )
    ]

    model_config = model_config_from_env()
    activation_backend: ActivationBackend
    if backend_name == "transformers":
        bundle = load_model_and_processor(model_config)
        activation_backend = TransformersHookBackend(bundle)
    elif backend_name == "nnsight":
        nnsight_bundle = load_nnsight_bundle(model_config)
        bundle = ModelBundle(
            model=nnsight_bundle.model,
            processor=nnsight_bundle.processor,
            tokenizer=nnsight_bundle.tokenizer,
            model_id=nnsight_bundle.model_id,
            config=nnsight_bundle.config,
        )
        activation_backend = NnsightActivationBackend(nnsight_bundle)
    else:
        raise typer.BadParameter('backend must be "transformers" or "nnsight".')

    layer_indices = parse_layer_spec(layers, activation_backend.decoder_layer_count())
    if storage_dtype == "float16":
        storage_dtype_value = "float16"
    elif storage_dtype == "float32":
        storage_dtype_value = "float32"
    else:
        raise typer.BadParameter('storage-dtype must be "float16" or "float32".')
    if compression not in {"gzip", "lzf", "none"}:
        raise typer.BadParameter('compression must be "gzip", "lzf", or "none".')
    output_path = _project_path(
        project_root,
        output_path or default_activation_output_path(activation_backend.model_id),
    )

    settings = ActivationExtractionSettings(
        layers=layer_indices,
        batch_size=batch_size,
        start=start,
        limit=limit,
        verify_masks=verify_masks,
        max_length=max_length,
        capture_logits=capture_logits,
        resume=resume,
        compression=compression,
        storage_dtype=storage_dtype_value,
    )

    summaries = []
    if named_datasets:
        for dataset in named_datasets:
            summaries.append(
                extract_dataset_activations(
                    bundle=bundle,
                    dataset=dataset,
                    output_path=output_path,
                    settings=settings,
                    overwrite=overwrite,
                    backend=activation_backend,
                )
            )
    else:
        for rollout_path in rollout_paths:
            summaries.append(
                extract_rollout_activations(
                    bundle=bundle,
                    rollout_path=rollout_path,
                    output_path=output_path,
                    settings=settings,
                    overwrite=overwrite,
                    backend=activation_backend,
                )
            )

    for summary in summaries:
        console.print(
            "[green]Extracted activations[/green] "
            f"task={summary.task} "
            f"examples={summary.examples_extracted}/{summary.examples_seen} "
            f"skipped_labels={summary.skipped_labels} "
            f"masked_tokens={summary.masked_tokens} "
            f"layers={list(summary.layers)} "
            f"path={summary.output_path}"
        )


@app.command("merge-activation-shards")
def merge_activation_shards(
    paths: list[Path] = typer.Argument(..., help="Activation HDF5 shard paths to merge."),
    output_path: Path = typer.Option(..., "--output", "-o", help="Final merged HDF5 output path."),
    overwrite: bool = typer.Option(False, help="Replace an existing merged output file."),
    compression: str = typer.Option(
        "lzf",
        "--compression",
        help='Output compression for concat merges: "gzip", "lzf", or "none". Copy-disjoint merges preserve source compression.',
    ),
    merge_strategy: str = typer.Option(
        "auto",
        "--merge-strategy",
        help='Merge strategy: "auto", "concat", or "copy-disjoint".',
    ),
    expected_queue_plan_id: str | None = typer.Option(
        None,
        "--expected-queue-plan-id",
        help="Require all input shards with queue metadata to match this queue plan id.",
    ),
    require_queue_plan_id: bool = typer.Option(
        False,
        "--require-queue-plan-id",
        help="Reject input shards that do not carry queue_plan_id metadata.",
    ),
    force_stale_merge_lock: bool = typer.Option(
        False,
        "--force-stale-merge-lock",
        help="Break an output merge lock only when it is on this host and the recorded PID is dead.",
    ),
) -> None:
    """Merge independently extracted activation HDF5 shards."""

    project_root = _resolve_project_root(None)
    resolved_paths = [_project_path(project_root, path) for path in paths]
    output_path = _project_path(project_root, output_path)
    if compression not in {"gzip", "lzf", "none"}:
        raise typer.BadParameter('compression must be "gzip", "lzf", or "none".')
    if merge_strategy not in {"auto", "concat", "copy-disjoint"}:
        raise typer.BadParameter('merge-strategy must be "auto", "concat", or "copy-disjoint".')
    summary = merge_activation_hdf5_shards(
        resolved_paths,
        output_path=output_path,
        overwrite=overwrite,
        compression=compression,
        merge_strategy=merge_strategy,
        expected_queue_plan_id=expected_queue_plan_id,
        require_queue_plan_id=require_queue_plan_id,
        force_stale_merge_lock=force_stale_merge_lock,
    )
    for task in summary.tasks:
        console.print(
            "[green]Merged activation shards[/green] "
            f"task={task} "
            f"examples={summary.examples_by_task[task]} "
            f"token_rows={summary.token_rows_by_task[task]} "
            f"shards={len(summary.shard_paths)} "
            f"path={summary.output_path}"
        )


@app.command("preflight-probes")
def preflight_probes(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Activation HDF5 path to inspect before probe training.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON path for the metadata-only probe preflight report.",
    ),
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Activation task to inspect. Repeatable. Defaults to every task in the HDF5.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root. Relative input/output paths resolve under this root.",
    ),
    test_size: float = typer.Option(
        0.25,
        min=0.01,
        max=0.99,
        help="Planned held-out fraction for each per-task stratified example split.",
    ),
    random_seed: int = typer.Option(0, help="Random seed for planned stratified example splits."),
    min_train_examples_per_class: int = typer.Option(
        20,
        min=1,
        help="Minimum smaller-class examples for a task to be a training source.",
    ),
    min_eval_examples_per_class: int = typer.Option(
        5,
        min=1,
        help="Minimum smaller-class examples for a task to be an eval-only target.",
    ),
    min_test_examples_per_class: int = typer.Option(
        5,
        min=1,
        help="Minimum smaller-class examples required in the planned held-out split.",
    ),
) -> None:
    """Write a metadata-only readiness report for later probe training."""

    project_root = _resolve_project_root(project_root)
    summary = preflight_probe_training(
        input_path=_project_path(project_root, input_path),
        output_path=_project_path(project_root, output_path),
        tasks=tasks,
        test_size=test_size,
        random_seed=random_seed,
        min_train_examples_per_class=min_train_examples_per_class,
        min_eval_examples_per_class=min_eval_examples_per_class,
        min_test_examples_per_class=min_test_examples_per_class,
    )
    console.print(
        "[green]Wrote probe preflight[/green] "
        f"trainable={len(summary.trainable_tasks)} "
        f"eval_only={len(summary.eval_only_tasks)} "
        f"blocked={len(summary.blocked_tasks)} "
        f"path={summary.output_path}"
    )


@app.command("train-probes")
def train_probes(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Activation HDF5 path produced by extract-activations.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output JSON path for probe metrics.",
    ),
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Activation task to train/evaluate. Repeatable. Defaults to every task in the HDF5.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root. Relative input/output paths resolve under this root.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        "-l",
        help='Decoder layers to train on: "all", comma list, or range.',
    ),
    test_size: float = typer.Option(
        0.25,
        min=0.01,
        max=0.99,
        help="Held-out fraction for each within-task example split.",
    ),
    random_seed: int = typer.Option(0, help="Random seed for stratified example splits and logistic regression."),
    max_iter: int = typer.Option(1000, min=1, help="Maximum logistic-regression iterations."),
    regularization_c: float = typer.Option(
        1.0,
        "--c",
        min=1e-9,
        help="Inverse L2 regularization strength for logistic regression.",
    ),
    train_general_domain_probe: bool = typer.Option(
        False,
        "--general-domain-probe",
        help="Also train one pooled general-domain probe per layer over the selected tasks.",
    ),
    general_task_class_cap: int = typer.Option(
        1000,
        "--general-task-class-cap",
        min=1,
        help="Maximum examples per task/label bucket for the general-domain probe.",
    ),
) -> None:
    """Train simple linear probes on mean-pooled answer-token activations."""

    project_root = _resolve_project_root(project_root)
    summary = train_probe_directions(
        input_path=_project_path(project_root, input_path),
        output_path=_project_path(project_root, output_path),
        tasks=tasks,
        layers=layers,
        test_size=test_size,
        random_seed=random_seed,
        max_iter=max_iter,
        regularization_c=regularization_c,
        train_general_domain_probe=train_general_domain_probe,
        general_task_class_cap=general_task_class_cap,
    )
    console.print(
        "[green]Trained probes[/green] "
        f"tasks={list(summary.tasks)} "
        f"layers={list(summary.layers)} "
        f"within_task={summary.within_task_results} "
        f"cross_task={summary.cross_task_results} "
        f"directions={summary.direction_results} "
        f"general_domain={summary.general_domain_results} "
        f"general_domain_directions={summary.general_domain_direction_results} "
        f"path={summary.output_path}"
    )
