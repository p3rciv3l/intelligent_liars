from __future__ import annotations

from pathlib import Path

import typer

from intelligent_liars.judging import (
    DEFAULT_JUDGE_CONFIG,
    DEFAULT_ROLLOUT_GRADING_FLUSH_EVERY,
    DEFAULT_ROLLOUT_GRADING_MAX_WORKERS,
)
from intelligent_liars.rollouts import (
    DEFAULT_ROLLOUT_GENERATION_SETTINGS,
    DEFAULT_ROLLOUT_PROMPT_SET,
    DEFAULT_ROLLOUT_TASKS,
    MODEL_SLUG,
)

from intelligent_liars.cli_common import (
    app,
    console,
    _bind_judge_config_to_project_root,
    _generation_settings,
    _infer_rollout_task_from_path,
    _project_path,
    _resolve_project_root,
    _rollout_task_needs_llm_judge,
    _selected_example_count,
    _selected_source_indices,
    _source_index_filter,
    _judge_config_from_options,
)


def _cli():
    from intelligent_liars import cli

    return cli


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
    overwrite: bool = typer.Option(
        False, help="Replace an existing output file instead of resuming."
    ),
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

    _cli().load_dotenv()
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
    _cli().seed_everything(seed)

    examples_by_task = {
        task: _cli().load_rollout_prompt_examples(
            task,
            project_root=project_root,
            prompt_set=prompt_set,
        )
        for task in task_list
    }

    model_config = _cli().model_config_from_env()
    model_id = _cli().resolve_model_id(model_config.model_name)
    model_slug = _cli().qwen_model_slug(model_id)
    pending_by_task: dict[str, set[int]] = {}
    for task, examples in examples_by_task.items():
        output_path = output_dir / f"{task}__{model_slug}.json"
        pending_by_task[task] = _cli().pending_rollout_source_indices(
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
    bundle = _cli().load_model_and_processor(model_config)

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
        summary = _cli().generate_rollout_task(
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
    limit: int | None = typer.Option(
        None, min=1, help="Limit rollout records to grade per file."
    ),
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

    _cli().load_dotenv()
    project_root = _resolve_project_root(project_root)
    output_dir = (
        _project_path(project_root, output_dir) if output_dir is not None else None
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
    # Resolve defaults lazily after project_root is known.
    rollout_paths = [
        _project_path(project_root, path)
        for path in (
            paths
            or [
                Path("data") / "rollouts" / f"{task}__{MODEL_SLUG}.json"
                for task in DEFAULT_ROLLOUT_TASKS
            ]
        )
    ]
    rollout_tasks = {
        path: _infer_rollout_task_from_path(path) for path in rollout_paths
    }
    if any(_rollout_task_needs_llm_judge(task) for task in rollout_tasks.values()):
        _cli().preflight_judge_config(judge_config)
    source_index_filter = _source_index_filter(
        source_indices=source_indices,
        source_start=source_start,
        source_limit=source_limit,
        record_start=start,
        record_limit=limit,
    )

    for path in rollout_paths:
        summary = _cli().grade_rollout_file(
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
    output_path: Path = typer.Option(
        ..., "--output-path", "-o", help="Final rollout JSON path."
    ),
    no_existing_output: bool = typer.Option(
        False,
        "--no-existing-output",
        help="Do not merge an existing output path before writing the merged file.",
    ),
) -> None:
    """Merge sharded rollout JSON files into one normal rollout file."""

    summary = _cli().merge_rollout_files(
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
    batch_size: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.batch_size, min=1
    ),
    max_new_tokens: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.max_new_tokens, min=1
    ),
    do_sample: bool = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.do_sample),
    temperature: float = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.temperature, min=0.0
    ),
    top_p: float = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_p, min=0.0, max=1.0
    ),
    top_k: int | None = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.top_k, min=0),
    seed: int = typer.Option(DEFAULT_ROLLOUT_GENERATION_SETTINGS.seed),
    start: int = typer.Option(0, min=0),
    limit: int | None = typer.Option(None, min=1),
    flush_every: int = typer.Option(
        DEFAULT_ROLLOUT_GENERATION_SETTINGS.flush_every, min=1
    ),
    overwrite_generation: bool = typer.Option(
        False, help="Regenerate rollout files instead of resuming."
    ),
    overwrite_grading: bool = typer.Option(
        False, help="Re-grade already populated labels."
    ),
    provider: str = typer.Option(DEFAULT_JUDGE_CONFIG.provider),
    judge_model: str | None = typer.Option(DEFAULT_JUDGE_CONFIG.model),
    max_workers: int = typer.Option(DEFAULT_ROLLOUT_GRADING_MAX_WORKERS, min=1),
    max_tokens: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_tokens, min=1),
    max_retries: int = typer.Option(DEFAULT_JUDGE_CONFIG.max_retries, min=1),
    timeout: float = typer.Option(DEFAULT_JUDGE_CONFIG.timeout, min=1.0),
    structured_outputs: bool = typer.Option(DEFAULT_JUDGE_CONFIG.structured_outputs),
    require_structured_outputs: bool = typer.Option(
        DEFAULT_JUDGE_CONFIG.require_structured_outputs
    ),
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

    _cli().load_dotenv()
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
        task: _cli().load_rollout_prompt_examples(
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
        preflight = _cli().preflight_judge_config(judge_config)
        console.print(
            "[green]OpenRouter judge preflight passed[/green] "
            f"alias={preflight.alias} resolved_model={preflight.resolved_model}"
        )

    _cli().seed_everything(seed)
    model_config = _cli().model_config_from_env()
    model_id = _cli().resolve_model_id(model_config.model_name)
    model_slug = _cli().qwen_model_slug(model_id)
    pending_by_task: dict[str, set[int]] = {}
    for task, examples in examples_by_task.items():
        output_path = output_dir / f"{task}__{model_slug}.json"
        pending_by_task[task] = _cli().pending_rollout_source_indices(
            task=task,
            output_path=output_path,
            model_id=model_id,
            examples=examples,
            settings=settings,
            overwrite=overwrite_generation,
            start=start,
            limit=limit,
        )
    bundle = (
        _cli().load_model_and_processor(model_config)
        if any(pending_by_task.values())
        else None
    )

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
            raise RuntimeError(
                "Internal error: pending rollouts require a loaded Qwen model."
            )
        summary = _cli().generate_rollout_task(
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
        grading_summary = _cli().grade_rollout_file(
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
