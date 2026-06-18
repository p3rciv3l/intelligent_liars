from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import cast

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from intelligent_liars.activations import (
    ActivationDataset,
    ActivationExtractionSettings,
    default_activation_output_path,
    extract_dataset_activations,
    extract_rollout_activations,
    parse_layer_spec,
)
from intelligent_liars.judging import (
    JudgeConfig,
    JudgeProvider,
    grade_insider_trading_transcripts as grade_insider_trading_transcripts_with_judge,
    grade_rollout_file,
)
from intelligent_liars.models import (
    load_model_and_processor,
    load_processor,
    model_config_from_env,
    qwen_model_load_description,
    resolve_model_id,
)
from intelligent_liars.activation_backends import TransformersHookBackend
from intelligent_liars.nnsight_backend import (
    NnsightActivationBackend,
    load_nnsight_bundle,
    trace_text_decoder_layer_once,
)
from intelligent_liars.rollouts import (
    DEFAULT_ROLLOUT_TASKS,
    GenerationSettings,
    LabelMode,
    MODEL_SLUG,
    generate_insider_trading_transcripts,
    generate_rollout_task,
    load_rollout_prompt_examples,
    qwen_model_slug,
    seed_everything,
)


app = typer.Typer(no_args_is_help=True)
console = Console()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


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
    if provider != "openrouter":
        raise typer.BadParameter('provider must be "openrouter". Use --judge-model for OpenRouter aliases or raw model IDs.')
    return JudgeConfig(
        provider=cast(JudgeProvider, provider),
        model=judge_model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        structured_outputs=structured_outputs,
        require_structured_outputs=require_structured_outputs,
        mock_response=mock_response,
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
    nnsight_layer: int = typer.Option(
        0,
        help="Decoder layer index to trace for --check-nnsight.",
    ),
) -> None:
    """Print the runtime state needed before running activation experiments."""
    load_dotenv()
    model_config = model_config_from_env()
    model_id = resolve_model_id(model_config.model_name)

    table = Table(title="Intelligent Liars environment")
    table.add_column("Check")
    table.add_column("Value")

    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    table.add_row("model_name", model_config.model_name)
    table.add_row("model_id", model_id)
    table.add_row("hf_home", model_config.cache_dir or "default")
    table.add_row("hf_token", "set" if os.getenv("HF_TOKEN") else "not set")
    table.add_row("model_load", qwen_model_load_description())
    table.add_row(
        "gpu_ids",
        ",".join(str(gpu_id) for gpu_id in model_config.gpu_ids)
        if model_config.gpu_ids
        else "default",
    )
    table.add_row("torch", _version("torch"))
    table.add_row("transformers", _version("transformers"))
    table.add_row("accelerate", _version("accelerate"))
    table.add_row("flash-attn", _version("flash-attn"))
    table.add_row("nnsight", _version("nnsight"))
    table.add_row("qwen-vl-utils", _version("qwen-vl-utils"))

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        table.add_row("cuda", str(cuda_available))
        if cuda_available:
            table.add_row("cuda_device_count", str(torch.cuda.device_count()))
            table.add_row("cuda_device_0", torch.cuda.get_device_name(0))
            table.add_row("cuda_version", torch.version.cuda or "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic path
        table.add_row("cuda", f"unavailable: {exc}")

    console.print(table)

    if require_cuda and not cuda_available:
        raise typer.BadParameter("CUDA is required for this run but is unavailable.")

    if check_processor:
        console.print(f"Loading processor for [bold]{model_id}[/bold]...")
        load_processor(model_config)
        console.print("[green]Processor loaded.[/green]")

    if check_model:
        console.print(f"Loading full model for [bold]{model_id}[/bold]...")
        load_model_and_processor(model_config)
        console.print("[green]Model loaded.[/green]")

    if check_nnsight:
        console.print(f"Loading NNsight wrapper for [bold]{model_id}[/bold]...")
        bundle = load_nnsight_bundle(model_config)
        console.print(f"Tracing decoder layer [bold]{nnsight_layer}[/bold]...")
        result = trace_text_decoder_layer_once(bundle, layer_idx=nnsight_layer)
        console.print(
            "[green]NNsight trace succeeded.[/green] "
            f"input_shape={result.input_shape} "
            f"layer_output_shape={result.layer_output_shape}"
        )


@app.command("generate-rollouts")
def generate_rollouts(
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Truth Spec rollout task to generate. Repeatable. Defaults to roleplaying and sandbagging.",
    ),
    project_root: Path = typer.Option(
        Path("."),
        help="Project root containing data/ and references/truth_spec/.",
    ),
    output_dir: Path = typer.Option(
        Path("data/rollouts"),
        help="Directory for Truth Spec-compatible rollout JSON files.",
    ),
    prompt_source_model: str | None = typer.Option(
        None,
        help="Optional reference rollout source model. Defaults to auto-discovery when only one source file matches.",
    ),
    batch_size: int = typer.Option(1, min=1, help="Generation batch size."),
    max_new_tokens: int = typer.Option(512, min=1, help="Maximum new tokens per completion."),
    do_sample: bool = typer.Option(False, help="Use sampling instead of greedy decoding."),
    temperature: float = typer.Option(0.6, min=0.0, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, min=0.0, max=1.0, help="Sampling nucleus top-p."),
    seed: int = typer.Option(0, help="Random seed for reproducible sampling."),
    start: int = typer.Option(0, min=0, help="Start source index."),
    limit: int | None = typer.Option(None, min=1, help="Limit examples per task."),
    flush_every: int = typer.Option(10, min=1, help="Write JSON after this many new completions."),
    overwrite: bool = typer.Option(False, help="Replace an existing output file instead of resuming."),
) -> None:
    """Generate Qwen rollouts on a GPU box and write Truth Spec-compatible JSON."""
    load_dotenv()
    project_root = project_root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    task_list = tuple(tasks) if tasks else DEFAULT_ROLLOUT_TASKS

    settings = GenerationSettings(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        flush_every=flush_every,
    )
    seed_everything(seed)

    model_config = model_config_from_env()
    bundle = load_model_and_processor(model_config)
    model_slug = qwen_model_slug(bundle.model_id)

    for task in task_list:
        examples = load_rollout_prompt_examples(
            task,
            project_root=project_root,
            prompt_source_model=prompt_source_model,
        )
        output_path = output_dir / f"{task}__{model_slug}.json"
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
    project_root: Path = typer.Option(
        Path("."),
        help="Project root containing data/ grading templates.",
    ),
    provider: str = typer.Option("openrouter", help='Judge transport. Currently only "openrouter" is supported.'),
    judge_model: str | None = typer.Option(
        None,
        help="OpenRouter model alias from prod_env/model_deployments.yaml or raw OpenRouter model ID.",
    ),
    max_workers: int = typer.Option(8, min=1, help="Concurrent judge calls for LLM-graded rollouts."),
    max_tokens: int = typer.Option(1000, min=1, help="Maximum judge output tokens."),
    max_retries: int = typer.Option(4, min=1, help="Judge API/parse retries."),
    timeout: float = typer.Option(120.0, min=1.0, help="OpenRouter request timeout in seconds."),
    structured_outputs: bool = typer.Option(True, help="Request JSON schema outputs for LLM judges."),
    require_structured_outputs: bool = typer.Option(
        True,
        help="Require routed OpenRouter providers to support structured output parameters.",
    ),
    start: int = typer.Option(0, min=0, help="Start rollout index."),
    limit: int | None = typer.Option(None, min=1, help="Limit rollout records to grade per file."),
    flush_every: int = typer.Option(25, min=1, help="Write JSON after this many completed judge calls."),
    overwrite: bool = typer.Option(False, help="Re-grade already populated labels."),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling an API.",
    ),
) -> None:
    """Grade generated rollout JSON files with the local Apollo judge port."""
    load_dotenv()
    project_root = project_root.resolve()
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
    rollout_paths = paths or [
        project_root / "data" / "rollouts" / f"{task}__{MODEL_SLUG}.json"
        for task in DEFAULT_ROLLOUT_TASKS
    ]

    for path in rollout_paths:
        path = path if path.is_absolute() else project_root / path
        summary = grade_rollout_file(
            path,
            project_root=project_root,
            config=judge_config,
            overwrite=overwrite,
            start=start,
            limit=limit,
            flush_every=flush_every,
            max_workers=max_workers,
        )
        console.print(
            "[green]Graded rollout file[/green] "
            f"task={summary.task} "
            f"graded={summary.graded_items} "
            f"skipped={summary.skipped_items} "
            f"path={summary.output_path}"
        )


@app.command("generate-insider-trading")
def generate_insider_trading(
    project_root: Path = typer.Option(
        Path("."),
        help="Project root containing data/insider_trading/prompts.",
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
    batch_size: int = typer.Option(1, min=1, help="Generation batch size. Insider flow uses sequential turns."),
    max_new_tokens: int = typer.Option(512, min=1, help="Maximum new tokens per assistant turn."),
    do_sample: bool = typer.Option(True, help="Use sampling. Recommended for multiple transcripts per prompt."),
    temperature: float = typer.Option(0.6, min=0.0, help="Sampling temperature."),
    top_p: float = typer.Option(0.95, min=0.0, max=1.0, help="Sampling nucleus top-p."),
    seed: int = typer.Option(0, help="Random seed for reproducible sampling."),
    flush_every: int = typer.Option(5, min=1, help="Write JSON after this many new transcripts."),
    label_mode: str = typer.Option(
        "unknown",
        help='Use "unknown" for ungraded data or "heuristic" only for smoke/debug labels.',
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing output file instead of resuming."),
) -> None:
    """Generate Qwen insider-trading transcripts from local prompt YAML files."""
    if label_mode not in {"unknown", "heuristic"}:
        raise typer.BadParameter('label-mode must be "unknown" or "heuristic".')
    validated_label_mode = cast(LabelMode, label_mode)

    load_dotenv()
    project_root = project_root.resolve()
    settings = GenerationSettings(
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        flush_every=flush_every,
    )
    seed_everything(seed)

    model_config = model_config_from_env()
    bundle = load_model_and_processor(model_config)
    model_slug = qwen_model_slug(bundle.model_id)
    output_path = output_path or project_root / "data" / "insider_trading" / f"{model_slug}-generations.json"
    if not output_path.is_absolute():
        output_path = project_root / output_path

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
    project_root: Path = typer.Option(
        Path("."),
        help="Project root containing data/insider_trading grading prompts.",
    ),
    input_path: Path | None = typer.Option(
        None,
        help="Transcript JSON path. Defaults to data/insider_trading/qwen3-vl-8b-thinking-generations.json.",
    ),
    output_path: Path | None = typer.Option(
        None,
        help="Output path. Defaults to in-place grading of input-path.",
    ),
    provider: str = typer.Option("openrouter", help='Judge transport. Currently only "openrouter" is supported.'),
    judge_model: str | None = typer.Option(
        None,
        help="OpenRouter model alias from prod_env/model_deployments.yaml or raw OpenRouter model ID.",
    ),
    max_tokens: int = typer.Option(1000, min=1, help="Maximum judge output tokens."),
    max_retries: int = typer.Option(4, min=1, help="Judge API/parse retries."),
    timeout: float = typer.Option(120.0, min=1.0, help="OpenRouter request timeout in seconds."),
    structured_outputs: bool = typer.Option(True, help="Request JSON schema outputs for LLM judges."),
    require_structured_outputs: bool = typer.Option(
        True,
        help="Require routed OpenRouter providers to support structured output parameters.",
    ),
    start: int = typer.Option(0, min=0, help="Start transcript index."),
    limit: int | None = typer.Option(None, min=1, help="Limit transcript records to grade."),
    flush_every: int = typer.Option(10, min=1, help="Write JSON after this many graded transcripts."),
    overwrite: bool = typer.Option(False, help="Re-grade transcripts that already have labels."),
    mock_response: str | None = typer.Option(
        None,
        help="Testing hook: use this literal judge response instead of calling an API.",
    ),
) -> None:
    """Grade generated insider-trading transcripts with Apollo's task-specific judge prompts."""
    load_dotenv()
    project_root = project_root.resolve()
    input_path = input_path or project_root / "data" / "insider_trading" / f"{MODEL_SLUG}-generations.json"
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if output_path is not None and not output_path.is_absolute():
        output_path = project_root / output_path

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
    project_root: Path = typer.Option(
        Path("."),
        help="Project root containing data/ and references/truth_spec/.",
    ),
    generated_model: str = typer.Option(
        MODEL_SLUG,
        help="Generated Qwen model slug used for rollout/transcript filenames.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HDF5 path. Defaults to results/extracted_feats_all_layers_{model_slug}.h5.",
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
        help="Reuse compatible existing task metadata/layers and write only missing requested layers.",
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
) -> None:
    """Extract decoder activations from rollout JSON paths or named Truth Spec datasets."""
    load_dotenv()
    project_root = project_root.resolve()

    if paths and tasks:
        raise typer.BadParameter("Use either --path rollout files or --task named datasets, not both.")

    rollout_paths = paths or []
    if not tasks and not rollout_paths:
        rollout_paths = [
            project_root / "data" / "rollouts" / f"{task}__{MODEL_SLUG}.json"
            for task in DEFAULT_ROLLOUT_TASKS
        ]
    rollout_paths = [
        path if path.is_absolute() else project_root / path
        for path in rollout_paths
    ]

    named_datasets: list[ActivationDataset] = []
    if tasks:
        for task in tasks:
            named_datasets.append(
                ActivationDataset.from_named_task(
                    task,
                    project_root=project_root,
                    generated_model=generated_model,
                )
            )

    model_config = model_config_from_env()
    if backend_name == "transformers":
        bundle = load_model_and_processor(model_config)
        activation_backend = TransformersHookBackend(bundle)
    elif backend_name == "nnsight":
        bundle = load_nnsight_bundle(model_config)
        activation_backend = NnsightActivationBackend(bundle)
    else:
        raise typer.BadParameter('backend must be "transformers" or "nnsight".')

    layer_indices = parse_layer_spec(layers, activation_backend.decoder_layer_count())
    output_path = output_path or default_activation_output_path(bundle.model_id)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    settings = ActivationExtractionSettings(
        layers=layer_indices,
        batch_size=batch_size,
        start=start,
        limit=limit,
        verify_masks=verify_masks,
        max_length=max_length,
        capture_logits=capture_logits,
        resume=resume,
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
