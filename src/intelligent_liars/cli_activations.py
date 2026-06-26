from __future__ import annotations

from pathlib import Path

import typer

from intelligent_liars.activation_backends import ActivationBackend
from intelligent_liars.rollouts import DEFAULT_ROLLOUT_TASKS, MODEL_SLUG

from intelligent_liars.cli_common import (
    app,
    console,
    _project_path,
    _resolve_project_root,
)


def _cli():
    from intelligent_liars import cli

    return cli


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
    limit: int | None = typer.Option(
        None, min=1, help="Limit rollout records per file."
    ),
    max_length: int | None = typer.Option(
        None, min=1, help="Optional tokenizer truncation length."
    ),
    verify_masks: bool = typer.Option(
        True, help="Decode masked tokens and verify answer-token alignment."
    ),
    overwrite: bool = typer.Option(
        False, help="Replace existing task datasets in the HDF5 file."
    ),
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

    _cli().load_dotenv()
    project_root = _resolve_project_root(project_root)

    if paths and tasks:
        raise typer.BadParameter(
            "Use either --path rollout files or --task named datasets, not both."
        )

    # Build named datasets before any model load; tests rely on source-data
    # validation failing early when a named task is unavailable.
    named_datasets = [
        _cli().ActivationDataset.from_named_task(
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
            or [
                Path("data") / "rollouts" / f"{task}__{MODEL_SLUG}.json"
                for task in DEFAULT_ROLLOUT_TASKS
            ]
        )
    ]

    model_config = _cli().model_config_from_env()
    activation_backend: ActivationBackend
    if backend_name == "transformers":
        bundle = _cli().load_model_and_processor(model_config)
        activation_backend = _cli().TransformersHookBackend(bundle)
    elif backend_name == "nnsight":
        nnsight_bundle = _cli().load_nnsight_bundle(model_config)
        bundle = _cli().ModelBundle(
            model=nnsight_bundle.model,
            processor=nnsight_bundle.processor,
            tokenizer=nnsight_bundle.tokenizer,
            model_id=nnsight_bundle.model_id,
            config=nnsight_bundle.config,
        )
        activation_backend = _cli().NnsightActivationBackend(nnsight_bundle)
    else:
        raise typer.BadParameter('backend must be "transformers" or "nnsight".')

    layer_indices = _cli().parse_layer_spec(
        layers, activation_backend.decoder_layer_count()
    )
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
        output_path
        or _cli().default_activation_output_path(activation_backend.model_id),
    )

    settings = _cli().ActivationExtractionSettings(
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
                _cli().extract_dataset_activations(
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
                _cli().extract_rollout_activations(
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
    paths: list[Path] = typer.Argument(
        ..., help="Activation HDF5 shard paths to merge."
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Final merged HDF5 output path."
    ),
    overwrite: bool = typer.Option(
        False, help="Replace an existing merged output file."
    ),
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
        raise typer.BadParameter(
            'merge-strategy must be "auto", "concat", or "copy-disjoint".'
        )
    summary = _cli().merge_activation_hdf5_shards(
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
