from __future__ import annotations

from pathlib import Path

import typer

from intelligent_liars.rollouts import MODEL_SLUG
from intelligent_liars.sycophancy import STEM_TASKS

from intelligent_liars.cli_common import (
    app,
    console,
    _project_path,
    _resolve_project_root,
)


def _cli():
    from intelligent_liars import cli

    return cli


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
    overwrite: bool = typer.Option(
        False, help="Regenerate existing raw per-task files."
    ),
) -> None:
    """Generate and pair Qwen-specific sycophancy source files."""

    _cli().load_dotenv()
    project_root = _resolve_project_root(project_root)
    selected_tasks = tuple(task or STEM_TASKS)
    unknown_tasks = sorted(set(selected_tasks) - set(STEM_TASKS))
    if unknown_tasks:
        raise typer.BadParameter(
            f"Unsupported MMLU STEM task(s): {', '.join(unknown_tasks)}"
        )
    if mode not in {"generate", "pair", "all"}:
        raise typer.BadParameter('mode must be "generate", "pair", or "all"')

    output_base = _project_path(project_root, output_dir)
    model_output_dir = output_base / generated_model

    if mode in {"generate", "all"}:
        model_config = _cli().model_config_from_env()
        model_id = _cli().resolve_model_id(model_config.model_name)
        expected_slug = _cli().qwen_model_slug(model_id)
        if generated_model != expected_slug:
            raise typer.BadParameter(
                f"generated-model must match loaded model slug {expected_slug!r}."
            )
        bundle = _cli().load_model_and_processor(model_config)
        generation_summary = _cli().generate_sycophancy_dataset(
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
        pairing_summary = _cli().pair_sycophancy_dataset(
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
