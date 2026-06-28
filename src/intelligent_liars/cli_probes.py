from __future__ import annotations

from pathlib import Path

import typer

from intelligent_liars.cli_common import (
    app,
    console,
    _project_path,
    _resolve_project_root,
)


def _cli():
    from intelligent_liars import cli

    return cli


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
    random_seed: int = typer.Option(
        0, help="Random seed for planned stratified example splits."
    ),
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
    summary = _cli().preflight_probe_training(
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
    random_seed: int = typer.Option(
        0, help="Random seed for stratified example splits and logistic regression."
    ),
    max_iter: int = typer.Option(
        1000, min=1, help="Maximum logistic-regression iterations."
    ),
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
    summary = _cli().train_probe_directions(
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


@app.command("cache-pooled-features")
def cache_pooled_features(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Raw activation HDF5 path produced by extract-activations.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output HDF5 path for pooled feature cache.",
    ),
    tasks: list[str] | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Activation task to cache. Repeatable. Defaults to every task in the HDF5.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root. Relative input/output paths resolve under this root.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        "-l",
        help='Decoder layers to cache: "all", comma list, or range.',
    ),
    compression: str | None = typer.Option(
        "lzf",
        help="HDF5 compression for cached feature datasets. Use empty string to disable.",
    ),
) -> None:
    """Build an HDF5 cache of mean-pooled per-example features."""

    project_root = _resolve_project_root(project_root)
    summary = _cli().build_pooled_feature_cache(
        input_path=_project_path(project_root, input_path),
        output_path=_project_path(project_root, output_path),
        tasks=tasks,
        layers=layers,
        compression=compression or None,
    )
    console.print(
        "[green]Built pooled feature cache[/green] "
        f"tasks={list(summary.tasks)} "
        f"layers={list(summary.layers)} "
        f"hidden_dim={summary.hidden_dim} "
        f"datasets={summary.feature_datasets} "
        f"path={summary.output_path}"
    )


@app.command("train-probes-from-cache")
def train_probes_from_cache(
    cache_path: Path = typer.Option(
        ...,
        "--cache",
        "-c",
        help="Pooled feature cache HDF5 path produced by cache-pooled-features.",
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
        help="Cached task to train/evaluate. Repeatable. Defaults to every task in the cache.",
    ),
    project_root: Path | None = typer.Option(
        None,
        help="Project root. Relative cache/output paths resolve under this root.",
    ),
    layers: str = typer.Option(
        "all",
        "--layers",
        "-l",
        help='Cached decoder layers to train on: "all", comma list, or range.',
    ),
    source_path: Path | None = typer.Option(
        None,
        "--source",
        help="Optional raw source HDF5 path to verify against cache provenance.",
    ),
    test_size: float = typer.Option(
        0.25,
        min=0.01,
        max=0.99,
        help="Held-out fraction for each within-task example split.",
    ),
    random_seed: int = typer.Option(
        0, help="Random seed for stratified example splits and logistic regression."
    ),
    max_iter: int = typer.Option(
        1000, min=1, help="Maximum logistic-regression iterations."
    ),
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
    """Train simple linear probes from a pooled-feature cache."""

    project_root = _resolve_project_root(project_root)
    summary = _cli().train_probe_directions_from_cache(
        cache_path=_project_path(project_root, cache_path),
        output_path=_project_path(project_root, output_path),
        tasks=tasks,
        layers=layers,
        source_path=_project_path(project_root, source_path) if source_path is not None else None,
        test_size=test_size,
        random_seed=random_seed,
        max_iter=max_iter,
        regularization_c=regularization_c,
        train_general_domain_probe=train_general_domain_probe,
        general_task_class_cap=general_task_class_cap,
    )
    console.print(
        "[green]Trained probes from cache[/green] "
        f"tasks={list(summary.tasks)} "
        f"layers={list(summary.layers)} "
        f"within_task={summary.within_task_results} "
        f"cross_task={summary.cross_task_results} "
        f"directions={summary.direction_results} "
        f"general_domain={summary.general_domain_results} "
        f"general_domain_directions={summary.general_domain_direction_results} "
        f"path={summary.output_path}"
    )
