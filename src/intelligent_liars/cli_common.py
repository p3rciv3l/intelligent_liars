from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console

from intelligent_liars.judging import JudgeConfig, infer_rollout_task
from intelligent_liars.rollouts import GenerationSettings, parse_task_name


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
    """Build a `JudgeConfig` from Typer option values."""

    if provider != "openrouter":
        raise typer.BadParameter(
            'provider must be "openrouter". Use --judge-model for OpenRouter aliases or raw model IDs.'
        )
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


def _bind_judge_config_to_project_root(
    config: JudgeConfig, project_root: Path
) -> JudgeConfig:
    return replace(config, model_config_path=project_root / "model_deployments.yaml")


def _project_path(project_root: Path, path: Path) -> Path:
    """Resolve a CLI path relative to the project root when needed."""

    return path if path.is_absolute() else project_root / path


def _find_project_root(start: Path | None = None) -> Path:
    """Find the nearest checkout root from a start directory."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "intelligent_liars"
        ).exists():
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
    """Collect repeated generation CLI options into one settings object."""

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
            raise typer.BadParameter(
                "--source-limit is required when using --source-start."
            )
        start = source_start or 0
        selected.update(range(start, start + source_limit))
    if selected and (record_start != 0 or record_limit is not None):
        raise typer.BadParameter(
            "Use either record --start/--limit or source-index filters, not both."
        )
    return selected or None


def _validate_insider_prompt_glob(project_root: Path, prompt_glob: str) -> None:
    if not sorted(project_root.glob(prompt_glob)):
        raise typer.BadParameter(
            f"No insider-trading prompt YAML files matched {prompt_glob!r} under {project_root}"
        )
