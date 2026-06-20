from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from intelligent_liars.models import (
    load_model_and_processor,
    load_processor,
    model_config_from_env,
    qwen_model_load_description,
    resolve_model_id,
)
from intelligent_liars.nnsight_backend import load_nnsight_bundle, trace_text_decoder_layer_once


PACKAGE_CHECKS = ("torch", "transformers", "accelerate", "flash-attn", "nnsight", "qwen-vl-utils")


def check_environment(
    *,
    console: Console,
    require_cuda: bool,
    check_processor: bool,
    check_model: bool,
    check_nnsight: bool,
    check_openrouter: bool,
    nnsight_layer: int,
) -> None:
    load_dotenv()
    model_config = model_config_from_env()
    model_id = resolve_model_id(model_config.model_name)

    table = Table(title="Intelligent Liars environment")
    table.add_column("Check")
    table.add_column("Value")

    rows = (
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("model_name", model_config.model_name),
        ("model_id", model_id),
        ("hf_home", model_config.cache_dir or "default"),
        ("hf_token", "set" if os.getenv("HF_TOKEN") else "not set"),
        ("model_load", qwen_model_load_description()),
        (
            "cuda_visible_devices",
            model_config.cuda_visible_devices or "default",
        ),
        *((package, _version(package)) for package in PACKAGE_CHECKS),
    )
    for name, value in rows:
        table.add_row(name, value)

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

    if (check_model or check_nnsight) and _version("flash-attn") == "not installed":
        raise typer.BadParameter(
            "flash-attn is required for Qwen model loading because attn_implementation is flash_attention_2. "
            "Install flash-attn on the CUDA box before --check-model or --check-nnsight."
        )

    if check_openrouter:
        from intelligent_liars.clients.openrouter_client import OpenRouterAPIError
        from intelligent_liars.judging import DEFAULT_JUDGE_CONFIG, preflight_judge_config

        console.print("[bold]Checking OpenRouter judge configuration...[/bold]")
        try:
            preflight = preflight_judge_config(DEFAULT_JUDGE_CONFIG)
        except (ValueError, FileNotFoundError, OpenRouterAPIError, RuntimeError) as exc:
            raise typer.BadParameter(f"OpenRouter judge preflight failed: {exc}") from exc
        console.print(
            "[green]OpenRouter judge preflight passed.[/green] "
            f"alias={preflight.alias} "
            f"resolved_model={preflight.resolved_model}"
        )

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


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
