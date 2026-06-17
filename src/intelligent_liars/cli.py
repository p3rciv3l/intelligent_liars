from __future__ import annotations

import importlib.metadata
import os
import platform
import sys

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


app = typer.Typer(no_args_is_help=True)
console = Console()


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


@app.command("check-env")
def check_env(
    require_cuda: bool = typer.Option(False, help="Fail if CUDA is unavailable."),
    check_processor: bool = typer.Option(
        False,
        help="Download/load only the Hugging Face processor for MODEL_NAME.",
    ),
) -> None:
    """Print the runtime state needed before running activation experiments."""
    load_dotenv()

    table = Table(title="Intelligent Liars environment")
    table.add_column("Check")
    table.add_column("Value")

    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    table.add_row("model", os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Thinking"))
    table.add_row("hf_home", os.getenv("HF_HOME", "default"))
    table.add_row("hf_token", "set" if os.getenv("HF_TOKEN") else "not set")
    table.add_row("torch", _version("torch"))
    table.add_row("transformers", _version("transformers"))
    table.add_row("accelerate", _version("accelerate"))
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
        from transformers import AutoProcessor

        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Thinking")
        console.print(f"Loading processor for [bold]{model_name}[/bold]...")
        AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        console.print("[green]Processor loaded.[/green]")
