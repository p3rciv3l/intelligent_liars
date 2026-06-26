from __future__ import annotations

import typer

from intelligent_liars.cli_common import app, console


def _cli():
    from intelligent_liars import cli

    return cli


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

    _cli().check_environment(
        console=console,
        require_cuda=require_cuda,
        check_processor=check_processor,
        check_model=check_model,
        check_nnsight=check_nnsight,
        check_openrouter=check_openrouter,
        nnsight_layer=nnsight_layer,
    )
