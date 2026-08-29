#!/usr/bin/env python3
"""Small, local-only tokens-per-second benchmark for the rollout model path.

The harness accepts an already-loaded ``ModelBundle`` so callers can reuse the
production processor and ``model.generate`` path without downloading weights,
contacting a service, or changing model state.  The CLI is intentionally a
fixture-free library entry point; a caller supplies a bundle from its worker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intelligent_liars.rollouts import (  # noqa: E402
    GenerationSettings,
    Message,
    _generation_kwargs,
    _model_input_device,
    _render_generation_conversation,
)


@dataclass(frozen=True)
class TPSBenchmarkResult:
    """Measured rates and runtime metadata; no performance claim is inferred."""

    batch_size: int
    repeats: int
    prompt_tokens: int
    generation_tokens: int
    elapsed_seconds: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    dtype: str
    attention_backend: str
    bottleneck: str


def _runtime_metadata(model: Any) -> tuple[str, str]:
    dtype = "unknown"
    try:
        dtype = str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration, TypeError):
        pass
    config = getattr(model, "config", None)
    attention = "unknown"
    for name in ("_attn_implementation", "attn_implementation", "attention_implementation"):
        value = getattr(config, name, None)
        if value:
            attention = str(value)
            break
    return dtype, attention


def _sync(model: Any) -> None:
    """Synchronize CUDA when available, while keeping CPU/fake models simple."""
    try:
        import torch

        device = _model_input_device(model)
        if getattr(device, "type", str(device)) == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except (ImportError, AttributeError, RuntimeError):
        return


def benchmark_bundle(
    bundle: Any,
    conversations: Sequence[Sequence[Message]],
    *,
    settings: GenerationSettings | None = None,
    warmup: int = 1,
    repeats: int = 3,
) -> TPSBenchmarkResult:
    """Benchmark the existing processor + ``model.generate`` path offline.

    Prompt and generated rates use the same wall-clock interval.  Thus the
    prompt rate describes prompt-token throughput during generation (prefill
    plus decode), and the ``bottleneck`` field is a diagnostic hint, not a
    profiler result.
    """
    if bundle.model is None:
        raise ValueError("benchmark requires a loaded bundle.model")
    if not conversations:
        raise ValueError("conversations must not be empty")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    settings = settings or GenerationSettings(batch_size=len(conversations), max_new_tokens=16, do_sample=False)
    if settings.batch_size != len(conversations):
        raise ValueError("settings.batch_size must equal len(conversations)")

    rendered = [_render_generation_conversation(bundle.processor, c) for c in conversations]
    inputs = bundle.processor(text=rendered, padding=True, return_tensors="pt")
    device = _model_input_device(bundle.model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    prompt_tokens = int(inputs["attention_mask"].sum().item()) if "attention_mask" in inputs else int(inputs["input_ids"].numel())
    kwargs = _generation_kwargs(settings)
    for _ in range(warmup):
        with _inference_context():
            bundle.model.generate(**inputs, **kwargs)
        _sync(bundle.model)
    elapsed = 0.0
    generation_tokens = 0
    for _ in range(repeats):
        _sync(bundle.model)
        started = time.perf_counter()
        with _inference_context():
            outputs = bundle.model.generate(**inputs, **kwargs)
        _sync(bundle.model)
        elapsed += time.perf_counter() - started
        generation_tokens += max(0, int(outputs.shape[1]) - int(inputs["input_ids"].shape[1])) * len(conversations)
    elapsed = max(elapsed, 1e-12)
    prompt_tps = prompt_tokens * repeats / elapsed
    generation_tps = generation_tokens / elapsed
    if generation_tokens == 0:
        bottleneck = "no_generation_tokens"
    elif prompt_tps < generation_tps * 0.5:
        bottleneck = "prompt_prefill"
    elif generation_tps < prompt_tps * 0.5:
        bottleneck = "autoregressive_decode"
    else:
        bottleneck = "balanced_or_input_overhead"
    dtype, attention = _runtime_metadata(bundle.model)
    return TPSBenchmarkResult(settings.batch_size, repeats, prompt_tokens * repeats, generation_tokens, elapsed, prompt_tps, generation_tps, dtype, attention, bottleneck)


class _inference_context:
    def __enter__(self):
        try:
            import torch
            self._ctx = torch.inference_mode()
            return self._ctx.__enter__()
        except ImportError:
            self._ctx = None
            return self

    def __exit__(self, *args):
        return self._ctx.__exit__(*args) if self._ctx is not None else False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print JSON when called by a wrapper")
    parser.add_argument("--fixture", action="store_true", help="run a deterministic CPU smoke benchmark")
    args = parser.parse_args()
    if not args.fixture:
        parser.error("loading a model is deliberately owned by the worker; use --fixture or import benchmark_bundle()")
    import torch
    from types import SimpleNamespace

    class FixtureProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return " ".join(str(message["content"]) for message in messages)

        def __call__(self, *, text, padding, return_tensors):
            width = max(len(item.split()) + 1 for item in text)
            ids = torch.zeros((len(text), width), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    class FixtureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(_attn_implementation="cpu_fixture")

        def generate(self, input_ids, **kwargs):
            suffix = torch.zeros((input_ids.shape[0], kwargs["max_new_tokens"]), dtype=torch.long)
            return torch.cat((input_ids, suffix), dim=1)

    result = benchmark_bundle(
        SimpleNamespace(model=FixtureModel(), processor=FixtureProcessor()),
        [[{"role": "user", "content": "offline benchmark"}]],
        settings=GenerationSettings(batch_size=1, max_new_tokens=8, do_sample=False),
    )
    print(json.dumps(asdict(result), sort_keys=True) if args.json else asdict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
