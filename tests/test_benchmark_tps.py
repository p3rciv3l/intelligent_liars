from __future__ import annotations

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from intelligent_liars.models import ModelBundle, ModelLoadConfig
from intelligent_liars.rollouts import GenerationSettings
_SPEC = importlib.util.spec_from_file_location("benchmark_tps", Path(__file__).parents[1] / "scripts/benchmark_tps.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
benchmark_bundle = _MODULE.benchmark_bundle


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(str(m["content"]) for m in messages)

    def __call__(self, *, text, padding, return_tensors):
        widths = [len(item.split()) + 1 for item in text]
        width = max(widths)
        ids = torch.zeros((len(text), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, size in enumerate(widths):
            mask[row, :size] = 1
        return {"input_ids": ids, "attention_mask": mask}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.config = SimpleNamespace(_attn_implementation="sdpa")

    def generate(self, input_ids, attention_mask, **kwargs):
        count = int(kwargs["max_new_tokens"])
        suffix = torch.full((input_ids.shape[0], count), 1, dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=1)


def bundle():
    return ModelBundle(Model(), Processor(), None, "fixture", ModelLoadConfig())


def test_reports_prompt_generation_rates_and_runtime_metadata():
    result = benchmark_bundle(
        bundle(),
        [[{"role": "user", "content": "one two"}], [{"role": "user", "content": "one"}]],
        settings=GenerationSettings(batch_size=2, max_new_tokens=3, do_sample=False),
        warmup=0,
        repeats=2,
    )
    assert result.batch_size == 2
    assert result.prompt_tokens > 0  # prompt token count is measured twice
    assert result.generation_tokens == 12
    assert result.prompt_tokens_per_second > 0
    assert result.generation_tokens_per_second > 0
    assert result.dtype == "torch.float32"
    assert result.attention_backend == "sdpa"
    assert result.bottleneck in {"prompt_prefill", "autoregressive_decode", "balanced_or_input_overhead"}


def test_rejects_processor_only_bundle():
    empty = SimpleNamespace(model=None, processor=Processor())
    with pytest.raises(ValueError, match="loaded bundle.model"):
        benchmark_bundle(empty, [[{"role": "user", "content": "hello"}]])


def test_rejects_batch_mismatch():
    with pytest.raises(ValueError, match="batch_size"):
        benchmark_bundle(
            bundle(),
            [[{"role": "user", "content": "hello"}]],
            settings=GenerationSettings(batch_size=2, max_new_tokens=1, do_sample=False),
        )
