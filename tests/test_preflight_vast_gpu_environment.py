from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_preflight_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "preflight_vast_gpu_environment.py"
    spec = importlib.util.spec_from_file_location("preflight_vast_gpu_environment_for_tests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    project_root = tmp_path / "workspace" / "intelligent_liars"
    project_root.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname='intelligent-liars'\n")
    defaults = {
        "project_root": project_root,
        "dev_shm": tmp_path / "dev_shm",
        "workspace": tmp_path / "workspace",
        "expected_gpus": 8,
        "min_gpu_memory_mib": 80_000,
        "min_dev_shm_free_gb": 120.0,
        "min_workspace_free_gb": 5.0,
        "require_env_file": False,
    }
    defaults["dev_shm"].mkdir(parents=True)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _a100s(count: int = 8):
    return [
        {
            "index": idx,
            "name": "NVIDIA A100-SXM4-80GB",
            "memory_total_mib": 81_920,
            "memory_free_mib": 80_000,
            "utilization_gpu_percent": 0,
        }
        for idx in range(count)
    ]


def test_preflight_accepts_clean_8xa100_environment(tmp_path, monkeypatch):
    preflight_module = _load_preflight_module()
    args = _args(tmp_path)
    monkeypatch.setenv("HF_HOME", str(args.dev_shm / "hf_home"))
    monkeypatch.setenv("TMPDIR", str(args.dev_shm / "tmp"))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    monkeypatch.setattr(preflight_module, "disk_free_gb", lambda path: 500.0)
    monkeypatch.setattr(preflight_module, "nvidia_smi_query", lambda: (_a100s(), None))

    report = preflight_module.preflight(args)

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["gpus"][0]["memory_total_mib"] == 81_920


def test_preflight_rejects_hf_home_outside_dev_shm(tmp_path, monkeypatch):
    preflight_module = _load_preflight_module()
    args = _args(tmp_path)
    monkeypatch.setenv("HF_HOME", str(args.workspace / "hf_home"))
    monkeypatch.setenv("TMPDIR", str(args.dev_shm / "tmp"))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    monkeypatch.setattr(preflight_module, "disk_free_gb", lambda path: 500.0)
    monkeypatch.setattr(preflight_module, "nvidia_smi_query", lambda: (_a100s(), None))

    report = preflight_module.preflight(args)

    assert report["ok"] is False
    assert any("HF_HOME must be under" in error for error in report["errors"])


def test_preflight_rejects_low_dev_shm_space(tmp_path, monkeypatch):
    preflight_module = _load_preflight_module()
    args = _args(tmp_path)
    monkeypatch.setenv("HF_HOME", str(args.dev_shm / "hf_home"))
    monkeypatch.setenv("TMPDIR", str(args.dev_shm / "tmp"))
    monkeypatch.setattr(
        preflight_module,
        "disk_free_gb",
        lambda path: 20.0 if path == args.dev_shm.resolve() else 500.0,
    )
    monkeypatch.setattr(preflight_module, "nvidia_smi_query", lambda: (_a100s(), None))

    report = preflight_module.preflight(args)

    assert report["ok"] is False
    assert any("need at least 120.0 GiB" in error for error in report["errors"])


def test_preflight_rejects_missing_gpus(tmp_path, monkeypatch):
    preflight_module = _load_preflight_module()
    args = _args(tmp_path)
    monkeypatch.setenv("HF_HOME", str(args.dev_shm / "hf_home"))
    monkeypatch.setenv("TMPDIR", str(args.dev_shm / "tmp"))
    monkeypatch.setattr(preflight_module, "disk_free_gb", lambda path: 500.0)
    monkeypatch.setattr(preflight_module, "nvidia_smi_query", lambda: (_a100s(4), None))

    report = preflight_module.preflight(args)

    assert report["ok"] is False
    assert "found 4 GPUs; expected at least 8" in report["errors"]
