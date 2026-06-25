from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest


def _load_renderer_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "render_vast_insider_resume.py"
    spec = importlib.util.spec_from_file_location("render_vast_insider_resume_for_tests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_resume_script_contains_preflight_audit_assertion_and_guarded_tmux():
    renderer = _load_renderer_module()
    config = renderer.ResumeConfig()

    script = renderer.render_resume_script(config)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "scripts/preflight_vast_gpu_environment.py" in script
    assert "scripts/run_insider_generation_dynamic.py audit" in script
    assert "resume-state assertion ok" in script
    assert "tmux has-session" in script
    assert "refusing to launch a second writer" in script
    assert "tmux new-session -d -s \"$SESSION\"" in script
    assert "--gpus 0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7" in script
    assert "--require-count 520" in script
    assert "snapshot_download('Qwen/Qwen3-VL-8B-Thinking')" in script


def test_render_resume_script_asserts_expected_406_114_state():
    renderer = _load_renderer_module()
    script = renderer.render_resume_script(renderer.ResumeConfig())

    assert '"done": 406' in script
    assert '"pending": 114' in script
    assert '"outputs": 406' in script
    assert '"valid_outputs": 406' in script


def test_render_resume_script_does_not_destroy_existing_queue():
    renderer = _load_renderer_module()
    script = renderer.render_resume_script(renderer.ResumeConfig())

    forbidden = ("rm -rf", "overwrite-queue", "clear_queue")
    assert all(text not in script for text in forbidden)


def test_write_output_is_atomic_and_executable(tmp_path):
    renderer = _load_renderer_module()
    path = tmp_path / "resume.sh"

    renderer.write_output(path, "#!/usr/bin/env bash\necho ok\n", overwrite=False)

    assert path.read_text().startswith("#!/usr/bin/env bash")
    assert path.stat().st_mode & stat.S_IXUSR
    with pytest.raises(FileExistsError):
        renderer.write_output(path, "echo replaced\n", overwrite=False)
    renderer.write_output(path, "echo replaced\n", overwrite=True)
    assert path.read_text() == "echo replaced\n"
