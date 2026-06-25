from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest


def _load_finish_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "render_vast_insider_finish.py"
    spec = importlib.util.spec_from_file_location("render_vast_insider_finish_for_tests", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_finish_script_contains_generation_grade_promote_extract_sequence():
    renderer = _load_finish_module()
    script = renderer.render_finish_script(renderer.FinishConfig())

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert "scripts/preflight_vast_gpu_environment.py" in script
    assert "scripts/run_insider_generation_dynamic.py audit" in script
    assert "finish-state assertion ok" in script
    assert "scripts/validate_insider_generation_json.py \"$S20_JSON\"" in script
    assert "--min-explicit 0" in script
    assert "--min-concealed 0" in script
    assert "src/intelligent_liars/cli.py grade-insider-trading" in script
    assert "--no-structured-outputs" in script
    assert "--no-require-structured-outputs" in script
    assert "--promote-to \"$PROMOTED_JSON\"" in script
    assert "src/intelligent_liars/cli.py extract-activations" in script
    assert "--task insider_trading__upscale" in script
    assert "scripts/validate_activation_hdf5.py \"$ACTIVATION_H5\"" in script


def test_finish_script_orders_promotion_before_extraction():
    renderer = _load_finish_module()
    script = renderer.render_finish_script(renderer.FinishConfig())

    assert script.index("--promote-to \"$PROMOTED_JSON\"") < script.index("src/intelligent_liars/cli.py extract-activations")
    assert script.index("src/intelligent_liars/cli.py grade-insider-trading") < script.index("--promote-to \"$PROMOTED_JSON\"")


def test_finish_script_refuses_existing_activation_by_default_and_never_trains_probes():
    renderer = _load_finish_module()
    script = renderer.render_finish_script(renderer.FinishConfig())

    assert "activation HDF5 already exists" in script
    assert "refusing to overwrite" in script
    assert "--overwrite-activation" in script
    assert "train-probes" not in script


def test_finish_script_can_render_intentional_activation_overwrite():
    renderer = _load_finish_module()
    script = renderer.render_finish_script(renderer.FinishConfig(overwrite_activation=True))

    assert "activation HDF5 already exists" not in script
    assert "--compression lzf --overwrite" in script


def test_write_output_is_atomic_and_executable(tmp_path):
    renderer = _load_finish_module()
    path = tmp_path / "finish.sh"

    renderer.write_output(path, "#!/usr/bin/env bash\necho ok\n", overwrite=False)

    assert path.read_text().startswith("#!/usr/bin/env bash")
    assert path.stat().st_mode & stat.S_IXUSR
    with pytest.raises(FileExistsError):
        renderer.write_output(path, "echo replaced\n", overwrite=False)
    renderer.write_output(path, "echo replaced\n", overwrite=True)
    assert path.read_text() == "echo replaced\n"
