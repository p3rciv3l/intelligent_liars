#!/usr/bin/env python3
"""Render a guarded remote finish script for insider-upscale activation extraction."""

from __future__ import annotations

import argparse
import os
import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT_GLOB = "data/insider_trading/prompts/**/*.yaml"
DEFAULT_EXPECTED_TASK = "insider_trading__upscale"


@dataclass(frozen=True)
class FinishConfig:
    project_root: str = "/workspace/intelligent_liars"
    run_dir: str = "/workspace/intelligent_liars/artifacts/insider_upscale_dynamic_20260624_s20_run"
    s20_json: str = "/dev/shm/intelligent_liars/data/insider_trading/qwen3-vl-8b-thinking-generations-s20.json"
    promoted_json: str = "/workspace/intelligent_liars/data/insider_trading/qwen3-vl-8b-thinking-generations.json"
    activation_h5: str = (
        "/dev/shm/intelligent_liars/artifacts/activations/"
        "activation_insider_upscale_20260624/"
        "extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5"
    )
    prompt_glob: str = DEFAULT_PROMPT_GLOB
    samples_per_prompt: int = 20
    expected_count: int = 520
    expected_task: str = DEFAULT_EXPECTED_TASK
    expected_gpus: int = 8
    min_gpu_memory_mib: int = 80_000
    min_dev_shm_free_gb: float = 120.0
    min_explicit: int = 1
    min_concealed: int = 1
    expected_layer_count: int = 36
    expected_hidden_dim: int = 4096
    finite_sample_rows: int = 64
    hf_home: str = "/dev/shm/hf_home"
    tmpdir: str = "/dev/shm/tmp"
    log_dir: str = "/dev/shm/intelligent_liars/logs"
    overwrite_activation: bool = False


def q(value: object) -> str:
    return shlex.quote(str(value))


def render_finish_script(config: FinishConfig) -> str:
    overwrite_extract = " --overwrite" if config.overwrite_activation else ""
    existing_h5_guard_placeholder = "__EXISTING_H5_GUARD__"
    existing_h5_guard = (
        ""
        if config.overwrite_activation
        else textwrap.dedent(
            """\
            if [ -e "$ACTIVATION_H5" ]; then
              echo "activation HDF5 already exists: $ACTIVATION_H5" >&2
              echo "refusing to overwrite; rerender with --overwrite-activation if replacement is intentional" >&2
              exit 1
            fi

            """
        )
    )
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        cd {q(config.project_root)}
        export HF_HOME={q(config.hf_home)}
        export TMPDIR={q(config.tmpdir)}
        export HF_HUB_DISABLE_XET=1
        export PYTHONPATH=src

        RUN_DIR={q(config.run_dir)}
        S20_JSON={q(config.s20_json)}
        PROMOTED_JSON={q(config.promoted_json)}
        ACTIVATION_H5={q(config.activation_h5)}
        LOG_DIR={q(config.log_dir)}

        mkdir -p "$HF_HOME" "$TMPDIR" "$LOG_DIR" "$(dirname "$PROMOTED_JSON")" "$(dirname "$ACTIVATION_H5")"

        if [ ! -d "$RUN_DIR" ]; then
          echo "missing run dir: $RUN_DIR" >&2
          exit 1
        fi
        if [ ! -s "$S20_JSON" ]; then
          echo "missing completed s20 JSON: $S20_JSON" >&2
          exit 1
        fi
        {existing_h5_guard_placeholder}echo "[1/6] remote environment preflight before activation extraction"
        uv run --no-sync python scripts/preflight_vast_gpu_environment.py \\
          --project-root {q(config.project_root)} \\
          --expected-gpus {config.expected_gpus} \\
          --min-gpu-memory-mib {config.min_gpu_memory_mib} \\
          --min-dev-shm-free-gb {config.min_dev_shm_free_gb} \\
          --require-env-file

        echo "[2/6] assert generation queue complete"
        uv run --no-sync python scripts/run_insider_generation_dynamic.py audit \\
          --project-root {q(config.project_root)} \\
          --run-dir "$RUN_DIR" \\
          --prompt-glob {q(config.prompt_glob)} \\
          --samples-per-prompt {config.samples_per_prompt} \\
          --label-mode unknown | tee "$LOG_DIR/insider_s20_finish_audit.json"

        python - "$LOG_DIR/insider_s20_finish_audit.json" <<'PY'
        import json
        import sys

        report = json.load(open(sys.argv[1]))
        state = report.get("state", {{}})
        expected = {{
            "planned": {config.expected_count},
            "pending": 0,
            "running": 0,
            "done": {config.expected_count},
            "failed": 0,
            "outputs": {config.expected_count},
            "valid_outputs": {config.expected_count},
        }}
        actual = {{
            "planned": report.get("planned"),
            "pending": state.get("pending"),
            "running": state.get("running"),
            "done": state.get("done"),
            "failed": state.get("failed"),
            "outputs": state.get("outputs"),
            "valid_outputs": report.get("valid_outputs"),
        }}
        errors = []
        if not report.get("ok"):
            errors.append(f"audit ok=false issues={{report.get('issues')}}")
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                errors.append(f"{{key}}={{actual.get(key)}} expected {{expected_value}}")
        if errors:
            print("finish-state assertion failed:", file=sys.stderr)
            for error in errors:
                print(f"- {{error}}", file=sys.stderr)
            sys.exit(1)
        print("finish-state assertion ok")
        PY

        echo "[3/6] validate complete ungraded s20 JSON before judge calls"
        uv run --no-sync python scripts/validate_insider_generation_json.py "$S20_JSON" \\
          --project-root {q(config.project_root)} \\
          --prompt-glob {q(config.prompt_glob)} \\
          --samples-per-prompt {config.samples_per_prompt} \\
          --expected-count {config.expected_count} \\
          --min-explicit 0 \\
          --min-concealed 0 | tee "$LOG_DIR/insider_s20_pregrade_validation.json"

        echo "[4/6] grade s20 insider JSON in place without structured outputs"
        uv run --no-sync python src/intelligent_liars/cli.py grade-insider-trading \\
          --project-root {q(config.project_root)} \\
          --input-path "$S20_JSON" \\
          --no-structured-outputs \\
          --no-require-structured-outputs \\
          2>&1 | tee "$LOG_DIR/insider_s20_grade.log"

        echo "[5/6] validate graded labels and promote canonical insider JSON"
        uv run --no-sync python scripts/validate_insider_generation_json.py "$S20_JSON" \\
          --project-root {q(config.project_root)} \\
          --prompt-glob {q(config.prompt_glob)} \\
          --samples-per-prompt {config.samples_per_prompt} \\
          --expected-count {config.expected_count} \\
          --require-graded \\
          --min-explicit {config.min_explicit} \\
          --min-concealed {config.min_concealed} \\
          --promote-to "$PROMOTED_JSON" \\
          --overwrite | tee "$LOG_DIR/insider_s20_graded_validation.json"

        echo "[6/6] extract and validate insider_upscale activations"
        uv run --no-sync python src/intelligent_liars/cli.py extract-activations \\
          --project-root {q(config.project_root)} \\
          --task {q(config.expected_task)} \\
          --output "$ACTIVATION_H5" \\
          --layers all \\
          --compression lzf{overwrite_extract} \\
          2>&1 | tee "$LOG_DIR/insider_upscale_activation_extract.log"

        uv run --no-sync python scripts/validate_activation_hdf5.py "$ACTIVATION_H5" \\
          --expected-task-count 1 \\
          --expected-tasks {q(config.expected_task)} \\
          --expected-layer-count {config.expected_layer_count} \\
          --expected-hidden-dim {config.expected_hidden_dim} \\
          --finite-check sample \\
          --finite-sample-rows {config.finite_sample_rows} \\
          --sha256 | tee "$LOG_DIR/insider_upscale_activation_validate.json"

        echo "INSIDER_UPSCALE_FINISH_DONE"
        echo "remote activation path: $ACTIVATION_H5"
        echo "local safe-fetch example:"
        echo "  PYTHONPATH=src uv run --no-sync python scripts/safe_fetch_artifact.py --ssh-target root@<host> --port <port> '$ACTIVATION_H5' artifacts/activations/activation_insider_upscale_20260624/extracted_feats_insider_upscale_qwen3-vl-8b-thinking.h5"
        """
    )
    return script.replace(existing_h5_guard_placeholder, existing_h5_guard)


def write_output(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(content)
    os.replace(tmp, path)
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a strict remote finish script for insider-upscale grading and activation extraction."
    )
    parser.add_argument("--project-root", default=FinishConfig.project_root)
    parser.add_argument("--run-dir", default=FinishConfig.run_dir)
    parser.add_argument("--s20-json", default=FinishConfig.s20_json)
    parser.add_argument("--promoted-json", default=FinishConfig.promoted_json)
    parser.add_argument("--activation-h5", default=FinishConfig.activation_h5)
    parser.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    parser.add_argument("--samples-per-prompt", type=int, default=20)
    parser.add_argument("--expected-count", type=int, default=520)
    parser.add_argument("--expected-task", default=DEFAULT_EXPECTED_TASK)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--min-gpu-memory-mib", type=int, default=80_000)
    parser.add_argument("--min-dev-shm-free-gb", type=float, default=120.0)
    parser.add_argument("--min-explicit", type=int, default=1)
    parser.add_argument("--min-concealed", type=int, default=1)
    parser.add_argument("--expected-layer-count", type=int, default=36)
    parser.add_argument("--expected-hidden-dim", type=int, default=4096)
    parser.add_argument("--finite-sample-rows", type=int, default=64)
    parser.add_argument("--hf-home", default=FinishConfig.hf_home)
    parser.add_argument("--tmpdir", default=FinishConfig.tmpdir)
    parser.add_argument("--log-dir", default=FinishConfig.log_dir)
    parser.add_argument("--overwrite-activation", action="store_true")
    parser.add_argument("--write", type=Path, help="Write the rendered script to this path instead of stdout.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> FinishConfig:
    return FinishConfig(
        project_root=args.project_root,
        run_dir=args.run_dir,
        s20_json=args.s20_json,
        promoted_json=args.promoted_json,
        activation_h5=args.activation_h5,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        expected_count=args.expected_count,
        expected_task=args.expected_task,
        expected_gpus=args.expected_gpus,
        min_gpu_memory_mib=args.min_gpu_memory_mib,
        min_dev_shm_free_gb=args.min_dev_shm_free_gb,
        min_explicit=args.min_explicit,
        min_concealed=args.min_concealed,
        expected_layer_count=args.expected_layer_count,
        expected_hidden_dim=args.expected_hidden_dim,
        finite_sample_rows=args.finite_sample_rows,
        hf_home=args.hf_home,
        tmpdir=args.tmpdir,
        log_dir=args.log_dir,
        overwrite_activation=args.overwrite_activation,
    )


def main() -> int:
    args = parse_args()
    script = render_finish_script(config_from_args(args))
    if args.write is not None:
        write_output(args.write, script, overwrite=args.overwrite)
        print(args.write)
    else:
        print(script, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
