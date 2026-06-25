#!/usr/bin/env python3
"""Render a guarded remote resume script for the insider-upscale s20 run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import textwrap
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GPUS = "0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7"
DEFAULT_PROMPT_GLOB = "data/insider_trading/prompts/**/*.yaml"
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"


@dataclass(frozen=True)
class ResumeConfig:
    project_root: str = "/workspace/intelligent_liars"
    run_dir: str = "/workspace/intelligent_liars/artifacts/insider_upscale_dynamic_20260624_s20_run"
    output: str = "/dev/shm/intelligent_liars/data/insider_trading/qwen3-vl-8b-thinking-generations-s20.json"
    prompt_glob: str = DEFAULT_PROMPT_GLOB
    samples_per_prompt: int = 20
    label_mode: str = "unknown"
    gpus: str = DEFAULT_GPUS
    session_name: str = "insider_upscale_s20_resume"
    expected_gpus: int = 8
    min_gpu_memory_mib: int = 80_000
    min_dev_shm_free_gb: float = 120.0
    expected_planned: int = 520
    expected_done: int = 406
    expected_pending: int = 114
    expected_running: int = 0
    expected_failed: int = 0
    expected_outputs: int = 406
    expected_valid_outputs: int = 406
    require_count: int = 520
    poll_seconds: int = 30
    hf_home: str = "/dev/shm/hf_home"
    tmpdir: str = "/dev/shm/tmp"
    model_id: str = DEFAULT_MODEL_ID
    audit_json: str = "/dev/shm/intelligent_liars/logs/insider_s20_resume_audit.json"
    log_path: str = "/dev/shm/intelligent_liars/logs/insider_s20_resume_supervisor.log"


def q(value: object) -> str:
    return shlex.quote(str(value))


def expected_audit(config: ResumeConfig) -> dict[str, int]:
    return {
        "planned": config.expected_planned,
        "pending": config.expected_pending,
        "running": config.expected_running,
        "done": config.expected_done,
        "failed": config.expected_failed,
        "outputs": config.expected_outputs,
        "valid_outputs": config.expected_valid_outputs,
    }


def render_supervisor_command(config: ResumeConfig) -> str:
    return textwrap.dedent(
        f"""
        set -euo pipefail
        cd {q(config.project_root)}
        export HF_HOME={q(config.hf_home)}
        export TMPDIR={q(config.tmpdir)}
        export HF_HUB_DISABLE_XET=1
        export PYTHONPATH=src
        uv run --no-sync python scripts/run_insider_generation_dynamic.py supervisor \\
          --project-root {q(config.project_root)} \\
          --run-dir {q(config.run_dir)} \\
          --prompt-glob {q(config.prompt_glob)} \\
          --samples-per-prompt {config.samples_per_prompt} \\
          --label-mode {q(config.label_mode)} \\
          --output {q(config.output)} \\
          --gpus {q(config.gpus)} \\
          --require-count {config.require_count} \\
          --poll-seconds {config.poll_seconds} \\
          2>&1 | tee {q(config.log_path)}
        """
    ).strip()


def render_resume_script(config: ResumeConfig) -> str:
    expected_json = json.dumps(expected_audit(config), sort_keys=True)
    supervisor_placeholder = "__SUPERVISOR_COMMAND__"
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
        OUT_PATH={q(config.output)}
        AUDIT_JSON={q(config.audit_json)}
        LOG_PATH={q(config.log_path)}
        SESSION={q(config.session_name)}

        mkdir -p "$HF_HOME" "$TMPDIR" "$(dirname "$OUT_PATH")" "$(dirname "$AUDIT_JSON")" "$(dirname "$LOG_PATH")"

        if [ ! -d "$RUN_DIR" ]; then
          echo "missing run dir: $RUN_DIR" >&2
          echo "copy the saved local snapshot back to this path before resuming:" >&2
          echo "  artifacts/run_snapshots/insider_upscale_dynamic_20260624_s20_run/" >&2
          exit 1
        fi

        if tmux has-session -t "$SESSION" 2>/dev/null; then
          echo "tmux session already exists: $SESSION" >&2
          echo "refusing to launch a second writer for the same queue" >&2
          exit 1
        fi

        echo "[1/5] remote environment preflight"
        uv run --no-sync python scripts/preflight_vast_gpu_environment.py \\
          --project-root {q(config.project_root)} \\
          --expected-gpus {config.expected_gpus} \\
          --min-gpu-memory-mib {config.min_gpu_memory_mib} \\
          --min-dev-shm-free-gb {config.min_dev_shm_free_gb} \\
          --require-env-file

        echo "[2/5] read-only queue audit"
        uv run --no-sync python scripts/run_insider_generation_dynamic.py audit \\
          --project-root {q(config.project_root)} \\
          --run-dir "$RUN_DIR" \\
          --prompt-glob {q(config.prompt_glob)} \\
          --samples-per-prompt {config.samples_per_prompt} \\
          --label-mode {q(config.label_mode)} | tee "$AUDIT_JSON"

        echo "[3/5] assert expected resume state"
        python - "$AUDIT_JSON" {q(expected_json)} <<'PY'
        import json
        import sys

        audit_path = sys.argv[1]
        expected = json.loads(sys.argv[2])
        report = json.load(open(audit_path))
        state = report.get("state", {{}})
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
            print("resume-state assertion failed:", file=sys.stderr)
            for error in errors:
                print(f"- {{error}}", file=sys.stderr)
            sys.exit(1)
        print("resume-state assertion ok")
        PY

        echo "[4/5] prefetch model into HF_HOME=$HF_HOME"
        uv run --no-sync python - <<'PY'
        from huggingface_hub import snapshot_download
        snapshot_download({config.model_id!r})
        print("PREFETCH_DONE")
        PY

        echo "[5/5] launch guarded resume in tmux: $SESSION"
        tmux new-session -d -s "$SESSION" {supervisor_placeholder}

        echo "launched: $SESSION"
        echo "monitor:"
        echo "  tmux attach -t $SESSION"
        echo "  tail -f $LOG_PATH"
        echo "  watch -n 30 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'"
        """
    )
    return script.replace(supervisor_placeholder, q(render_supervisor_command(config)))


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
        description="Render a strict remote bash script for resuming the insider-upscale s20 queue."
    )
    parser.add_argument("--project-root", default=ResumeConfig.project_root)
    parser.add_argument("--run-dir", default=ResumeConfig.run_dir)
    parser.add_argument("--output-json", dest="output", default=ResumeConfig.output)
    parser.add_argument("--prompt-glob", default=DEFAULT_PROMPT_GLOB)
    parser.add_argument("--samples-per-prompt", type=int, default=20)
    parser.add_argument("--label-mode", default="unknown")
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--session-name", default=ResumeConfig.session_name)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--min-gpu-memory-mib", type=int, default=80_000)
    parser.add_argument("--min-dev-shm-free-gb", type=float, default=120.0)
    parser.add_argument("--expected-planned", type=int, default=520)
    parser.add_argument("--expected-done", type=int, default=406)
    parser.add_argument("--expected-pending", type=int, default=114)
    parser.add_argument("--expected-running", type=int, default=0)
    parser.add_argument("--expected-failed", type=int, default=0)
    parser.add_argument("--expected-outputs", type=int, default=406)
    parser.add_argument("--expected-valid-outputs", type=int, default=406)
    parser.add_argument("--require-count", type=int, default=520)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--hf-home", default=ResumeConfig.hf_home)
    parser.add_argument("--tmpdir", default=ResumeConfig.tmpdir)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--audit-json", default=ResumeConfig.audit_json)
    parser.add_argument("--log-path", default=ResumeConfig.log_path)
    parser.add_argument("--write", type=Path, help="Write the rendered script to this path instead of stdout.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ResumeConfig:
    return ResumeConfig(
        project_root=args.project_root,
        run_dir=args.run_dir,
        output=args.output,
        prompt_glob=args.prompt_glob,
        samples_per_prompt=args.samples_per_prompt,
        label_mode=args.label_mode,
        gpus=args.gpus,
        session_name=args.session_name,
        expected_gpus=args.expected_gpus,
        min_gpu_memory_mib=args.min_gpu_memory_mib,
        min_dev_shm_free_gb=args.min_dev_shm_free_gb,
        expected_planned=args.expected_planned,
        expected_done=args.expected_done,
        expected_pending=args.expected_pending,
        expected_running=args.expected_running,
        expected_failed=args.expected_failed,
        expected_outputs=args.expected_outputs,
        expected_valid_outputs=args.expected_valid_outputs,
        require_count=args.require_count,
        poll_seconds=args.poll_seconds,
        hf_home=args.hf_home,
        tmpdir=args.tmpdir,
        model_id=args.model_id,
        audit_json=args.audit_json,
        log_path=args.log_path,
    )


def main() -> int:
    args = parse_args()
    script = render_resume_script(config_from_args(args))
    if args.write is not None:
        write_output(args.write, script, overwrite=args.overwrite)
        print(args.write)
    else:
        print(script, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
