# Truth-editing study runtime

This image is the Linux/CUDA environment for persistent truth-editing Optuna
studies. It deliberately keeps CUDA-only FlashAttention out of `pyproject.toml`,
so the ordinary project remains installable and testable on macOS CPU or MPS.

Local development:

```bash
uv sync --frozen --extra study
uv run --frozen --extra study pytest -q tests/test_truth_editing_study.py
```

Production image build:

```bash
docker build \
  --platform linux/amd64 \
  --file docker/truth-editing/Dockerfile \
  --tag truth-editing:cu124-torch251-transformers4571-fa274post1-optuna490-wandb0290-boto34018-r3 \
  docker/truth-editing
```

The source checkout is mounted at `/workspace/intelligent_liars`; it is not
copied into the dependency image. Run the full on-host validation before the
study process:

```bash
python /opt/truth-editing/validate_runtime.py
```

That validates the installed package identities, local construction of the
off-host S3 checkpoint client, CUDA 12.4, BF16 execution, the FlashAttention
CUDA kernel, compute capability, memory, and writable cache.
The image build itself runs the metadata and package checks without requiring a
GPU.

`requirements.lock` is generated from `requirements.in` using
`compile_lock.sh`. Torch, Triton, and NVIDIA runtime packages are intentionally
excluded because the digest-pinned base image supplies that coherent ABI.
