#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

expected_uv_version="0.6.6"
actual_uv_version="$(uv --version | awk '{print $2}')"
if [[ "$actual_uv_version" != "$expected_uv_version" ]]; then
  echo "expected uv $expected_uv_version, got $actual_uv_version" >&2
  exit 2
fi

# Torch, Triton, and CUDA libraries are provided by the digest-pinned base.
# Excluding them prevents a PyPI CUDA family from overlaying that known stack.
uv pip compile requirements.in \
  --python-version 3.11 \
  --python-platform x86_64-manylinux_2_17 \
  --index-url https://pypi.org/simple \
  --no-emit-package torch \
  --no-emit-package triton \
  --no-emit-package nvidia-cublas-cu12 \
  --no-emit-package nvidia-cuda-cupti-cu12 \
  --no-emit-package nvidia-cuda-nvrtc-cu12 \
  --no-emit-package nvidia-cuda-runtime-cu12 \
  --no-emit-package nvidia-cudnn-cu12 \
  --no-emit-package nvidia-cufft-cu12 \
  --no-emit-package nvidia-curand-cu12 \
  --no-emit-package nvidia-cusolver-cu12 \
  --no-emit-package nvidia-cusparse-cu12 \
  --no-emit-package nvidia-nccl-cu12 \
  --no-emit-package nvidia-nvjitlink-cu12 \
  --no-emit-package nvidia-nvtx-cu12 \
  --generate-hashes \
  --output-file requirements.lock \
  --custom-compile-command "docker/truth-editing/compile_lock.sh"
