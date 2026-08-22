#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Torch, Triton, and CUDA libraries come from the digest-pinned PyTorch base.
# Excluding the whole CUDA dependency family prevents PyPI's CUDA 12.4 wheels
# from silently overlaying the qualified CUDA 12.1 image.
uv pip compile requirements.in \
  --python-version 3.11 \
  --python-platform x86_64-manylinux_2_17 \
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
  --custom-compile-command "see docker/tinylora-step5/compile_lock.sh"
