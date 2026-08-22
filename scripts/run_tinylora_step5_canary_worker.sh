#!/usr/bin/env bash
set -euo pipefail

runtime_image_digest="${1:?runtime image digest is required}"
if [[ ! "$runtime_image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "runtime image digest must be immutable" >&2
  exit 2
fi

PYTHONPATH=src python scripts/hydrate_tinylora_step5_inputs.py \
  --url-manifest-url-file /run/secrets/step5-input-url-manifest-url \
  --inputs-dir /workspace/inputs \
  --cache-dir /workspace/cache/huggingface \
  --receipt /workspace/controller/input_hydration_receipt.json \
  --expected-frozen-inputs-archive-sha256 09009258f4b2422aa12e547568536d9f5e15d14afc64e23ce8f86c33d1000439 \
  --expected-frozen-inputs-completion-sha256 579664fd8e59f8d81f0c6d71b469d5f1011811100e981887df02af53e59f4882 \
  --expected-model-completion-sha256 4ccd7c940e1a72800d0153e809edab69e91c7cdd4d79627a1bb98076a0ef7275 \
  --expected-model-content-sha256 bbca6a8b09a56f0c538887b82b9594b0c0945c5fbcde54f39eda153a9f64eda8 \
  --expected-model-manifest-sha256 6ec207494fed0658ebaf31f1083f77110caae45f8c2392354e767ef54c78dd07 \
  --expected-model-revision 92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b \
  --expected-pixmo-archive-sha256 284ec9d7d1f3e4833f8e71e6028e5f7d0ce039d2749742b220c488d2de2e6d55 \
  --expected-pixmo-completion-sha256 a68b54be7e5a46014d8f6c97e8e7e1e527898c02c9e0b2542f7129ba72d9f876 \
  --expected-pixmo-content-sha256 430de1b25babb4fcd462ed7cf0476bce9f8c4e2b8fc3872c843eea332c1e56bc \
  --expected-pixmo-manifest-sha256 b1bcd5034e3751a58d531ef2897bfa49b8da0a3c848a9bb976e2f08f2ad19226 \
  --expected-plan-sha256 5282aaf0696098970de476e6812534e4c75c2434d7ec369495f5a311d6c09f99 \
  --expected-probe-qualification-file-sha256 6f409af64387a2a2ee1ca7f21e7fcdc7fbad1eeffa22672b8872277f45937ff2 \
  --expected-probe-qualification-receipt-sha256 f21781fdadab2eab6773d3e324d7500132e1f5f9e4bb38696c50837a07693b54

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python scripts/run_tinylora_step5_screen.py \
  --plan /workspace/inputs/step5_v1/manifest.json \
  --probe /workspace/inputs/probes/step5_grouped_ensemble_v1/probes/legacy-grouped-regularizer.json \
  --pixmo-bundle /workspace/inputs/pixmo \
  --arm tinylora_dim13 \
  --output-dir /workspace/runs/step5-canary-v1/tinylora_dim13 \
  --mode prerequisites \
  --cache-dir /workspace/cache/huggingface \
  --max-steps 20 \
  --max-length 2048 \
  --learning-rate 0.0002 \
  --gradient-accumulation 8 \
  --checkpoint-every 5 \
  --checkpoint-minutes 5 \
  --development-per-objective 0 \
  --seed 20260822 \
  --projection-seed 42 \
  --runtime-image-digest "$runtime_image_digest" \
  --durability-controller-public-key /workspace/inputs/controller-public.pem \
  --durability-verifier-command "python scripts/publish_step5_checkpoint_worker.py --exchange-dir /workspace/checkpoint-bridge --controller-public-key /workspace/inputs/controller-public.pem --timeout-seconds 1200" \
  > /workspace/controller/worker.log 2>&1

PYTHONPATH=src python scripts/finalize_tinylora_step5_artifacts.py \
  --artifact-root /workspace/artifacts/step5-canary-v1 \
  --expected-inventory /workspace/workload/configs/tinylora_step5_canary_expected_artifacts_v1.json \
  --run-id step5-canary-v1 \
  --durable-uri s3://intelligent-liars-osworld-29a36b2b927f4078b2ac95ca83ec9c21/step5-artifacts/v2/step5-canary-v1/attempt-1/artifacts.tar \
  --archive-path /workspace/artifacts/step5-canary-v1.tar \
  --presigned-put-url-file /run/secrets/step5-artifact-put-url \
  --file result.json=/workspace/runs/step5-canary-v1/tinylora_dim13/result.json \
  --file prerequisite_receipt.json=/workspace/runs/step5-canary-v1/tinylora_dim13/prerequisite_receipt.json \
  --file tinylora_dim13_basis.pt=/workspace/runs/step5-canary-v1/tinylora_dim13/tinylora_dim13_basis.pt \
  --tree-archive checkpoint_store.tar=/workspace/runs/step5-canary-v1/tinylora_dim13/checkpoint_store \
  --file host_qualification.json=/workspace/controller/host_qualification.json \
  --file input_hydration_receipt.json=/workspace/controller/input_hydration_receipt.json \
  --file source_bundle_receipt.json=/workspace/workload/SOURCE_BUNDLE_MANIFEST.json \
  --file worker.log=/workspace/controller/worker.log \
  --generate-canary-summary canary_summary.json
