# Truth-Direction Intervention Pipeline

The intervention pipeline separates a probe direction from the operation applied
relative to that direction. For probe vector `w`, intercept `b`, and activation
`h`, the signed coordinate is:

```text
z = wᵀh + b
```

Positive scores follow the repository convention and point from honest toward
deceptive activations.

## Supported Runtime Operators

| Method | Coordinate map |
|---|---|
| `scalar_addition` | `z' = z + delta` |
| `affine_projection` | `z' = target` |
| `full_reflection` | `z' = -z` |
| `partial_reflection` | `z' = (1 - 2 strength) z` |
| `one_sided_reflection` | Reflect only honest or deceptive scores |
| `bounded_remap` | Piecewise-linear bounded input/output remapping |
| `bounded_margin_clamp` | Minimum bounded adjustment needed to reach a signed margin |

Every map changes only the component parallel to `w`. Components orthogonal to
`w` are preserved. `max_score_delta` can bound any method's per-token movement
and is required for `bounded_margin_clamp`.

`last_token` is the default token scope. During prefill it changes only the last
prompt token, which lets the intervention affect the first generated token;
during cached decoding it changes the current state. `all` changes every prefill
and decode activation.

## Seeded Matched-Random Control

Every method can replace the probe direction with a deterministic random vector
that is orthogonal to the probe direction and has the same norm. Set
`--matched-random-seed` when building the intervention. The bundle stores the
seed and reconstructs the control direction deterministically.

## Build a Portable Intervention

After pulling the selected probe result from DVC:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars build-intervention \
  --probe artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json \
  --output artifacts/interventions/layer21_one_sided_reflection.json \
  --layer 21 \
  --intervention-layer 21 \
  --method one_sided_reflection \
  --selected-side honest \
  --reflection-strength 1
```

The JSON bundle includes the direction, intercept, sign convention, operator
settings, and source provenance. It does not require the probe artifact at
runtime.

Use it from Python:

```python
from pathlib import Path

from intelligent_liars.interventions import RuntimeIntervention, load_intervention_bundle
from intelligent_liars.models import load_model_and_processor

bundle = load_intervention_bundle(
    Path("artifacts/interventions/layer21_one_sided_reflection.json")
)
model_bundle = load_model_and_processor()

with RuntimeIntervention(model_bundle.model, bundle):
    outputs = model_bundle.model.generate(**inputs)
```

Hooks are removed when the context exits, including when generation raises.

## Create an Experimental Writer-Edited Checkpoint

Persistent rank-one writer edits are a separate experiment from the exact
runtime operators. Build an explicit all-token, zero-intercept bundle and apply
it to Qwen's selected attention-output and MLP-down matrices:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars build-intervention \
  --probe artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json \
  --output artifacts/interventions/layer21_homogeneous_writer_reflection.json \
  --layer 21 \
  --method full_reflection \
  --homogeneous-writer-edit

PYTHONPATH=src uv run --no-sync intelligent-liars materialize-writer-edit \
  --intervention artifacts/interventions/layer21_homogeneous_writer_reflection.json \
  --output artifacts/models/qwen3-vl-8b-thinking-layer21-reflected
```

The output is a normal Hugging Face checkpoint plus `intervention.json` and
`writer_edit_materialization.json` manifests. It can be loaded without this
repository's runtime hooks.

Only these homogeneous writer operations can currently be saved:

- zero-target projection through the origin;
- full writer reflection;
- partial writer reflection.

Scalar addition, non-zero affine projection, one-sided reflection, bounded
remapping, and margin clamping depend on the live activation and require runtime
hooks. Qwen writer projections are bias-free, so a non-zero affine intercept
cannot be represented exactly as a stock weight-only checkpoint.

Writer-space reflection and attenuation are not baked versions of the named
residual-state transforms: the incoming residual bypass is unchanged. The
current materializer intentionally edits only selected decoder layers'
attention-output and MLP-down writer matrices. Even zero-target projection is
equivalent to global directional ablation only when every residual-stream writer
is edited. The separate command name and manifest record this distinction.

The saved checkpoint remains BF16. Quantization, GPU sizing, and small-RTX
deployment are intentionally deferred and unvalidated in this pipeline.

## Required Evaluation Controls

Every run should retain:

- unmodified model baseline;
- seeded matched-random direction;
- identical decoding settings;
- probe-score traces before and after intervention;
- refusal, formatting, repetition, and invalid-answer rates;
- MMLU/GPQA accuracy and OSWorld/ALFWorld preservation metrics.

The current direction was trained on mean-pooled completed answer tokens. A
behavioral effect at generation time remains a causal hypothesis until the
runtime intervention is evaluated.

## Compile Interventions Into Standalone Qwen Models

Runtime hooks are only used offline to generate teacher completions. Each
teacher is then distilled into language-model LoRA weights, merged into the
base model, and saved as an ordinary Qwen3-VL checkpoint. The resulting model
loads with the stock Hugging Face Qwen class and requires no intervention code.

Build the standard seven-method suite plus a matched-random reflection control:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars build-intervention-suite \
  --probe artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json \
  --output artifacts/interventions/layer21_suite \
  --layer 21
```

Prepare a JSON prompt file. Either a top-level list or `{ "records": [...] }`
is accepted. Every row needs a stable `id` or `source_index` and `messages` or
`input_messages`:

```json
[
  {
    "id": "mmlu-0001",
    "messages": [
      {"role": "user", "content": "Question and answer choices..."}
    ]
  }
]
```

Generate one unmodified preservation teacher over a representative mix of
general, action-format, and OSWorld trajectory prompts:

```bash
REVISION=$(PYTHONPATH=src uv run --no-sync python -c \
  'from huggingface_hub import HfApi; print(HfApi().model_info("Qwen/Qwen3-VL-8B-Thinking").sha)')

PYTHONPATH=src uv run --no-sync intelligent-liars generate-intervention-teacher \
  --prompts examples/distillation/preservation_prompts.example.json \
  --output artifacts/intervention_fleet/preservation_teacher.json \
  --base-revision "$REVISION"
```

The checked-in `.example.json` files are schema-valid smoke inputs. Replace or
extend them with the full intervention set and held-out-from-evaluation OSWorld
trajectories before a scientific run.

Plan all standalone variants:

```bash
args=()
for path in artifacts/interventions/layer21_suite/*.json; do
  args+=(--intervention "$path")
done

PYTHONPATH=src uv run --no-sync intelligent-liars plan-intervention-model-fleet \
  "${args[@]}" \
  --prompts examples/distillation/intervention_prompts.example.json \
  --preservation-teacher artifacts/intervention_fleet/preservation_teacher.json \
  --output-root artifacts/intervention_fleet \
  --plan artifacts/intervention_fleet/plan.json \
  --base-revision "$REVISION"
```

Place the plan and output root on a filesystem shared by the workers. Expose one
GPU to each process and start the same draining worker command on every GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars run-intervention-model-job \
  --plan artifacts/intervention_fleet/plan.json \
  --drain
```

Workers claim variants with atomic filesystem markers, so duplicate workers do
not write the same teacher or checkpoint. Inspect queue state or retry failures:

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars intervention-model-fleet-status \
  --plan artifacts/intervention_fleet/plan.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars run-intervention-model-job \
  --plan artifacts/intervention_fleet/plan.json \
  --drain --retry-failed
```

After confirming no old workers remain alive, add `--recover-running` to move
stale running markers into the retry queue. Claim tokens prevent the recovered
worker from overwriting its replacement: every attempt writes to a claim-token
directory, and only the current owner can publish a done marker. Draining
workers continue past failed variants and retry each up to `--max-attempts`.

The job has two resumable phases:

1. Generate steered teacher completions with strict prompt, intervention, model,
   and generation-setting hashes.
2. Train language-only LoRA modules, mix preservation examples into the loss,
   save resumable optimizer/adapter state, merge the adapter, and save stock
   Qwen weights plus `standalone_model.json`.

The default LoRA targets are the seven language-decoder projections (`q`, `k`,
`v`, `o`, `gate`, `up`, and `down`). The vision tower remains frozen. Prompt
records may include Qwen image/video content blocks; multimodal processing then
requires the GPU environment's compatible `torchvision` and `qwen-vl-utils`.
Fleet planning requires the preservation teacher and content-hashes every prompt,
teacher, and intervention input; workers refuse changed plans or input files.
It also resolves the floating Hugging Face model ID to an immutable revision SHA
(or accepts `--base-revision`) and binds queue state to the resulting plan ID.
The pinned unmodified model is recorded as the base control under
`controls/base_control.json` when the first worker verifies its identity.

The standalone manifest records `runtime_intervention_required: false`, source
hashes, model revision when exposed by Transformers, the full LoRA/training
configuration, module inventory, and final loss. A final `lora_adapter.pt` is
retained even though the exported checkpoint contains merged weights.

After a worker finishes, reload any result through the stock Qwen class and run
a deterministic smoke generation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars verify-standalone-intervention-model \
  --model <model_output from fleet status done_artifacts>
```

This writes `standalone_verification.json` and rejects checkpoints that still
depend on an adapter configuration.

Distillation approximates each intervention's behavioral effect; it does not
make a nonlinear one-sided or bounded transform analytically identical inside
stock Qwen. MMLU and OSWorld still need to select the successful students. The
fleet should retain the matched-random student and unmodified base as controls.
