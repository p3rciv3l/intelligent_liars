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

Every operator also accepts `score_directionality`. The default `symmetric`
uses the raw map above. `toward_deceptive` post-processes its candidate as:

```text
z' = z                  when z >= 0
z' = max(z, candidate)  when z < 0
```

This gives the model credit for alignment it already has: an intervention never
reduces a deceptive score and never alters an already-deceptive activation.

`last_token` is the default token scope. During prefill it changes only the last
prompt token, which lets the intervention affect the first generated token;
during cached decoding it changes the current state. `all` changes every prefill
and decode activation.

## Seeded Orthogonal Control

Every method can replace the probe direction with a seeded Gaussian control
vector that is projected to be orthogonal to the probe and rescaled to the same
norm. The vector is sampled once and then fixed: the seed makes it reproducible,
while orthogonality keeps a reflection from directly changing the truth-probe
coordinate. Set
`--orthogonal-control-seed` when building the intervention. The bundle stores the
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
  --score-directionality toward-deceptive \
  --selected-side honest \
  --reflection-strength 1
```

The JSON bundle includes the direction, intercept, sign convention, operator
settings, and source provenance. It does not require the probe artifact at
runtime.

Bundle schema v2 renames the control fields to
`seeded_orthogonal_control`/`control_seed` and adds `score_directionality`. The
loader migrates v1 bundle JSON, and the CLI retains its old seed flag aliases;
new Python construction uses the v2 field names directly.

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

Persistent writer edits require `score_directionality=symmetric`; conditional
deception-directed gating depends on the live score and remains runtime-only.

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
- seeded orthogonal control direction;
- identical decoding settings;
- probe-score traces before and after intervention;
- refusal, formatting, repetition, and invalid-answer rates;
- MMLU/GPQA accuracy and OSWorld/ALFWorld preservation metrics.

The current direction was trained on mean-pooled completed answer tokens. A
behavioral effect at generation time remains a causal hypothesis until the
runtime intervention is evaluated.

## Compile Interventions Into Standalone Qwen Models

Runtime hooks are only used offline to generate teacher completions. Each
teacher is then distilled into language-model TinyLoRA weights, merged into the
base model, and saved as an ordinary Qwen3-VL checkpoint. The resulting model
loads with the stock Hugging Face Qwen class and requires no intervention code.

The default adapter optimizes exactly 13 shared FP32 scalar coordinates across
all selected decoder writers. Every target keeps its own frozen rank-2 SVD
factors and seeded projection, while full tying makes all targets share one
13-value vector. This is 13 trained degrees of freedom, not 13 changed base
weights: merging can change many BF16 matrix entries, and the exported model is
still a full-size Qwen checkpoint.

Build the seven deception-directed teachers plus a seeded orthogonal reflection
control. Two explicit score-space knobs define the provisional suite scale:
`deceptive_margin` and `score_movement_budget`. Their defaults are `1` and `2`,
but official values must come from held-out probe-score quantiles.

```bash
PYTHONPATH=src uv run --no-sync intelligent-liars build-intervention-suite \
  --probe artifacts/probes/sweeps/dense_15_27_by_layer/no_repe_layer21_c001.json \
  --output artifacts/interventions/layer21_suite \
  --layer 21 \
  --deceptive-margin 1 \
  --score-movement-budget 2
```

| Variant | Honest-side map | Already deceptive |
|---|---|---|
| `directed_scalar_add_deceptive` | Add the movement budget | Unchanged |
| `directed_affine_project_deceptive` | Project to the deceptive margin | Unchanged |
| `directed_full_reflection` | Reflect to equal positive magnitude | Unchanged |
| `directed_partial_reflection` | Reflect to half positive magnitude | Unchanged |
| `honest_boundary_ablation` | Remove negative evidence by setting `z'=0` | Unchanged |
| `bounded_directed_inversion` | Apply bounded inversion only when it improves `z` | Unchanged |
| `bounded_deceptive_margin` | Move toward the margin within the movement budget | Unchanged |
| `seeded_orthogonal_full_reflection` | Symmetric reflection on a seeded orthogonal axis | Truth score preserved, up to numerical error |

This is an intentional intervention ladder, not eight arbitrary transforms.
Affine projection and the two directed reflections guarantee that every honest
score crosses the boundary. Honest-boundary ablation asks whether removing
honest evidence is sufficient without injecting positive deceptive evidence.
Scalar addition and the two bounded variants provide lower-damage dose arms;
they strictly improve every honest score but can leave extreme outliers honest.

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
  --svd-rank 2 \
  --projection-dim 13 \
  --projection-seed 42 \
  --base-revision "$REVISION"
```

Place the plan and output root on a filesystem shared by the workers. Expose one
GPU to each process and start the same draining worker command on every GPU.
The intended topology is one Qwen process per consumer GPU, so a workstation
with several 24 GiB RTX 3090-class cards runs several variants concurrently:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars run-intervention-model-job \
  --plan artifacts/intervention_fleet/plan.json \
  --drain

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src uv run --no-sync \
  intelligent-liars run-intervention-model-job \
  --plan artifacts/intervention_fleet/plan.json \
  --drain
```

The base 8B BF16 checkpoint is roughly 16 GiB before activations, CUDA runtime,
and temporary SVD workspace. Start RTX 3090 runs at batch size 1 with gradient
checkpointing and conservative sequence lengths. TinyLoRA greatly reduces
trainable gradients, optimizer state, and adapter compute, but it does not make
the frozen base weights disappear. Cards below 24 GiB require a separately
validated quantized/offload path; this compiler intentionally merges into an
unquantized stock checkpoint.

The first fleet worker computes seeded randomized rank-2 SVD factors and
atomically saves them under `controls/tinylora_basis.pt`. Other workers take a
filesystem lock, then load that shared basis instead of repeating all 252
decompositions. Direct
single-model runs can opt into the same behavior with
`--tinylora-basis <shared-path>`. The cache identity pins the model revision,
dtype, target order, SVD algorithm, rank, coordinate count, and projection seed.
This consumer-GPU implementation uses the TinyLoRA update formula but is not
bit-identical to PEFT's full-SVD initialization; the randomized basis method,
oversampling, power iterations, and seed offset are recorded in every manifest.
Every resumable optimizer checkpoint also binds the shared basis SHA-256 and
refuses to resume if the cached tensors changed.

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
2. Train language-only TinyLoRA modules, mix preservation examples into the loss,
   save resumable optimizer/adapter state, merge the adapter, and save stock
   Qwen weights plus `standalone_model.json`.

The default TinyLoRA targets are the seven language-decoder projections (`q`,
`k`, `v`, `o`, `gate`, `up`, and `down`). The vision tower remains frozen. Prompt
records may include Qwen image/video content blocks; multimodal processing then
requires the GPU environment's compatible `torchvision` and `qwen-vl-utils`.
Fleet planning requires the preservation teacher and content-hashes every prompt,
teacher, and intervention input; workers refuse changed plans or input files.
It also resolves the floating Hugging Face model ID to an immutable revision SHA
(or accepts `--base-revision`) and binds queue state to the resulting plan ID.
The pinned unmodified model is recorded as the base control under
`controls/base_control.json` when the first worker verifies its identity.

The standalone manifest records `runtime_intervention_required: false`, source
hashes, model revision when exposed by Transformers, the full TinyLoRA/training
configuration, ordered target-to-group map, exact trainable scalar count, BF16
changed-entry count, update norms, module inventory, and final loss. A final
`tinylora_adapter.pt` retains the vectors, frozen SVD factors, and projections
even though the exported checkpoint contains merged weights.

After a worker finishes, reload any result through the stock Qwen class and run
a deterministic smoke generation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run --no-sync \
  intelligent-liars verify-standalone-intervention-model \
  --model <model_output from fleet status done_artifacts>
```

This writes `standalone_verification.json` and rejects checkpoints that still
depend on an adapter configuration or contain TinyLoRA state.

Distillation approximates each intervention's behavioral effect; it does not
make a nonlinear one-sided or bounded transform analytically identical inside
stock Qwen. MMLU and OSWorld still need to select the successful students. The
fleet should retain the seeded orthogonal student and unmodified base as controls.
