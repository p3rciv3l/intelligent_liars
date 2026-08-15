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
