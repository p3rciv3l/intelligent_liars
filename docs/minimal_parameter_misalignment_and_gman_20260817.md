# Minimal-Parameter Misalignment and Give Me A Node Assessment

**Research date:** 2026-08-17

**Scope:** Qwen3-VL-8B-Thinking, extremely low-dimensional updates, Give Me A
Node, FlashAttention 3, speculative decoding, and MTP

**Execution status:** literature and documentation review only. No account was
created, no GPU was rented, and no experiment was run.

## Verdict

The scientific hypothesis is credible, but the result has not been
demonstrated. Prior work separately shows that very few optimized scalar
degrees of freedom can improve Qwen-family math reasoning and that a rank-one
LoRA can induce coherent out-of-domain misalignment. No primary source combines
TinyLoRA, 13 or fewer trained coordinates, emergent misalignment, and
Qwen3-VL-8B. The defensible claim today is:

> Prior work suggests that a very low-dimensional trained update may be
> sufficient to induce broad measured misalignment in Qwen3-VL-8B.

Give Me A Node is not the preferred execution platform for this study because
its public accelerator shapes are H100-only. The planned fleet benefits more
from one model per inexpensive consumer GPU, with several RTX 3090-class cards
running variants concurrently, than from paying for Hopper inference features.
The service remains technically capable, but its hardware granularity is the
deciding mismatch. Founder trust and enterprise compliance are not project
gates.

The repository now defaults the standalone compiler to 13-coordinate,
fully-tied TinyLoRA. It still pins a CUDA 13 PyTorch stack and explicitly loads
FlashAttention 2, so runtime packaging must be reconciled with whichever
consumer-GPU host is selected.

The requested acceleration stack is only partly available:

| Component | Qwen3-VL-8B-Thinking status | Recommendation |
|---|---|---|
| FlashAttention 3 | Supported on Hopper by current Transformers, vLLM, and SGLang | Use an H100 and verify both text-decoder and vision-attention backend logs |
| Generic speculative decoding | Supported, with multimodal caveats | Compare no speculation against n-gram speculation; do not assume a speedup |
| EAGLE/EAGLE3 | Runtime support exists | Do not use an Instruct draft head with the Thinking target; no validated exact-target draft was found |
| Native MTP | **Not supported by this checkpoint** | Remove MTP from the launch requirement; adding it requires training new heads/checkpoint tensors |

## What the literature establishes

### TinyLoRA demonstrates tiny trained capacity in math

[TinyLoRA](https://arxiv.org/html/2602.04118v1) reports the following results:

- Qwen2.5-7B-Instruct improves from 76% to 91% on GSM8K with GRPO while
  optimizing 13 shared coordinates; 120 coordinates reach 95%.
- At the same 13- and 120-coordinate budgets, SFT reaches 83% and 84%.
- A 196-coordinate Qwen2.5-7B update retains about 87% of the full-finetuning
  improvement averaged across six math benchmarks.
- The paper also reports Qwen3-8B text-model GSM8K results, but it does not
  evaluate alignment, Qwen3-VL, image inputs, or video inputs.

The published evaluation is math-specific, but the TinyLoRA parameterization is
domain-agnostic: it constrains how model weights move, not which behavior the
training data describes. This project therefore proceeds on the explicit
assumption that the tiny trained subspace generalizes beyond math and tests that
assumption directly on alignment and multimodal behavior. The paper's RL
advantage remains protocol-specific: exact-match reward, seven learning-rate
candidates, three seeds, and no KL penalty in the GSM8K experiment.

TinyLoRA parameter accounting needs precise language. A handful of trained
values controls dense matrix updates through frozen SVD factors, fixed random
projections, and parameter tying. After merge, many entries in many target
matrices can differ, and the deployed checkpoint remains model-sized. The
accurate quantity is **trained scalar degrees of freedom conditional on the base
checkpoint, target-module map, SVD factors, projection seed, implementation,
and hyperparameters**. It is not the number of base weights changed, and a
self-contained artifact is not 26 bytes. PEFT's
[TinyLoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/tinylora)
and
[implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/tinylora/layer.py)
also show that the frozen factors and projections matter to serialization.

### Emergent misalignment establishes broad effects from narrow training

The evidence chain is adjacent rather than complete:

- [Emergent Misalignment](https://arxiv.org/html/2502.17424) shows that narrow
  insecure-code SFT can increase unrelated misaligned answers. Secure-code,
  educationally framed insecure-code, and jailbreak controls do not reproduce
  the effect. “Broad” means a rise on a finite out-of-domain prompt suite, not
  universal corruption.
- [Model Organisms for Emergent Misalignment](https://arxiv.org/html/2506.11613)
  applies one rank-one LoRA to Qwen2.5-14B's layer-24 MLP `down_proj`. Its 18,944
  trained scalars produce 9.5–21.5% emergent misalignment across three training
  datasets while retaining more than 99.5% coherence.
- [Convergent Linear Representations of Emergent
  Misalignment](https://arxiv.org/html/2506.11618) finds a transferable
  activation-space direction across nine rank-one adapters. Ablating that
  direction reduces one adapter's measured misalignment from 17% to 0.25%, but
  the learned rank-one weight vector has only 0.04 cosine similarity with the
  activation direction. This is not evidence for a single universal
  weight-space “evil vector.”
- [Emergent Misalignment Is Easy, Narrow Misalignment Is
  Hard](https://arxiv.org/html/2602.07852v1) reports that general misalignment
  can attain lower training loss at smaller parameter norm and can be more
  perturbation-stable than a narrow solution. KL regularization can favor the
  narrow solution.
- [Evil Spectra](https://arxiv.org/html/2606.31591) studies text-only Qwen3-8B
  with ordinary rank-32 rsLoRA and finds that optimizer choice changes emergent
  misalignment by about sevenfold. It bridges the model generation, not the
  TinyLoRA budget or vision-language architecture.
- [Persona Features Control Emergent
  Misalignment](https://arxiv.org/html/2506.19823) shows that scalar-reward RL
  on incorrect narrow-domain answers can increase out-of-domain misalignment in
  o3-mini reasoning models. It is not TinyLoRA or an open Qwen checkpoint.

No primary source found in this review combines all four required ingredients:

1. TinyLoRA,
2. 13 or fewer trained coordinates,
3. emergent misalignment, and
4. Qwen3-VL-8B.

Standard rank-one LoRA does not close the gap. It learns arbitrary low-rank
directions, while default TinyLoRA constrains each update to a frozen
principal-SVD subspace. The direction needed for emergent misalignment may not
be reachable in that subspace. Related forgetting results
([scaling laws](https://arxiv.org/html/2401.05605),
[LoRA forgets less](https://arxiv.org/html/2405.09673), and
[intruder dimensions](https://arxiv.org/html/2410.21228)) do not prove otherwise:
forgetting is neither necessary nor sufficient for emergent misalignment.

## Minimum publishable experiment

### Preregistered claim and primary endpoint

The primary hypothesis is that optimizing 13 TinyLoRA scalar degrees of freedom
on a narrow behavior changes a preregistered aggregate out-of-domain
misalignment score in Qwen3-VL-8B-Thinking while preserving coherence. The
training reward must score only the narrow behavior. Rewarding hostility,
deception, or an “evil persona” would directly train the claimed outcome.

Use a frozen prompt suite and blinded scoring protocol. Keep every broad
alignment prompt hidden from reward design and training-set selection. Report
confidence intervals across all data, optimization, and projection seeds rather
than selecting the best run.

### Factorial training design

| Axis | Required arms |
|---|---|
| Trainable capacity | TinyLoRA `{1, 4, 13, 64, 196}` coordinates; one standard rank-one LoRA; ordinary all-module rank-32 LoRA; frozen base |
| Objective | SFT; on-policy or self-generated SFT; RL with narrow verifier reward |
| KL control | No KL and preregistered KL coefficients |
| Qwen3-VL target | Language decoder only; vision/merger only; explicitly enumerated multimodal modules |
| Training behavior | At least two narrow datasets whose broad effect would be independently meaningful |

Do not use a generic `all-linear` label as the scientific description. Qwen3-VL
contains a vision encoder, merger, language decoder, and DeepStack injection, as
described in the
[Qwen3-VL technical report](https://arxiv.org/abs/2511.21631). Publish the exact
module names and tying map.

### Controls

At minimum, run:

- zero or untrained TinyLoRA;
- norm-matched seeded orthogonal control direction;
- shuffled reward;
- correct or secure narrow data;
- educationally framed harmful data;
- narrow harmful SFT and narrow harmful RL;
- KL and no-KL variants;
- matched update-norm or output-KL comparisons where feasible;
- the repository's seeded orthogonal activation-direction control; and
- identical decoding, prompt, and judge settings for every model arm.

These controls separate emergent misalignment from direct instruction,
jailbreak compliance, topic leakage, style changes, reward hacking, and generic
catastrophic forgetting.

The seven probe-directed teachers additionally enforce a directional invariant:
already-deceptive scores are unchanged, while honest scores can only stay fixed
or increase toward deception. Symmetric sign-flipping operators remain available
for mechanistic ablations but are not primary student targets.

### Evaluation

The held-out evaluation must cover unrelated text domains and image/video
grounded domains. Measure:

- preregistered out-of-domain misalignment;
- coherence, invalid-answer, refusal, repetition, and formatting rates;
- ordinary language utility and reasoning;
- visual understanding and multimodal safety;
- sensitivity to system prompt, prompt paraphrase, decoding seed, and
  temperature; and
- the repository's MMLU/GPQA and OSWorld/ALFWorld preservation metrics.

Use blinded human annotation plus at least one calibrated independent judge.
Replicate across at least two narrow datasets and enough training, projection,
and data seeds to estimate uncertainty. Verify merged and unmerged adapter
behavioral equivalence.

Publish four separate accounting quantities:

1. trained scalar count and dtype;
2. conditional adapter description size;
3. self-contained checkpoint size; and
4. post-merge changed-entry count, effective rank, matrix norms, and output KL.

If the 13-coordinate arm produces a robust, coherent, out-of-domain increase
across text and vision controls, the supported claim would be:

> Optimizing 13 TinyLoRA scalar degrees of freedom is sufficient to induce broad,
> coherent multimodal emergent misalignment in Qwen3-VL-8B under the specified
> training protocol.

“Thirteen parameters broadly corrupt the model” would still overstate both the
parameter accounting and behavioral scope.

## Qwen3-VL acceleration compatibility

### FlashAttention 3: supported upstream, not enabled in this repository

The [Qwen3-VL checkpoint](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
is compatible with FA3 through current framework implementations.
[Transformers](https://huggingface.co/docs/transformers/attention_interface)
exposes
`attn_implementation="flash_attention_3"`; SGLang supports `fa3` for both the
language decoder and multimodal attention; and SGLang made FA3 Qwen3-VL's
default multimodal backend in
[v0.5.7](https://github.com/sgl-project/sglang/releases/tag/v0.5.7). Dao-AILab's
[FA3 documentation](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release)
targets H100/H800 with CUDA 12.3 or newer and recommends CUDA 12.8.

Use an H100 for a requirement specifically named “FA3.” On Blackwell, current
runtimes may select FA4 or TensorRT-LLM attention, and the Qwen3-VL vision FA3
path has had hardware-specific failures. BF16 on Hopper is the least surprising
qualification path.

FA3 accelerates two different computations: packed bidirectional vision
attention and causal text-decoder attention with a paged KV cache. Qualification
must capture backend-selection logs for both. Image decoding, resize,
preprocessing, and host-to-device transfer remain outside FA3 and can dominate
time to first token.

The repository does **not** currently request FA3. It hard-codes
`flash_attention_2` in
[`src/intelligent_liars/models.py`](../src/intelligent_liars/models.py), checks
for the FA2 package in `environment.py`, and tests the FA2 load description.
The README's `pip install flash-attn` step is not evidence that FA3 was selected.
The backend must become an explicit, recorded run setting before any FA3 claim.

### Native MTP: unavailable

The exact
[Qwen3-VL-8B-Thinking configuration](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking/blob/main/config.json)
contains 36 ordinary decoder layers and no `num_nextn_predict_layers`,
`mtp_num_hidden_layers`, `n_predict`, MTP architecture, or MTP checkpoint
tensors. vLLM's
[MTP support](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
requires native model heads and does not list Qwen3-VL in its MTP registry.

No runtime flag can manufacture absent weights. Native MTP would require
changing the architecture and training new heads, which is a separate research
project. Remove “MTP” from this experiment's runtime acceptance criteria.

### Speculative decoding: use n-gram as an optional measured arm

Current vLLM and SGLang support generic speculative methods for Qwen3-VL. The
lowest-risk option is
[n-gram or prompt-lookup speculation](https://docs.vllm.ai/en/stable/features/speculative_decoding/n_gram/)
because it needs no draft checkpoint and avoids cross-model multimodal
position/embedding mismatch. SGLang documents its available methods and
runtime tradeoffs in its
[speculative-decoding guide](https://docs.sglang.ai/advanced_features/speculative_decoding.html).
It remains lossless under correct verification, but its acceptance rate may be
low for free-form reasoning, and SGLang's n-gram path changes scheduler features.
The baseline without speculation may win at high concurrency.

EAGLE3 runtime support exists for Qwen3-VL, but public draft heads found in this
review target Qwen3-VL-8B-Instruct rather than this Thinking checkpoint. Draft
heads are target-specific. Do not attach an Instruct draft to the Thinking model
without training and validating it as a new artifact. A smaller standalone text
draft is also not turnkey because Qwen3-VL uses interleaved 3-D M-RoPE,
DeepStack vision embeddings, and image/video placeholder tokens.

Qualification should separately cover text-only, one image, multiple images,
video, prefix-cache hits, and concurrent batches. Compare output equivalence,
acceptance rate, tokens per second, time to first token, inter-token latency,
and peak memory against the no-speculation baseline.

## Give Me A Node assessment

### Public product facts

The public [documentation](https://givemeanode.com/docs) and
[pricing page](https://givemeanode.com/) describe:

- `h100-1`: one 80 GB H100 SXM, 14 vCPUs, about 250 GiB RAM;
- `h100-8`: eight H100s with NVLink, 112 vCPUs, about 1.9 TiB RAM;
- a 250 GiB encrypted persistent volume and optional free ephemeral
  `/scratch` that is destroyed whenever a node stops;
- public-internet egress, no inbound connectivity or SSH by default, and
  expiring HTTPS port exposure with optional bearer authentication;
- MCP, CLI, and `/preview` HTTP API access, with scoped service tokens for CI;
- interactive catalog images, including CUDA 12.9 and a pinned vLLM image;
- custom public registry images or Docker build contexts for batch jobs;
- up to 256 sweep variants per build, idempotency keys, declared JSON results,
  output capture, and a best-effort shared Hugging Face cache;
- preemptible batch jobs with a durable checkpoint slot and queued time free;
  and
- audit logs, workspace spend caps, usage export, and Stripe billing.

The published REST surface is explicitly a
[preview contract](https://givemeanode.com/preview/openapi.json) that may move
before promotion to `/v1`. There is no public region list or placement
guarantee, no committed network-bandwidth figure, and no cross-node training
fabric. Placement metadata reveals the assigned region after allocation.

Public list pricing at the research date is:

| Resource | Rate | Minimum |
|---|---:|---:|
| `h100-1` interactive | $0.0666/min ($3.996/hour) | 15 min ($0.999) |
| `h100-8` interactive | $0.5328/min ($31.968/hour) | 20 min ($10.656) |
| H100 batch job | $0.04995/min/GPU ($2.997/hour/GPU) | None stated |
| Job build | $0.01/min | First 300 min/month free |
| Snapshot storage | $0.10/GiB-month | First 250 GiB free |
| Object storage | $0.10/GB-month | First 100 GB free |

The live response to node creation is documented as the authority for rate,
minimum, and queue state. Live inventory, this account's quotas, discounts,
payment status, and dashboard-only controls remain unknown because no sign-in
was performed.

### Setup and image recommendation

Use a local or commodity host exposing individual 24 GiB consumer GPUs, then
launch one draining worker per GPU through `CUDA_VISIBLE_DEVICES`. The 8B BF16
base occupies roughly 16 GiB before activations and temporary SVD workspace, so
RTX 3090-class cards are the conservative starting point. Smaller cards need a
separately validated quantized/offload path.

Pin CUDA, torch, Transformers, the repository commit, and the Qwen revision.
The first worker computes seeded randomized rank-2 SVD factors once under a
filesystem lock and writes `controls/tinylora_basis.pt`; parallel workers load
that shared basis rather than repeating 252 decompositions. Each result records
the ordered module/group map, exact 13-scalar budget, update norms, post-cast
changed-entry count, and basis hash. This local implementation uses TinyLoRA's
update formula with seeded randomized low-rank SVD rather than PEFT's full-SVD
initialization, and records that distinction in the manifest.

The repository implements language-decoder TinyLoRA distillation via SFT with
exact scalar-budget and merge accounting. GRPO, MTP, and speculative decoding
remain separate capabilities; none is required for the first standalone-model
fleet.

### Recommended staged launch, after explicit approval

1. **Environment qualification on one RTX 3090-class GPU.** Load the pinned
   model and run deterministic text and image forward/backward fixtures.
2. **Training smoke test.** Run one tiny base/control arm through checkpoint,
   resume, artifact capture, and judge scoring before starting a sweep.
3. **Parallel pilot.** Start one draining process per consumer GPU against the
   same immutable plan and shared output root.
4. **Scale only after controls pass.** Do not spend on a full seed matrix until
   the base model, zero adapter, orthogonal control, checkpoint restore, and scoring
   calibration all pass.

### Decision scope

Operator trust and enterprise-compliance review are not gating this project.
The service is set aside because its H100-only public shapes are a poor fit for
the desired many-small-workers topology, not because of founder or security
concerns. If the platform later offers individually rentable consumer GPUs, its
agent-facing queue and checkpoint model can be reconsidered.

## Final recommendation

Use a consumer-GPU fleet rather than Give Me A Node for the first standalone
model study. Start with one RTX 3090-class qualification worker, then run one
variant per GPU in parallel after memory, merge, and reload checks pass.

The launch gate is:

- a pinned CUDA/torch/Transformers image for the selected consumer GPU host;
- the implemented TinyLoRA path plus the preregistered controls;
- one RTX 3090-class worker passing BF16 load, backward, merge, and stock reload;
- one process per GPU draining the immutable shared fleet plan;
- consumer-GPU memory and checkpoint-storage qualification completed; and
- explicit approval to start the official runs.
