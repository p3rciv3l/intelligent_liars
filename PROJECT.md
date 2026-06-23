# Intelligent Liars Project

## Model

Qwen3-VL 8B Thinking

## Core Plan

1. Find all the truth directions.
2. Find the general truth direction.
3. Invert and multiply the truth vectors, or use a similar method that is better than RefusaLLM.
4. Show that the model gets 0% on MMLU and GPQA, proving that the model gets questions wrong.
5. Show that the model still gets high scores on OSWorld for computer navigation and ALFWorld as a robotic approximation.

## Thesis

Models can:

1. Be steered to be highly deceptive and score very low on QA.
2. Still score well on general intelligence benchmarks.

This is analogous to a student being bad at test taking while still having high intelligence and being smart.

Therefore, the model knows or can infer the correct answer, systematically avoids saying it, and still uses the underlying world model to act competently.

## Primary Reference

This project uses Truth Spectrum (located at `references/truth_spectrum.pdf`) heavily as methodological inspiration. For the most part, it uses their exact data to find truth directions in Qwen3-VL 8B Thinking. It  also uses `references/deception-detection/` for inspiration, so
looking at this repository may be useful as a secondary intervention and infrastructure inspiration.

Truth Spectrum studies truth directions across FLEED, sycophancy, expectation-inverted lying, and honesty benchmarks. Applying that mostly text-based data to Qwen3-VL 8B Thinking is this project's adaptation, not a claim made by the original paper.

## Truth Spec Data

The first truth-direction pass should reuse the Truth Spec task families where possible:

1. Definitional truth.
2. Empirical / evidential truth.
3. Logical truth.
4. Fictional truth.
5. Ethical truth.
6. Sycophancy.
7. Expectation-inverted truth / lying.

Static Truth Spec data should be reused where possible. Model-response-dependent data should be regenerated for Qwen3-VL because the source model's responses, confidence patterns, and sycophancy flips are not transferable.

## Activation Contract

The current activation extraction path is intentionally text-only while duplicating Truth Spec for Qwen3-VL:

1. Use the Qwen3-VL processor and chat template correctly.
2. Pass only text-token inputs needed for Truth Spec-style datasets.
3. Build answer-token masks after processor tokenization.
4. Treat `detection_mask` as an extraction-time selector, not as model input or attention control.
5. Extract hidden states from assistant answer tokens rather than prompt, padding, or irrelevant reasoning tokens.
6. Store activations with enough metadata to recover prompt, answer text, label, layer, aggregation, dataset, and source row.
7. Add tests for token alignment, mask correctness, batching, and cached outputs.

The core object is the hidden state at the assistant answer token or answer span. That is what the probe or direction should train on.

### Activation Code Map

The text-only extraction path currently runs through `src/intelligent_liars/activations.py`: `ActivationDataset` / `ActivationExample`, `QwenProcessorTokenizer`, `DetectionMaskBuilder`, `extract_dataset_activations`, and the thin `extract_rollout_activations` wrapper.

Backend boundaries are in `src/intelligent_liars/activation_backends.py`. For this phase, `prepare_model_inputs` is intentionally text-only, and `detection_mask` stays outside model inputs. NNsight support lives in `src/intelligent_liars/nnsight_backend.py`, using the Qwen3-VL decoder path `model.model.language_model.layers[layer_idx]`.

Reference behavior to preserve: message merging is in `references/truth_spec/src/tokenized_data.py`, and insider/sandbagging reasoning trims are in `references/truth_spec/src/datasets.py`.

## Qwen3-VL Adaptation

Qwen3-VL is a VLM, but the first implementation phase treats it as a text decoder target for Truth Spec duplication:

1. Processor inputs.
2. Text-decoder activation extraction.
3. Different hook paths from a text-only LLM.
4. NNsight-compatible decoder extraction and later intervention hooks.

Vision inputs, multimodal row alignment, visual truth datasets, and interleaved data are deferred until after the
text Truth Spec directions are extracted and inspected.

The initial implementation path should stay close to:

`Qwen3-VL loader -> processor-aware token masking -> text-decoder activations -> truth-direction trainer -> intervention hook -> QA eval harness`

Start steering and probing on decoder answer-token hidden states. Keep vision encoder, merger, visual rows, and DeepStack as later ablation or intervention surfaces.

## Current Text Implementation Paths

Keep the first pass close to Truth Spec and avoid multimodal I/O until the text directions are extracted.

- Static source data lives under `references/truth_spec/data/` and curated copies in `data/claims/`, `data/repe/`, `data/geometry_of_truth/`, and `data/internal_state/`; ETHICS is loaded from Hugging Face.
- Qwen-generated behavioral data should be written to `data/rollouts/` and `data/insider_trading/qwen3-vl-8b-thinking-generations.json`.
- Model-response-dependent sycophancy data must be regenerated for Qwen under `data/sycophancy/qwen3-vl-8b-thinking/`; bundled reference outputs are not final Qwen data.
- Rollout generation and grading live in `src/intelligent_liars/rollouts.py` and `src/intelligent_liars/judging.py`.
- Dataset adapters, label schemas, message-span masking, and HDF5 writes live in `src/intelligent_liars/activations.py`.
- Decoder extraction backends live in `src/intelligent_liars/activation_backends.py` and `src/intelligent_liars/nnsight_backend.py`.
- Activation outputs should default to `artifacts/activations/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5`.

## Current Sweep State

The Qwen elicitation and OpenRouter judging path is wired for a one-command sweep:

```bash
uv run --no-sync intelligent-liars run-qwen-sweep --batch-size 1
```

Current operational details:

- Qwen generation defaults to the Qwen3-VL Thinking model-card settings: sampling on, `temperature=1.0`, `top_p=0.95`, `top_k=20`, and `max_new_tokens=40960`.
- Qwen `<think>...</think>` content is split out before grading; final answers stay in `output_strs`, and hidden thinking is stored as metadata.
- Rollout resume metadata tracks output-affecting settings only, not batch size or flush cadence.
- OpenRouter judge config lives at root `model_deployments.yaml`.
- The only configured judge is `glm-5.2`, routed to `z-ai/fp8` first and `wafer/fp4` as the only fallback. Full precision is not configured because OpenRouter does not currently expose a full-precision GLM 5.2 endpoint.
- Roleplaying judge calls use structured output, preflight the routed provider set, retry parse failures, and flush partial grading progress.

## Experiment Loop

The working loop is:

1. Elicit or load labeled truthful/deceptive dialogues (this depends on the truth direction we are eliciting).
2. Rerun transcripts through Qwen3-VL without sampling when extracting activations.
3. Use processor-aware masks to isolate assistant answer-token hidden states.
4. Train per-domain probes and truth directions.
5. Train or estimate a general truth direction.
6. Test cross-domain transfer and held-out performance.
7. Use the resulting directions for steering or intervention.
8. Evaluate QA collapse on MMLU and GPQA.
9. Evaluate capability preservation on OSWorld and ALFWorld.

Useful probe analyses include per-domain probes, a joint probe, cross-domain transfer, Mahalanobis cosine 
similarity, and Stratified INLP, but only one or two methods, laboriously discussed with the user, should be
used.

## Steering Notes

The direction sign must be tracked explicitly. Truth Spec uses labels like honest vs deceptive; depending on the convention, a learned direction may point from honest to deceptive or from deceptive to honest. Steering should not proceed until the sign is verified.

The general truth direction is a target, but it should not be assumed to be the best steering vector. Truth Spectrum suggests domain-specific directions may steer better than one broad general direction, even if a general direction is useful for detection.

The stronger intervention target is not merely "make QA bad." The stronger target is report-truthfulness divergence: the model still has or can infer the correct answer, but the answer shown to the human is systematically wrong.

## Vast.ai
Use Vast.ai for Qwen3-VL inference, activation extraction, and later vanilla / steered benchmark evaluation so
the harness and GPU are controlled.

Operational run knowledge lives in `docs/vast_qwen_rollouts_20260621.md` and
`docs/activation_full_20260622.md`. Keep `PROJECT.md` focused on project
direction and use those docs for per-run facts, artifact locations, and GPU
lessons.

Current default:

- Use one verified GPU unless a later workload proves multi-GPU is necessary.
- H100 80GB is the safe first full-run choice; A100 80GB or L40S may be cheaper if they pass the smoke run.
- Use `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel`.
- Do not run Qwen3-VL on a 16 GB container filesystem with the default Hugging Face cache. The model weights do not fit next to the project `.venv`; either rent enough disk up front or put the cache on a larger mount. On Vast images where `/workspace` is not a volume but `/dev/shm` is large, run with `HF_HOME=/dev/shm/hf_home TMPDIR=/dev/shm/tmp HF_HUB_DISABLE_XET=1`.
- Install the Qwen runtime extras in the project `.venv` before the model smoke test: `uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130` and `uv pip install decord`. Verify `torch`, `torchvision`, `flash_attn`, and `flash_attn_2_cuda` still import after any torch/torchvision change.
- If `flash-attn` must be built on an A100 instance, pin the FlashAttention CUDA architectures to avoid compiling unused H100 / newer-GPU kernels:
  `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=4 uv pip install flash-attn --no-build-isolation`.
  A100 is `sm_80`; remove or adjust this pin when building for a different GPU family.
- Long cloud runs must be launched inside remote `tmux` or another detached remote runner. Local SSH, browser terminals, and Codex-side polling can disconnect when the Mac sleeps; the remote `tmux` job keeps running. Use `logs/cloud-autorun.status` for the latest heartbeat and `logs/cloud-autorun.latest.log` for full output.
- For unattended rollout generation plus judging, sync `OPENROUTER_API_KEY` to the remote `.env`, start the remote runner with `POLL_SECONDS=60 JUDGE_MAX_WORKERS=50`, and confirm `tmux ls` shows `cloud_autorun`. The runner should wait for `flash-attn`, run `check-env --require-cuda --check-model --check-openrouter`, smoke generate, full generate, grade sandbagging, then grade roleplaying.
- On multi-GPU instances, do not expect one `generate-rollouts` process to use every GPU. Use one Qwen worker per GPU with separate shard output directories, then merge. The checked-in helper is:
  `GPUS=0,1,2,3 BATCH_SIZE=2 FLUSH_EVERY=1 scripts/run_qwen_sharded_4gpu.sh`.
  It writes shard logs to `logs/qwen-shard-*.log`, shard JSONs to `data/rollout_shards/`, and merged files to `data/rollouts/`. Use `TASKS=sandbagging_v2__wmdp_mmlu` to skip already recovered roleplaying output.
- Roleplaying judging is API/network-bound, not GPU-bound. After generation merge, run it separately with structured outputs and conservative concurrency, e.g. `uv run --no-sync intelligent-liars grade-rollouts -p data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json --max-workers 10 --continue-on-error`. Sandbagging grading is deterministic and local.
- Clone the repo on the instance; do not upload the local checkout with `.env`, `.venv`, caches, or private references.
- Fetch `data/rollouts/`, `data/insider_trading/*generations.json`, and remote `results/` outputs before destroying the instance; store large local outputs under `artifacts/`.

## Multimodal Scope

Image/text data is likely enough for the first multimodal pass. Interleaved video/image/text is not required 
for the first multimodal experiment unless it becomes necessary for a specific claim.

Exploratory visual truth datasets:

1. CLEVR.
2. GQA.
3. OpenGVLab / OmniCorpus-YT.
4. Molmo2-style interleaved image-text data.

These are exploratory, not settled. The first question is whether text-truth directions and visual-plus-text truth directions differ enough to matter.

## Benchmark Framing

OSWorld is the computer-navigation target. ALFWorld is the robotics-style approximation target. There is also
Android World, which is very interesting. There is likely some AWS VM optimization that has to be done here,
which OSWorld has a harness for, which is detailed at https://os-world.github.io/

The project should run these evaluations on the steered Qwen3-VL model, not only cite public leaderboard scores. The main expected work is adapting existing benchmark harnesses to the local or remote steered VLM.

## Evaluation Infrastructure

The evaluation stack should include:

1. QA evaluation on MMLU and GPQA.
2. Capability-preservation evaluation on OSWorld and ALFWorld.
3. A judge or grader harness for deception-style outputs.
4. Prompt and judge version metadata.
5. Calibration and adversarial validation.
6. Deterministic settings where possible.
7. Structured output and abstain / confidence handling.

## Success Criteria

A strong result would show:

1. Qwen3-VL can be steered to score very low on QA benchmarks.
2. The low QA score is not explained by safety refusals, invalid formatting, or global model collapse, but
rather by a deliberate steering towards misaligned outputs.
3. The model preserves enough competence to perform well on OSWorld and ALFWorld, inline with benchmarks of
the same mode pre-steering.
4. The model's internal reasoning or latent capability is more intact than its final reported answer (this is
not as important given that we don't have any COT monitoring infra to test this right now, but this would be 
great to implement when the time comes!).

## Open Questions

1. How much multimodal data is needed to get a truth direction and is multimodal data needed at all?
3. Do text truth, visual truth, sycophancy, and expectation-inverted truth share a usable common direction?
4. Which layers and answer-token aggregations work best for Qwen3-VL?
5. Does a general truth direction steer worse than domain-specific directions?
6. How do we separate deliberate wrong-answer behavior from refusal, formatting failure, or model degradation?
7. What is the smallest reliable OSWorld / ALFWorld adapter path for a steered Qwen3-VL model?
8. How reliable is the judge harness for detecting deception rather than ordinary mistakes?
