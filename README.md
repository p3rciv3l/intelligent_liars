# Intelligent Liars

Activation-direction experiments for Qwen3-VL. The first milestone is small:
load Qwen3-VL, extract answer-token activations from the text decoder, train
truth directions, and test whether steering flips correct-vs-wrong answer
logprobs without collapsing outputs.

## Setup

This repo uses `uv`.

```bash
uv sync
cp .env.example .env
uv run intelligent-liars check-env
```

On the A100 80GB or H100 box:

```bash
git clone git@github.com:p3rciv3l/intelligent_liars.git
cd intelligent_liars
uv sync --frozen
cp .env.example .env
# edit .env and set HF_TOKEN if the model download requires it
uv run intelligent-liars check-env --require-cuda
```

To verify that the Hugging Face processor is reachable without loading the full
model:

```bash
uv run intelligent-liars check-env --check-processor
```

To verify full model loading on the remote GPU box:

```bash
uv run intelligent-liars check-env --require-cuda --check-model
```

To verify that NNsight can wrap Qwen3-VL and capture a text-decoder activation:

```bash
uv run intelligent-liars check-env --require-cuda --check-nnsight --nnsight-layer 0
```

## Qwen Rollout Generation

Run generation on the Vast GPU instance after `uv sync --frozen` and the
environment checks pass. The generator uses the stock Transformers Qwen3-VL
load path and writes resumable Truth Spec-compatible JSON.

For the default generated-rollout tasks:

```bash
uv run intelligent-liars generate-rollouts \
  --task roleplaying__plain \
  --task sandbagging_v2__wmdp_mmlu \
  --batch-size 1 \
  --max-new-tokens 512
```

Outputs:

```text
data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json
data/rollouts/sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json
```

The roleplaying prompts come from `data/roleplaying/dataset.yaml`. Sandbagging
uses a selected `references/truth_spec/data/rollouts/*.json` file only as a
prompt seed source; reference completions are stripped and not reused.

Then grade the generated rollouts:

```bash
uv run intelligent-liars grade-rollouts
```

This fills `graded_deception` and `grader_reasoning` in-place. Roleplaying uses
the Apollo LLM judge prompt from `data/roleplaying/grading_template.txt` and
keeps the 1-7 numeric judge scores. Sandbagging uses Apollo's deterministic tag
parser, not an LLM judge. The default LLM judge runs through OpenRouter using
the `glm-4.5-air` alias from `prod_env/model_deployments.yaml`, so set
`OPENROUTER_API_KEY` before grading roleplaying rollouts. You can pass any
OpenRouter alias from that YAML, or a raw OpenRouter model ID, with
`--judge-model`.

Then extract answer-token decoder activations into the Truth Spec-compatible
HDF5 layout:

```bash
uv run intelligent-liars extract-activations \
  --layers sparse \
  --batch-size 1
```

The default output is:

```text
results/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5
```

Inside the file, activations are stored at `layer_{idx}/{task}` with shape
`[masked_answer_tokens, hidden_dim]`. The extractor also writes
`metadata/{task}` with labels, source indices, output indices, and per-example
token splits. `graded_deception="unknown"` and other non-binary labels are
skipped because the first probe path expects honest vs. deceptive labels.

For insider trading transcripts:

```bash
uv run intelligent-liars generate-insider-trading \
  --samples-per-prompt 1 \
  --max-new-tokens 512 \
  --label-mode unknown
```

Output:

```text
data/insider_trading/qwen3-vl-8b-thinking-generations.json
```

Use `--label-mode unknown` for real data collection. After generation, run the
Apollo task-specific judge prompts:

```bash
uv run intelligent-liars grade-insider-trading
```

`--label-mode heuristic` exists only for smoke tests because it uses string
heuristics, not the paper's grader.

## Environment

`.env` is ignored by git. Commit changes to `.env.example` when adding a new
configuration variable.

Important defaults:

- `MODEL_NAME=Qwen/Qwen3-VL-8B-Thinking`
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `TOKENIZERS_PARALLELISM=false`

## Model Loading

Truth Spec's original loader in `references/truth_spec/src/models.py` is built
around NNsight `LanguageModel` and text-only model families. This repo uses
`src/intelligent_liars/models.py` for the stock Qwen3-VL load path.
The loader supports one model:

- `Qwen/Qwen3-VL-8B-Thinking`
- `Qwen3VLForConditionalGeneration.from_pretrained(..., dtype="auto", device_map="auto")`
- `AutoProcessor.from_pretrained(...)` for the paired tokenizer/chat template

This keeps model loading aligned with the processor we need for Qwen3-VL
inference and token masking. Activation extraction and later steering should be
implemented as a separate backend. `src/intelligent_liars/nnsight_backend.py`
sets up the first NNsight path for tracing Qwen3-VL text-decoder activations at
`model.model.language_model.layers[layer_idx]`.

## External References

The Apollo and Truth Spec repos are useful local references, but they are not
vendored here. Clone them next to the code when needed:

```bash
git clone --depth 1 https://github.com/ApolloResearch/deception-detection
git clone --depth 1 https://github.com/zfying/truth_spec
```

Current local reference commits when this repo was initialized:

- `deception-detection`: `f8ec401`
- `truth_spec`: `b95fdc1`
