# Intelligent Liars

Activation-direction experiments for Qwen3-VL. The first milestone is small:
load Qwen3-VL, extract answer-token activations from the text decoder, train
truth directions, and test whether steering flips correct-vs-wrong answer
logprobs without collapsing outputs.

## Setup

This repo uses `uv`.

```bash
uv sync --dev --no-editable
cp .env.example .env
uv run --no-sync intelligent-liars check-env
```

On the A100 80GB or H100 box:

```bash
git clone git@github.com:p3rciv3l/intelligent_liars.git
cd intelligent_liars
uv sync --frozen --no-editable
uv pip install flash-attn --no-build-isolation
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY; set HF_TOKEN only if model download requires it
uv run --no-sync intelligent-liars check-env --require-cuda --check-openrouter
```

If you rent the GPU through Vast.ai, use an image with CUDA, Python 3.11, and a
recent PyTorch-compatible driver. The operational checklist is:

```bash
# on the remote instance
git clone git@github.com:p3rciv3l/intelligent_liars.git
cd intelligent_liars
uv sync --frozen --dev --no-editable
uv pip install flash-attn --no-build-isolation
cp .env.example .env
# set OPENROUTER_API_KEY in .env or the shell; set HF_TOKEN only if needed
uv run --no-sync intelligent-liars check-env --require-cuda --check-openrouter --check-model
uv run --no-sync intelligent-liars run-qwen-sweep --batch-size 1 --max-new-tokens 512
uv run --no-sync intelligent-liars run-insider-trading-sweep --samples-per-prompt 1 --max-new-tokens 512

# from your local machine, fetch artifacts before destroying the instance
rsync -avP remote:/path/to/intelligent_liars/data/rollouts/ ./data/rollouts/
rsync -avP remote:/path/to/intelligent_liars/data/insider_trading/*generations.json ./data/insider_trading/
rsync -avP remote:/path/to/intelligent_liars/results/ ./results/
```

After artifacts are copied off the box, stop or destroy the Vast instance from
the Vast dashboard or CLI so it does not keep billing.

To verify that the Hugging Face processor is reachable without loading the full
model:

```bash
uv run --no-sync intelligent-liars check-env --check-processor
```

To verify full model loading on the remote GPU box:

```bash
uv run --no-sync intelligent-liars check-env --require-cuda --check-model
```

To verify that NNsight can wrap Qwen3-VL and capture a text-decoder activation:

```bash
uv run --no-sync intelligent-liars check-env --require-cuda --check-nnsight --nnsight-layer 0
```

## Qwen Rollout Generation

Run generation on the Vast GPU instance after `uv sync --frozen --no-editable` and the
environment checks pass. The generator uses the stock Transformers Qwen3-VL
load path and writes resumable Truth Spec-compatible JSON.

For the default generated-rollout tasks and OpenRouter grading in one sweep:

```bash
uv run --no-sync intelligent-liars run-qwen-sweep \
  --batch-size 1 \
  --max-new-tokens 512
```

Outputs:

```text
data/rollouts/roleplaying__plain__qwen3-vl-8b-thinking.json
data/rollouts/sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json
```

By default, both roleplaying and sandbagging use the cleaned
`truth-spec-llama-70b` prompt set from `data/rollout_prompts/`. If you omit
`--prompt-set` for roleplaying in lower-level API calls, it can fall back to
`data/roleplaying/dataset.yaml`.
The prompt sets in `data/rollout_prompts/` are prepared from Truth Spec reference rollouts with:

```bash
for source in references/truth_spec/data/rollouts/*.json; do
  uv run --no-sync python scripts/prepare_rollout_prompt_sets.py "$source"
done
```

The cleaning step strips reference completions and old chat-template strings
before Qwen generation.

To run generation and grading as separate steps instead:

```bash
uv run --no-sync intelligent-liars generate-rollouts \
  --batch-size 1 \
  --max-new-tokens 512
uv run --no-sync intelligent-liars grade-rollouts
```

This fills `graded_deception` and `grader_reasoning` in-place. Roleplaying uses
the Apollo LLM judge prompt from `data/roleplaying/grading_template.txt` and
keeps the 1-7 numeric judge scores. Sandbagging uses Apollo's deterministic tag
parser, not an LLM judge. The default LLM judge runs through OpenRouter using
the `glm-5.2` alias from `model_deployments.yaml`, so set
`OPENROUTER_API_KEY` before grading roleplaying rollouts. You can pass any
OpenRouter alias from that YAML, or a raw OpenRouter model ID, with
`--judge-model`.

Then extract answer-token decoder activations into the Truth Spec-compatible
HDF5 layout:

```bash
uv run --no-sync intelligent-liars extract-activations \
  --layers sparse \
  --batch-size 1
```

The default output is:

```text
results/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5
```

Inside the file, activations are stored at `layer_{idx}/{task}` with shape
`[masked_answer_tokens, hidden_dim]`. The extractor also writes
`metadata/{task}` with labels, source indices, output indices, rendered prompts,
messages, raw labels, label schemas, source datasets, per-example metadata, and
token splits. `graded_deception="unknown"` and other non-binary labels now fail
fast if they leave no honest/deceptive examples, because the first probe path
expects usable binary labels.

For insider trading transcripts:

```bash
uv run --no-sync intelligent-liars run-insider-trading-sweep \
  --samples-per-prompt 1 \
  --max-new-tokens 512
```

Output:

```text
data/insider_trading/qwen3-vl-8b-thinking-generations.json
```

This command preflights the OpenRouter judge before loading Qwen, generates the
transcripts with unknown labels, then runs the Apollo task-specific judge
prompts on the generated file. To run generation and grading as separate steps:

```bash
uv run --no-sync intelligent-liars generate-insider-trading \
  --samples-per-prompt 1 \
  --max-new-tokens 512 \
  --label-mode unknown
uv run --no-sync intelligent-liars grade-insider-trading
```

`--label-mode heuristic` exists only for smoke tests because it uses string
heuristics, not the paper's grader.

## Environment

`.env` is ignored by git. The only required value for the current judge path is
`OPENROUTER_API_KEY`. `HF_TOKEN` is optional and only needed if Hugging Face
requires authentication for model downloads.

Do not put remote-GPU tuning in the default `.env`. Set values like `HF_HOME`,
`CUDA_VISIBLE_DEVICES`, `PYTORCH_CUDA_ALLOC_CONF`, or `TOKENIZERS_PARALLELISM`
per machine/run when needed.

## Model Loading

Truth Spec's original loader in `references/truth_spec/src/models.py` is built
around NNsight `LanguageModel` and text-only model families. This repo uses
`src/intelligent_liars/models.py` for the stock Qwen3-VL load path.
The loader hardcodes one supported model:

- `Qwen/Qwen3-VL-8B-Thinking`
- `Qwen3VLForConditionalGeneration.from_pretrained(..., dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")`
- `AutoProcessor.from_pretrained(...)` for the paired tokenizer/chat template

This keeps model loading aligned with the processor we need for Qwen3-VL
inference and token masking. Activation extraction and later steering should be
implemented as a separate backend. `src/intelligent_liars/nnsight_backend.py`
sets up the first NNsight path for tracing Qwen3-VL text-decoder activations at
`model.model.language_model.layers[layer_idx]`. Install `flash-attn` separately
on the CUDA box before `--check-model` or `--check-nnsight`; it is not a base
dependency because it is hardware-specific.

## External References

The Apollo and Truth Spec repos are useful local references, but they are not
vendored here. Clone them under `references/` when needed:

```bash
mkdir -p references
git clone --depth 1 https://github.com/ApolloResearch/deception-detection references/deception-detection
git clone --depth 1 https://github.com/zfying/truth_spec references/truth_spec
```

Current local reference commits when this repo was initialized:

- `deception-detection`: `f8ec401`
- `truth_spec`: `b95fdc1`
