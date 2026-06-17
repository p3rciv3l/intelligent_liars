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

## Environment

`.env` is ignored by git. Commit changes to `.env.example` when adding a new
configuration variable.

Important defaults:

- `MODEL_NAME=Qwen/Qwen3-VL-8B-Thinking`
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `TOKENIZERS_PARALLELISM=false`

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
