# Qwen3-VL Token/Mask Builder Reference

## Core Distinction

Wrong mental model:

`detection_mask` is a model input that tells Qwen3-VL what to care about.

Correct model:

`detection_mask` is our extraction-time row selector. Qwen3-VL computes hidden states for the whole sequence. Afterward, we use `detection_mask` to keep only the assistant answer-token rows.

## Objects

| Object | Shape | Used by | Meaning |
| --- | --- | --- | --- |
| `input_ids` | `[batch, seq]` | model | Integer token ids for the formatted prompt/answer sequence. |
| `attention_mask` | `[batch, seq]` | model | Which positions are real tokens versus padding. |
| `pixel_values`, `image_grid_thw` | vision-specific | model | Image tensors and visual layout metadata for multimodal prompts. |
| `detection_mask` | `[batch, seq]` | our extraction code | Which token positions should become probe-training rows. |
| `label` | `[batch]` or per example | probe trainer | True/false class for the selected activation rows. Not shown to the model. |

## Toy Example

Dialogue:

```text
User: What is the capital of France?
Assistant: Paris
```

Toy processor output:

```python
input_ids      = [101, 202, 431, 88, 62, 901, 19, 777, 303, 8842, 102, 0]
token_text     = ["<bos>", "<user>", "What", "is", "the", "capital", "of", "France?", "<assistant>", "Paris", "<eos>", "<pad>"]
attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
detection_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

`attention_mask` marks all real non-pad tokens.

`detection_mask` marks only `Paris`.

## Robust Answer-Position Method

For text-only examples, offset mappings can work.

For multimodal Qwen3-VL examples, prefer answer-subsequence matching against the final processor-produced `input_ids`.

```python
answer_ids = processor.tokenizer(
    answer_text,
    add_special_tokens=False,
).input_ids

positions = find_subsequence(inputs["input_ids"][0].tolist(), answer_ids)

detection_mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
detection_mask[0, positions] = True

decoded = processor.tokenizer.decode(inputs["input_ids"][0][detection_mask[0]])
assert answer_text.strip() in decoded
```

## Activation Extraction

The model produces hidden states:

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    use_cache=False,
)

hidden_states = outputs.hidden_states
```

Conceptual shape:

```text
hidden_states: [batch, seq, layer, hidden]
```

Then:

```python
X = hidden_states[detection_mask]
y = labels_for_examples
```

`X` is what the truth probe trains on.

## Why The Mask Matters

If the mask selects answer tokens, the probe can learn truth-related answer representations.

If the mask selects prompt tokens, role tokens, image placeholder tokens, or padding, the probe can learn formatting, topic, modality, or length artifacts instead.

## Implementation Checklist

1. Build final model input with `AutoProcessor`, not a plain tokenizer for multimodal examples.
2. Keep all processor-returned tensors needed by Qwen3-VL.
3. Tokenize the answer text by itself.
4. Find answer ids inside final `input_ids`.
5. Set `detection_mask` true only at those positions.
6. Decode masked ids and verify they match the answer.
7. Run model forward with `output_hidden_states=True`.
8. Index hidden states using `detection_mask`.
9. Train probe on selected rows and true/false labels.

## One-Sentence Compression

The processor prepares what Qwen3-VL reads; the detection mask records which answer-token hidden states we keep after Qwen3-VL reads it.
