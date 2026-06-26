from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from intelligent_liars.activation_types import (
    ActivationExample,
    DetectionMask,
    LocalDetectionSpan,
    MergedMessageSpan,
    ProcessorBatch,
    RenderedDetectionExample,
    _content_text,
)
from intelligent_liars.rollouts import Message


class QwenProcessorTokenizer:
    """Processor wrapper for text-only Truth Spec activation inputs."""

    def __init__(self, *, processor: Any, tokenizer: Any):
        self.processor = processor
        self.tokenizer = tokenizer

    def build_batch(
        self,
        examples: Sequence[ActivationExample],
        *,
        max_length: int | None = None,
    ) -> ProcessorBatch:
        rendered_examples = tuple(
            render_activation_example(self.processor, example.messages)
            for example in examples
        )
        inputs = _build_qwen_inputs(
            processor=self.processor,
            tokenizer=self.tokenizer,
            examples=examples,
            rendered_examples=rendered_examples,
            max_length=max_length,
        )
        return ProcessorBatch(
            examples=tuple(examples),
            inputs=inputs,
            expected_texts=tuple(
                rendered.detected_text for rendered in rendered_examples
            ),
            detected_texts=tuple(
                rendered.detected_text for rendered in rendered_examples
            ),
            char_spans=tuple(rendered.char_spans for rendered in rendered_examples),
            rendered_texts=tuple(
                rendered.rendered_text for rendered in rendered_examples
            ),
            preserved_input_keys=tuple(sorted(inputs)),
        )


class DetectionMaskBuilder:
    """Build extraction-only answer-token masks from processor-tokenized inputs."""

    def __init__(self, *, tokenizer: Any, verify: bool = True):
        self.tokenizer = tokenizer
        self.verify = verify

    def build(self, batch: ProcessorBatch) -> DetectionMask:
        mask, decoded_texts, token_positions = _build_message_detection_mask_detail(
            tokenizer=self.tokenizer,
            input_ids=batch.inputs["input_ids"],
            offset_mapping=batch.inputs.get("offset_mapping"),
            detected_texts=batch.detected_texts,
            char_spans=batch.char_spans,
            attention_mask=batch.inputs.get("attention_mask"),
        )
        if self.verify:
            verify_decoded_masks(
                expected_texts=batch.expected_texts,
                decoded_texts=decoded_texts,
            )
        return DetectionMask(
            tensor=mask,
            decoded_texts=tuple(decoded_texts),
            detected_texts=batch.detected_texts,
            char_spans=batch.char_spans,
            token_positions=tuple(tuple(positions) for positions in token_positions),
        )


def build_answer_detection_mask(
    *,
    tokenizer: Any,
    input_ids: Any,
    expected_texts: Sequence[str],
    attention_mask: Any | None = None,
) -> tuple[Any, list[str]]:
    """Mask assistant answer token spans by subsequence search in final processor input IDs."""
    mask, decoded_texts, _token_positions = _build_answer_detection_mask_detail(
        tokenizer=tokenizer,
        input_ids=input_ids,
        expected_texts=expected_texts,
        attention_mask=attention_mask,
    )
    return mask, decoded_texts


def _build_answer_detection_mask_detail(
    *,
    tokenizer: Any,
    input_ids: Any,
    expected_texts: Sequence[str],
    attention_mask: Any | None = None,
) -> tuple[Any, list[str], list[list[int]]]:
    """Return answer-token masks plus original token positions for feature metadata."""
    import torch

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    decoded_texts: list[str] = []
    token_positions: list[list[int]] = []
    for row_idx, expected_text in enumerate(expected_texts):
        row_ids = input_ids[row_idx].detach().cpu().tolist()
        usable = _usable_position_mask(
            input_ids=input_ids[row_idx],
            attention_mask=None if attention_mask is None else attention_mask[row_idx],
        )
        token_span = _find_expected_text_span(
            tokenizer=tokenizer,
            row_ids=row_ids,
            usable_mask=usable.detach().cpu().tolist(),
            expected_text=expected_text,
        )
        if token_span is None:
            decoded_full = tokenizer.decode(
                [
                    token_id
                    for token_id, keep in zip(row_ids, usable.detach().cpu().tolist())
                    if keep
                ],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            raise ValueError(
                "Could not align expected assistant text to token IDs. "
                f"Expected {expected_text[:120]!r}; decoded usable input starts {decoded_full[:240]!r}"
            )
        start, end = token_span
        mask[row_idx, start:end] = True
        mask[row_idx] &= usable
        positions = (
            torch.nonzero(mask[row_idx], as_tuple=False)
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        token_positions.append([int(position) for position in positions])
        decoded_texts.append(
            tokenizer.decode(
                input_ids[row_idx][mask[row_idx]].detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return mask, decoded_texts, token_positions


def _build_message_detection_mask_detail(
    *,
    tokenizer: Any,
    input_ids: Any,
    offset_mapping: Any,
    detected_texts: Sequence[str],
    char_spans: Sequence[Sequence[tuple[int, int]]],
    attention_mask: Any | None = None,
) -> tuple[Any, list[str], list[list[int]]]:
    """Build masks from rendered-text character spans, not global answer-token search."""
    import torch

    if offset_mapping is None:
        raise ValueError(
            "Offset mapping is required for message-span activation masking."
        )
    offsets = (
        offset_mapping
        if hasattr(offset_mapping, "to")
        else torch.as_tensor(offset_mapping)
    )
    offsets = offsets.to(input_ids.device) if hasattr(input_ids, "device") else offsets

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    decoded_texts: list[str] = []
    token_positions: list[list[int]] = []
    for row_idx, spans in enumerate(char_spans):
        usable = _usable_position_mask(
            input_ids=input_ids[row_idx],
            attention_mask=None if attention_mask is None else attention_mask[row_idx],
        )
        row_offsets = offsets[row_idx]
        token_starts = row_offsets[:, 0]
        token_ends = row_offsets[:, 1]
        row_mask = torch.zeros_like(input_ids[row_idx], dtype=torch.bool)
        for start_char, end_char in spans:
            row_mask |= (
                (token_ends > int(start_char))
                & (token_starts < int(end_char))
                & (token_ends > token_starts)
            )
        row_mask &= usable
        mask[row_idx] = row_mask
        positions = (
            torch.nonzero(row_mask, as_tuple=False).flatten().detach().cpu().tolist()
        )
        token_positions.append([int(position) for position in positions])
        decoded_texts.append(
            tokenizer.decode(
                input_ids[row_idx][row_mask].detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        if detected_texts[row_idx] and not positions:
            raise ValueError(
                f"Detection mask for row {row_idx} is empty despite detected text."
            )
    return mask, decoded_texts, token_positions


def verify_decoded_masks(
    *,
    expected_texts: Sequence[str],
    decoded_texts: Sequence[str],
) -> None:
    for idx, (expected, decoded) in enumerate(
        zip(expected_texts, decoded_texts, strict=True)
    ):
        if _decoded_detection_text_matches(expected=expected, decoded=decoded):
            continue
        raise ValueError(
            f"Detection mask mismatch at batch row {idx}: expected {expected[:160]!r}, got {decoded[:160]!r}"
        )


def _decoded_detection_text_matches(*, expected: str, decoded: str) -> bool:
    expected_norm = _normalise_for_compare(expected)
    decoded_norm = _normalise_for_compare(decoded)
    if expected_norm == decoded_norm:
        return True
    if not expected_norm:
        return not decoded_norm
    start = decoded_norm.find(expected_norm)
    if start < 0:
        return False
    prefix = decoded_norm[:start]
    suffix = decoded_norm[start + len(expected_norm) :]
    return _is_quote_boundary_extra(prefix) and _is_quote_boundary_extra(suffix)


def _is_quote_boundary_extra(text: str) -> bool:
    # Qwen tokenization can include neighboring punctuation with a detected
    # punctuation-only span, e.g. expected "." -> "!." or "%.".
    return all(char.isspace() or not char.isalnum() for char in text)


def render_activation_example(
    processor: Any, messages: Sequence[Message]
) -> RenderedDetectionExample:
    merged_messages, spans = merge_consecutive_messages_with_spans(messages)
    rendered_text = processor.apply_chat_template(
        _qwen_messages(merged_messages),
        tokenize=False,
        add_generation_prompt=False,
    )
    absolute_spans = _absolute_detection_spans(
        rendered_text=rendered_text,
        merged_messages=merged_messages,
        spans=spans,
    )
    detected_text = "".join(span.text for span in spans)
    return RenderedDetectionExample(
        messages=tuple(merged_messages),
        rendered_text=rendered_text,
        detected_text=detected_text,
        char_spans=tuple((span.start, span.end) for span in absolute_spans),
    )


def merge_consecutive_messages_with_spans(
    messages: Sequence[Message],
) -> tuple[list[Message], list[MergedMessageSpan]]:
    merged: list[Message] = []
    spans: list[MergedMessageSpan] = []
    for message in messages:
        role = str(message["role"])
        content = message.get("content", "")
        detect = bool(message.get("detect", False))
        text = _content_text(content)
        if merged and merged[-1]["role"] == role:
            merged_index = len(merged) - 1
            start = len(_content_text(merged[-1].get("content", "")))
            merged[-1]["content"] = _append_content(
                merged[-1].get("content", ""), content
            )
            merged[-1]["detect"] = bool(merged[-1].get("detect", False)) or detect
        else:
            merged_index = len(merged)
            start = 0
            merged.append({"role": role, "content": content, "detect": detect})
        if detect and text:
            spans.append(
                MergedMessageSpan(merged_index, start, start + len(text), text)
            )
    return merged, spans


def _absolute_detection_spans(
    *,
    rendered_text: str,
    merged_messages: Sequence[Message],
    spans: Sequence[MergedMessageSpan],
) -> list[LocalDetectionSpan]:
    spans_by_message: dict[int, list[MergedMessageSpan]] = {}
    for span in spans:
        spans_by_message.setdefault(span.message_index, []).append(span)

    absolute: list[LocalDetectionSpan] = []
    search_cursor = 0
    for message_index, message in enumerate(merged_messages):
        content_text = _content_text(message.get("content", ""))
        if not content_text:
            continue
        start_char, end_char = _find_content_flexible(
            rendered_text, content_text, search_cursor
        )
        if start_char < 0:
            raise ValueError(
                "Could not find merged message content in rendered chat template. "
                f"Content starts {content_text[:120]!r}; rendered starts {rendered_text[:240]!r}"
            )
        for span in spans_by_message.get(message_index, []):
            absolute.append(
                LocalDetectionSpan(
                    start=start_char + span.start,
                    end=start_char + span.end,
                    text=span.text,
                )
            )
        search_cursor = end_char
    return absolute


def _find_content_flexible(
    full_text: str, content: str, start_pos: int = 0
) -> tuple[int, int]:
    exact = full_text.find(content, start_pos)
    if exact >= 0:
        return exact, exact + len(content)
    content_stripped = content.strip()
    if not content_stripped:
        return -1, -1
    escaped = re.escape(content_stripped)
    pattern = re.sub(r"\\ ", r"\\s+", escaped)
    match = re.search(pattern, full_text[start_pos:])
    if match:
        return start_pos + match.start(), start_pos + match.end()
    return -1, -1


def _append_content(left: Any, right: Any) -> Any:
    if isinstance(left, list) or isinstance(right, list):
        return _content_blocks(left) + _content_blocks(right)
    return str(left) + str(right)


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return list(content)
    if isinstance(content, Mapping):
        return [dict(content)]
    return [{"type": "text", "text": str(content)}]


def _build_qwen_inputs(
    *,
    processor: Any,
    tokenizer: Any,
    examples: Sequence[ActivationExample],
    rendered_examples: Sequence[RenderedDetectionExample],
    max_length: int | None,
) -> dict[str, Any]:
    del examples
    messages = [_qwen_messages(rendered.messages) for rendered in rendered_examples]
    rendered_texts = [rendered.rendered_text for rendered in rendered_examples]
    tokenized = tokenizer(
        rendered_texts,
        padding=True,
        truncation=max_length is not None,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "truncation": max_length is not None,
                "max_length": max_length,
            },
        )
    except TypeError:
        inputs = processor(
            text=rendered_texts,
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_tensors="pt",
        )
    merged_inputs = dict(inputs)
    tokenized_inputs = dict(tokenized)
    if not _same_token_ids(
        merged_inputs.get("input_ids"), tokenized_inputs.get("input_ids")
    ):
        raise ValueError(
            "Processor and tokenizer input_ids differ, so offset mappings cannot be aligned safely."
        )
    else:
        merged_inputs["offset_mapping"] = tokenized_inputs["offset_mapping"]
        if (
            "attention_mask" not in merged_inputs
            and "attention_mask" in tokenized_inputs
        ):
            merged_inputs["attention_mask"] = tokenized_inputs["attention_mask"]
    return merged_inputs


def _same_token_ids(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return _tensorish_to_list(left) == _tensorish_to_list(right)


def _tensorish_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _qwen_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": str(message["role"]),
            "content": _qwen_content(message.get("content", "")),
        }
        for message in messages
    ]


def _qwen_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _find_expected_text_span(
    *,
    tokenizer: Any,
    row_ids: Sequence[int],
    usable_mask: Sequence[bool],
    expected_text: str,
) -> tuple[int, int] | None:
    candidates = _token_candidates(tokenizer, expected_text)
    best: tuple[int, int] | None = None
    for candidate in candidates:
        if not candidate:
            continue
        for start in range(0, len(row_ids) - len(candidate) + 1):
            end = start + len(candidate)
            if not all(usable_mask[start:end]):
                continue
            if list(row_ids[start:end]) == candidate:
                best = (start, end)
    return best


def _token_candidates(tokenizer: Any, text: str) -> list[list[int]]:
    variants = [text, text.strip(), "\n" + text, " " + text]
    candidates: list[list[int]] = []
    for variant in variants:
        if not variant:
            continue
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if token_ids and token_ids not in candidates:
            candidates.append(token_ids)
    return candidates


def _usable_position_mask(
    *,
    input_ids: Any,
    attention_mask: Any | None,
) -> Any:
    import torch

    usable = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        usable &= attention_mask.to(dtype=torch.bool)
    return usable


def _shift_mask_for_next_token_logits(detection_mask: Any) -> Any:
    import torch

    logit_mask = torch.zeros_like(detection_mask, dtype=torch.bool)
    logit_mask[:, :-1] = detection_mask[:, 1:]
    return logit_mask


def _normalise_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
