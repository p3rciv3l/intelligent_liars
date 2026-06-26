from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import intelligent_liars.activations as activations_module
from intelligent_liars.activations import (
    ActivationDataset,
    ActivationExample,
    ActivationExtractionSettings,
    DECEPTIVE,
    HONEST,
    LabelSchema,
    RenderedDetectionExample,
    STEM_TASKS_PART_2,
    SKIP,
    build_answer_detection_mask,
    default_activation_output_path,
    extract_rollout_activations,
    extract_dataset_activations,
    load_named_activation_examples,
    load_rollout_activation_examples,
    merge_activation_hdf5_shards,
    normalise_deception_label,
    parse_layer_spec,
    render_activation_example,
    verify_decoded_masks,
    write_activation_hdf5,
    _build_qwen_inputs,
    _build_message_detection_mask_detail,
)
from intelligent_liars.activation_backends import prepare_model_inputs, qwen3_vl_decoder_surface
from intelligent_liars.models import DEFAULT_MODEL_ID, ModelBundle, ModelLoadConfig
from intelligent_liars.run_control import LockHeldError, acquire_lock, lock_payload
from intelligent_liars.nnsight_backend import NnsightActivationBackend, _save_selected_rows


def test_parse_layer_spec_supports_all_sparse_lists_and_ranges():
    assert parse_layer_spec("all", 4) == (0, 1, 2, 3)
    assert parse_layer_spec("sparse", 12) == (3, 7, 11)
    assert parse_layer_spec("0, 2, 2, 4-5", 8) == (0, 2, 4, 5)

    with pytest.raises(ValueError):
        parse_layer_spec("8", 8)


def test_default_activation_output_path_matches_project_contract():
    assert (
        default_activation_output_path(DEFAULT_MODEL_ID)
        == Path("artifacts/activations/extracted_feats_all_layers_qwen3-vl-8b-thinking.h5")
    )


def test_normalise_deception_label_matches_truth_spec_rollout_labels():
    assert normalise_deception_label("honest") == HONEST
    assert normalise_deception_label("deceptive") == DECEPTIVE
    assert normalise_deception_label("unknown") == SKIP
    assert normalise_deception_label("explicit", schema=LabelSchema.INSIDER_REPORT) == HONEST
    assert normalise_deception_label("concealed", schema=LabelSchema.INSIDER_REPORT) == DECEPTIVE
    assert normalise_deception_label("skip", schema=LabelSchema.INSIDER_REPORT) == SKIP
    assert normalise_deception_label("refusal", schema=LabelSchema.INSIDER_REPORT) == SKIP
    assert normalise_deception_label(7, schema=LabelSchema.ROLEPLAYING_1_TO_7) == DECEPTIVE
    assert normalise_deception_label(1, schema=LabelSchema.ROLEPLAYING_1_TO_7) == HONEST
    assert normalise_deception_label(4, schema=LabelSchema.ROLEPLAYING_1_TO_7) == SKIP
    assert normalise_deception_label(1, schema=LabelSchema.TRUTH_SPEC_NUMERIC) == DECEPTIVE
    with pytest.raises(ValueError):
        normalise_deception_label(7)
    with pytest.raises(ValueError):
        normalise_deception_label(1)
    with pytest.raises(ValueError):
        normalise_deception_label(True, schema=LabelSchema.STRING_DECEPTION)
    with pytest.raises(ValueError):
        normalise_deception_label("concealed", schema=LabelSchema.STRING_DECEPTION)


def test_load_rollout_activation_examples_replaces_empty_assistant_placeholder(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [
                            {"role": "system", "content": "sys", "detect": False},
                            {"role": "user", "content": "question", "detect": False},
                            {"role": "assistant", "content": "", "detect": True},
                        ],
                        "output_strs": ["answer"],
                        "graded_deception": [7],
                        "metadata": {"source_index": 13},
                    }
                ]
            }
        )
    )

    examples = load_rollout_activation_examples(rollout_path)

    assert len(examples) == 1
    assert examples[0].task == "roleplaying__plain"
    assert examples[0].source_index == 13
    assert examples[0].label == DECEPTIVE
    assert [message["role"] for message in examples[0].messages] == ["system", "user", "assistant"]
    assert examples[0].messages[-1] == {"role": "assistant", "content": "answer", "detect": True}
    assert examples[0].label_schema == LabelSchema.ROLEPLAYING_1_TO_7
    assert examples[0].raw_label == 7


def test_geometry_of_truth_loader_strips_terminal_sentence_punctuation(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "geometry_of_truth"
    data_dir.mkdir(parents=True)
    (data_dir / "cities.csv").write_text("statement,label\nSpringfield is in Oregon!,1\n", encoding="utf-8")
    monkeypatch.setattr(activations_module, "GOT_BEST_CSVS", ("cities.csv",))

    examples = load_named_activation_examples("geometry_of_truth__best", project_root=tmp_path)

    assert len(examples) == 1
    assert examples[0].messages[-2]["content"] == "Springfield is in Oregon"
    assert examples[0].messages[-1] == {"role": "assistant", "content": ".", "detect": True}


def test_render_activation_example_merges_assistant_prefix_and_masks_only_detected_span():
    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            del tokenize, add_generation_prompt
            return "".join(f"<{message['role']}>{message['content'][0]['text']}" for message in messages)

    rendered = render_activation_example(
        FakeProcessor(),
        [
            {"role": "user", "content": "Why?", "detect": False},
            {"role": "assistant", "content": "Student:", "detect": False},
            {"role": "assistant", "content": " I procrastinated.", "detect": True},
        ],
    )

    assert rendered.rendered_text.count("<assistant>") == 1
    assert rendered.detected_text == " I procrastinated."
    start, end = rendered.char_spans[0]
    assert rendered.rendered_text[start:end] == " I procrastinated."


def test_load_rollout_activation_examples_keeps_prefix_context_out_of_detected_text(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [
                            {"role": "user", "content": "Why?", "detect": False},
                            {"role": "assistant", "content": "Student:", "detect": False},
                            {"role": "assistant", "content": "", "detect": True},
                        ],
                        "output_messages": [
                            {"role": "assistant", "content": " I procrastinated.", "detect": True}
                        ],
                        "output_strs": [" I procrastinated."],
                        "graded_deception": [7],
                        "metadata": {"source_index": 3},
                    }
                ]
            }
        )
    )

    example = load_rollout_activation_examples(rollout_path)[0]

    assert example.detected_text == " I procrastinated."
    assert example.expected_text == " I procrastinated."
    assert example.messages[-2]["content"] == ""
    assert example.messages[-1] == {
        "role": "assistant",
        "content": " I procrastinated.",
        "detect": True,
    }


def test_load_rollout_activation_examples_selects_matching_output_message(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question", "detect": False}],
                        "output_strs": ["first", "second"],
                        "output_messages": [
                            {"role": "assistant", "content": "first", "detect": True},
                            {"role": "assistant", "content": "second", "detect": True},
                        ],
                        "graded_deception": [7, 1],
                    }
                ]
            }
        )
    )

    examples = load_rollout_activation_examples(rollout_path)

    assert [example.output_index for example in examples] == [0, 1]
    assert [example.detected_text for example in examples] == ["first", "second"]
    assert [example.messages[-1]["content"] for example in examples] == ["first", "second"]
    assert [example.raw_label for example in examples] == [7, 1]
    assert [example.label for example in examples] == [DECEPTIVE, HONEST]


def test_load_rollout_activation_examples_rejects_detected_non_assistant_output(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question", "detect": False}],
                        "output_strs": ["answer"],
                        "output_messages": [{"role": "user", "content": "not an answer", "detect": True}],
                        "graded_deception": [7],
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="Only assistant output messages"):
        load_rollout_activation_examples(rollout_path)


def test_load_rollout_activation_examples_defaults_non_assistant_output_to_not_detected(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question", "detect": False}],
                        "output_strs": ["answer"],
                        "output_messages": [
                            {"role": "user", "content": "followup"},
                            {"role": "assistant", "content": "answer"},
                        ],
                        "graded_deception": [7],
                    }
                ]
            }
        )
    )

    example = load_rollout_activation_examples(rollout_path)[0]

    assert example.messages[-2] == {"role": "user", "content": "followup", "detect": False}
    assert example.messages[-1] == {"role": "assistant", "content": "answer", "detect": True}


def test_load_rollout_activation_examples_reconstructs_each_output_str(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question", "detect": False}],
                        "output_strs": ["first", "second"],
                        "graded_deception": [7, 1],
                    }
                ]
            }
        )
    )

    examples = load_rollout_activation_examples(rollout_path)

    assert [example.detected_text for example in examples] == ["first", "second"]
    assert [example.messages[-1]["content"] for example in examples] == ["first", "second"]


def test_load_named_claims_examples_maps_claim_and_negation(tmp_path):
    claims_dir = tmp_path / "data" / "claims"
    claims_dir.mkdir(parents=True)
    (claims_dir / "claims__definitional_gemini_600_full.csv").write_text(
        "Claim,Negated Claim\nA triangle has three sides.,A triangle has four sides.\n"
    )

    examples = load_named_activation_examples(
        "claims__definitional_gemini_600_full",
        project_root=tmp_path,
    )

    assert [example.label for example in examples] == [HONEST, DECEPTIVE]
    assert [example.detected_text for example in examples] == [
        "A triangle has three sides.",
        "A triangle has four sides.",
    ]


def test_load_named_repe_if_dishonest_inverts_truth_labels(tmp_path):
    repe_dir = tmp_path / "data" / "repe"
    repe_dir.mkdir(parents=True)
    (repe_dir / "true_false_facts.csv").write_text(
        "statement,label\nThe sky is blue.,1\nThe sky is green.,0\n"
    )

    examples = load_named_activation_examples(
        "repe_honesty__IF_dishonest",
        project_root=tmp_path,
    )

    assert [example.label for example in examples] == [DECEPTIVE, HONEST]
    assert all("dishonest person" in example.messages[0]["content"] for example in examples)
    assert [example.detected_text for example in examples] == ["The sky is blue.", "The sky is green."]


def test_sycophancy_requires_generated_model_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/sycophancy/qwen3-vl-8b-thinking"):
        load_named_activation_examples(
            "sycophancy__mmlu_stem_conf_part_2",
            project_root=tmp_path,
            generated_model="qwen3-vl-8b-thinking",
        )


def test_sycophancy_qwen_generated_source_is_allowed(tmp_path):
    _write_sycophancy_files(tmp_path / "data" / "sycophancy" / "qwen3-vl-8b-thinking")

    examples = load_named_activation_examples(
        "sycophancy__mmlu_stem_conf_part_2",
        project_root=tmp_path,
        generated_model="qwen3-vl-8b-thinking",
    )

    assert len(examples) == 2
    assert [example.label for example in examples] == [DECEPTIVE, HONEST]
    assert {example.metadata["sycophancy_source_kind"] for example in examples} == {"generated"}
    assert {example.metadata["generated_model"] for example in examples} == {"qwen3-vl-8b-thinking"}


def test_combined_propagates_sycophancy_source_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "intelligent_liars.activations.COMBINED_COMPONENT_TASKS",
        ("sycophancy__mmlu_stem_conf_part_2",),
    )

    with pytest.raises(FileNotFoundError, match="data/sycophancy/qwen3-vl-8b-thinking"):
        load_named_activation_examples(
            "combined__unit",
            project_root=tmp_path,
            generated_model="qwen3-vl-8b-thinking",
        )


def test_sandbagging_rollout_trim_detects_final_step_or_skips(tmp_path):
    rollout_path = tmp_path / "sandbagging_v2__wmdp_mmlu__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["reasoning <step4>final answer"],
                        "graded_deception": ["deceptive"],
                    },
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["reasoning only"],
                        "graded_deception": ["honest"],
                    },
                ]
            }
        )
    )

    examples = load_rollout_activation_examples(rollout_path)

    assert examples[0].label == DECEPTIVE
    assert examples[0].detected_text == "<step4>final answer"
    assert examples[0].messages[-2] == {"role": "assistant", "content": "reasoning ", "detect": False}
    assert examples[1].label == SKIP


def test_insider_trading_rollout_trim_missing_marker_skips_labels(tmp_path):
    insider_path = tmp_path / "insider_trading__upscale__qwen3-vl-8b-thinking.json"
    insider_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["I will conceal the trade."],
                        "graded_deception": ["concealed"],
                    }
                ]
            }
        )
    )
    doubledown_path = tmp_path / "insider_trading_doubledown__upscale__qwen3-vl-8b-thinking.json"
    doubledown_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["No, I did not know."],
                        "graded_deception": ["deceptive"],
                    }
                ]
            }
        )
    )

    assert load_rollout_activation_examples(insider_path)[0].label == SKIP
    assert load_rollout_activation_examples(doubledown_path)[0].label == SKIP


def test_insider_trading_doubledown_uses_string_deception_labels(tmp_path):
    transcript_path = tmp_path / "data" / "insider_trading" / "qwen3-vl-8b-thinking-generations.json"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "transcript": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "trade?"},
                        {"role": "assistant", "content": "trade"},
                        {"role": "user", "content": "report?"},
                        {"role": "assistant", "content": "report"},
                        {"role": "user", "content": "did you know?"},
                        {"role": "assistant", "content": "thinking Action: send_message No, I did not."},
                    ],
                    "metadata": {
                        "report_label": "concealed",
                        "doubling_down_label": "deceptive",
                    },
                }
            ]
        )
    )

    examples = load_named_activation_examples(
        "insider_trading_doubledown__upscale",
        project_root=tmp_path,
    )

    assert len(examples) == 1
    assert examples[0].label == DECEPTIVE
    assert examples[0].label_schema == LabelSchema.STRING_DECEPTION
    assert examples[0].raw_label == "deceptive"
    assert examples[0].detected_text == "Action: send_message No, I did not."


def test_doubledown_rollout_labels_do_not_use_report_schema(tmp_path):
    rollout_path = tmp_path / "insider_trading_doubledown__onpolicy__qwen3-vl-8b-thinking.json"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["thinking Action: send_message No, I did not."],
                        "graded_deception": ["deceptive"],
                    }
                ]
            }
        )
    )

    examples = load_rollout_activation_examples(rollout_path)

    assert examples[0].label == DECEPTIVE
    assert examples[0].label_schema == LabelSchema.STRING_DECEPTION


def test_named_insider_task_prefers_rollout_file_when_available(tmp_path):
    transcript_path = tmp_path / "data" / "insider_trading" / "qwen3-vl-8b-thinking-generations.json"
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "transcript": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "Action: send_message honest report"},
                    ],
                    "metadata": {"report_label": "explicit"},
                }
            ]
        )
    )
    rollout_path = tmp_path / "data" / "rollouts" / "insider_trading__onpolicy__qwen3-vl-8b-thinking.json"
    rollout_path.parent.mkdir(parents=True)
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["thinking before report. Action: send_message concealed report"],
                        "graded_deception": ["concealed"],
                        "metadata": {"source_index": 7},
                    }
                ]
            }
        )
    )

    examples = load_named_activation_examples("insider_trading__onpolicy", project_root=tmp_path)

    assert len(examples) == 1
    assert examples[0].label == DECEPTIVE
    assert examples[0].label_schema == LabelSchema.INSIDER_REPORT
    assert examples[0].source_dataset == str(rollout_path)
    assert examples[0].detected_text == "Action: send_message concealed report"


def test_extract_rollout_activations_fails_on_all_ungraded_rollouts_without_writing(tmp_path):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    output_path = tmp_path / "acts.h5"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["answer"],
                        "graded_deception": ["unknown"],
                    }
                ]
            }
        )
    )
    bundle = ModelBundle(
        model=object(),
        processor=object(),
        tokenizer=object(),
        model_id=DEFAULT_MODEL_ID,
        config=ModelLoadConfig(),
    )

    with pytest.raises(ValueError, match="Run grading first"):
        extract_rollout_activations(
            bundle=bundle,
            rollout_path=rollout_path,
            output_path=output_path,
            settings=ActivationExtractionSettings(layers=(0,)),
        )

    assert not output_path.exists()


def test_extract_rollout_activations_rejects_empty_detected_answer_for_labeled_example(tmp_path):
    output_path = tmp_path / "acts.h5"
    dataset = ActivationDataset(
        task="roleplaying__plain",
        examples=(
            ActivationExample(
                task="roleplaying__plain",
                source_index=0,
                output_index=0,
                messages=[{"role": "assistant", "content": "", "detect": True}],
                label=DECEPTIVE,
                label_schema=LabelSchema.ROLEPLAYING_1_TO_7,
                raw_label=7,
            ),
        ),
    )
    bundle = ModelBundle(
        model=None,
        processor=object(),
        tokenizer=object(),
        model_id=DEFAULT_MODEL_ID,
        config=ModelLoadConfig(),
    )
    backend = SimpleNamespace(
        name="fake-backend",
        capture=lambda **kwargs: SimpleNamespace(activations_by_layer={0: []}, logits=None),
        surface_for_decoder_layer=lambda layer: SimpleNamespace(name=f"layer_{layer}"),
    )

    def fake_build_batch(self, examples, max_length=None):
        del self, max_length
        return SimpleNamespace(
            examples=tuple(examples),
            inputs={"input_ids": "ids"},
            preserved_input_keys=(),
            rendered_texts=("",),
        )

    def fake_build_mask(self, batch):
        del self, batch
        return SimpleNamespace(
            tensor="mask",
            decoded_texts=("",),
            detected_texts=("",),
            char_spans=((),),
            token_positions=((),),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(activations_module.QwenProcessorTokenizer, "build_batch", fake_build_batch)
    monkeypatch.setattr(activations_module.DetectionMaskBuilder, "build", fake_build_mask)

    try:
        with pytest.raises(ValueError, match="No answer tokens selected"):
            extract_dataset_activations(
                bundle=bundle,
                dataset=dataset,
                output_path=output_path,
                settings=ActivationExtractionSettings(layers=(0,)),
                backend=backend,
            )
    finally:
        monkeypatch.undo()

    assert not output_path.exists()


def test_extract_rollout_activations_delegates_to_dataset_path(tmp_path, monkeypatch):
    rollout_path = tmp_path / "roleplaying__plain__qwen3-vl-8b-thinking.json"
    output_path = tmp_path / "acts.h5"
    rollout_path.write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "input_messages": [{"role": "user", "content": "question"}],
                        "output_strs": ["answer"],
                        "graded_deception": [7],
                    }
                ]
            }
        )
    )
    bundle = ModelBundle(
        model=None,
        processor=object(),
        tokenizer=object(),
        model_id=DEFAULT_MODEL_ID,
        config=ModelLoadConfig(),
    )
    settings = ActivationExtractionSettings(layers=(0,), resume=True)
    backend = SimpleNamespace(name="fake-backend")
    received = {}

    def fake_extract_dataset_activations(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            task=kwargs["dataset"].task,
            output_path=kwargs["output_path"],
            examples_seen=len(kwargs["dataset"]),
            examples_extracted=len(kwargs["dataset"].labeled_for_probe()),
            skipped_labels=0,
            masked_tokens=0,
            layers=kwargs["settings"].layers,
            backend=kwargs["backend"].name,
        )

    monkeypatch.setattr(activations_module, "extract_dataset_activations", fake_extract_dataset_activations)

    summary = extract_rollout_activations(
        bundle=bundle,
        rollout_path=rollout_path,
        output_path=output_path,
        settings=settings,
        overwrite=True,
        backend=backend,
    )

    assert isinstance(received["dataset"], ActivationDataset)
    assert received["bundle"] is bundle
    assert received["output_path"] == output_path
    assert received["settings"] is settings
    assert received["overwrite"] is True
    assert received["backend"] is backend
    assert received["dataset"].source_path == rollout_path
    assert received["dataset"].dataset_id == str(rollout_path)
    assert summary.task == "roleplaying__plain"


def test_build_answer_detection_mask_excludes_padding():
    torch = pytest.importorskip("torch")

    class TinyTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            table = {"yes": [2], " no": [3], "no": [3]}
            return table.get(text, table.get(text.strip(), []))

        def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            inv = {2: "yes", 3: "no", 11: "user", 12: "assistant"}
            return " ".join(inv.get(int(token_id), "?") for token_id in token_ids)

    input_ids = torch.tensor([[0, 11, 12, 2, 0], [0, 11, 12, 3, 0]])
    attention_mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])

    mask, decoded = build_answer_detection_mask(
        tokenizer=TinyTokenizer(),
        input_ids=input_ids,
        expected_texts=["yes", "no"],
        attention_mask=attention_mask,
    )

    assert mask.tolist() == [
        [False, False, False, True, False],
        [False, False, False, True, False],
    ]
    assert decoded == ["yes", "no"]


def test_message_detection_mask_uses_char_spans_not_first_matching_tokens():
    torch = pytest.importorskip("torch")

    class TinyTokenizer:
        def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            inv = {1: "user", 2: "answer", 3: "assistant"}
            return " ".join(inv.get(int(token_id), "?") for token_id in token_ids)

    input_ids = torch.tensor([[1, 2, 3, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    offsets = torch.tensor([[(0, 4), (5, 11), (12, 21), (22, 28), (0, 0)]])

    mask, decoded, positions = _build_message_detection_mask_detail(
        tokenizer=TinyTokenizer(),
        input_ids=input_ids,
        offset_mapping=offsets,
        detected_texts=["answer"],
        char_spans=[((22, 28),)],
        attention_mask=attention_mask,
    )

    assert mask.tolist() == [[False, False, False, True, False]]
    assert positions == [[3]]
    assert decoded == ["answer"]


def test_verify_decoded_masks_rejects_alphanumeric_extra_decoded_text():
    with pytest.raises(ValueError, match="Detection mask mismatch"):
        verify_decoded_masks(expected_texts=["."], decoded_texts=["x."])


def test_verify_decoded_masks_allows_quote_boundary_extra():
    verify_decoded_masks(expected_texts=["."], decoded_texts=["’."])
    verify_decoded_masks(expected_texts=["."], decoded_texts=["!."])
    verify_decoded_masks(expected_texts=["."], decoded_texts=["%."])
    verify_decoded_masks(expected_texts=["answer"], decoded_texts=['"answer"'])


def test_build_qwen_inputs_rejects_processor_tokenizer_mismatch():
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            del texts, kwargs
            return {
                "input_ids": [[1, 2]],
                "attention_mask": [[1, 1]],
                "offset_mapping": [[(0, 1), (1, 2)]],
            }

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return {"input_ids": [[1, 99]], "attention_mask": [[1, 1]]}

    with pytest.raises(ValueError, match="Processor and tokenizer input_ids differ"):
        _build_qwen_inputs(
            processor=FakeProcessor(),
            tokenizer=FakeTokenizer(),
            examples=[],
            rendered_examples=[
                RenderedDetectionExample(
                    messages=({"role": "user", "content": "hi", "detect": False},),
                    rendered_text="hi",
                    detected_text="i",
                    char_spans=((1, 2),),
                )
            ],
            max_length=None,
        )


def test_qwen_decoder_surface_names_canonical_decoder_answer_tokens():
    surface = qwen3_vl_decoder_surface(17)

    assert surface.name == "decoder_layer_17_answer_tokens"
    assert surface.path == "model.model.language_model.layers[17]"
    assert surface.token_selector == "answer_tokens"


def test_prepare_model_inputs_is_text_only_and_drops_processor_extras():
    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def to(self, device):
            return (self.value, device)

    class FakeModel:
        device = "cuda:0"

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            processor_extra=None,
            use_cache=None,
        ):
            del input_ids, attention_mask, processor_extra, use_cache

    prepared = prepare_model_inputs(
        {
            "input_ids": FakeTensor("ids"),
            "attention_mask": FakeTensor("mask"),
            "offset_mapping": FakeTensor("offsets"),
            "detection_mask": FakeTensor("selector"),
            "logit_mask": FakeTensor("logits"),
            "overflow_to_sample_mapping": FakeTensor("overflow"),
            "processor_extra": FakeTensor("drop-extra"),
            "unexpected": FakeTensor("drop"),
        },
        FakeModel(),
    )

    assert set(prepared) == {"input_ids", "attention_mask"}
    assert "detection_mask" not in prepared
    assert "offset_mapping" not in prepared
    assert "logit_mask" not in prepared
    assert "overflow_to_sample_mapping" not in prepared
    assert "processor_extra" not in prepared


def test_nnsight_save_selected_rows_saves_masked_shape():
    torch = pytest.importorskip("torch")
    saved_shapes = []

    proxy = FakeSaveTensor(torch.arange(24).reshape(2, 3, 4), saved_shapes)
    selector = torch.tensor([[False, True, False], [True, False, True]])

    saved = _save_selected_rows((proxy,), selector)

    assert saved_shapes == [(3, 4)]
    assert saved.value.tolist() == proxy.tensor[selector].tolist()


def test_nnsight_capture_filters_trace_inputs_and_saves_masked_rows():
    torch = pytest.importorskip("torch")
    saved_shapes = []
    hidden = torch.arange(24).reshape(2, 3, 4)
    logits = torch.arange(60).reshape(2, 3, 10)
    detection_mask = torch.tensor([[False, True, False], [True, False, True]])
    logit_mask = torch.tensor([[True, False, False], [False, True, False]])

    class FakeTrace:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeNnsightModel:
        device = "cpu"

        def __init__(self):
            self.trace_inputs = None
            self.model = SimpleNamespace(
                language_model=SimpleNamespace(
                    layers=[SimpleNamespace(output=(FakeSaveTensor(hidden, saved_shapes),))]
                )
            )
            self.lm_head = SimpleNamespace(output=FakeSaveTensor(logits, saved_shapes))

        def forward(self, input_ids=None, attention_mask=None, processor_extra=None):
            del input_ids, attention_mask, processor_extra

        def trace(self, inputs, scan=False, validate=False):
            del scan, validate
            self.trace_inputs = dict(inputs)
            return FakeTrace()

    model = FakeNnsightModel()
    backend = NnsightActivationBackend(
        SimpleNamespace(
            model=model,
            processor=object(),
            tokenizer=object(),
            model_id=DEFAULT_MODEL_ID,
            config=ModelLoadConfig(),
        )
    )

    trace = backend.capture(
        inputs={
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "offset_mapping": torch.zeros(2, 3, 2, dtype=torch.long),
            "detection_mask": detection_mask,
            "logit_mask": logit_mask,
            "overflow_to_sample_mapping": torch.tensor([0]),
            "processor_extra": torch.ones(1, 2),
        },
        detection_mask=detection_mask,
        layers=(0,),
        capture_logits=True,
        logit_mask=logit_mask,
    )

    assert set(model.trace_inputs) == {"input_ids", "attention_mask"}
    assert saved_shapes == [(3, 4), (2, 10)]
    assert trace.activations_by_layer[0].tolist() == hidden[detection_mask].tolist()
    assert trace.logits.tolist() == logits[logit_mask].tolist()


def test_write_activation_hdf5_stages_metadata_and_resumes_missing_layers(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")

    output_path = tmp_path / "acts.h5"
    common_kwargs = {
        "output_path": output_path,
        "task": "roleplaying__plain",
        "source_indices": [10, 11],
        "output_indices": [0, 0],
        "labels": [HONEST, DECEPTIVE],
        "example_indices": [0, 1],
        "token_positions": [4, 7],
        "example_splits": [0, 1, 2],
        "example_source_indices": [10, 11],
        "example_output_indices": [0, 0],
        "example_labels": [HONEST, DECEPTIVE],
        "example_token_counts": [1, 1],
        "messages_json": ['[{"role": "user", "content": "q"}]', '[{"role": "user", "content": "q2"}]'],
        "rendered_texts": ["<user>q<assistant>yes", "<user>q2<assistant>no"],
        "source_datasets": ["roleplaying__plain", "roleplaying__plain"],
        "raw_labels_json": ["1", "7"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value, LabelSchema.ROLEPLAYING_1_TO_7.value],
        "example_metadata_json": ['{"source_index": 10}', '{"source_index": 11}'],
        "decoded_answer_texts": ["yes", "no"],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens", 1: "decoder_layer_1_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "rollouts.json",
        "processor_id": DEFAULT_MODEL_ID,
        "task_metadata_attrs": {
            "sycophancy_source_kind": "generated",
            "sycophancy_generated_model": "qwen3-vl-8b-thinking",
        },
    }

    write_activation_hdf5(
        layers=(0,),
        activations_by_layer={0: [torch.ones(2, 3)]},
        overwrite=False,
        **common_kwargs,
    )

    write_activation_hdf5(
        layers=(0, 1),
        activations_by_layer={0: [torch.ones(2, 3)], 1: [torch.zeros(2, 3)]},
        overwrite=False,
        resume=True,
        **common_kwargs,
    )

    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["format"] == "qwen_answer_token_activations_v2"
        assert handle.attrs["source_output_path"] == str(output_path)
        assert handle.attrs["created_at"]
        assert handle["layer_0/roleplaying__plain"].shape == (2, 3)
        assert handle["layer_1/roleplaying__plain"].shape == (2, 3)
        assert handle["metadata/roleplaying__plain/token_positions"][...].tolist() == [4, 7]
        assert (
            handle["layer_1/roleplaying__plain"].attrs["surface"]
            == "decoder_layer_1_answer_tokens"
        )
        assert handle["metadata/roleplaying__plain"].attrs["sycophancy_source_kind"] == "generated"
        assert handle["metadata/roleplaying__plain"].attrs["sycophancy_generated_model"] == "qwen3-vl-8b-thinking"
        assert handle["metadata/roleplaying__plain"].attrs["aggregation"] == "token_rows/no_pooling"
        assert handle["metadata/roleplaying__plain/messages_json"][0].decode() == '[{"role": "user", "content": "q"}]'
        assert handle["metadata/roleplaying__plain/label_schemas"][1].decode() == "roleplaying_1_to_7"


def test_write_activation_hdf5_resume_rejects_changed_existing_layer_data(tmp_path):
    torch = pytest.importorskip("torch")

    output_path = tmp_path / "acts.h5"
    common_kwargs = {
        "output_path": output_path,
        "task": "roleplaying__plain",
        "source_indices": [10],
        "output_indices": [0],
        "labels": [HONEST],
        "example_indices": [0],
        "token_positions": [4],
        "example_splits": [0, 1],
        "example_source_indices": [10],
        "example_output_indices": [0],
        "example_labels": [HONEST],
        "example_token_counts": [1],
        "messages_json": ['[{"role": "assistant", "content": "yes"}]'],
        "rendered_texts": ["<assistant>yes"],
        "source_datasets": ["roleplaying__plain"],
        "raw_labels_json": ["1"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value],
        "example_metadata_json": ['{"source_index": 10}'],
        "decoded_answer_texts": ["yes"],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens", 1: "decoder_layer_1_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "rollouts.json",
        "processor_id": DEFAULT_MODEL_ID,
    }

    write_activation_hdf5(
        layers=(0,),
        activations_by_layer={0: [torch.ones(1, 3)]},
        overwrite=False,
        **common_kwargs,
    )

    with pytest.raises(FileExistsError, match="existing data differs"):
        write_activation_hdf5(
            layers=(0, 1),
            activations_by_layer={0: [torch.zeros(1, 3)], 1: [torch.zeros(1, 3)]},
            overwrite=False,
            resume=True,
            **common_kwargs,
        )


def test_write_activation_hdf5_overwrite_removes_stale_task_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")

    output_path = tmp_path / "acts.h5"
    common_kwargs = {
        "output_path": output_path,
        "task": "roleplaying__plain",
        "source_indices": [10],
        "output_indices": [0],
        "labels": [HONEST],
        "example_indices": [0],
        "token_positions": [4],
        "example_splits": [0, 1],
        "example_source_indices": [10],
        "example_output_indices": [0],
        "example_labels": [HONEST],
        "example_token_counts": [1],
        "messages_json": ['[{"role": "assistant", "content": "yes"}]'],
        "rendered_texts": ["<assistant>yes"],
        "source_datasets": ["roleplaying__plain"],
        "raw_labels_json": ["1"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value],
        "example_metadata_json": ['{"source_index": 10}'],
        "decoded_answer_texts": ["yes"],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens", 1: "decoder_layer_1_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "rollouts.json",
        "processor_id": DEFAULT_MODEL_ID,
    }

    write_activation_hdf5(
        layers=(0, 1),
        activations_by_layer={0: [torch.ones(1, 3)], 1: [torch.zeros(1, 3)]},
        logits_by_batch=[torch.ones(1, 5)],
        overwrite=False,
        **common_kwargs,
    )

    write_activation_hdf5(
        layers=(0,),
        activations_by_layer={0: [torch.full((1, 3), 2.0)]},
        overwrite=True,
        **common_kwargs,
    )

    with h5py.File(output_path, "r") as handle:
        assert "roleplaying__plain" in handle["layer_0"]
        assert "roleplaying__plain" not in handle["layer_1"]
        assert "roleplaying__plain" not in handle.get("logits", {})
        assert handle["layer_0/roleplaying__plain"][...].tolist() == [[2.0, 2.0, 2.0]]


def test_merge_activation_hdf5_shards_concatenates_examples_and_layers(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")

    shard_1 = tmp_path / "shard-1.h5"
    shard_2 = tmp_path / "shard-2.h5"
    output_path = tmp_path / "merged.h5"

    common_kwargs = {
        "task": "roleplaying__plain",
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "roleplaying.json",
        "processor_id": DEFAULT_MODEL_ID,
        "overwrite": False,
    }
    write_activation_hdf5(
        output_path=shard_1,
        layers=(0,),
        activations_by_layer={0: [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]},
        source_indices=[10, 10],
        output_indices=[0, 0],
        labels=[HONEST, HONEST],
        example_indices=[0, 0],
        token_positions=[5, 6],
        example_splits=[0, 2],
        example_source_indices=[10],
        example_output_indices=[0],
        example_labels=[HONEST],
        example_token_counts=[2],
        detected_answer_texts=["yes"],
        decoded_answer_texts=["yes"],
        char_spans_json=["[[0, 3]]"],
        messages_json=['[{"role": "assistant", "content": "yes"}]'],
        rendered_texts=["<assistant>yes"],
        source_datasets=["roleplaying.json"],
        raw_labels_json=["1"],
        label_schemas=[LabelSchema.ROLEPLAYING_1_TO_7.value],
        example_metadata_json=['{"source_index": 10}'],
        **common_kwargs,
    )
    write_activation_hdf5(
        output_path=shard_2,
        layers=(0,),
        activations_by_layer={0: [torch.tensor([[5.0, 6.0]])]},
        source_indices=[11],
        output_indices=[0],
        labels=[DECEPTIVE],
        example_indices=[0],
        token_positions=[7],
        example_splits=[0, 1],
        example_source_indices=[11],
        example_output_indices=[0],
        example_labels=[DECEPTIVE],
        example_token_counts=[1],
        detected_answer_texts=["no"],
        decoded_answer_texts=["no"],
        char_spans_json=["[[0, 2]]"],
        messages_json=['[{"role": "assistant", "content": "no"}]'],
        rendered_texts=["<assistant>no"],
        source_datasets=["roleplaying.json"],
        raw_labels_json=["7"],
        label_schemas=[LabelSchema.ROLEPLAYING_1_TO_7.value],
        example_metadata_json=['{"source_index": 11}'],
        **common_kwargs,
    )

    summary = merge_activation_hdf5_shards([shard_1, shard_2], output_path=output_path)

    assert summary.tasks == ("roleplaying__plain",)
    assert summary.examples_by_task["roleplaying__plain"] == 2
    assert summary.token_rows_by_task["roleplaying__plain"] == 3
    with h5py.File(output_path, "r") as handle:
        assert handle["layer_0/roleplaying__plain"].shape == (3, 2)
        assert handle["metadata/roleplaying__plain/example_splits"][...].tolist() == [0, 2, 3]
        assert handle["metadata/roleplaying__plain/example_indices"][...].tolist() == [0, 0, 1]
        assert handle["metadata/roleplaying__plain/example_source_indices"][...].tolist() == [10, 11]
        assert handle["metadata/roleplaying__plain"].attrs["merged_shard_count"] == 2


def test_merge_activation_hdf5_shards_auto_copies_disjoint_tasks(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")

    shard_1 = tmp_path / "shard-1.h5"
    shard_2 = tmp_path / "shard-2.h5"
    output_path = tmp_path / "merged.h5"

    common_kwargs = {
        "layers": (0,),
        "output_indices": [0],
        "labels": [HONEST],
        "example_indices": [0],
        "token_positions": [5],
        "example_splits": [0, 1],
        "example_output_indices": [0],
        "example_labels": [HONEST],
        "example_token_counts": [1],
        "detected_answer_texts": ["yes"],
        "decoded_answer_texts": ["yes"],
        "char_spans_json": ["[[0, 3]]"],
        "messages_json": ['[{"role": "assistant", "content": "yes"}]'],
        "rendered_texts": ["<assistant>yes"],
        "raw_labels_json": ["1"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "processor_id": DEFAULT_MODEL_ID,
        "overwrite": False,
    }
    write_activation_hdf5(
        output_path=shard_1,
        task="roleplaying__plain",
        activations_by_layer={0: [torch.tensor([[1.0, 2.0]])]},
        source_indices=[10],
        example_source_indices=[10],
        source_datasets=["roleplaying.json"],
        example_metadata_json=['{"source_index": 10}'],
        dataset_id="roleplaying.json",
        compression="lzf",
        **common_kwargs,
    )
    with h5py.File(shard_1, "a") as shard:
        shard["metadata"].create_dataset("task_names", data=[b"roleplaying__plain"])
    write_activation_hdf5(
        output_path=shard_2,
        task="sandbagging_v2__wmdp_mmlu",
        activations_by_layer={0: [torch.tensor([[3.0, 4.0]])]},
        source_indices=[20],
        example_source_indices=[20],
        source_datasets=["sandbagging.json"],
        example_metadata_json=['{"source_index": 20}'],
        dataset_id="sandbagging.json",
        compression="none",
        **common_kwargs,
    )

    summary = merge_activation_hdf5_shards(
        [shard_1, shard_2],
        output_path=output_path,
        compression="gzip",
    )

    assert summary.tasks == ("roleplaying__plain", "sandbagging_v2__wmdp_mmlu")
    assert summary.examples_by_task == {
        "roleplaying__plain": 1,
        "sandbagging_v2__wmdp_mmlu": 1,
    }
    assert summary.token_rows_by_task == {
        "roleplaying__plain": 1,
        "sandbagging_v2__wmdp_mmlu": 1,
    }
    with h5py.File(output_path, "r") as handle:
        assert handle.attrs["merged_strategy"] == "copy-disjoint"
        assert handle.attrs["merged_compression"] == "preserved"
        assert "task_names" not in handle["metadata"]
        assert handle["layer_0/roleplaying__plain"].compression == "lzf"
        assert handle["layer_0/sandbagging_v2__wmdp_mmlu"].compression is None
        assert handle["metadata/roleplaying__plain"].attrs["merged_shard_count"] == 1
        assert handle["metadata/sandbagging_v2__wmdp_mmlu"].attrs["merged_shard_count"] == 1


def test_merge_activation_hdf5_shards_copy_disjoint_rejects_overlapping_tasks(tmp_path):
    torch = pytest.importorskip("torch")

    shard_1 = tmp_path / "shard-1.h5"
    shard_2 = tmp_path / "shard-2.h5"
    common_kwargs = {
        "task": "roleplaying__plain",
        "layers": (0,),
        "output_indices": [0],
        "labels": [HONEST],
        "example_indices": [0],
        "token_positions": [5],
        "example_splits": [0, 1],
        "example_output_indices": [0],
        "example_labels": [HONEST],
        "example_token_counts": [1],
        "detected_answer_texts": ["yes"],
        "decoded_answer_texts": ["yes"],
        "char_spans_json": ["[[0, 3]]"],
        "messages_json": ['[{"role": "assistant", "content": "yes"}]'],
        "rendered_texts": ["<assistant>yes"],
        "source_datasets": ["roleplaying.json"],
        "raw_labels_json": ["1"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "roleplaying.json",
        "processor_id": DEFAULT_MODEL_ID,
        "compression": "none",
        "overwrite": False,
    }
    write_activation_hdf5(
        output_path=shard_1,
        activations_by_layer={0: [torch.tensor([[1.0, 2.0]])]},
        source_indices=[10],
        example_source_indices=[10],
        example_metadata_json=['{"source_index": 10}'],
        **common_kwargs,
    )
    write_activation_hdf5(
        output_path=shard_2,
        activations_by_layer={0: [torch.tensor([[3.0, 4.0]])]},
        source_indices=[11],
        example_source_indices=[11],
        example_metadata_json=['{"source_index": 11}'],
        **common_kwargs,
    )

    with pytest.raises(FileExistsError, match="copy-disjoint"):
        merge_activation_hdf5_shards(
            [shard_1, shard_2],
            output_path=tmp_path / "merged.h5",
            merge_strategy="copy-disjoint",
        )


def test_merge_activation_hdf5_shards_rejects_duplicate_examples(tmp_path):
    torch = pytest.importorskip("torch")

    shard_1 = tmp_path / "shard-1.h5"
    shard_2 = tmp_path / "shard-2.h5"
    common_kwargs = {
        "task": "roleplaying__plain",
        "layers": (0,),
        "source_indices": [10],
        "output_indices": [0],
        "labels": [HONEST],
        "example_indices": [0],
        "token_positions": [5],
        "example_splits": [0, 1],
        "example_source_indices": [10],
        "example_output_indices": [0],
        "example_labels": [HONEST],
        "example_token_counts": [1],
        "detected_answer_texts": ["yes"],
        "decoded_answer_texts": ["yes"],
        "char_spans_json": ["[[0, 3]]"],
        "messages_json": ['[{"role": "assistant", "content": "yes"}]'],
        "rendered_texts": ["<assistant>yes"],
        "source_datasets": ["roleplaying.json"],
        "raw_labels_json": ["1"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value],
        "example_metadata_json": ['{"source_index": 10}'],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "roleplaying.json",
        "processor_id": DEFAULT_MODEL_ID,
        "overwrite": False,
    }
    write_activation_hdf5(
        output_path=shard_1,
        activations_by_layer={0: [torch.tensor([[1.0, 2.0]])]},
        **common_kwargs,
    )
    write_activation_hdf5(
        output_path=shard_2,
        activations_by_layer={0: [torch.tensor([[3.0, 4.0]])]},
        **common_kwargs,
    )

    with pytest.raises(FileExistsError, match="duplicate example"):
        merge_activation_hdf5_shards([shard_1, shard_2], output_path=tmp_path / "merged.h5")


def test_write_activation_hdf5_respects_compression_mode(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")

    none_path = tmp_path / "acts_none.h5"
    write_activation_hdf5(
        output_path=none_path,
        task="roleplaying__plain",
        layers=(0,),
        activations_by_layer={0: [torch.tensor([[1.0, 2.0]])]},
        source_indices=[10],
        output_indices=[0],
        labels=[HONEST],
        example_indices=[0],
        token_positions=[4],
        example_splits=[0, 1],
        example_source_indices=[10],
        example_output_indices=[0],
        example_labels=[HONEST],
        example_token_counts=[1],
        detected_answer_texts=["yes"],
        decoded_answer_texts=["yes"],
        char_spans_json=["[[0, 3]]"],
        messages_json=['[{"role": "assistant", "content": "yes"}]'],
        rendered_texts=["<assistant>yes"],
        source_datasets=["roleplaying.json"],
        raw_labels_json=["1"],
        label_schemas=[LabelSchema.ROLEPLAYING_1_TO_7.value],
        example_metadata_json=['{"source_index": 10}'],
        model_id=DEFAULT_MODEL_ID,
        backend_name="unit-test",
        surface_names={0: "decoder_layer_0_answer_tokens"},
        preserved_input_keys=("attention_mask", "input_ids"),
        dataset_id="roleplaying.json",
        processor_id=DEFAULT_MODEL_ID,
        compression="none",
        overwrite=False,
    )

    lzf_path = tmp_path / "acts_lzf.h5"
    write_activation_hdf5(
        output_path=lzf_path,
        task="roleplaying__plain",
        layers=(0,),
        activations_by_layer={0: [torch.tensor([[3.0, 4.0]])]},
        source_indices=[11],
        output_indices=[0],
        labels=[DECEPTIVE],
        example_indices=[0],
        token_positions=[5],
        example_splits=[0, 1],
        example_source_indices=[11],
        example_output_indices=[0],
        example_labels=[DECEPTIVE],
        example_token_counts=[1],
        detected_answer_texts=["no"],
        decoded_answer_texts=["no"],
        char_spans_json=["[[0, 2]]"],
        messages_json=['[{"role": "assistant", "content": "no"}]'],
        rendered_texts=["<assistant>no"],
        source_datasets=["roleplaying.json"],
        raw_labels_json=["0"],
        label_schemas=[LabelSchema.ROLEPLAYING_1_TO_7.value],
        example_metadata_json=['{"source_index": 11}'],
        model_id=DEFAULT_MODEL_ID,
        backend_name="unit-test",
        surface_names={0: "decoder_layer_0_answer_tokens"},
        preserved_input_keys=("attention_mask", "input_ids"),
        dataset_id="roleplaying.json",
        processor_id=DEFAULT_MODEL_ID,
        compression="lzf",
        overwrite=False,
    )

    with h5py.File(none_path, "r") as none_h5:
        assert none_h5["layer_0/roleplaying__plain"].compression is None
    with h5py.File(lzf_path, "r") as lzf_h5:
        assert lzf_h5["layer_0/roleplaying__plain"].compression == "lzf"


def test_merge_activation_hdf5_shards_respects_compression_mode(tmp_path):
    torch = pytest.importorskip("torch")
    h5py = pytest.importorskip("h5py")

    shard_1 = tmp_path / "shard-1.h5"
    shard_2 = tmp_path / "shard-2.h5"
    common_kwargs = {
        "task": "roleplaying__plain",
        "layers": (0,),
        "output_indices": [0],
        "labels": [HONEST],
        "example_indices": [0],
        "token_positions": [5],
        "example_splits": [0, 1],
        "example_labels": [HONEST],
        "example_token_counts": [1],
        "detected_answer_texts": ["yes"],
        "decoded_answer_texts": ["yes"],
        "char_spans_json": ["[[0, 3]]"],
        "messages_json": ['[{"role": "assistant", "content": "yes"}]'],
        "rendered_texts": ["<assistant>yes"],
        "source_datasets": ["roleplaying.json"],
        "raw_labels_json": ["1"],
        "label_schemas": [LabelSchema.ROLEPLAYING_1_TO_7.value],
        "example_metadata_json": ['{"source_index": 10}'],
        "model_id": DEFAULT_MODEL_ID,
        "backend_name": "unit-test",
        "surface_names": {0: "decoder_layer_0_answer_tokens"},
        "preserved_input_keys": ("attention_mask", "input_ids"),
        "dataset_id": "roleplaying.json",
        "processor_id": DEFAULT_MODEL_ID,
        "overwrite": False,
    }

    write_activation_hdf5(
        output_path=shard_1,
        activations_by_layer={0: [torch.tensor([[1.0, 2.0]])]},
        source_indices=[10],
        example_source_indices=[10],
        example_output_indices=[0],
        compression="none",
        **common_kwargs,
    )
    shard_2_kwargs = common_kwargs | {"example_metadata_json": ['{"source_index": 11}']}
    write_activation_hdf5(
        output_path=shard_2,
        activations_by_layer={0: [torch.tensor([[3.0, 4.0]])]},
        source_indices=[11],
        example_source_indices=[11],
        example_output_indices=[0],
        compression="none",
        **shard_2_kwargs,
    )

    merged_path = tmp_path / "merged.h5"
    summary = merge_activation_hdf5_shards(
        [shard_1, shard_2],
        output_path=merged_path,
        overwrite=True,
        compression="lzf",
    )

    assert summary.output_path == merged_path
    with h5py.File(merged_path, "r") as merged:
        assert merged["layer_0/roleplaying__plain"].compression == "lzf"
        assert merged.attrs["merged_compression"] == "lzf"


def test_merge_activation_hdf5_shards_refuses_live_merge_lock(tmp_path):
    torch = pytest.importorskip("torch")

    shard = tmp_path / "shard.h5"
    merged_path = tmp_path / "merged.h5"
    write_activation_hdf5(
        output_path=shard,
        task="roleplaying__plain",
        layers=(0,),
        activations_by_layer={0: [torch.tensor([[1.0, 2.0]])]},
        source_indices=[10],
        output_indices=[0],
        labels=[HONEST],
        example_indices=[0],
        token_positions=[5],
        example_splits=[0, 1],
        example_source_indices=[10],
        example_output_indices=[0],
        example_labels=[HONEST],
        example_token_counts=[1],
        detected_answer_texts=["yes"],
        decoded_answer_texts=["yes"],
        char_spans_json=["[[0, 3]]"],
        messages_json=['[{"role": "assistant", "content": "yes"}]'],
        rendered_texts=["<assistant>yes"],
        source_datasets=["roleplaying.json"],
        raw_labels_json=["1"],
        label_schemas=[LabelSchema.ROLEPLAYING_1_TO_7.value],
        example_metadata_json=['{"source_index": 10}'],
        model_id=DEFAULT_MODEL_ID,
        backend_name="unit-test",
        surface_names={0: "decoder_layer_0_answer_tokens"},
        preserved_input_keys=("attention_mask", "input_ids"),
        dataset_id="roleplaying.json",
        processor_id=DEFAULT_MODEL_ID,
        overwrite=False,
    )
    lock = acquire_lock(
        merged_path.with_name(f"{merged_path.name}.merge.lock"),
        lock_payload(run_id="merge-1", queue_plan_id=None),
    )
    try:
        with pytest.raises(LockHeldError, match="Lock exists"):
            merge_activation_hdf5_shards([shard], output_path=merged_path)
    finally:
        lock.release()


class FakeSaveTensor:
    def __init__(self, tensor, saved_shapes):
        self.tensor = tensor
        self.saved_shapes = saved_shapes

    @property
    def device(self):
        return self.tensor.device

    @property
    def value(self):
        return self.tensor

    def __getitem__(self, selector):
        return FakeSaveTensor(self.tensor[selector], self.saved_shapes)

    def save(self):
        self.saved_shapes.append(tuple(self.tensor.shape))
        return self


def _write_sycophancy_files(source_dir):
    source_dir.mkdir(parents=True)
    sample = {
        "model_answer_with_bio": 1,
        "bio_messages": [{"content": "Question with biography"}],
        "question": "Question?",
        "correct_answer": 0,
        "sycophancy_occurred": True,
    }
    for index, stem_task in enumerate(STEM_TASKS_PART_2):
        payload = {
            "positive": [sample] if index == 0 else [],
            "negative": [dict(sample, sycophancy_occurred=False)] if index == 0 else [],
        }
        (source_dir / f"conf_paired_mmlu_sycophancy_{stem_task}.json").write_text(json.dumps(payload))
