"""Arm C loads and returns 11 sigmoid outputs (tiny local model, no download)."""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")  # noqa: F841


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({f"t{i}": i for i in range(4, 100)})
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(tokenizer_object=tk, model_max_length=512)


def _tiny_model(collections):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.for_model(
        "bert",
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        num_labels=len(collections),
        problem_type="multi_label_classification",
        id2label={i: c.name for i, c in enumerate(collections)},
        label2id={c.name: i for i, c in enumerate(collections)},
    )
    return AutoModelForSequenceClassification.from_config(config)


def test_forward_multilabel(collections, splits):
    from crb.arms.classifier import predict_labels

    model = _tiny_model(collections)
    tokenizer = _tiny_tokenizer()

    blocks = splits["test"][:3]
    labels, probs = predict_labels(model, tokenizer, blocks, collections)
    assert len(labels) == 3
    assert len(probs) == 3
    for row in probs:
        assert len(row) == 11
        assert all(0.0 <= p <= 1.0 for p in row)
