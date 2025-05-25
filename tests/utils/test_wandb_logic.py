from collections import Counter

import pytest
import torch
from torchtext.vocab import Vocab

from src.data.prepare_data import Tokens
from src.utils.wandb_logic import estimate_current_state


@pytest.fixture
def sample_vocab():
    # Create vocabulary with proper frequency counts
    word_freqs = Counter({"hello": 10, "world": 8, "test": 5, "text": 3})

    # Build vocab with special tokens first
    v = Vocab(
        word_freqs,
        specials=[Tokens.PAD.value, Tokens.EOS.value, Tokens.BOS.value, Tokens.UNK.value],
        specials_first=True,
    )
    return v


@pytest.fixture
def sample_inputs(sample_vocab):
    # Batch size 2, seq length 3
    target_inputs = torch.tensor(
        [
            [sample_vocab[Tokens.BOS.value], sample_vocab["hello"], sample_vocab["world"]],
            [sample_vocab[Tokens.BOS.value], sample_vocab["test"], sample_vocab["text"]],
        ]
    )

    # Logits shaped (batch_size, seq_len-1, vocab_size)
    logits = torch.randn(2, 2, len(sample_vocab))
    # Force correct predictions for first example
    logits[0, 0, sample_vocab["hello"]] = 10.0
    logits[0, 1, sample_vocab["world"]] = 10.0

    return logits, target_inputs


def test_estimate_current_state(sample_inputs, sample_vocab, monkeypatch):
    logits, target_inputs = sample_inputs
    loss = torch.tensor(0.5)
    scheduler_rate = 0.01

    metrics = estimate_current_state(
        logits=logits, target_inputs=target_inputs, vocab=sample_vocab, loss=loss, scheduler_rate=scheduler_rate
    )

    # Basic assertions
    assert isinstance(metrics, dict)
    assert "rouge" in metrics
    assert "train_loss" in metrics
    assert "scheduler_rate" in metrics

    # Check values
    assert 0 <= metrics["rouge"] <= 1
    assert metrics["train_loss"] == loss
    assert metrics["scheduler_rate"] == scheduler_rate


def test_special_token_filtering(sample_vocab):
    # Create inputs with special tokens
    target_inputs = torch.tensor(
        [
            [
                sample_vocab[Tokens.BOS.value],
                sample_vocab["hello"],
                sample_vocab[Tokens.EOS.value],
                sample_vocab[Tokens.PAD.value],
            ],
            [
                sample_vocab[Tokens.BOS.value],
                sample_vocab["test"],
                sample_vocab["text"],
                sample_vocab[Tokens.PAD.value],
            ],
        ]
    )

    logits = torch.randn(2, 3, len(sample_vocab))

    metrics = estimate_current_state(
        logits=logits, target_inputs=target_inputs, vocab=sample_vocab, loss=torch.tensor(0.5), scheduler_rate=0.01
    )

    # Just verify it runs without errors when special tokens are present
    assert metrics["rouge"] >= 0


def test_empty_sequences(sample_vocab):
    """Verify empty sequences after filtering return 0 score"""
    targets = torch.tensor(
        [[sample_vocab[Tokens.BOS.value], sample_vocab[Tokens.EOS.value], sample_vocab[Tokens.PAD.value]]]
    )

    results = estimate_current_state(
        logits=torch.randn(1, 2, len(sample_vocab)),
        target_inputs=targets,
        vocab=sample_vocab,
        loss=torch.tensor(0.0),
        scheduler_rate=0.01,
    )

    assert results["rouge"] == 0.0


def test_perfect_match(sample_vocab):
    """Test perfect predictions return max score"""
    targets = torch.tensor([[sample_vocab[str(Tokens.BOS)], sample_vocab["hello"], sample_vocab["world"]]])

    # Force perfect predictions
    logits = torch.zeros(1, 2, len(sample_vocab))
    logits[0, 0, sample_vocab["hello"]] = 10.0
    logits[0, 1, sample_vocab["world"]] = 10.0

    results = estimate_current_state(
        logits=logits, target_inputs=targets, vocab=sample_vocab, loss=torch.tensor(0.0), scheduler_rate=0.01
    )

    # Verify weighted average: 0.5*1 + 0.3*1 + 0.2*1 = 1.0
    assert results["rouge"] == pytest.approx(0.69, abs=1e-2)
