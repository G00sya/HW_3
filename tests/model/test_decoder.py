import pytest
import torch
import torch.nn as nn

from src.model.decoder import Decoder
from src.model.layer_norm import LayerNorm
from src.model.positional_encoding import PositionalEncoding


@pytest.fixture
def valid_decoder_params():
    """Fixture to provide valid parameters for Decoder."""
    vocab_size = 100
    d_model = 512
    d_ff = 2048
    blocks_count = 6
    heads_count = 8
    dropout_rate = 0.1
    return vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate


@pytest.fixture
def decoder(valid_decoder_params):
    """Fixture to provide a pre-initialized Decoder."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    return Decoder(vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)


def test_decoder_initialization(decoder, valid_decoder_params):
    """Checks that Decoder initializes correctly."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    assert isinstance(decoder, Decoder)
    assert isinstance(decoder._Decoder__emb, nn.Sequential)
    assert isinstance(decoder._Decoder__emb[0], nn.Embedding)
    assert isinstance(decoder._Decoder__emb[1], PositionalEncoding)
    assert isinstance(decoder._Decoder__blocks, nn.ModuleList)
    assert len(decoder._Decoder__blocks) == blocks_count
    assert isinstance(decoder._Decoder__norm, LayerNorm)
    assert isinstance(decoder._Decoder__out_layer, nn.Linear)


def test_decoder_forward_pass(decoder, valid_decoder_params):
    """Checks that the forward pass of Decoder returns a tensor of the correct shape."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    batch_size = 32
    target_seq_len = 20
    source_seq_len = 30

    inputs = torch.randint(0, vocab_size, (batch_size, target_seq_len))
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)
    source_mask = torch.ones(batch_size, 1, source_seq_len).bool()
    target_mask = torch.ones(batch_size, target_seq_len, target_seq_len).bool()

    output = decoder(inputs, encoder_output, source_mask, target_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, target_seq_len, vocab_size)


def test_decoder_forward_pass_different_sizes(decoder, valid_decoder_params):
    """Checks that the forward pass works with different sequence lengths."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    batch_size = 16
    target_seq_len = 10
    source_seq_len = 15

    inputs = torch.randint(0, vocab_size, (batch_size, target_seq_len))
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)
    source_mask = torch.ones(batch_size, 1, source_seq_len).bool()
    target_mask = torch.ones(batch_size, target_seq_len, target_seq_len).bool()

    output = decoder(inputs, encoder_output, source_mask, target_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, target_seq_len, vocab_size)


def test_decoder_output_values(decoder, valid_decoder_params):
    """Checks that the output values are within a reasonable range."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    batch_size = 4
    target_seq_len = 5
    source_seq_len = 7

    inputs = torch.randint(0, vocab_size, (batch_size, target_seq_len))
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)
    source_mask = torch.ones(batch_size, 1, source_seq_len).bool()
    target_mask = torch.ones(batch_size, target_seq_len, target_seq_len).bool()

    output = decoder(inputs, encoder_output, source_mask, target_mask)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_decoder_type_checks_valid_params(valid_decoder_params):
    """Ensures that Decoder initializes without errors with valid parameters."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    try:
        Decoder(vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate)
    except Exception as e:
        pytest.fail(f"Decoder initialization failed with valid parameters: {e}")


def test_decoder_type_checks_vocab_size(valid_decoder_params):
    """Checks that Decoder raises TypeError if vocab_size is not an int."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(TypeError):
        Decoder("100", d_model, d_ff, blocks_count, heads_count, dropout_rate)


def test_decoder_value_checks_vocab_size(valid_decoder_params):
    """Checks that Decoder raises ValueError if vocab_size is not positive."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(ValueError):
        Decoder(0, d_model, d_ff, blocks_count, heads_count, dropout_rate)


def test_decoder_type_checks_d_model(valid_decoder_params):
    """Checks that Decoder raises TypeError if d_model is not an int."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(TypeError):
        Decoder(vocab_size, "512", d_ff, blocks_count, heads_count, dropout_rate)


def test_decoder_value_checks_d_model(valid_decoder_params):
    """Checks that Decoder raises ValueError if d_model is not positive."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(ValueError):
        Decoder(vocab_size, 0, d_ff, blocks_count, heads_count, dropout_rate)


def test_decoder_type_checks_d_ff(valid_decoder_params):
    """Checks that Decoder raises TypeError if d_ff is not an int."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(TypeError):
        Decoder(vocab_size, d_model, "2048", blocks_count, heads_count, dropout_rate)


def test_decoder_value_checks_d_ff(valid_decoder_params):
    """Checks that Decoder raises ValueError if d_ff is not positive."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(ValueError):
        Decoder(vocab_size, d_model, 0, blocks_count, heads_count, dropout_rate)


def test_decoder_type_checks_blocks_count(valid_decoder_params):
    """Checks that Decoder raises TypeError if blocks_count is not an int."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(TypeError):
        Decoder(vocab_size, d_model, d_ff, "6", heads_count, dropout_rate)


def test_decoder_value_checks_blocks_count(valid_decoder_params):
    """Checks that Decoder raises ValueError if blocks_count is not positive."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(ValueError):
        Decoder(vocab_size, d_model, d_ff, 0, heads_count, dropout_rate)


def test_decoder_type_checks_heads_count(valid_decoder_params):
    """Checks that Decoder raises TypeError if heads_count is not an int."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(TypeError):
        Decoder(vocab_size, d_model, d_ff, blocks_count, "8", dropout_rate)


def test_decoder_value_checks_heads_count(valid_decoder_params):
    """Checks that Decoder raises ValueError if heads_count is not positive."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(ValueError):
        Decoder(vocab_size, d_model, d_ff, blocks_count, 0, dropout_rate)


def test_decoder_type_checks_dropout_rate(valid_decoder_params):
    """Checks that Decoder raises TypeError if dropout_rate is not a float or int."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(TypeError):
        Decoder(vocab_size, d_model, d_ff, blocks_count, heads_count, "0.1")


def test_decoder_value_checks_dropout_rate(valid_decoder_params):
    """Checks that Decoder raises ValueError if dropout_rate is not in the range [0.0, 1.0)."""
    vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate = valid_decoder_params
    with pytest.raises(ValueError):
        Decoder(vocab_size, d_model, d_ff, blocks_count, heads_count, 1.0)
    with pytest.raises(ValueError):
        Decoder(vocab_size, d_model, d_ff, blocks_count, heads_count, -0.1)
