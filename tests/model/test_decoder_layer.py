import pytest
import torch

from src.model.decoder_layer import DecoderLayer
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positionwise_feed_forward import PositionwiseFeedForward
from src.model.residual_block import ResidualBlock


@pytest.fixture
def valid_decoder_layer_params():
    """Fixture to provide valid parameters for DecoderLayer."""
    size = 512
    heads_count = 8
    dropout_rate = 0.1
    self_attn = MultiHeadedAttention(heads_count, size, dropout_rate)
    encoder_attn = MultiHeadedAttention(heads_count, size, dropout_rate)
    feed_forward = PositionwiseFeedForward(size, 2048, dropout_rate)
    return size, self_attn, encoder_attn, feed_forward, dropout_rate


@pytest.fixture
def decoder_layer(valid_decoder_layer_params):
    """Fixture to provide a pre-initialized DecoderLayer."""
    size, self_attn, encoder_attn, feed_forward, dropout_rate = valid_decoder_layer_params
    return DecoderLayer(size, self_attn, encoder_attn, feed_forward, dropout_rate)


def test_decoder_layer_initialization(decoder_layer):
    """Checks that DecoderLayer initializes correctly."""
    assert isinstance(decoder_layer, DecoderLayer)
    assert isinstance(decoder_layer._DecoderLayer__self_attn, MultiHeadedAttention)
    assert isinstance(decoder_layer._DecoderLayer__encoder_attn, MultiHeadedAttention)
    assert isinstance(decoder_layer._DecoderLayer__feed_forward, PositionwiseFeedForward)
    assert isinstance(decoder_layer._DecoderLayer__self_attention_block, ResidualBlock)
    assert isinstance(decoder_layer._DecoderLayer__attention_block, ResidualBlock)
    assert isinstance(decoder_layer._DecoderLayer__feed_forward_block, ResidualBlock)


def test_decoder_layer_forward_pass(decoder_layer):
    """Checks that the forward pass of DecoderLayer returns a tensor of the correct shape."""
    size = 512
    batch_size = 32
    target_seq_len = 20
    source_seq_len = 30

    inputs = torch.randn(batch_size, target_seq_len, size)
    encoder_output = torch.randn(batch_size, source_seq_len, size)
    source_mask = torch.ones(batch_size, 1, source_seq_len).bool()
    target_mask = torch.ones(batch_size, target_seq_len, target_seq_len).bool()

    output = decoder_layer(inputs, encoder_output, source_mask, target_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, target_seq_len, size)


def test_decoder_layer_forward_pass_different_sizes(decoder_layer):
    """Checks that the forward pass works with different sequence lengths."""
    size = 512
    batch_size = 16
    target_seq_len = 10
    source_seq_len = 15

    inputs = torch.randn(batch_size, target_seq_len, size)
    encoder_output = torch.randn(batch_size, source_seq_len, size)
    source_mask = torch.ones(batch_size, 1, source_seq_len).bool()
    target_mask = torch.ones(batch_size, target_seq_len, target_seq_len).bool()

    output = decoder_layer(inputs, encoder_output, source_mask, target_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, target_seq_len, size)


def test_decoder_layer_forward_pass_with_masks(decoder_layer):
    """Checks that the forward pass works with masks."""
    size = 512
    batch_size = 8
    target_seq_len = 5
    source_seq_len = 7

    inputs = torch.randn(batch_size, target_seq_len, size)
    encoder_output = torch.randn(batch_size, source_seq_len, size)

    source_mask = torch.randint(0, 2, (batch_size, 1, source_seq_len)).bool()
    target_mask = torch.randint(0, 2, (batch_size, target_seq_len, target_seq_len)).bool()

    output = decoder_layer(inputs, encoder_output, source_mask, target_mask)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, target_seq_len, size)
