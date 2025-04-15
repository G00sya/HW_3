import pytest
import torch

from src.model.decoder_layer import DecoderLayer
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positionwise_feed_forward import PositionwiseFeedForward
from src.model.residual_block import ResidualBlock


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


def test_initialization(valid_decoder_layer_params):
    """Checks that DecoderLayer initializes correctly with valid parameters."""
    size, self_attn, encoder_attn, feed_forward, dropout_rate = valid_decoder_layer_params
    decoder_layer = DecoderLayer(size, self_attn, encoder_attn, feed_forward, dropout_rate)
    assert isinstance(decoder_layer, DecoderLayer)


def test_decoder_layer_initialization_type_errors():
    """Checks that DecoderLayer raises TypeError for invalid parameter types."""
    with pytest.raises(TypeError):
        # All invalid types
        DecoderLayer(size="invalid", self_attn=None, encoder_attn=None, feed_forward=None, dropout_rate=None)

    size = 512
    self_attn = MultiHeadedAttention(heads_count=8, d_model=size)
    encoder_attn = MultiHeadedAttention(heads_count=8, d_model=size)
    feed_forward = PositionwiseFeedForward(d_model=size, d_ff=2048)
    dropout_rate = 0.1

    with pytest.raises(TypeError):
        DecoderLayer(
            size="invalid",
            self_attn=self_attn,
            encoder_attn=encoder_attn,
            feed_forward=feed_forward,
            dropout_rate=dropout_rate,
        )
    with pytest.raises(TypeError):
        DecoderLayer(
            size=size,
            self_attn="invalid",
            encoder_attn=encoder_attn,
            feed_forward=feed_forward,
            dropout_rate=dropout_rate,
        )
    with pytest.raises(TypeError):
        DecoderLayer(
            size=size, self_attn=self_attn, encoder_attn="invalid", feed_forward=feed_forward, dropout_rate=dropout_rate
        )
    with pytest.raises(TypeError):
        DecoderLayer(
            size=size, self_attn=self_attn, encoder_attn=encoder_attn, feed_forward="invalid", dropout_rate=dropout_rate
        )
    with pytest.raises(TypeError):
        DecoderLayer(
            size=size, self_attn=self_attn, encoder_attn=encoder_attn, feed_forward=feed_forward, dropout_rate="invalid"
        )


def test_decoder_layer_initialization_value_errors(valid_decoder_layer_params):
    """Checks that DecoderLayer raises ValueError for invalid parameter values."""
    size, self_attn, encoder_attn, feed_forward, dropout_rate = valid_decoder_layer_params

    # Invalid dropout_rate
    with pytest.raises(ValueError):
        DecoderLayer(
            size=size, self_attn=self_attn, encoder_attn=encoder_attn, feed_forward=feed_forward, dropout_rate=1.1
        )
