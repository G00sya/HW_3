from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from src.model.encoder_block import EncoderBlock
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.residual_block import ResidualBlock


class TestEncoderBlock:
    def test_init(self, init_encoder_block):
        """Test that the EncoderBlock class is initialized correctly."""
        assert hasattr(init_encoder_block, "_EncoderBlock__self_attn"), "Self-attention module is not defined"
        assert isinstance(
            init_encoder_block._EncoderBlock__self_attn, MultiHeadedAttention
        ), "Self-attention is not of type MultiHeadedAttention"
        assert hasattr(init_encoder_block, "_EncoderBlock__feed_forward"), "Feed-forward module is not defined"
        assert isinstance(
            init_encoder_block._EncoderBlock__feed_forward, nn.Linear
        ), "Feed-forward is not of type nn.Linear"
        assert hasattr(
            init_encoder_block, "_EncoderBlock__self_attention_block"
        ), "Self-attention ResidualBlock is not defined"
        assert isinstance(
            init_encoder_block._EncoderBlock__self_attention_block, ResidualBlock
        ), "Self-attention ResidualBlock is not of type ResidualBlock"
        assert hasattr(
            init_encoder_block, "_EncoderBlock__feed_forward_block"
        ), "Feed-forward ResidualBlock is not defined"
        assert isinstance(
            init_encoder_block._EncoderBlock__feed_forward_block, ResidualBlock
        ), "Feed-forward ResidualBlock is not of type ResidualBlock"

    def test_raise_error_with_invalid_value(self, init_encoder_block, encoder_block_sample_tensors):
        """Test that an error is raised when invalid value is given."""
        mock_nn_module = MagicMock(spec=nn.Module)
        with pytest.raises(TypeError):
            EncoderBlock(size=45.5, self_attn=mock_nn_module, feed_forward=mock_nn_module, dropout_rate=0.1)
        with pytest.raises(ValueError):
            EncoderBlock(size=-64, self_attn=mock_nn_module, feed_forward=mock_nn_module, dropout_rate=0.1)
        with pytest.raises(TypeError):
            EncoderBlock(size=64, self_attn=mock_nn_module, feed_forward=mock_nn_module, dropout_rate="dropout_rate")
        with pytest.raises(ValueError):
            EncoderBlock(size=64, self_attn=mock_nn_module, feed_forward=mock_nn_module, dropout_rate=11.1)
        with pytest.raises(TypeError):
            EncoderBlock(size=64, self_attn="self_attn", feed_forward=mock_nn_module, dropout_rate=0.1)
        with pytest.raises(TypeError):
            EncoderBlock(size=64, self_attn=mock_nn_module, feed_forward="feed_forward", dropout_rate=0.1)
        with pytest.raises(TypeError):
            init_encoder_block([0, 1], encoder_block_sample_tensors[1])
        with pytest.raises(TypeError):
            init_encoder_block(encoder_block_sample_tensors[0], [2, 3])

    def test_forward_shape(self, init_encoder_block, encoder_block_sample_tensors):
        """Test the output shape of the EncoderBlock's forward method."""
        inputs, mask = encoder_block_sample_tensors
        output = init_encoder_block(inputs, mask)

        batch_size, seq_len, size = inputs.shape
        assert output.shape == (batch_size, seq_len, size), "Output shape is incorrect"

    def test_forward_pass(self, init_encoder_block, encoder_block_sample_tensors):
        """Test a simple forward pass through the EncoderBlock."""
        inputs, mask = encoder_block_sample_tensors
        output = init_encoder_block(inputs, mask)

        # Check that the output is not the same as the input (basic functional check)
        assert not torch.equal(inputs, output), "Output should not be identical to input"
