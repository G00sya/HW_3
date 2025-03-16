from unittest.mock import MagicMock

import torch
import torch.nn as nn

from src.model.residual_block import ResidualBlock


class TestResidualBlock:
    def test_init(self, init_residual_block) -> None:
        """
        Test the initialization of the ResidualBlock.
        """
        residual_block, d_model, dropout_rate = init_residual_block

        assert residual_block._norm is not None, "LayerNorm should be initialized"
        assert residual_block._dropout is not None, "Dropout should be initialized"

        # Check if LayerNorm has the correct size
        assert residual_block._norm._gamma.shape == (d_model,), "LayerNorm size is incorrect"
        assert residual_block._norm._beta.shape == (d_model,), "LayerNorm size is incorrect"

        # Check if Dropout has the correct dropout rate
        assert residual_block._dropout.p == dropout_rate, "Dropout rate is incorrect"

    def test_residual_block_forward_calls(self):
        """
        Tests that the sublayers are called with the correct arguments.
        """
        size = 128
        dropout_rate = 0.1

        batch_size = 4
        seq_len = 32
        input_tensor = torch.randn(batch_size, seq_len, size)

        # Create MagicMock instances for the layers
        norm_mock = MagicMock(spec=nn.LayerNorm)
        dropout_mock = MagicMock(spec=nn.Dropout)

        # Create a ResidualBlock with the mocked layers
        residual_block = ResidualBlock(size, dropout_rate)
        residual_block._norm = norm_mock
        residual_block._dropout = dropout_mock

        # Create a Linear model as sublayer
        linear_layer = nn.Linear(size, size)
        sublayer = MagicMock(spec=linear_layer)

        residual_block.forward(input_tensor, sublayer)

        # Check that LayerNorm was called with the input tensor
        residual_block._norm.assert_called_once_with(input_tensor)

        # Check that the sublayer was called with the normalized input
        normalized_input = residual_block._norm.return_value  # Get the return value of LayerNorm
        sublayer.assert_called_once_with(normalized_input)

        # Check that Dropout was called with the sublayer output
        sublayer_output = sublayer.return_value
        residual_block._dropout.assert_called_once_with(sublayer_output)
