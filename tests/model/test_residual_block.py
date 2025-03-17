from unittest.mock import MagicMock

import pytest
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

    def test_init_size_type_error(self):
        """
        Test that TypeError is raised when size is not an int.
        """
        with pytest.raises(TypeError):
            ResidualBlock(size="128", dropout_rate=0.1)

    def test_init_size_value_error(self):
        """
        Test that ValueError is raised when size is not positive.
        """
        with pytest.raises(ValueError):
            ResidualBlock(size=-128, dropout_rate=0.1)

    def test_init_dropout_rate_type_error(self):
        """
        Test that TypeError is raised when dropout_rate is not a float or int.
        """
        with pytest.raises(TypeError):
            ResidualBlock(size=128, dropout_rate="0.1")

    def test_init_dropout_rate_value_error_too_small(self):
        """
        Test that ValueError is raised when dropout_rate is less than 0.
        """
        with pytest.raises(ValueError):
            ResidualBlock(size=128, dropout_rate=-0.1)

    def test_init_dropout_rate_value_error_too_large(self):
        """
        Test that ValueError is raised when dropout_rate is greater than 1.
        """
        with pytest.raises(ValueError):
            ResidualBlock(size=128, dropout_rate=1.1)

    def test_forward_inputs_type_error(self, init_residual_block):
        """
        Test that TypeError is raised when inputs is not a torch.Tensor.
        """
        residual_block, d_model, dropout_rate = init_residual_block
        sublayer = nn.Linear(d_model, d_model)

        with pytest.raises(TypeError):
            residual_block.forward(inputs="not a tensor", sublayer=sublayer)

    def test_forward_sublayer_type_error(self, init_residual_block):
        """
        Test that TypeError is raised when sublayer is not an nn.Module.
        """
        residual_block, d_model, dropout_rate = init_residual_block
        inputs = torch.randn(4, 32, 2)

        with pytest.raises(TypeError):
            residual_block.forward(inputs=inputs, sublayer="not a module")

    def test_residual_block_forward_calls(self, init_residual_block):
        """
        Tests that the sublayers are called with the correct arguments.
        """
        # Create MagicMock instances for the layers
        norm_mock = MagicMock(spec=nn.LayerNorm)
        dropout_mock = MagicMock(spec=nn.Dropout)

        # Create a ResidualBlock
        residual_block, d_model, dropout_rate = init_residual_block

        # Create input tensor
        batch_size = 4
        seq_len = 32
        input_tensor = torch.randn(batch_size, seq_len, d_model)

        # Create a ResidualBlock's mocks
        residual_block._norm = norm_mock
        residual_block._dropout = dropout_mock
        residual_block._dropout.return_value = input_tensor

        # Create a Linear model as sublayer
        sublayer = MagicMock(spec=nn.Linear(d_model, d_model))

        residual_block.forward(input_tensor, sublayer)

        # Check that LayerNorm was called with the input tensor
        residual_block._norm.assert_called_once_with(input_tensor)

        # Check that the sublayer was called with the normalized input
        normalized_input = residual_block._norm.return_value  # Get the return value of LayerNorm
        sublayer.assert_called_once_with(normalized_input)

        # Check that Dropout was called with the sublayer output
        sublayer_output = sublayer.return_value
        residual_block._dropout.assert_called_once_with(sublayer_output)
