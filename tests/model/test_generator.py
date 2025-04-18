import pytest
import torch
import torch.nn as nn

from src.model.generator import Generator


class TestGenerator:
    def test_initialization(self, init_generator):
        """
        Test init function of Generator.
        """
        generator, _, _ = init_generator
        assert isinstance(generator, Generator)
        assert isinstance(generator._Generator__proj, nn.Linear)
        assert generator._Generator__proj.in_features == 512
        assert generator._Generator__proj.out_features == 10000

    def test_generator_init_invalid_d_model(self):
        """
        Test with invalid d_model type (str).
        """
        d_model = "512"
        target_vocab_size = 10000

        with pytest.raises(TypeError, match="Expected d_model to be of type int, but got str"):
            Generator(d_model, target_vocab_size)

    def test_generator_init_invalid_target_vocab_size(self):
        """
        Test with invalid target_vocab_size type (str).
        """
        d_model = 512
        target_vocab_size = "10000"

        with pytest.raises(TypeError, match="Expected target_vocab_size to be of type int, but got str"):
            Generator(d_model, target_vocab_size)

    def test_forward_pass(self, init_generator, init_input_for_generator):
        """
        Test forward output for dimensionality and Softmax distribution.
        """
        generator, _, target_vocab_size = init_generator
        x = init_input_for_generator
        output = generator(x)
        batch_size = x.shape[0]
        assert output.shape == (batch_size, target_vocab_size)
        assert torch.allclose(output.exp().sum(dim=-1), torch.ones(batch_size))  # Check that sum of exponents == 1

    def test_output_range(self, init_generator, init_input_for_generator):
        """
        Test output values are less than zero.
        """
        generator, _, _ = init_generator
        x = init_input_for_generator
        output = generator(x)
        assert torch.all(output <= 0), "Log softmax output should be <= 0"
