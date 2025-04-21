import pytest
import torch

from src.model.encoder import Encoder


class TestEncoder:
    @pytest.mark.parametrize(
        "param, invalid_value, expected_error",
        [
            ("d_model", -10, ValueError),
            ("d_model", 512.0, TypeError),
            ("d_ff", -2048, ValueError),
            ("d_ff", 2048.0, TypeError),
            ("blocks_count", -6, ValueError),
            ("blocks_count", 6.0, TypeError),
            ("heads_count", -8, ValueError),
            ("heads_count", 8.0, TypeError),
            ("dropout_rate", -0.1, ValueError),
            ("dropout_rate", 1.0, ValueError),
            ("dropout_rate", "0.1", TypeError),
            ("shared_embedding", torch.nn.Embedding(1000, 512), TypeError),
        ],
    )
    def test_init_validation(self, param, invalid_value, expected_error, init_encoder):
        """Test that invalid parameters raise the correct errors."""
        kwargs = {
            "d_model": 512,
            "d_ff": 2048,
            "blocks_count": 6,
            "heads_count": 8,
            "dropout_rate": 0.1,
            "shared_embedding": init_encoder[1],
            param: invalid_value,
        }

        with pytest.raises(expected_error):
            Encoder(**kwargs)

    def test_forward_shape(self, init_encoder, encoder_sample_tensors):
        """Test that forward returns the expected output shape."""
        batch_size, seq_len, inputs = encoder_sample_tensors
        mask = torch.ones((batch_size, seq_len, seq_len))
        encoder, _, d_model = init_encoder

        output = encoder(inputs, mask)
        assert output.shape == (
            batch_size,
            seq_len,
            d_model,
        ), "Output should be size of (batch, seq_len, d_model)"

    def test_forward_with_custom_mask(self, init_encoder, encoder_sample_tensors):
        """Test that the mask is applied correctly."""
        batch_size, seq_len, inputs = encoder_sample_tensors
        encoder = init_encoder[0]

        mask = torch.ones((batch_size, seq_len, seq_len))  # Mask out the last token
        mask[:, :, -1] = 0  # No attention to last token

        output = encoder(inputs, mask)
        assert not torch.isnan(output).any(), "Mask shouldn't cause NaNs"
