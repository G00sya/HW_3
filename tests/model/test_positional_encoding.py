import pytest
import torch
import torch.nn as nn

from src.model.positional_encoding import PositionalEncoding


class TestPositionalEncoding:
    def test_init(self) -> None:
        """
        Test the initialization of the PositionalEncoding module.
        Checks if the attributes are initialized correctly.
        """
        d_model = 512
        dropout = 0.1
        max_len = 100

        pe = PositionalEncoding(d_model, dropout, max_len)

        assert isinstance(pe.dropout, nn.Dropout)
        assert pe.dropout.p == dropout
        assert pe.pe.shape == (1, max_len, d_model)

    def test_forward(self, init_positional_encoding) -> None:
        """
        Test the forward pass of the PositionalEncoding module.
        Checks if the output tensor has the correct shape and if the positional encodings are added correctly.
        """
        pe, d_model, dropout, max_len = init_positional_encoding
        batch_size = 32
        seq_len = 50

        inputs = torch.randn(batch_size, seq_len, d_model)

        output = pe(inputs)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.equal(inputs, output)

    def test_dropout_effect(self, init_positional_encoding) -> None:
        """
        Test that dropout layer is actually applied
        """
        pe, d_model, dropout, max_len = init_positional_encoding
        batch_size = 32
        seq_len = 50

        pe.train()
        inputs = torch.randn(batch_size, seq_len, d_model)

        output_train = pe(inputs)
        num_zeros_train = torch.sum(torch.abs(output_train) < 1e-6).item()

        assert num_zeros_train > 0, "Dropout layer is not zeroing out elements during training mode."

    def test_valid_initialization(self):
        """Test successful initialization."""
        pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=100)
        assert isinstance(pe.dropout, nn.Dropout)

    def test_invalid_d_model_type(self):
        """Test for invalid d_model type."""
        with pytest.raises(TypeError, match="d_model must be an int"):
            PositionalEncoding(d_model="512", dropout=0.1, max_len=100)

    def test_invalid_dropout_type(self):
        """Test for invalid dropout type."""
        with pytest.raises(TypeError, match="dropout must be a float"):
            PositionalEncoding(d_model=512, dropout="0.1", max_len=100)

    def test_invalid_max_len_type(self):
        """Test for invalid max_len type."""
        with pytest.raises(TypeError, match="max_len must be an int"):
            PositionalEncoding(d_model=512, dropout=0.1, max_len="100")

    def test_invalid_dropout_value(self):
        """Test for invalid dropout value (outside the range 0-1)."""
        with pytest.raises(ValueError, match="Dropout rate must be between 0 and 1"):
            PositionalEncoding(d_model=512, dropout=1.1, max_len=100)
