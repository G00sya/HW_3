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

    def test_forward(self) -> None:
        """
        Test the forward pass of the PositionalEncoding module.
        Checks if the output tensor has the correct shape and if the positional encodings are added correctly.
        """
        d_model = 512
        dropout = 0.1
        max_len = 100
        batch_size = 32
        seq_len = 50

        pe = PositionalEncoding(d_model, dropout, max_len)
        inputs = torch.randn(batch_size, seq_len, d_model)

        output = pe(inputs)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.equal(inputs, output)

    def test_dropout_effect(self) -> None:
        """
        Test that dropout layer is actually applied
        """
        d_model = 512
        dropout = 0.5
        max_len = 100
        batch_size = 32
        seq_len = 50

        pe = PositionalEncoding(d_model, dropout, max_len)
        pe.train()
        inputs = torch.randn(batch_size, seq_len, d_model)

        output_train = pe(inputs)
        num_zeros_train = torch.sum(torch.abs(output_train) < 1e-6).item()

        assert num_zeros_train > 0, "Dropout layer is not zeroing out elements during training mode."
