import pytest
import torch
import torch.nn as nn

from src.model.positionwise_feed_forward import PositionwiseFeedForward


@pytest.fixture
def init_positionwise_feed_forward():
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    seq_len = 10

    ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
    inputs = torch.randn(batch_size, seq_len, d_model)
    return ffn, inputs


class TestPositionwiseFeedForward:
    def test_init(self) -> None:
        d_model = 512
        d_ff = 2048
        dropout = 0.1

        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        assert isinstance(ffn.w_1, nn.Linear)
        assert isinstance(ffn.w_2, nn.Linear)
        assert isinstance(ffn.dropout, nn.Dropout)

        assert ffn.w_1.in_features == d_model
        assert ffn.w_1.out_features == d_ff
        assert ffn.w_2.in_features == d_ff
        assert ffn.w_2.out_features == d_model
        assert ffn.dropout.p == dropout

    def test_forward(self, init_positionwise_feed_forward) -> None:
        ffn, inputs = init_positionwise_feed_forward
        output = ffn(inputs)
        assert output.shape == inputs.shape

    def test_dropout_effect(self, init_positionwise_feed_forward) -> None:
        ffn, _ = init_positionwise_feed_forward
        d_model = 512
        batch_size = 32
        seq_len = 10

        ffn.train()
        inputs = torch.randn(batch_size, seq_len, d_model)
        output_train = ffn(inputs)

        num_zeros_train = torch.sum(output_train == 0).item()
        assert num_zeros_train > 0, "Dropout layer is not zeroing out elements during training mode."

        ffn.eval()
        output_eval = ffn(inputs)
        assert not torch.equal(output_train, output_eval), "The outputs shouldn't be the same."
