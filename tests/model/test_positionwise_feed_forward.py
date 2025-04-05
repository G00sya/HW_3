import torch
import torch.nn as nn

from src.model.positionwise_feed_forward import PositionwiseFeedForward


class TestPositionwiseFeedForward:
    def test_init(self) -> None:
        d_model = 512
        d_ff = 2048
        dropout = 0.1

        ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        assert isinstance(ffn._w_1, nn.Linear)
        assert isinstance(ffn._w_2, nn.Linear)
        assert isinstance(ffn._dropout, nn.Dropout)

        assert ffn._w_1.in_features == d_model
        assert ffn._w_1.out_features == d_ff
        assert ffn._w_2.in_features == d_ff
        assert ffn._w_2.out_features == d_model
        assert ffn._dropout.p == dropout

    def test_forward(self, init_positionwise_feed_forward) -> None:
        ffn, inputs = init_positionwise_feed_forward
        output = ffn(inputs)
        assert output.shape == inputs.shape

    def test_dropout_effect(self, init_positionwise_feed_forward) -> None:
        ffn, _ = init_positionwise_feed_forward
        d_model = 512
        batch_size = 32
        seq_len = 10

        inputs = torch.randn(batch_size, seq_len, d_model)
        output_train = ffn(inputs)

        ffn.eval()
        output_eval = ffn(inputs)
        assert not torch.equal(output_train, output_eval), "The outputs shouldn't be the same."
