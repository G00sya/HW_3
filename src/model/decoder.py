import torch.nn as nn
from layer_norm import LayerNorm
from multi_headed_attention import MultiHeadedAttention
from positional_encoding import PositionalEncoding
from positionwise_feed_forward import PositionwiseFeedForward
from residual_block import ResidualBlock


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout_rate):
        super().__init__()

        self.__self_attn = self_attn
        self.__encoder_attn = encoder_attn
        self.__feed_forward = feed_forward
        self.__self_attention_block = ResidualBlock(size, dropout_rate)
        self.__attention_block = ResidualBlock(size, dropout_rate)
        self.__feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        outputs = self.__self_attention_block(
            inputs, lambda inputs: self.__self_attn(inputs, inputs, inputs, target_mask)
        )
        outputs = self._attention_block(
            outputs, lambda inputs: self.__encoder_attn(inputs, encoder_output, encoder_output, source_mask)
        )
        return self.__feed_forward_block(outputs, self.__feed_forward)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate):
        super().__init__()

        self.__emb = nn.Sequential(nn.Embedding(vocab_size, d_model), PositionalEncoding(d_model, dropout_rate))

        def create_decoder_block():
            return DecoderLayer(
                size=d_model,
                self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
                encoder_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
                feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
                dropout_rate=dropout_rate,
            )

        self.__blocks = nn.ModuleList([create_decoder_block() for _ in range(blocks_count)])
        self.__norm = LayerNorm(d_model)
        self.__out_layer = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        inputs = self.__emb(inputs)
        for block in self.__blocks:
            inputs = block(inputs, encoder_output, source_mask, target_mask)
        return self.__out_layer(self.__norm(inputs))
