import torch
import torch.nn as nn

from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positionwise_feed_forward import PositionwiseFeedForward
from src.model.residual_block import ResidualBlock


class DecoderLayer(nn.Module):
    """
    Represents a single layer within the Decoder.  Each layer consists of self-attention,
    encoder-decoder attention, and a feed-forward network.
    """

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        encoder_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float,
    ):
        """
        Initializes the DecoderLayer.

        :param size: The dimensionality of the layer's embeddings and hidden states (d_model).
        :param self_attn: The self-attention module (MultiHeadedAttention)
        for attending to the decoder's own output.
        :param encoder_attn: The encoder-decoder attention module (MultiHeadedAttention)
         for attending to the encoder's output.
        :param feed_forward: The feed-forward network (PositionwiseFeedForward)
        for further processing.
        :param dropout_rate: The dropout probability.
        """
        super().__init__()

        if not isinstance(size, int):
            raise TypeError(f"size must be an int, but got {type(size)}")
        if not isinstance(self_attn, MultiHeadedAttention):
            raise TypeError(f"self_attn must be a MultiHeadedAttention, but got {type(self_attn)}")
        if not isinstance(encoder_attn, MultiHeadedAttention):
            raise TypeError(f"encoder_attn must be a MultiHeadedAttention, but got {type(encoder_attn)}")
        if not isinstance(feed_forward, PositionwiseFeedForward):
            raise TypeError(f"feed_forward must be a PositionwiseFeedForward, but got {type(feed_forward)}")
        if not isinstance(dropout_rate, float):
            raise TypeError(f"dropout_rate must be a float, but got {type(dropout_rate)}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")

        self.__self_attn = self_attn
        self.__encoder_attn = encoder_attn
        self.__feed_forward = feed_forward
        self.__self_attention_block = ResidualBlock(size, dropout_rate)
        self.__attention_block = ResidualBlock(size, dropout_rate)
        self.__feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(
        self, inputs: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs a forward pass through the DecoderLayer.

        :param inputs: The input tensor to the decoder layer.  Shape: (batch_size, target_seq_len, d_model).
        :param encoder_output: The output tensor from the encoder. Shape: (batch_size, source_seq_len, d_model).
        :param source_mask:  The mask for the source sequence, indicating which positions are valid (not padding).
        Shape: (batch_size, 1, source_seq_len).
        :param target_mask: The mask for the target sequence,
        indicating which positions are valid and preventing future positions from being attended to (look-ahead mask).
        Shape: (batch_size, target_seq_len, target_seq_len).
        :return: The output tensor after processing by the decoder layer.  Shape: (batch_size, target_seq_len, d_model).
        """
        outputs = self.__self_attention_block(
            inputs, lambda inputs: self.__self_attn(inputs, inputs, inputs, target_mask)
        )
        outputs = self.__attention_block(
            outputs, lambda inputs: self.__encoder_attn(inputs, encoder_output, encoder_output, source_mask)
        )
        return self.__feed_forward_block(outputs, self.__feed_forward)
