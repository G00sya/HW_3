import torch.nn as nn

from src.model.residual_block import ResidualBlock


class DecoderLayer(nn.Module):
    """
    Represents a single layer within the Decoder.  Each layer consists of self-attention,
    encoder-decoder attention, and a feed-forward network.
    """

    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout_rate):
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

        self.__self_attn = self_attn
        self.__encoder_attn = encoder_attn
        self.__feed_forward = feed_forward
        self.__self_attention_block = ResidualBlock(size, dropout_rate)
        self.__attention_block = ResidualBlock(size, dropout_rate)
        self.__feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
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
        outputs = self._attention_block(
            outputs, lambda inputs: self.__encoder_attn(inputs, encoder_output, encoder_output, source_mask)
        )
        return self.__feed_forward_block(outputs, self.__feed_forward)
