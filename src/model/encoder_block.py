import torch
import torch.nn as nn

from src.model.residual_block import ResidualBlock
from src.utils.lambda_wrapper import LambdaWrapper


class EncoderBlock(nn.Module):
    """
    An Encoder Block consisting of a self-attention layer followed by a feed-forward network.
    This is a standard building block in Transformer encoder architectures.
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout_rate: float):
        """
        Initializes the EncoderBlock module.

        :param size: Dimensionality of the input and output tensors.
        :param self_attn: Self-attention module (e.g., MultiHeadedAttention).
        :param feed_forward: Feed-forward network module (e.g., a sequence of linear layers).
        :param dropout_rate: Dropout rate to apply within the residual blocks.
        """
        super().__init__()

        self.__self_attn = self_attn
        self.__feed_forward = feed_forward
        self.__self_attention_block = ResidualBlock(size, dropout_rate)
        self.__feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the output of the EncoderBlock.

        :param inputs: Input tensor. Shape: (batch_size, seq_len, size)
        :param mask: Mask to apply to the self-attention mechanism. Shape: (batch_size, seq_len, seq_len)

        :return: Output tensor. Shape: (batch_size, seq_len, size)
        """
        outputs = self.__self_attention_block(inputs, LambdaWrapper(lambda x: self.__self_attn(x, x, x, mask)))
        return self.__feed_forward_block(outputs, self.__feed_forward)
