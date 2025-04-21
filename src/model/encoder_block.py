import torch
import torch.nn as nn

from src.model.residual_block import ResidualBlock


class EncoderBlock(nn.Module):
    """
    An Encoder Block consisting of a self-attention layer followed by a feed-forward network.
    This is a standard building block in Transformer encoder architectures.
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout_rate: float | int):
        """
        Initializes the EncoderBlock module.

        :param size: Dimensionality of the input and output tensors.
        :param self_attn: Self-attention module (e.g., MultiHeadedAttention).
        :param feed_forward: Feed-forward network module (e.g., a sequence of linear layers).
        :param dropout_rate: Dropout rate to apply within the residual blocks.
        """
        super().__init__()

        if not isinstance(size, int):
            raise TypeError(f"Size must be an int, but got {type(size)}")
        if size <= 0:
            raise ValueError(f"Size must be a positive number, but got {size}")
        if not isinstance(dropout_rate, float) and not isinstance(dropout_rate, int):
            raise TypeError(f"Dropout_rate must be a float or int, but got {type(dropout_rate)}")
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"Dropout_rate must be between 0 and 1, but got {dropout_rate}")
        if not isinstance(self_attn, nn.Module):
            raise TypeError(f"self_attn must be an nn.Module, but got {type(self_attn)}")
        if not isinstance(feed_forward, nn.Module):
            raise TypeError(f"feed_forward must be an nn.Module, but got {type(feed_forward)}")

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
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Inputs must be a torch.Tensor, but got {type(inputs)}")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"Mask must be an torch.Tensor, but got {type(mask)}")

        outputs = self.__self_attention_block(inputs, lambda x: self.__self_attn(x, x, x, mask))
        return self.__feed_forward_block(outputs, self.__feed_forward)
