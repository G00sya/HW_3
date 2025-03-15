import torch
import torch.nn as nn

from src.model.layer_norm import LayerNorm


class ResidualBlock(nn.Module):
    """
    Add to function result its argument to avoid the problem of a vanishing gradient.
    """

    def __init__(self, size: int, dropout_rate: float):
        """
        :param size: Dimensionality of the model’s internal representations (embeddings).
        :param dropout_rate: The probability of dropping out (setting to 0) a neuron’s output during the forward pass.
        """
        super().__init__()
        self._norm = LayerNorm(size)
        self._dropout = nn.Dropout(
            dropout_rate
        )  # Prevent overfitting by randomly dropping out neurons during training

    def forward(self, inputs: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies a residual connection around a sublayer.
        This method normalizes the input, passes it through a sublayer, applies dropout, and then adds the result back
        to the original input.

        # TODO check shapes
        :param inputs: The input tensor to the residual block. Shape: (batch_size, seq_len, d_model).
        :param sublayer: The sublayer to be applied (a feedforward network or attention mechanism).
        :return: The output tensor after applying the residual connection. Shape: (batch_size, seq_len, d_model)
        """
        normalized = self._norm(inputs)
        sublayer_result = sublayer(normalized)
        dropout_result = self._dropout(sublayer_result)
        return inputs + dropout_result
