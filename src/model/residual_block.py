import torch
import torch.nn as nn

from src.model.layer_norm import LayerNorm


class ResidualBlock(nn.Module):
    """
    Add to function result its argument to avoid the problem of a vanishing gradient.
    """

    def __init__(self, size: int, dropout_rate: float | int):
        """
        :param size: Dimensionality of the model's internal representations (embeddings). Must be a positive integer.
        :param dropout_rate: The probability of dropping out (setting to 0) a neuron's output during the forward pass.
                             Must be a float or int between 0 and 1.
        :raises TypeError: If `size` is not an integer.
        :raises ValueError: If `size` is not positive.
        :raises TypeError: If `dropout_rate` is not a float or int.
        :raises ValueError: If `dropout_rate` is not between 0 and 1.
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

        self.__norm = LayerNorm(size)  # Assuming LayerNorm takes size as an argument
        self.__dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Applies a residual connection around a sublayer.
        This method normalizes the input, passes it through a sublayer, applies dropout, and then adds the result back
        to the original input.

        # TODO check shapes
        :param inputs: The input tensor to the residual block. Shape: (batch_size, seq_len, d_model).
        :param sublayer: The sublayer to be applied (a feedforward network or attention mechanism).
        :raises TypeError: If `inputs` is not a torch.Tensor.
        :raises TypeError: If `sublayer` is not an nn.Module.
        :return: The output tensor after applying the residual connection. Shape: (batch_size, seq_len, d_model)
        """

        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Inputs must be a torch.Tensor, but got {type(inputs)}")

        normalized = self.__norm(inputs)
        sublayer_result = sublayer(normalized)
        dropout_result = self.__dropout(sublayer_result)
        return torch.add(inputs, dropout_result)
