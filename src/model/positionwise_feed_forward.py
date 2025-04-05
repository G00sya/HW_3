import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float | int = 0.1):
        """
        Initializes the PositionwiseFeedForward module.
        This module implements a feed-forward network applied to each position separately.
        :param d_model:  Input and output dimensionality.
        :param d_ff: Inner layer dimensionality.
        :param dropout: Dropout rate.
        """
        super().__init__()

        if not isinstance(d_model, int):
            raise TypeError(f"Dimensionality of input and output embeddings must be an int, but got {type(d_model)}.")
        if not isinstance(d_ff, int):
            raise TypeError(f"Dimensionality of inner layer must be an int, but got {type(d_ff)}.")
        if not isinstance(dropout, float) and not isinstance(dropout, int):
            raise TypeError(f"Dropout rate must be a float or an int, but got {type(dropout)}.")
        if not 0 <= dropout <= 1:
            raise ValueError(f"Dropout rate must be between 0 and 1, but got {dropout}.")

        self.__w_1 = nn.Linear(d_model, d_ff)
        self.__w_2 = nn.Linear(d_ff, d_model)
        self.__dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies the position-wise feed-forward network to the input tensor.
        :param inputs: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.__w_2(self.__dropout(F.relu(self.__w_1(inputs))))
