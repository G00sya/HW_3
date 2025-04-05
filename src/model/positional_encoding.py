import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.
        Computes the positional encodings once in log space, and then stores them.
        :param d_model: the dimension of the model (embedding size).
        :param dropout: the dropout value. (Dropout value)
        :param max_len: the maximum length of the input sequence. (Maximum length of the input sequence)
        """
        super().__init__()

        if not isinstance(d_model, int):
            raise TypeError(f"d_model must be an int, but got {type(d_model)}")
        if not isinstance(dropout, float):
            raise TypeError(f"dropout must be a float, but got {type(dropout)}")
        if not isinstance(max_len, int):
            raise TypeError(f"max_len must be an int, but got {type(max_len)}")
        if not 0 <= dropout <= 1:
            raise ValueError(f"Dropout rate must be between 0 and 1, but got {dropout}.")

        self.dropout = nn.Dropout(p=dropout)

        # Create a tensor (matrix) to store positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create a tensor with positions (from 0 to max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create a tensor to compute the divisor (div_term)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # Calculate sine values for even columns
        pe[:, 0::2] = torch.sin(position * div_term)

        # Calculate cosine values for odd columns
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a dummy dimension (batch dimension) to the beginning
        pe = pe.unsqueeze(0)

        # Register the pe tensor as a buffer (not a model parameter)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input. (Applies positional encoding to the input)
        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Add positional encodings to the input tensor
        x = x + self.pe[:, : x.size(1)]

        # Apply Dropout
        return self.dropout(x)
