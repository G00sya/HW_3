import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        Initializes the PositionwiseFeedForward module.
        This module implements a feed-forward network applied to each position separately.
        :param d_model:  Input and output dimensionality.
        :param d_ff: Inner layer dimensionality.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Applies the position-wise feed-forward network to the input tensor.
        :param inputs: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor of shape (batch_size, seq_len, d_model).
        """
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))
