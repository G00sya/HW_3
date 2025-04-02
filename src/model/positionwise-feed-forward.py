import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.w_2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, inputs: torch.Tensor):
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))
