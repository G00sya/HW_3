import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism. This is a core building block of the Transformer architecture.
    """

    def __init__(self, dropout_rate: float):
        """
        Initializes the attention module.

        :param dropout_rate: Dropout rate to apply to the attention weights. Regularization technique.
        """
        super().__init__()

        self._dropout = nn.Dropout(dropout_rate)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the attention weights and applies them to the values.

        :param query: Query tensor.  Shape: (batch_size, num_heads, seq_len_q, d_k)
        :param key: Key tensor. Shape: (batch_size, num_heads, seq_len_k, d_k)
        :param value: Value tensor. Shape: (batch_size, num_heads, seq_len_k, d_v)  d_v might be d_k sometimes
        :param mask: Optional mask to block certain positions from attending.
                     Shape: (batch_size, 1, seq_len_q, seq_len_k) or (1, 1, seq_len_q, seq_len_k)
                     Use 1 for values we don't want to mask.

        :return: A tuple containing:
            - The weighted values (attention output). Shape: (batch_size, num_heads, seq_len_q, d_v)
            - The attention probabilities (attention weights). Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)
        p_attn = functional.softmax(scores, dim=-1)
        p_attn = self._dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
