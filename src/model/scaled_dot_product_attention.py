import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism. This is a core building block of the Transformer architecture.
    """

    def __init__(self, dropout_rate: float | int):
        """
        Initializes the attention module.

        :param dropout_rate: Dropout rate to apply to the attention weights. Regularization technique.
        """
        super().__init__()

        if not isinstance(dropout_rate, float) and not isinstance(dropout_rate, int):
            raise TypeError(f"Dropout_rate must be a float or int, but got {type(dropout_rate)}")
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"Dropout_rate must be between 0 and 1, but got {dropout_rate}")

        self.__dropout = nn.Dropout(dropout_rate)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the attention weights and applies them to the values.

        :param query: Query tensor.  Shape: (batch_size, num_heads, seq_len_q, d_k).
                      seq_len_q - sequence length of query.
        :param key: Key tensor. Shape: (batch_size, num_heads, seq_len_k, d_k).
                    seq_len_k - sequence length of key.
        :param value: Value tensor. Shape: (batch_size, num_heads, seq_len_k, d_v)  d_v might be d_k sometimes.
        :param mask: Optional mask to block certain positions from attending.
                     Shape: (batch_size, 1, seq_len_q, seq_len_k). Use 1 for values we don't want to mask.

        :return: A tuple containing:
            - The weighted values (attention output). Shape: (batch_size, num_heads, seq_len_q, d_v).
            - The attention probabilities (attention weights). Shape: (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # We set scores to a large negative value in order to effectively force the softmax to assign near-zero
            # probabilities to the masked positions. This is crucial for preventing those positions from influencing
            # the attention mechanism.
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.__dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
