import torch
import torch.nn as nn

from src.model.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadedAttention(nn.Module):
    """
    Multi-Headed Attention mechanism.  Allows the model to attend to information
    from different representation subspaces at different positions.
    """

    def __init__(self, heads_count: int, d_model: int, dropout_rate: float | int = 0.1):
        """
        Initializes the MultiHeadedAttention module.

        :param heads_count: Number of attention heads.  Must evenly divide d_model.
        :param d_model: Dimensionality of the input and output embeddings.
        :param dropout_rate: Dropout rate to apply to the attention weights.
        """
        super().__init__()

        if not isinstance(heads_count, int):
            raise TypeError(f"Heads count must be an int, but got {type(heads_count)}.")
        if not isinstance(d_model, int):
            raise TypeError(f"Dimensionality of input and output embeddings must be an int, but got {type(d_model)}.")
        if not isinstance(dropout_rate, float) and not isinstance(dropout_rate, int):
            raise TypeError(f"Dropout_rate must be a float or an int, but got {type(dropout_rate)}.")
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"Dropout_rate must be between 0 and 1, but got {dropout_rate}.")
        if d_model % heads_count != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads_count ({heads_count}).")

        self.__d_k = d_model // heads_count
        self.__heads_count = heads_count
        self.__attention = ScaledDotProductAttention(dropout_rate)
        self.__attn_probs = None

        # Linear layers for projecting the query, key, value, and output.
        self.__w_q = nn.Linear(d_model, d_model)
        self.__w_k = nn.Linear(d_model, d_model)
        self.__w_v = nn.Linear(d_model, d_model)
        self.__w_o = nn.Linear(d_model, d_model)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Computes the multi-headed attention output.

        :param query: Query tensor. Shape: (batch_size, seq_len_q, d_model).
                      seq_len_q - sequence length of the query tensor.
        :param key: Key tensor. Shape: (batch_size, seq_len_k, d_model).
                    seq_len_k - sequence length of the key tensor.
        :param value: Value tensor. Shape: (batch_size, seq_len_v, d_model).
                      seq_len_v - sequence length of the value tensor.
        :param mask: Optional mask to prevent certain positions from attending.
                     Shape: (batch_size, 1, seq_len_q, seq_len_k). Use 1 for values we don't want to mask.

        :return: Multi-headed attention output. Shape: (batch_size, seq_len_q, d_model).
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add an extra dimension to the mask tensor so it can be easily applied across.
            # all attention heads. Now: (batch_size, 1, 1, seq_len_k).
        nbatches = query.size(0)  # Batch size.

        # 1) Project the query, key, and value tensors through their respective linear layers.
        # 2) Split the results into `heads_count` heads.
        # 3) Transpose the results to have shape (batch_size, num_heads, seq_len, d_k).
        query = self.__w_q(query).view(nbatches, -1, self.__heads_count, self.__d_k).transpose(1, 2)
        key = self.__w_k(key).view(nbatches, -1, self.__heads_count, self.__d_k).transpose(1, 2)
        value = self.__w_v(value).view(nbatches, -1, self.__heads_count, self.__d_k).transpose(1, 2)

        # Apply the Scaled Dot-Product Attention to the projected and split queries, keys, and values.
        x, self.__attn_probs = self.__attention(query, key, value, mask)

        # 1) Transpose back.
        # 2) Concatenate the heads.
        # 3) Apply the final linear layer.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.__heads_count * self.__d_k)
        x = self.__w_o(x)
        return x
