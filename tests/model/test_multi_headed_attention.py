import pytest
import torch
import torch.nn as nn

from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.scaled_dot_product_attention import ScaledDotProductAttention


class TestMultiHeadedAttention:
    def test_init(self, init_multi_headed_attention):
        """Test that the MultiHeadedAttention class is initialized correctly."""
        assert hasattr(init_multi_headed_attention, "_w_q"), "Linear layer _w_q is not defined"
        assert isinstance(init_multi_headed_attention._w_q, nn.Linear), "_w_q is not of type nn.Linear"
        assert hasattr(init_multi_headed_attention, "_w_k"), "Linear layer _w_k is not defined"
        assert isinstance(init_multi_headed_attention._w_k, nn.Linear), "_w_k is not of type nn.Linear"
        assert hasattr(init_multi_headed_attention, "_w_v"), "Linear layer _w_v is not defined"
        assert isinstance(init_multi_headed_attention._w_v, nn.Linear), "_w_v is not of type nn.Linear"
        assert hasattr(init_multi_headed_attention, "_w_o"), "Linear layer _w_o is not defined"
        assert isinstance(init_multi_headed_attention._w_o, nn.Linear), "_w_o is not of type nn.Linear"
        assert hasattr(init_multi_headed_attention, "_attention"), "ScaledDotProductAttention is not defined"
        assert isinstance(
            init_multi_headed_attention._attention, ScaledDotProductAttention
        ), "_attention is not of type ScaledDotProductAttention"
        assert (
            init_multi_headed_attention._d_k
            == init_multi_headed_attention._w_q.in_features // init_multi_headed_attention._heads_count
        ), "d_k is not calculated correctly"
        assert init_multi_headed_attention._heads_count == 8, "Number of heads is not set correctly in the module"

    def test_d_model_not_divisible_by_heads_count(self):
        """Test that a ValueError is raised when d_model is not divisible by heads_count."""
        heads_count = 3
        d_model = 10
        dropout_rate = 0.1

        # UCheck if a ValueError is raised.
        with pytest.raises(ValueError):
            MultiHeadedAttention(heads_count=heads_count, d_model=d_model, dropout_rate=dropout_rate)

    def test_shapes(self, init_multi_headed_attention, multi_headed_attention_sample_tensors):
        """Test the output shapes of the MultiHeadedAttention module."""
        query, key, value = multi_headed_attention_sample_tensors
        output = init_multi_headed_attention(query, key, value)

        batch_size, seq_len_q, d_model = query.shape
        assert output.shape == (batch_size, seq_len_q, d_model), "Output shape is incorrect"

    def test_masking(self, init_multi_headed_attention, multi_headed_attention_sample_tensors):
        """Test that the masking mechanism works correctly."""
        query, key, value = multi_headed_attention_sample_tensors
        batch_size, seq_len_q = query.shape[:2]
        seq_len_k = key.shape[1]

        # Create a mask to mask out the first half of the key sequence.
        mask = torch.ones(batch_size, seq_len_q, seq_len_k)
        mask[:, :, : seq_len_k // 2] = 0

        # Apply the mask to the attention mechanism.
        init_multi_headed_attention(query, key, value, mask)

        # Retrieve the attention probabilities.
        attn_probs = init_multi_headed_attention._attn_probs

        # Ensure that the attention probabilities for the masked positions are zero.
        assert torch.all(
            attn_probs[:, :, :, : seq_len_k // 2] == 0
        ), "Attention probabilities are not zero for masked positions"
