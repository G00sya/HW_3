import pytest
import torch
import torch.nn as nn

from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.scaled_dot_product_attention import ScaledDotProductAttention


class TestMultiHeadedAttention:
    def test_init(self, init_multi_headed_attention):
        """Test that the MultiHeadedAttention class is initialized correctly."""
        multi_headed_attention, head_count = init_multi_headed_attention
        assert hasattr(multi_headed_attention, "_MultiHeadedAttention__w_q"), "Linear layer __w_q is not defined."
        assert isinstance(
            multi_headed_attention._MultiHeadedAttention__w_q, nn.Linear
        ), "__w_q is not of type nn.Linear."
        assert hasattr(multi_headed_attention, "_MultiHeadedAttention__w_k"), "Linear layer __w_k is not defined."
        assert isinstance(
            multi_headed_attention._MultiHeadedAttention__w_k, nn.Linear
        ), "__w_k is not of type nn.Linear."
        assert hasattr(multi_headed_attention, "_MultiHeadedAttention__w_v"), "Linear layer __w_v is not defined."
        assert isinstance(
            multi_headed_attention._MultiHeadedAttention__w_v, nn.Linear
        ), "__w_v is not of type nn.Linear."
        assert hasattr(multi_headed_attention, "_MultiHeadedAttention__w_o"), "Linear layer __w_o is not defined."
        assert isinstance(
            multi_headed_attention._MultiHeadedAttention__w_o, nn.Linear
        ), "__w_o is not of type nn.Linear."
        assert hasattr(
            multi_headed_attention, "_MultiHeadedAttention__attention"
        ), "ScaledDotProductAttention is not defined."
        assert isinstance(
            multi_headed_attention._MultiHeadedAttention__attention, ScaledDotProductAttention
        ), "__attention is not of type ScaledDotProductAttention."
        assert (
            multi_headed_attention._MultiHeadedAttention__d_k
            == multi_headed_attention._MultiHeadedAttention__w_q.in_features
            // multi_headed_attention._MultiHeadedAttention__heads_count
        ), "__d_k is not calculated correctly."
        assert (
            multi_headed_attention._MultiHeadedAttention__heads_count == head_count
        ), "Number of heads is not set correctly in the module."

    def test_raise_error_with_invalid_value(self):
        """Test that an error is raised when invalid value is given."""
        with pytest.raises(ValueError):
            MultiHeadedAttention(heads_count=3, d_model=10, dropout_rate=0.1)
        with pytest.raises(TypeError):
            MultiHeadedAttention(heads_count=0.3, d_model=10, dropout_rate=0.1)
        with pytest.raises(TypeError):
            MultiHeadedAttention(heads_count=3, d_model=10.5, dropout_rate=0.1)
        with pytest.raises(TypeError):
            MultiHeadedAttention(heads_count=3, d_model=10, dropout_rate=1)

    def test_shapes(self, init_multi_headed_attention, multi_headed_attention_sample_tensors):
        """Test the output shapes of the MultiHeadedAttention module."""
        multi_headed_attention, _ = init_multi_headed_attention
        query, key, value = multi_headed_attention_sample_tensors
        output = multi_headed_attention(query, key, value)

        batch_size, seq_len_q, d_model = query.shape
        assert output.shape == (batch_size, seq_len_q, d_model), "Output shape is incorrect."

    def test_masking(self, init_multi_headed_attention, multi_headed_attention_sample_tensors):
        """Test that the masking mechanism works correctly."""
        multi_headed_attention, _ = init_multi_headed_attention
        query, key, value = multi_headed_attention_sample_tensors
        batch_size, seq_len_q = query.shape[:2]
        seq_len_k = key.shape[1]

        # Create a mask to mask out the first half of the key sequence.
        mask = torch.ones(batch_size, seq_len_q, seq_len_k)
        mask[:, :, : seq_len_k // 2] = 0

        # Apply the mask to the attention mechanism.
        multi_headed_attention(query, key, value, mask)

        # Retrieve the attention probabilities.
        attn_probs = multi_headed_attention._MultiHeadedAttention__attn_probs

        # Ensure that the attention probabilities for the masked positions are zero.
        assert torch.all(
            attn_probs[:, :, :, : seq_len_k // 2] == 0
        ), "Attention probabilities are not zero for masked positions."
