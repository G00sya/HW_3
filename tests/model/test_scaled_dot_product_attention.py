import torch


class TestScaledDotProductAttention:
    def test_init(self, init_scaled_dot_product_attention):
        """Tests that the ScaledDotProductAttention class is initialized correctly."""
        scaled_dot_product_attention, dropout_rate = init_scaled_dot_product_attention
        assert hasattr(scaled_dot_product_attention, "_dropout"), "Dropout layer is not defined as _dropout."
        assert isinstance(
            scaled_dot_product_attention._dropout, torch.nn.Dropout
        ), "The dropout layer is not of type nn.Dropout"
        assert scaled_dot_product_attention._dropout.p == 0.1, "Dropout rate is not set correctly in the module."

    def test_shapes(self, init_scaled_dot_product_attention, scaled_dot_product_attention_sample_tensors):
        scaled_dot_product_attention, _ = init_scaled_dot_product_attention
        """Test that the attention module produces the expected output shapes."""
        query, key, value = scaled_dot_product_attention_sample_tensors
        mask = None
        output, attn_weights = scaled_dot_product_attention(query, key, value, mask)

        batch_size, num_heads, seq_len_q = query.shape[:3]
        d_v = value.size(-1)
        seq_len_k = key.size(-2)

        assert output.shape == (batch_size, num_heads, seq_len_q, d_v), "Output shape is incorrect."
        assert attn_weights.shape == (
            batch_size,
            num_heads,
            seq_len_q,
            seq_len_k,
        ), "Attention weights shape is incorrect."

    def test_masking(self, init_scaled_dot_product_attention, scaled_dot_product_attention_sample_tensors):
        """Test that the attention module correctly applies masking."""
        scaled_dot_product_attention, _ = init_scaled_dot_product_attention
        query, key, value = scaled_dot_product_attention_sample_tensors
        batch_size, num_heads, seq_len_q = query.shape[:3]
        seq_len_k = key.shape[2]

        # Create a mask where the first half of the sequence is masked out.
        mask = torch.ones(batch_size, 1, seq_len_q, seq_len_k)
        mask[:, :, :, : seq_len_k // 2] = 0  # Mask out the first half of the keys.

        output, attn_weights = scaled_dot_product_attention(query, key, value, mask)

        # Check that the attention weights are close to zero in the masked positions.
        assert torch.all(
            attn_weights[:, :, :, : seq_len_k // 2] < 1e-6
        ), "Masking failed: attention weights not near zero."

        # Calculate the mean attention weight in the masked and unmasked regions.
        masked_weights = attn_weights[:, :, :, : seq_len_k // 2]
        unmasked_weights = attn_weights[:, :, :, seq_len_k // 2 :]

        mean_masked_weight = masked_weights.mean()
        mean_unmasked_weight = unmasked_weights.mean()

        # Assert that the ratio of mean weights is small enough.
        # (Masked weights should be much smaller than unmasked).
        relative_ratio = mean_masked_weight / mean_unmasked_weight
        assert relative_ratio < 0.1, f"Masking failed: relative ratio is {relative_ratio:.3f}, expected less than 0.1."
