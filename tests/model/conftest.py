import pytest
import torch

from src.model.layer_norm import LayerNorm
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.scaled_dot_product_attention import ScaledDotProductAttention


@pytest.fixture
def init_layer_norm() -> (LayerNorm, torch.Tensor):
    """
    Init LayerNorm and feature_vector.
    """
    d_model = 2  # Dimensionality of the model’s internal representations (embeddings).
    layer_norm = LayerNorm(features=d_model)
    feature_vector = torch.arange(start=1, end=d_model + 1, dtype=torch.float32)
    return layer_norm, feature_vector


@pytest.fixture
def init_scaled_dot_product_attention() -> ScaledDotProductAttention:
    dropout_rate = 0.1
    return ScaledDotProductAttention(dropout_rate=dropout_rate)


@pytest.fixture
def scaled_dot_product_attention_sample_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 2
    num_heads = 4
    seq_len_q = 8
    seq_len_k = 10
    d_k = 32
    d_v = 64

    query = torch.randn(batch_size, num_heads, seq_len_q, d_k)
    key = torch.randn(batch_size, num_heads, seq_len_k, d_k)
    value = torch.randn(batch_size, num_heads, seq_len_k, d_v)
    return query, key, value


@pytest.fixture
def init_multi_headed_attention() -> MultiHeadedAttention:
    heads_count = 8
    d_model = 64
    dropout_rate = 0.1
    return MultiHeadedAttention(heads_count=heads_count, d_model=d_model, dropout_rate=dropout_rate)


@pytest.fixture
def multi_headed_attention_sample_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = 2
    seq_len_q = 10
    seq_len_k = 12
    d_model = 64

    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_k, d_model)
    value = torch.randn(batch_size, seq_len_k, d_model)
    return query, key, value
