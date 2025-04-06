import pytest
import torch

from src.model.encoder_block import EncoderBlock
from src.model.layer_norm import LayerNorm
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.residual_block import ResidualBlock
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


@pytest.fixture()
def init_residual_block() -> (ResidualBlock, int, float):
    """
    Init ResidualBlock.
    """
    d_model = 2
    dropout_rate = 0.2
    residual_block = ResidualBlock(size=d_model, dropout_rate=dropout_rate)
    return residual_block, d_model, dropout_rate


@pytest.fixture
def init_scaled_dot_product_attention() -> tuple[ScaledDotProductAttention, float]:
    dropout_rate = 0.1
    return ScaledDotProductAttention(dropout_rate=dropout_rate), dropout_rate


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
def init_multi_headed_attention() -> tuple[MultiHeadedAttention, int]:
    heads_count = 8
    d_model = 64
    dropout_rate = 0.1
    return MultiHeadedAttention(heads_count=heads_count, d_model=d_model, dropout_rate=dropout_rate), heads_count


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


@pytest.fixture
def init_encoder_block() -> EncoderBlock:
    size = 64
    heads_count = 8
    dropout_rate = 0.1

    self_attn = MultiHeadedAttention(heads_count=heads_count, d_model=size, dropout_rate=dropout_rate)
    feed_forward = torch.nn.Linear(size, size)
    return EncoderBlock(size=size, self_attn=self_attn, feed_forward=feed_forward, dropout_rate=dropout_rate)


@pytest.fixture
def encoder_block_sample_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = 2
    seq_len = 10
    size = 64

    inputs = torch.randn(batch_size, seq_len, size)
    mask = torch.ones(batch_size, seq_len, seq_len)
    return inputs, mask
