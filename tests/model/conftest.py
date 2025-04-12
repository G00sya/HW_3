import pytest
import torch

from src.model.encoder import Encoder
from src.model.encoder_block import EncoderBlock
from src.model.layer_norm import LayerNorm
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positional_encoding import PositionalEncoding
from src.model.positionwise_feed_forward import PositionwiseFeedForward
from src.model.residual_block import ResidualBlock
from src.model.scaled_dot_product_attention import ScaledDotProductAttention
from src.utils.shared_embedding import SharedEmbedding


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
def init_positionwise_feed_forward() -> (PositionwiseFeedForward, torch.Tensor):
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    seq_len = 10

    ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
    inputs = torch.randn(batch_size, seq_len, d_model)
    return ffn, inputs


@pytest.fixture()
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
def init_positional_encoding() -> (PositionalEncoding, int, float, int):
    """
    Initializes the PositionalEncoding module with predefined parameters.
    :return: Tuple[PositionalEncoding, int, float, int]: A tuple containing an instance of PositionalEncoding,
    d_model, dropout, and max_len.
    """
    d_model = 512
    dropout = 0.1
    max_len = 100
    batch_size = 32
    seq_len = 50

    inputs = torch.randn(batch_size, seq_len, d_model)
    return PositionalEncoding(d_model, dropout, max_len), d_model, dropout, max_len, inputs


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


@pytest.fixture
def init_encoder() -> tuple[Encoder, SharedEmbedding, int]:
    d_model = 512
    shared_embedding = SharedEmbedding(vocab_size=1000, d_model=d_model)
    return (
        Encoder(
            d_model=d_model,
            d_ff=2048,
            blocks_count=6,
            heads_count=8,
            dropout_rate=0.1,
            shared_embedding=shared_embedding,
        ),
        shared_embedding,
        d_model,
    )


@pytest.fixture
def encoder_sample_tensors() -> tuple[int, int, torch.Tensor]:
    batch_size, seq_len = 2, 10
    return batch_size, seq_len, torch.randint(0, 1000, (batch_size, seq_len))
