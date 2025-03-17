import pytest

from src.utils.shared_embedding import SharedEmbedding


@pytest.fixture()
def init_shared_embedding_no_padding_idx():
    vocab_size = 10
    d_model = 5
    shared_embedding = SharedEmbedding(vocab_size, d_model)
    return shared_embedding, vocab_size, d_model
