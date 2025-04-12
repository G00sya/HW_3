import pytest
import torch.optim as optim
from torch import nn

from src.utils.shared_embedding import SharedEmbedding


@pytest.fixture()
def init_shared_embedding_no_padding_idx():
    vocab_size = 10
    d_model = 5
    shared_embedding = SharedEmbedding(vocab_size, d_model)
    return shared_embedding, vocab_size, d_model


@pytest.fixture
def simple_model_for_noam_opt():
    """Fixture providing a simple model with one parameter."""
    return nn.Linear(10, 10)


@pytest.fixture
def sample_optimizer(simple_model_for_noam_opt):
    """Fixture providing an Adam optimizer with zero initial LR."""
    return optim.Adam(simple_model_for_noam_opt.parameters(), lr=0)
