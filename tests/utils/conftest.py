import pytest
import torch

from src.utils.shared_embedding import SharedEmbedding


@pytest.fixture()
def init_shared_embedding_no_padding_idx():
    vocab_size = 10
    d_model = 5
    shared_embedding = SharedEmbedding(vocab_size, d_model)
    return shared_embedding, vocab_size, d_model


@pytest.fixture
def test_glove_file(tmp_path):  # Added tmp_path argument
    """
    Fixture to create a dummy GloVe file for testing and clean it up afterwards using pytest's tmp_path.
    """
    test_embedding_path = tmp_path / "test_glove.txt"  # Create a Path object
    test_vocab_size = 10
    test_d_model = 5

    # create some random embeddings:
    test_embedding_matrix = torch.randn(test_vocab_size, test_d_model)

    with open(str(test_embedding_path), "w") as f:  # Convert path to string
        for i in range(test_vocab_size):
            embedding_str = " ".join(str(x) for x in test_embedding_matrix[i].tolist())
            f.write(f"word{i} {embedding_str}\n")

    return str(test_embedding_path), test_vocab_size, test_d_model  # Return the string representation of the path


@pytest.fixture
def label_smoothing_loss_sample_data():
    pred = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([2, 1], dtype=torch.long)
    return pred, target
