from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from navec import Navec

from src.utils.shared_embedding import SharedEmbedding, create_pretrained_embedding


class TestSharedEmbedding:
    def test_init(self):
        """
        Tests the SharedEmbedding class initialization without pretrained embedding.
        """
        vocab_size = 10
        d_model = 5
        padding_idx = 0

        # Create an instance of SharedEmbedding.
        shared_embedding = SharedEmbedding(vocab_size, d_model, padding_idx)
        assert (
            shared_embedding._SharedEmbedding__embedding.num_embeddings == vocab_size
        ), "Wrong parameters of nn.Embedding."
        assert (
            shared_embedding._SharedEmbedding__embedding.embedding_dim == d_model
        ), "Wrong parameters of nn.Embedding."

        # Create an input tensor.
        input_tensor = torch.tensor([1, 2, 3, 4, 5])

        # Transform the input tensor.
        output_tensor = shared_embedding(input_tensor)

        # Check the shape of the output tensor.
        assert output_tensor.shape == (5, d_model), "Incorrect shape of the output tensor."

        # Check that the padding tokens are zeros (almost zeros due to machine precision).
        padding_embedding = shared_embedding(torch.tensor([padding_idx]))
        assert torch.allclose(padding_embedding, torch.zeros(1, d_model)), "Padding embedding should be zero."

        # Check that the transformation is the same for the same token.
        token_index = 1
        token1_embedding = shared_embedding(torch.tensor([token_index]))
        token2_embedding = shared_embedding(torch.tensor([token_index]))
        assert torch.allclose(token1_embedding, token2_embedding), "Embeddings for the same token should be the same."

    def test_init_with_pretrained_embedding(self):
        """
        Tests the SharedEmbedding class initialization with pretrained embedding.
        """
        vocab_size = 10
        d_model = 5
        padding_idx = 0

        pretrained_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx
        )  # Example pre-trained embedding
        shared_embedding = SharedEmbedding(
            vocab_size=vocab_size, d_model=d_model, padding_idx=padding_idx, pretrained_embedding=pretrained_embedding
        )
        assert (
            shared_embedding._SharedEmbedding__embedding.num_embeddings == vocab_size
        ), "Wrong parameters of nn.Embedding."
        assert (
            shared_embedding._SharedEmbedding__embedding.embedding_dim == d_model
        ), "Wrong parameters of nn.Embedding."

        # Create an input tensor.
        input_tensor = torch.tensor([1, 2, 3, 4, 5])

        # Transform the input tensor.
        output_tensor = shared_embedding(input_tensor)

        # Check the shape of the output tensor.
        assert output_tensor.shape == (5, d_model), "Incorrect shape of the output tensor."

        # Check that the transformation is the same for the same token.
        token_index = 1
        token1_embedding = shared_embedding(torch.tensor([token_index]))
        token2_embedding = shared_embedding(torch.tensor([token_index]))
        assert torch.allclose(token1_embedding, token2_embedding), "Embeddings for the same token should be the same."

    def test_invalid_pretrained_embedding(self):
        pretrained_embedding = "embedding"
        with pytest.raises(TypeError):
            SharedEmbedding(vocab_size=0, d_model=0, padding_idx=0, pretrained_embedding=pretrained_embedding)

    def test_init_vocab_size_type_error(self):
        """
        Test that TypeError is raised when vocab_size is not an int.
        """
        with pytest.raises(TypeError):
            SharedEmbedding(vocab_size="10", d_model=5)

    def test_init_vocab_size_value_error(self):
        """
        Test that ValueError is raised when vocab_size is not positive.
        """
        with pytest.raises(ValueError):
            SharedEmbedding(vocab_size=-10, d_model=5)

    def test_init_d_model_type_error(self):
        """
        Test that TypeError is raised when d_model is not an int.
        """
        with pytest.raises(TypeError):
            SharedEmbedding(vocab_size=10, d_model="5")

    def test_init_d_model_value_error(self):
        """
        Test that ValueError is raised when d_model is not positive.
        """
        with pytest.raises(ValueError):
            SharedEmbedding(vocab_size=10, d_model=-5)

    def test_init_padding_idx_type_error(self):
        """
        Test that TypeError is raised when padding_idx is not an int or None.
        """
        with pytest.raises(TypeError):
            SharedEmbedding(vocab_size=10, d_model=5, padding_idx="0")

    def test_init_padding_idx_value_error(self):
        """
        Test that ValueError is raised when padding_idx is negative.
        """
        with pytest.raises(ValueError):
            SharedEmbedding(vocab_size=10, d_model=5, padding_idx=-1)

    def test_forward_x_type_error(self):
        """
        Test that TypeError is raised when x is not a torch.Tensor.
        """
        shared_embedding = SharedEmbedding(vocab_size=10, d_model=5)
        with pytest.raises(TypeError):
            shared_embedding.forward(x="not a tensor")

    def test_batched_input(self, init_shared_embedding_no_padding_idx):
        """
        Test SharedEmbedding with batched input.
        """
        shared_embedding, vocab_size, d_model = init_shared_embedding_no_padding_idx

        # Create a batched input tensor.
        batch_size = 2
        seq_len = 3
        input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Embed the batched input.
        output_tensor = shared_embedding(input_tensor)

        # Check the shape of the output tensor.
        assert output_tensor.shape == (batch_size, seq_len, d_model), "Incorrect shape for batched output."

    def test_no_padding(self, init_shared_embedding_no_padding_idx):
        """
        Test SharedEmbedding without padding index.
        """
        shared_embedding, vocab_size, d_model = init_shared_embedding_no_padding_idx

        input_tensor = torch.tensor([0, 1, 2])
        output_tensor = shared_embedding(input_tensor)

        assert output_tensor.shape == (3, d_model), "Incorrect shape without padding."

    def test_different_dtype(self, init_shared_embedding_no_padding_idx):
        """
        Test SharedEmbedding with different dtype of input tensor.
        """
        shared_embedding, vocab_size, d_model = init_shared_embedding_no_padding_idx

        input_tensor = torch.tensor([0, 1, 2], dtype=torch.int64)
        output_tensor = shared_embedding(input_tensor)

        assert output_tensor.shape == (3, d_model), "Incorrect shape with specified dtype."

    @patch("navec.Navec.load")  # Mock Navec.load within your_module
    def test_create_pretrained_embedding(self, mock_navec_load):
        """Tests create_pretrained_embedding function with mocked Navec."""

        # Mock Navec Object and its Attributes
        mock_navec = MagicMock(spec=Navec)

        # Mock navec.vocab
        mock_vocab = MagicMock()
        mock_vocab.word_ids = {}
        mock_navec.vocab = mock_vocab
        mock_navec.vocab.pad_id = 0

        # Mock navec.pq
        mock_pq = MagicMock()
        mock_pq.shape = (1000, 300)
        mock_pq.indexes = np.zeros(5)
        mock_pq.codes = np.zeros((5, 5, 5))
        mock_navec.pq = mock_pq

        # Mock navec.meta
        mock_navec.meta = MagicMock()
        mock_navec.meta.id = MagicMock()

        # Set the return value of Navec.load to the mocked object
        mock_navec_load.return_value = mock_navec

        # Call the function
        path = "dummy/path/to/embedding.vec"  # or any path
        shared_embedding, returned_navec = create_pretrained_embedding(path)

        # Check Navec.load was called with the correct path
        mock_navec_load.assert_called_once_with(path)

        # Check that SharedEmbedding was created with correct parameters
        assert isinstance(shared_embedding, SharedEmbedding)
        # Ensure correct return value
        assert returned_navec is mock_navec
