from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from gensim.models import KeyedVectors

from src.utils.shared_embedding import SharedEmbedding, _build_keyed_vectors, create_pretrained_embedding


class TestSharedEmbedding:
    @pytest.fixture
    def mock_kv(self):
        """Mock KeyedVectors with 10 words, dim=300"""
        kv = KeyedVectors(vector_size=300)
        kv.add_vectors([f"word{i}" for i in range(10)], np.random.rand(10, 300))
        return kv

    def test_init_with_pretrained(self, mock_kv):
        """Test initialization with pretrained embeddings"""
        emb = SharedEmbedding(vocab_size=10, d_model=300, pretrained_embedding=mock_kv)
        assert emb._SharedEmbedding__embedding.weight.shape == (10, 300)

    def test_init_with_random(self):
        """Test random initialization"""
        emb = SharedEmbedding(vocab_size=100, d_model=200)
        assert emb._SharedEmbedding__embedding.num_embeddings == 100
        assert emb._SharedEmbedding__embedding.embedding_dim == 200

    def test_forward_pass(self, mock_kv):
        """Test forward pass with valid input"""
        emb = SharedEmbedding(vocab_size=10, d_model=300, pretrained_embedding=mock_kv)
        x = torch.LongTensor([1, 2, 3])
        out = emb(x)
        assert out.shape == (3, 300)

    @pytest.mark.parametrize("bad_input", ["string", 123, None])
    def test_forward_bad_input(self, bad_input):
        """Test forward with invalid input types"""
        emb = SharedEmbedding(vocab_size=100, d_model=200)
        with pytest.raises(TypeError):
            emb(bad_input)

    @patch("navec.Navec.load")
    def test_create_pretrained(self, mock_load):
        """Test pretrained embedding creation"""
        mock_navec = MagicMock()
        mock_navec.pq.shape = (1000, 300)
        mock_navec.vocab.pad_id = 0
        mock_navec.vocab.unk_id = 1
        mock_navec.pq.dim = 300
        mock_navec.vocab.words = [f"word{i}" for i in range(1000)]
        mock_navec.pq.unpack.return_value = np.random.rand(1000, 300)
        mock_load.return_value = mock_navec

        emb, kv, pad_idx, unk_idx = create_pretrained_embedding("dummy_path")
        assert isinstance(emb, SharedEmbedding)
        assert pad_idx == 0
        assert unk_idx == 1

    def test_build_keyed_vectors(self, mock_kv):
        """Test KeyedVectors construction"""
        mock_navec = MagicMock()
        mock_navec.pq.dim = 300
        mock_navec.vocab.words = [f"word{i}" for i in range(10)]
        mock_navec.pq.unpack.return_value = np.random.rand(10, 300)

        kv = _build_keyed_vectors(mock_navec)
        assert kv.vector_size == 300
        assert len(kv) == 12  # 10 words + 2 special tokens

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
