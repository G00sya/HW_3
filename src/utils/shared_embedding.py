import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from navec import Navec

from src.data.prepare_data import Tokens


class SharedEmbedding(nn.Module):
    """
    A wrapper for nn.Embedding, allowing the same nn.Embedding object to be
    used for transforming multiple inputs with different token IDs.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int | None = None,
        pretrained_embedding: KeyedVectors | None = None,
    ):
        """
        Initializes SharedEmbedding.

        :param vocab_size: The size of the vocabulary (number of embeddings). Must be a positive integer.
        :param d_model: The dimensionality of each embedding. Must be a positive integer.
        :param padding_idx: The index of the token that should be padding. Must be a non-negative integer or None.
        :param pretrained_embedding: Pre-trained embedding to initialize the embedding layer with.
                                    Must be a KeyedVectors or None.
                                    If provided, vocab_size and d_model will be inferred from the shape of this tensor.
                                    If None, a new embedding layer will be initialized randomly.

        :raises TypeError: If `vocab_size` is not an integer.
        :raises ValueError: If `vocab_size` is not positive.
        :raises TypeError: If `d_model` is not an integer.
        :raises ValueError: If `d_model` is not positive.
        :raises TypeError: If `padding_idx` is not an integer or None.
        :raises ValueError: If `padding_idx` is negative.
        :raises TypeError: If `pretrained_embedding` is not a torch.Tensor or None.
        :raises ValueError: If `pretrained_embedding` is provided
                                and its shape is incompatible with vocab_size and d_model.
        """
        super().__init__()

        if pretrained_embedding is not None:
            if not isinstance(pretrained_embedding, KeyedVectors):
                raise TypeError(
                    f"pretrained_embedding must be a KeyedVectors or None, but got {type(pretrained_embedding)}"
                )

            self.__embedding = self._create_embedding_layer(pretrained_embedding, d_model, padding_idx, False)
        else:
            if not isinstance(vocab_size, int):
                raise TypeError(f"vocab_size must be an int, but got {type(vocab_size)}")
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be a positive number, but got {vocab_size}")

            if not isinstance(d_model, int):
                raise TypeError(f"d_model must be an int, but got {type(d_model)}")
            if d_model <= 0:
                raise ValueError(f"d_model must be a positive number, but got {d_model}")

            if padding_idx is not None and not isinstance(padding_idx, int):
                raise TypeError(f"padding_idx must be an int or None, but got {type(padding_idx)}")
            if padding_idx is not None and padding_idx < 0:
                raise ValueError(f"padding_idx must be a non-negative number or None, but got {padding_idx}")

            self.__embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def _create_embedding_layer(
        self, keyed_vectors: KeyedVectors, embedding_dim: int, padding_idx: int, freeze: bool
    ) -> nn.Embedding:
        """
        Convert KeyedVectors model to nn.Embedding.
        :param keyed_vectors: KeyedVectors model to convert.
        :param embedding_dim: d_model in KeyedVectors.
        :param padding_idx: Index of padding.
        :param freeze: If True it doesn't change weights during training, changes otherwise.
        :return: Embedding object.
        """
        embedding = nn.Embedding(
            num_embeddings=len(keyed_vectors),
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        # Load pretrained weights
        embedding.weight.data = torch.FloatTensor(keyed_vectors.vectors)
        # Freeze if needed
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input tensor using the internal nn.Embedding object.

        :param x: The input tensor containing token IDs. Shape: (...,).
        :raises TypeError: If `x` is not a torch.Tensor.
        :return: The output tensor containing embeddings for each token in the input tensor.
                 Shape: (..., embedding_dim).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")

        return self.__embedding(x)


def create_pretrained_embedding(path: str) -> (SharedEmbedding, KeyedVectors, int, int):
    """
    Create shared embedding instance with pretrained embedding which is red using path.

    :param path: Full path to pretrained embedding file.
    :return: Instance of SharedEmbedding and pretrained_embedding. pad_id and unk_id.
    """
    navec = Navec.load(path)
    vocab_size, d_model = map(int, navec.pq.shape)
    pad_idx = navec.vocab.pad_id
    unk_idx = navec.vocab.unk_id
    pretrained_embedding = _build_keyed_vectors(navec)

    shared_embedding = SharedEmbedding(
        vocab_size=vocab_size, d_model=d_model, padding_idx=pad_idx, pretrained_embedding=pretrained_embedding
    )

    return shared_embedding, pretrained_embedding, pad_idx, unk_idx


def _build_keyed_vectors(navec) -> KeyedVectors:
    """
    Builds a KeyedVectors object with pre-computed weights from Navec model.
    :return: KeyedVectors model.
    """
    vector_size = navec.pq.dim  # Assuming self.pq.dim is the embedding dimension

    # Get word list and weights
    word_list = navec.vocab.words
    weights = navec.pq.unpack()  # Assuming self.pq.unpack() returns a NumPy array of weights

    # Create a dictionary mapping words to their vectors
    word_vectors = {}
    for word, vector in zip(word_list, weights):
        word_vectors[word] = vector

    # Create KeyedVectors object
    model = KeyedVectors(vector_size)

    # Load the precomputed vectors into the KeyedVectors object.
    model.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))  # load the weights

    # Add special tokens
    np.random.seed(42)
    special_tokens = {
        Tokens.BOS.value: np.random.normal(scale=0.1, size=vector_size),
        Tokens.EOS.value: np.random.normal(scale=0.1, size=vector_size),
    }
    for token, vector in special_tokens.items():
        if token not in model:
            model.add_vector(token, vector)

    return model
