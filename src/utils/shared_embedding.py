import torch
import torch.nn as nn


class SharedEmbedding(nn.Module):
    """
    A wrapper for nn.Embedding, allowing the same nn.Embedding object to be
    used for transforming multiple inputs with different token IDs.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None):
        """
        :param vocab_size: The size of the vocabulary (number of embeddings).
        :param d_model: The dimensionality of each embedding.
        :param padding_idx: The index of the token that should be padding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input tensor using the internal nn.Embedding object.

        :param x: The input tensor containing token IDs. Shape: (...,).
        :return: The output tensor containing embeddings for each token in the input tensor.
                 Shape: (..., embedding_dim).
        """
        return self.embedding(x)
