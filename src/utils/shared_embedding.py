import torch
import torch.nn as nn


class SharedEmbedding(nn.Module):
    """
    A wrapper for nn.Embedding, allowing the same nn.Embedding object to be
    used for transforming multiple inputs with different token IDs.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None):
        """
        Initializes SharedEmbedding.

        :param vocab_size: The size of the vocabulary (number of embeddings). Must be a positive integer.
        :param d_model: The dimensionality of each embedding. Must be a positive integer.
        :param padding_idx: The index of the token that should be padding. Must be a non-negative integer or None.

        :raises TypeError: If `vocab_size` is not an integer.
        :raises ValueError: If `vocab_size` is not positive.
        :raises TypeError: If `d_model` is not an integer.
        :raises ValueError: If `d_model` is not positive.
        :raises TypeError: If `padding_idx` is not an integer or None.
        :raises ValueError: If `padding_idx` is negative.
        """
        super().__init__()

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

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

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

        return self.embedding(x)
