import torch
import torch.nn as nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding


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
        pretrained_embedding: nn.Module | None = None,
    ):
        """
        Initializes SharedEmbedding.

        :param vocab_size: The size of the vocabulary (number of embeddings). Must be a positive integer.
        :param d_model: The dimensionality of each embedding. Must be a positive integer.
        :param padding_idx: The index of the token that should be padding. Must be a non-negative integer or None.
        :param pretrained_embedding: Pre-trained embedding to initialize the embedding layer with.
                                    Must be a nn.Module or None.
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
            if not isinstance(pretrained_embedding, nn.Module):
                raise TypeError(
                    f"pretrained_embedding must be a nn.Module or None, but got {type(pretrained_embedding)}"
                )
            self.__embedding = pretrained_embedding
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


def create_pretrained_embedding(path: str) -> (SharedEmbedding, int, int, int, Navec):
    """
    Create shared embedding instance with pretrained embedding which is red using path.

    :param path: Full path to pretrained embedding file.
    :return: Instance of SharedEmbedding and pretrained_embedding.
    """
    navec = Navec.load(path)
    vocab_size, d_model = map(int, navec.pq.shape)
    pad_idx = navec.vocab.pad_id
    pretrained_embedding = NavecEmbedding(navec)

    shared_embedding = SharedEmbedding(
        vocab_size=vocab_size, d_model=d_model, padding_idx=pad_idx, pretrained_embedding=pretrained_embedding
    )

    return shared_embedding, navec
