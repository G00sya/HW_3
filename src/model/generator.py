import torch
import torch.nn as nn
import torch.nn.functional as func


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step. Last module in architecture.
    """

    def __init__(self, d_model: int, target_vocab_size: int):
        """
        :param d_model: Dimensionality of embedding.
        :param target_vocab_size: Size of vocabulary of resulted text.
        """
        if not isinstance(d_model, int):
            raise TypeError(f"Expected d_model to be of type int, but got {type(d_model).__name__}")
        if not isinstance(target_vocab_size, int):
            raise TypeError(f"Expected target_vocab_size to be of type int, but got {type(target_vocab_size).__name__}")

        super(Generator, self).__init__()
        self.__proj = nn.Linear(d_model, target_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        """
        Apply Linear and Softmax layer to output of a decoder.

        :param x: Input in Generator. Size: (d_model).
        :return: Vector of probabilities for each word in vocabulary. Size: (target_vocab_size).
        """
        proj = self.__proj(x)
        return func.log_softmax(proj, dim=-1)
