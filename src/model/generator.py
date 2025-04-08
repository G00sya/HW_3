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
