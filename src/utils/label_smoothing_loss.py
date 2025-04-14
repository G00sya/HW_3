from enum import StrEnum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReductionStrEnum(StrEnum):
    MEAN: auto()
    SUM: auto()
    NONE: auto()


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss with padding index support.

    This implements label smoothing similar to the cal_loss function provided,
    which smooths the target labels by mixing the true label distribution
    with a uniform distribution over all classes.

    Shape:
        - Input: (N, C) where C = number of classes
        - Target: (N,) where each value is 0 ≤ targets[i] ≤ C-1
        - Output: scalar if reduction is 'mean' or 'sum', otherwise (N,)
    """

    def __init__(
        self,
        pad_idx: int | None = None,
        smoothing: float | None = 0.1,
        reduction: ReductionStrEnum = ReductionStrEnum.MEAN,
    ):
        """
        Initializes the Label Smoothing Loss.

        :param pad_idx: Index to ignore when computing loss (for padding tokens). Default: None
        :param smoothing: Smoothing factor (between 0 and 1). Default: 0.1
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(LabelSmoothingLoss, self).__init__()

        if not isinstance(pad_idx, int) and pad_idx is not None:
            raise TypeError(f"pad_idx must be an int, but got {type(pad_idx)}.")
        if not isinstance(smoothing, float) and smoothing is not None:
            raise TypeError(f"smoothing must be a float, but got {type(smoothing)}.")
        if smoothing < 0 or smoothing > 1:
            raise ValueError("Smoothing must be between 0 and 1.")
        if not isinstance(reduction, ReductionStrEnum):
            raise TypeError(f"reduction must be ReductionStrEnum element, but got {type(reduction)}.")

        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        """
        Compute the label smoothing loss.

        :param pred: Predictions from model (before softmax).
        :param target: Ground truth labels.

        :return: Computed loss value.
        """
        target = target.contiguous().view(-1)
        n_class = pred.size(1)

        # Create one-hot vectors
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)

        # Apply label smoothing
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)

        # Compute log probabilities
        log_prb = F.log_softmax(pred, dim=1)

        # Handle padding
        non_pad_mask = target.ne(self.pad_idx) if self.pad_idx is not None else None

        # Compute loss
        loss = -(one_hot * log_prb).sum(dim=1)

        if non_pad_mask is not None:
            loss = loss.masked_select(non_pad_mask)

        # Apply reduction
        if self.reduction == ReductionStrEnum.SUM:
            return loss.sum()
        elif self.reduction == ReductionStrEnum.MEAN:
            return loss.mean() if non_pad_mask is None else loss.sum() / non_pad_mask.sum()
        elif self.reduction == ReductionStrEnum.NONE:
            return loss
