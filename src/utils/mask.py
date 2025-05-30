import torch
import torch.nn as nn


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Creates a triangular mask to prevent attention to future positions.

    This function generates a mask that is used during the decoding phase of
    sequence-to-sequence models to prevent the model from attending to
    future tokens in the target sequence.

    :param size: The size (length) of the target sequence.
    :return: A tensor of shape (1, size, size) with True in the valid positions and False otherwise.
    """
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    mask = torch.ones(size, size, dtype=torch.bool, device=DEVICE).triu_(diagonal=1)
    return mask.unsqueeze(0) == 0


def make_mask(
    source_inputs: torch.Tensor,
    target_inputs: torch.Tensor,
    pad_idx: int,
    unk_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates masks for source and target sequences.

    This function generates two masks:
    1.  `source_mask`:  Prevents attention to padding and unknown tokens in the source sequence.
    2.  `target_mask`: Prevents attention to padding and unknown tokens and future tokens in the target sequence.

    :param source_inputs: Source input sequence. Shape: (batch_size, source_seq_len).
    :param target_inputs: Target input sequence. Shape: (batch_size, target_seq_len).
    :param pad_idx: The index of the padding token in the vocabulary.
    :param unk_idx: The index of the unknown token in the vocabulary.
    :return: A tuple containing:
            - source_mask: A boolean tensor of shape (batch_size, 1, source_seq_len) indicating
            which source tokens are not padding or unknown.
            - target_mask: A boolean tensor of shape (batch_size, target_seq_len, target_seq_len)
            indicating which target tokens are valid (not padding or unknown and not future tokens).
    """
    source_mask = (source_inputs != pad_idx) & (source_inputs != unk_idx)
    source_mask = source_mask.unsqueeze(-2)

    target_mask = (target_inputs != pad_idx) & (target_inputs != unk_idx)
    target_mask = target_mask.unsqueeze(-2)

    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
    return source_mask, target_mask


def convert_batch(
    batch: nn.Module, pad_idx: int, unk_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Creates masks for source and target sequences.

    This function generates two masks:
    1.  `source_mask`:  Prevents attention to padding and unknown tokens in the source sequence.
    2.  `target_mask`: Prevents attention to padding and unknown tokens and future tokens in the target sequence.

    :param batch: Batch of data.
    :param pad_idx: The index of the padding token in the vocabulary.
    :param unk_idx: The index of the unknown token in the vocabulary.

    :return: A tuple containing:
        - source_mask: A boolean tensor of shape (batch_size, 1, source_seq_len) indicating
          which source tokens are not padding or unknown.
        - target_mask: A boolean tensor of shape (batch_size, target_seq_len, target_seq_len)
          indicating which target tokens are valid (not padding or unknown and not future tokens).
    """
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, unk_idx)

    return source_inputs, target_inputs, source_mask, target_mask
