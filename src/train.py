import math
import os
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.data.prepare_data import Data
from src.model.encoder_decoder import EncoderDecoder
from src.utils.mask import convert_batch
from src.utils.noam_opt import NoamOpt
from src.utils.shared_embedding import create_pretrained_embedding

tqdm.get_lock().locks = []


def do_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_iter: Iterable[tuple[Tensor, Tensor, Tensor, Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    name: Optional[str] = None,
) -> float:
    """
    Performs a single training or validation epoch with progress tracking.

    :param model: Neural network model to train/evaluate. Must implement forward().
    :param criterion: Loss function (e.g., CrossEntropyLoss).
    :param data_iter: Iterator yielding batches of (source, target) pairs.
    :param optimizer: Optimizer for parameter updates. None for validation.
    :param name: Prefix for progress bar descriptions (e.g., "Train").

    :return: Average loss across all batches in the epoch.
    """
    epoch_loss = 0

    is_train = optimizer is not None
    name = name or ""
    model.train(is_train)

    batches_count = len(data_iter)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)
                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])

                logits = logits.contiguous().view(-1, logits.shape[-1])
                target = target_inputs[:, 1:].contiguous().view(-1)
                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description(
                    "{:>5s} Loss = {:.5f}, PPX = {:.2f}".format(name, loss.item(), math.exp(loss.item()))
                )

            progress_bar.set_description(
                "{:>5s} Loss = {:.5f}, PPX = {:.2f}".format(
                    name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count)
                )
            )
            progress_bar.refresh()

    return epoch_loss / batches_count


def fit(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_iter: Iterable[tuple[Tensor, Tensor, Tensor, Tensor]],
    epochs_count: int = 1,
    val_iter: Optional[Iterable[tuple[Tensor, Tensor, Tensor, Tensor]]] = None,
) -> tuple[list[float], float]:
    """
    Trains the model for specified number of epochs with optional validation.

    :param model: Neural network model to train.
    :param criterion: Loss function used for optimization.
    :param optimizer: Optimizer for parameter updates.
    :param train_iter: Training data iterator yielding batches of tensors
           (source, target, source_mask, target_mask).
    :param epochs_count: Number of complete passes through the training data. Default: 1.
    :param val_iter: Optional validation data iterator with same format as train_iter. Default: None.
    :return: Tuple containing (training_losses, best_validation_loss).
             training_losses: List of average training losses per epoch.
             best_validation_loss: Lowest validation loss encountered (inf if no validation).
    """
    best_val_loss = float("inf")
    train_losses = []  # Track training losses per epoch

    for epoch in range(epochs_count):
        name_prefix = f"[{epoch + 1} / {epochs_count}] "
        train_loss = do_epoch(model, criterion, train_iter, optimizer, name_prefix + "Train:")
        train_losses.append(train_loss)  # Store training loss

        if val_iter is not None:
            val_loss = do_epoch(model, criterion, val_iter, None, name_prefix + "  Val:")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    return train_losses, best_val_loss  # Return training and validation losses


if __name__ == "__main__":
    # Initialize SharedEmbedding with glove embedding
    shared_embedding = create_pretrained_embedding(path="./embeddings/glove.6B.300d.txt", padding_idx=0)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    data = Data()
    train_iter, test_iter = data.init_dataset(os.path.join("data", "raw", "news.csv"))
    model = EncoderDecoder(
        source_vocab_size=len(data.word_field.vocab), target_vocab_size=len(data.word_field.vocab)
    ).to(DEVICE)

    pad_idx = data.word_field.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(DEVICE)

    optimizer = NoamOpt(model.d_model)

    fit(model, criterion, optimizer, train_iter, epochs_count=30, val_iter=test_iter)
