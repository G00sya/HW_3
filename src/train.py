import math
import os

import torch
import torch.optim as optim
from torchtext.data import BucketIterator
from tqdm.auto import tqdm

from src.data.prepare_data import Data
from src.model.encoder_decoder import EncoderDecoder
from src.utils.device import setup_device
from src.utils.label_smoothing_loss import LabelSmoothingLoss
from src.utils.mask import convert_batch
from src.utils.noam_opt import NoamOpt
from src.utils.shared_embedding import create_pretrained_embedding

tqdm.get_lock().locks = []


def do_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_iter: BucketIterator,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: NoamOpt | None = None,
    name: str | None = None,
) -> float:
    """
    Performs a single training or validation epoch with progress tracking.

    :param model: Neural network model to train/evaluate. Must implement forward().
    :param criterion: Loss function (e.g., CrossEntropyLoss).
    :param data_iter: Iterator yielding batches of (source, target) pairs.
    :param optimizer: Optimizer for parameter updates. None for validation.
    :param scheduler: Scheduler for learning rate managing. None for validation.
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
                    optimizer.zero_grad()
                    loss.backward()
                    scheduler.step()
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
    scheduler: NoamOpt,
    train_iter: BucketIterator,
    epochs_count: int = 1,
    val_iter: BucketIterator | None = None,
) -> tuple[list[float], float]:
    """
    Trains the model for specified number of epochs with optional validation.

    :param model: Neural network model to train.
    :param criterion: Loss function used for optimization.
    :param optimizer: Optimizer for parameter updates.
    :param scheduler: Scheduler for learning rate managing.
    :param train_iter: Training data BucketIterator yielding batches of tensors
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
        train_loss = do_epoch(model, criterion, train_iter, optimizer, scheduler, name_prefix + "Train:")
        train_losses.append(train_loss)  # Store training loss

        if val_iter is not None:
            val_loss = do_epoch(model, criterion, val_iter, None, None, name_prefix + "  Val:")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    return train_losses, best_val_loss  # Return training and validation losses


if __name__ == "__main__":
    DEVICE = setup_device()
    # Initialize SharedEmbedding with glove embedding
    shared_embedding, vocab_size, d_model = create_pretrained_embedding(
        path="../embeddings/glove.6B.300d.txt", padding_idx=0
    )
    data = Data()
    train_iter, test_iter = data.init_dataset(os.path.join("..", "data", "raw", "news.csv"))
    model = EncoderDecoder(
        target_vocab_size=vocab_size,
        shared_embedding=shared_embedding,
        d_model=d_model,
        heads_count=10,  # Because we use d_model=300, and it must be divisible by heads_count
    ).to(DEVICE)

    pad_idx = data.word_field.vocab.stoi["<pad>"]
    criterion = LabelSmoothingLoss(pad_idx=pad_idx).to(DEVICE)

    optimizer = optim.Adam(model.parameters())
    scheduler = NoamOpt(model.d_model, optimizer)

    fit(model, criterion, optimizer, scheduler, train_iter, epochs_count=30, val_iter=test_iter)
