import math

import torch
from tqdm.auto import tqdm

from src.utils.mask import convert_batch
from src.utils.shared_embedding import create_pretrained_embedding

tqdm.get_lock().locks = []

if __name__ == "__main__":
    # Initialize SharedEmbedding with glove embedding
    shared_embedding = create_pretrained_embedding(path="./embeddings/glove.6B.300d.txt", padding_idx=0)

    def do_epoch(model, criterion, data_iter, optimizer=None, name=None):
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

    def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None):
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
