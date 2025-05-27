import torch
from torchtext.vocab import Vocab

from src.data.prepare_data import Tokens
from src.utils.rouge_metric import RussianRouge


def estimate_current_state(
    logits: torch.Tensor,
    target_inputs: torch.Tensor,
    vocab: Vocab,
    loss: torch.Tensor,
    scheduler_rate: float,
) -> dict[str, torch.Tensor]:
    """
    Calculate metrics for current state of training.
    :param logits: Output of model with shape (batch_size, sequence_length-1, vocabulary_size).
    :param target_inputs: Ground truth in numericalized format, shape: (batch_size, sequence_length-1).
    :param vocab: Vocabulary of the model.
    :param loss: Loss tensor for current state.
    :param scheduler_rate: Current value of scheduler rate.
    :return: Dictionary for wandb: metrics = {'test_acc': accuracy, 'train_loss': loss}.
    """
    # Convert logits to predicted tokens
    predicted_indices = torch.argmax(logits, dim=-1)  # Shape: (16, 11)

    min_word_count = target_inputs.shape[-1]
    # Convert target inputs to strings (skip first token if it's BOS)
    target_strings = []
    for seq in target_inputs:
        tokens = [vocab.itos[token] for token in seq if vocab.itos[token] not in [Tokens.PAD.value, Tokens.EOS.value]]
        target_strings.append(" ".join(tokens))
        min_word_count = min(min_word_count, len(tokens))

    # Convert Predictions to Strings
    predicted_strings = []
    for seq in predicted_indices:
        tokens = [vocab.itos[token] for token in seq if vocab.itos[token] not in [Tokens.PAD.value, Tokens.EOS.value]]
        predicted_strings.append(" ".join(tokens))
        min_word_count = min(min_word_count, len(tokens))

    # Calculate ROUGE Scores
    rouge = RussianRouge(ngram_sizes=(1, 2, min_word_count), use_lemmatization=True)
    total_score = 0
    for pred, target in zip(predicted_strings, target_strings):
        scores = rouge.compute_score(pred, target)
        rouge_l = scores[f"rouge-{min_word_count}"]["f1"]
        rouge_1 = scores["rouge-1"]["f1"]
        rouge_2 = scores["rouge-2"]["f1"]
        total_score += 0.5 * rouge_l + 0.3 * rouge_2 + 0.2 * rouge_1
    mean_rouge = total_score / len(target_strings)

    metrics = {"rouge": mean_rouge, "train_loss": loss, "scheduler_rate": scheduler_rate}
    return metrics
