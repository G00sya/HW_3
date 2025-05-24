import torch


def calculate_rouge() -> float:
    return 2


def estimate_current_state(
    loss: torch.Tensor,
    scheduler_rate: float,
) -> dict[str, torch.Tensor]:
    """
    Calculate metrics for current state of training.

    :param loss: Loss tensor for current state.
    :param scheduler_rate: Current value of scheduler rate.
    :return: Dictionary for wandb: metrics = {'test_acc': accuracy, 'train_loss': loss}.
    """
    rouge = calculate_rouge()

    metrics = {"rouge": rouge, "train_loss": loss, "scheduler_rate": scheduler_rate}
    return metrics
