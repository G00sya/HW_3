import torch.optim as optim


class NoamOpt:
    """
    Noam learning rate scheduler.
    Implements the learning rate schedule from the paper: "Attention Is All You Need" (Vaswani et al., 2017).

    The learning rate follows a warmup schedule where it increases linearly
    for the first `warmup` steps, then decreases proportionally to the inverse
    square root of the step number.
    """

    def __init__(self, model_size: int, optimizer: optim.Optimizer, factor: float = 2, warmup: int = 4000):
        """
        Initialize the Noam learning rate scheduler.

        :param model_size: The dimensionality of the model's embeddings (d_model).
        :param optimizer: The optimizer to apply scheduling to.
        :param factor: Scaling factor for the learning rate.
        :param warmup: Number of warmup steps where LR increases.
        """
        if not isinstance(model_size, int):
            raise TypeError(f"model_size must be an integer, but got {type(model_size)}.")
        if model_size <= 0:
            raise ValueError(f"model_size must be a positive integer, but got {model_size}.")
        if not isinstance(factor, (float, int)):
            raise TypeError(f"factor must be an integer or float, but got {type(factor)}.")
        if factor <= 0:
            raise ValueError(f"factor must be a positive integer or float, but got {factor}.")
        if not isinstance(warmup, int):
            raise TypeError(f"warmup must be an integer, but got {type(warmup)}.")
        if warmup <= 0:
            raise ValueError(f"warmup must be a positive integer, but got {warmup}.")
        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError(f"optimizer must be torch.optim.Optimizer object, but got {type(optimizer)}.")

        self.optimizer = optimizer
        self.__step = 0  # Internal step counter
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def step(self) -> None:
        """
        Update the learning rate and step the optimizer.
        Should be called after each parameter update.
        """
        self.__step += 1
        rate = self.rate()

        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate

    def rate(self, step: int | None = None) -> float:
        """
        Compute the learning rate for a given step.

        :param step: If provided, compute rate for this step. Otherwise, uses internal counter.
        :return: The computed learning rate.
        Formula:
            rate = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        """
        if step is None:
            step = self.__step

        # Calculate components of the Noam schedule
        warmup_factor = step * (self.warmup**-1.5)
        decay_factor = step**-0.5
        scale_factor = self.model_size**-0.5

        return self.factor * scale_factor * min(decay_factor, warmup_factor)
