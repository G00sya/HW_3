import torch


class LambdaWrapper(torch.nn.Module):
    """
    A PyTorch Module wrapper for lambda functions to enable compatibility with ResidualBlock.
    """

    def __init__(self, lambda_func):
        """Initializes the LambdaWrapper with a callable function.

        :param lambda_func: Function to wrap. Must accept a tensor and return a tensor.
        """
        super().__init__()
        self.lambda_func = lambda_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the wrapped function on the input tensor.

        :param x: Input tensor of shape (*, d_model) where d_model is the feature dimension.

        :return: Transformed output tensor of same shape as input.

        Note:
            The input will typically be layer-normalized when used with ResidualBlock.
        """
        return self.lambda_func(x)
