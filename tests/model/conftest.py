import pytest

from src.model.residual_block import ResidualBlock


@pytest.fixture()
def init_residual_block() -> (ResidualBlock, int, float):
    """
    Init ResidualBlock.
    """
    d_model = 2
    dropout_rate = 0.2
    residual_block = ResidualBlock(size=d_model, dropout_rate=dropout_rate)
    return residual_block, d_model, dropout_rate
