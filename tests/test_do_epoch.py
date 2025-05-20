from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.train import do_epoch
from src.utils.noam_opt import NoamOpt


class MockDataIter:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


@pytest.fixture
def setup():
    # Setup a more complete mock structure
    model = MagicMock(spec=nn.Module)
    criterion = MagicMock(spec=nn.Module)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    scheduler = MagicMock(spec=NoamOpt)

    # Create proper batch structure with requires_grad for training
    batch = (
        torch.randn(2, 10, 512),  # source_inputs
        torch.randint(0, 100, (2, 12)),  # target_inputs
        torch.ones(2, 10),  # source_mask
        torch.ones(2, 12, 12),  # target_mask
    )

    # Create mock data iterator with proper length
    data_iter = MockDataIter([batch, batch])  # 2 identical batches

    # Mock convert_batch to return the batch as-is
    with patch("src.train.convert_batch") as mock_convert:
        mock_convert.side_effect = lambda x: x  # Just return the batch unchanged
        yield model, criterion, optimizer, 1, scheduler, data_iter


def test_training_mode(setup):
    model, criterion, optimizer, epoch_number, scheduler, data_iter = setup

    # Create a tensor that requires grad for training
    mock_output = torch.randn(2 * 11, 100, requires_grad=True)
    model.forward.return_value = mock_output
    criterion.return_value = torch.tensor(1.23, requires_grad=True)

    loss = do_epoch(model, criterion, data_iter, epoch_number, optimizer, scheduler, "Train", False)

    # Verify training mode behaviors
    model.train.assert_called_once_with(True)
    assert optimizer.zero_grad.call_count == 2
    assert optimizer.step.call_count == 2
    assert isinstance(loss, float)


def test_validation_mode(setup):
    model, criterion, _, epoch_number, scheduler, data_iter = setup
    model.forward.return_value = torch.randn(2 * 11, 100)
    criterion.return_value = torch.tensor(2.34)

    loss = do_epoch(model, criterion, data_iter, epoch_number, None, scheduler, "Val")

    model.train.assert_called_once_with(False)
    assert isinstance(loss, float)


def test_batch_processing(setup):
    model, criterion, _, epoch_number, scheduler, data_iter = setup
    model.forward.return_value = torch.randn(2 * 11, 100)
    criterion.return_value = torch.tensor(1.0)

    do_epoch(model, criterion, data_iter, epoch_number, None, scheduler, None)

    assert model.forward.call_count == 2


def test_empty_iterator():
    model = MagicMock(spec=nn.Module)
    criterion = MagicMock(spec=nn.Module)
    empty_iter = MockDataIter([])  # Empty iterator
    epoch_number = 1

    with pytest.raises(ZeroDivisionError):
        do_epoch(model, criterion, empty_iter, epoch_number, None)
