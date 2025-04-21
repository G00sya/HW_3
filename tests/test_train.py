import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.train import do_epoch


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
    optimizer.optimizer = MagicMock()

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
        yield model, criterion, optimizer, data_iter


def test_training_mode(setup):
    model, criterion, optimizer, data_iter = setup

    # Create a tensor that requires grad for training
    mock_output = torch.randn(2 * 11, 100, requires_grad=True)
    model.forward.return_value = mock_output
    criterion.return_value = torch.tensor(1.23, requires_grad=True)

    loss = do_epoch(model, criterion, data_iter, optimizer, "Train")

    # Verify training mode behaviors
    model.train.assert_called_once_with(True)
    assert optimizer.optimizer.zero_grad.call_count == 2
    assert optimizer.step.call_count == 2
    assert criterion.call_count == 2
    assert isinstance(loss, float)
    assert math.isclose(loss, 1.23, rel_tol=1e-3)


def test_validation_mode(setup):
    model, criterion, _, data_iter = setup
    model.forward.return_value = torch.randn(2 * 11, 100)
    criterion.return_value = torch.tensor(2.34)

    loss = do_epoch(model, criterion, data_iter, None, "Val")

    model.train.assert_called_once_with(False)
    assert isinstance(loss, float)
    assert math.isclose(loss, 2.34, rel_tol=1e-3)


def test_batch_processing(setup):
    model, criterion, _, data_iter = setup
    model.forward.return_value = torch.randn(2 * 11, 100)
    criterion.return_value = torch.tensor(1.0)

    loss = do_epoch(model, criterion, data_iter, None)

    assert model.forward.call_count == 2
    assert criterion.call_count == 2
    assert math.isclose(loss, 1.0, rel_tol=1e-3)


# def test_progress_bar(setup):
#     model, criterion, _, data_iter = setup
#     model.forward.return_value = torch.randn(2 * 11, 100)
#     criterion.return_value = torch.tensor(1.5)
#
#     # Create a mock progress bar instance
#     mock_pbar = MagicMock()
#     mock_pbar.update = Mock()
#     mock_pbar.set_description = Mock()
#
#     # Patch tqdm to return our mock progress bar
#     with patch('tqdm.tqdm', return_value=mock_pbar) as mock_tqdm:
#         do_epoch(model, criterion, data_iter, None, "Test")
#
#         # Verify tqdm was called with correct total
#         mock_tqdm.assert_called_once_with(total=len(data_iter))
#
#         # Verify progress updates
#         assert mock_pbar.update.call_count == 2
#         assert mock_pbar.set_description.call_count >= 2


def test_loss_calculation(setup):
    model, criterion, _, data_iter = setup
    model.forward.return_value = torch.randn(2 * 11, 100)
    criterion.side_effect = [torch.tensor(1.0), torch.tensor(2.0)]

    loss = do_epoch(model, criterion, data_iter, None)

    assert math.isclose(loss, 1.5, rel_tol=1e-3)


# def test_teacher_forcing(setup):
#     model, criterion, _, data_iter = setup
#     batch = data_iter.batches[0]
#
#     # Configure mock to return real tensor
#     model.forward.return_value = torch.randn(2 * 11, 100)
#
#     # Create a dummy progress bar that won't cause formatting issues
#     mock_pbar = MagicMock()
#     mock_pbar.update = Mock()
#     mock_pbar.set_description = Mock(return_value=None)  # Returns None instead of mock
#
#     # Patch both possible tqdm import paths
#     try:
#         with patch('src.train.tqdm', return_value=mock_pbar), \
#                 patch('src.train.math.exp', return_value=1.0):
#             do_epoch(model, criterion, data_iter, None)
#     except AttributeError:
#         with patch('src.train.tqdm.tqdm', return_value=mock_pbar), \
#                 patch('src.train.math.exp', return_value=1.0):
#             do_epoch(model, criterion, data_iter, None)
#
#     # Verify teacher forcing implementation
#     called_target = model.forward.call_args[0][1]
#     expected_target = batch[1][:, :-1]
#     assert torch.equal(called_target, expected_target)


def test_empty_iterator():
    model = MagicMock(spec=nn.Module)
    criterion = MagicMock(spec=nn.Module)
    empty_iter = MockDataIter([])  # Empty iterator

    with pytest.raises(ZeroDivisionError):
        do_epoch(model, criterion, empty_iter, None)
