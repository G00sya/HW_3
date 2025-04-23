from unittest.mock import MagicMock, patch

import pytest
import torch

from src.train import fit


@pytest.fixture
def setup():
    # Setup mock objects
    model = MagicMock(spec=torch.nn.Module)
    criterion = MagicMock(spec=torch.nn.Module)
    optimizer = MagicMock(spec=torch.optim.Optimizer)

    # Create mock data iterators
    train_batch = (
        torch.randn(2, 10, 512),  # source
        torch.randint(0, 100, (2, 12)),  # target
        torch.ones(2, 10),  # source_mask
        torch.ones(2, 12, 12),  # target_mask
    )
    val_batch = (torch.randn(2, 5, 512), torch.randint(0, 100, (2, 8)), torch.ones(2, 5), torch.ones(2, 8, 8))

    train_iter = [train_batch, train_batch]  # 2 training batches
    val_iter = [val_batch, val_batch]  # 2 validation batches

    return model, criterion, optimizer, train_iter, val_iter


def test_fit_training_only(setup):
    model, criterion, optimizer, train_iter, _ = setup

    with patch("src.train.do_epoch") as mock_do_epoch:
        mock_do_epoch.return_value = 0.5
        train_losses, best_val_loss = fit(model, criterion, optimizer, train_iter, epochs_count=3)
        assert len(train_losses) == 3
        assert best_val_loss == float("inf")


def test_fit_with_validation(setup):
    model, criterion, optimizer, train_iter, val_iter = setup

    with patch("src.train.do_epoch") as mock_do_epoch:
        mock_do_epoch.side_effect = [0.5, 0.6, 0.4, 0.5, 0.3, 0.4]
        train_losses, best_val_loss = fit(model, criterion, optimizer, train_iter, epochs_count=3, val_iter=val_iter)
        assert train_losses == [0.5, 0.4, 0.3]
        assert best_val_loss == 0.4


def test_fit_multiple_epochs(setup):
    model, criterion, optimizer, train_iter, val_iter = setup

    with patch("src.train.do_epoch") as mock_do_epoch:
        mock_do_epoch.side_effect = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

        train_losses, best_val_loss = fit(model, criterion, optimizer, train_iter, epochs_count=3, val_iter=val_iter)

        # Verify best validation loss tracking
        assert best_val_loss == 0.05
        assert train_losses == [0.5, 0.3, 0.1]


def test_fit_no_validation_best_loss(setup):
    model, criterion, optimizer, train_iter, _ = setup

    with patch("src.train.do_epoch") as mock_do_epoch:
        mock_do_epoch.return_value = 0.5

        train_losses, best_val_loss = fit(model, criterion, optimizer, train_iter, epochs_count=2)

        # Should return inf when no validation
        assert best_val_loss == float("inf")
        assert train_losses == [0.5, 0.5]


def test_fit_empty_validation(setup):
    model, criterion, optimizer, train_iter, _ = setup

    with patch("src.train.do_epoch") as mock_do_epoch:
        mock_do_epoch.return_value = 0.5
        train_losses, best_val_loss = fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None)
        assert best_val_loss == float("inf")
        assert len(train_losses) == 1
