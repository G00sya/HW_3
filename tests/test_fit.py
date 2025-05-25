from unittest.mock import MagicMock, patch

import pytest

from src.train import fit


class TestFit:
    @pytest.fixture
    def mock_components(self):
        """Setup mock components for testing"""
        model = MagicMock()
        model.save_model = MagicMock()

        criterion = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()

        train_iter = MagicMock()
        val_iter = MagicMock()

        return {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_iter": train_iter,
            "val_iter": val_iter,
        }

    @patch("src.train.do_epoch")
    def test_fit_training_only(self, mock_do_epoch, mock_components):
        """Test training without validation"""
        mock_do_epoch.return_value = 0.5  # Mock loss value

        train_losses, best_val_loss = fit(
            model=mock_components["model"],
            criterion=mock_components["criterion"],
            optimizer=mock_components["optimizer"],
            scheduler=mock_components["scheduler"],
            train_iter=mock_components["train_iter"],
            pad_idx=0,
            unk_idx=1,
            epochs_count=2,
            val_iter=None,
        )

        assert len(train_losses) == 2
        assert best_val_loss == float("inf")
        assert mock_do_epoch.call_count == 2
        mock_components["model"].save_model.assert_not_called()

    @patch("src.train.do_epoch")
    def test_fit_with_validation(self, mock_do_epoch, mock_components):
        """Test training with validation"""
        # Alternate between train and val losses
        mock_do_epoch.side_effect = [0.5, 0.4, 0.3, 0.35]  # train, val, train, val

        train_losses, best_val_loss = fit(
            model=mock_components["model"],
            criterion=mock_components["criterion"],
            optimizer=mock_components["optimizer"],
            scheduler=mock_components["scheduler"],
            train_iter=mock_components["train_iter"],
            pad_idx=0,
            unk_idx=1,
            epochs_count=2,
            val_iter=mock_components["val_iter"],
        )

        assert len(train_losses) == 2
        assert best_val_loss == 0.35
        assert mock_do_epoch.call_count == 4
        mock_components["model"].save_model.assert_not_called()

    @patch("src.train.do_epoch")
    def test_best_val_loss_tracking(self, mock_do_epoch, mock_components):
        """Test validation loss tracking"""
        mock_do_epoch.side_effect = [0.5, 0.4, 0.3, 0.35, 0.2, 0.1]  # Alternating train/val

        _, best_val_loss = fit(
            model=mock_components["model"],
            criterion=mock_components["criterion"],
            optimizer=mock_components["optimizer"],
            scheduler=mock_components["scheduler"],
            train_iter=mock_components["train_iter"],
            pad_idx=0,
            unk_idx=1,
            epochs_count=3,
            val_iter=mock_components["val_iter"],
        )

        assert best_val_loss == 0.1
