from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.train import do_epoch


class TestDoEpoch:
    @pytest.fixture
    def mock_components(self):
        """Setup mock components with proper tensor handling"""
        # Test dimensions
        batch_size = 4
        seq_len = 10
        vocab_size = 1000

        # Create a model mock that handles tensor operations
        model = MagicMock()
        logits = torch.randn(batch_size, seq_len - 1, vocab_size, requires_grad=True)
        model.forward.return_value = logits

        # Make contiguous() and view() return proper tensors
        # model.return_value.contiguous.return_value = logits.contiguous()
        # model.return_value.contiguous.return_value.view.return_value = logits.view(-1, vocab_size)

        # Create a mock batch
        batch = MagicMock()
        batch.source = torch.randint(0, vocab_size, (seq_len, batch_size))
        batch.target = torch.randint(0, vocab_size, (seq_len, batch_size))

        # Configure data iterator
        data_iter = MagicMock()
        data_iter.__iter__.return_value = [batch]  # Yield one batch
        data_iter.__len__.return_value = 1  # One batch total

        # Other components
        criterion = nn.CrossEntropyLoss()
        optimizer = MagicMock()
        scheduler = MagicMock()

        return {
            "model": model,
            "criterion": criterion,
            "data_iter": data_iter,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "batch": batch,  # Expose the mock batch for verification
        }

    def test_batch_processing(self, mock_components):
        """Test complete batch processing pipeline"""
        with patch("src.utils.mask.convert_batch") as mock_convert:
            # Setup convert_batch mock
            source_mask = torch.ones(2, 1, 10, dtype=torch.bool)
            target_mask = torch.ones(2, 10, 10, dtype=torch.bool)
            mock_convert.return_value = (
                mock_components["batch"].source,
                mock_components["batch"].target,
                source_mask,
                target_mask,
            )

            # Run epoch
            loss = do_epoch(
                model=mock_components["model"],
                criterion=mock_components["criterion"],
                data_iter=mock_components["data_iter"],
                epoch_number=1,
                pad_idx=0,
                unk_idx=1,
                optimizer=None,
                use_wandb=False,
            )

            # Verify
            assert isinstance(loss, float)

    def test_training_mode(self, mock_components):
        """Test optimizer steps in training mode"""
        with patch("src.utils.mask.convert_batch"):
            do_epoch(
                model=mock_components["model"],
                criterion=mock_components["criterion"],
                data_iter=mock_components["data_iter"],
                epoch_number=1,
                pad_idx=0,
                unk_idx=1,
                optimizer=mock_components["optimizer"],
                scheduler=mock_components["scheduler"],
                use_wandb=False,
            )

            mock_components["optimizer"].zero_grad.assert_called_once()
            mock_components["optimizer"].step.assert_called_once()
            mock_components["scheduler"].step.assert_called_once()
