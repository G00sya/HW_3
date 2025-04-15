import pytest
import torch

from src.utils.label_smoothing_loss import LabelSmoothingLoss, ReductionStrEnum


class TestLabelSmoothingLossErrors:
    def test_init(self):
        """Test that LabelSmoothingLoss initializes correctly."""
        loss_fn = LabelSmoothingLoss(pad_idx=0, smoothing=0.2, reduction=ReductionStrEnum.SUM)
        assert loss_fn.pad_idx == 0
        assert loss_fn.smoothing == 0.2
        assert loss_fn.reduction == ReductionStrEnum.SUM

    @pytest.mark.parametrize(
        "param, invalid_value, expected_error",
        [
            ("pad_idx", 1.0, TypeError),
            ("smoothing", -1.0, ValueError),
            ("smoothing", "1.0", TypeError),
            ("reduction", "mean", TypeError),
        ],
    )
    def test_init_validation(self, param, invalid_value, expected_error):
        """Test that invalid parameters raise the correct errors."""
        kwargs = {
            "pad_idx": 1,
            "smoothing": 0.1,
            "reduction": ReductionStrEnum.MEAN,
            param: invalid_value,
        }
        with pytest.raises(expected_error):
            LabelSmoothingLoss(**kwargs)

    def test_forward_with_padding(self, label_smoothing_loss_sample_data):
        """Test forward pass with padding index."""
        pred, target = label_smoothing_loss_sample_data
        loss_fn = LabelSmoothingLoss(pad_idx=0)
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "Loss should be positive when ignoring padding"

    def test_forward_smoothing_effect_reduction_mean(self, label_smoothing_loss_sample_data):
        """Test that smoothing produces different loss values."""
        pred, target = label_smoothing_loss_sample_data

        loss_fn_no_smooth = LabelSmoothingLoss(smoothing=0.0, reduction=ReductionStrEnum.MEAN)
        loss_no_smooth = loss_fn_no_smooth(pred, target)

        loss_fn_smooth = LabelSmoothingLoss(smoothing=0.1, reduction=ReductionStrEnum.MEAN)
        loss_smooth = loss_fn_smooth(pred, target)

        assert not torch.isclose(loss_smooth, loss_no_smooth), "Smoothing should affect the loss value."

        assert loss_smooth > 0, "Loss should be positive."
        assert not torch.isnan(loss_smooth), "Loss should not be NaN."

    def test_forward_reduction_sum(self, label_smoothing_loss_sample_data):
        """Test forward pass with sum reduction."""
        pred, target = label_smoothing_loss_sample_data
        loss_fn = LabelSmoothingLoss(reduction=ReductionStrEnum.SUM)
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "Sum reduction should return positive scalar loss value/"

    def test_forward_reduction_none(self, label_smoothing_loss_sample_data):
        """Test forward pass with no reduction."""
        pred, target = label_smoothing_loss_sample_data
        loss_fn = LabelSmoothingLoss(reduction=ReductionStrEnum.NONE)
        loss = loss_fn(pred, target)
        assert loss.shape == (pred.shape[0],), "No reduction should return loss per sample (batch_size,)."
