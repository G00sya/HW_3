from unittest.mock import MagicMock

import pytest
import torch.optim as optim
from torch import nn

from src.utils.noam_opt import NoamOpt


class TestNoamOpt:
    def test_initialization(self):
        """Test that NoamOpt initializes correctly."""
        optimizer = MagicMock(spec=optim.Adam)
        scheduler = NoamOpt(model_size=512, optimizer=optimizer, warmup=4000, factor=2)
        assert scheduler._NoamOpt__step == 0
        assert scheduler.warmup == 4000
        assert scheduler.factor == 2
        assert scheduler.model_size == 512

    @pytest.mark.parametrize(
        "param, invalid_value, expected_error",
        [
            ("model_size", -512, ValueError),
            ("model_size", 512.0, TypeError),
            ("factor", -1.0, ValueError),
            ("factor", "2", TypeError),
            ("warmup", 0, ValueError),
            ("warmup", 4000.0, TypeError),
            ("optimizer", nn.Linear(10, 10), TypeError),
        ],
    )
    def test_init_validation_errors(self, sample_optimizer, param, invalid_value, expected_error):
        """Test that invalid parameters raise correct ValueError types."""
        kwargs = {"model_size": 512, "factor": 2.0, "warmup": 4000, "optimizer": sample_optimizer, param: invalid_value}

        with pytest.raises(expected_error):
            NoamOpt(**kwargs)

    def test_rate_calculation(self):
        """Test the learning rate calculation at different steps."""
        optimizer = MagicMock(spec=optim.Adam)
        scheduler = NoamOpt(model_size=512, optimizer=optimizer, warmup=4000, factor=2)

        # During warmup (step < warmup)
        warmup_rate = scheduler.rate(step=1000)
        expected = 2 * (512**-0.5) * (1000 * (4000**-1.5))
        assert pytest.approx(warmup_rate) == expected

        # After warmup (step > warmup)
        post_warmup_rate = scheduler.rate(step=5000)
        expected = 2 * (512**-0.5) * (5000**-0.5)
        assert pytest.approx(post_warmup_rate) == expected

    def test_step_updates(self, sample_optimizer):
        """Test that step() updates the learning rate correctly."""
        scheduler = NoamOpt(model_size=512, optimizer=sample_optimizer)

        # Before any steps
        assert scheduler._NoamOpt__step == 0
        assert sample_optimizer.param_groups[0]["lr"] == 0

        # After first step
        scheduler.step()
        assert scheduler._NoamOpt__step == 1
        assert sample_optimizer.param_groups[0]["lr"] == scheduler.rate(1)

        # After second step
        scheduler.step()
        assert scheduler._NoamOpt__step == 2
        assert sample_optimizer.param_groups[0]["lr"] == scheduler.rate(2)

    def test_multiple_param_groups(self, simple_model_for_noam_opt):
        """Test behavior with multiple parameter groups."""
        # Create optimizer with different LR for two groups
        optimizer = optim.Adam(
            [{"params": simple_model_for_noam_opt.weight, "lr": 0}, {"params": simple_model_for_noam_opt.bias, "lr": 0}]
        )

        scheduler = NoamOpt(model_size=512, optimizer=optimizer)
        scheduler.step()

        # Both groups should get the same scheduled LR
        assert optimizer.param_groups[0]["lr"] == scheduler.rate(1)
        assert optimizer.param_groups[1]["lr"] == scheduler.rate(1)
        assert optimizer.param_groups[0]["lr"] == optimizer.param_groups[1]["lr"]
