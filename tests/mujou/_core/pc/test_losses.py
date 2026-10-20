import pytest
import torch
from torch import nn

from mujou._core.pc.losses import cross_entropy, mse


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


class TestMse:
    def test_half_squared_error_of_each_sample(self):
        g = _generator()
        y_hat = torch.randn(3, 2, 4, generator=g, dtype=torch.float64)
        y = torch.randn(3, 2, 4, generator=g, dtype=torch.float64)
        ell = mse(y_hat, y)
        assert ell.shape == (3,)
        for b in range(3):
            expected = 0.5 * sum(
                (y[b, i, j] - y_hat[b, i, j]).item() ** 2
                for i in range(2)
                for j in range(4)
            )
            assert ell[b].item() == pytest.approx(expected, rel=1e-14)

    def test_one_value_per_sample(self):
        y_hat = torch.tensor([1.0, 2.0], dtype=torch.float64)
        y = torch.tensor([0.0, 0.0], dtype=torch.float64)
        assert torch.equal(mse(y_hat, y), torch.tensor([0.5, 2.0], dtype=torch.float64))

    def test_shapes_must_match(self):
        with pytest.raises(ValueError, match="same shape"):
            mse(torch.zeros(3, 1), torch.zeros(3))


class TestCrossEntropy:
    def test_mean_is_the_mean_loss_of_torch(self):
        # The per-sample loss has no 1/B in it.
        g = _generator()
        y_hat = torch.randn(5, 4, generator=g, dtype=torch.float64)
        y = torch.randint(0, 4, (5,), generator=g)
        ell = cross_entropy(y_hat, y)
        assert ell.shape == (5,)
        torch.testing.assert_close(
            ell.mean(), nn.CrossEntropyLoss()(y_hat, y), rtol=1e-14, atol=0
        )

    def test_probability_targets(self):
        g = _generator()
        y_hat = torch.randn(5, 4, generator=g, dtype=torch.float64)
        y = torch.softmax(torch.randn(5, 4, generator=g, dtype=torch.float64), dim=1)
        expected = -(y * torch.log_softmax(y_hat, dim=1)).sum(dim=1)
        torch.testing.assert_close(
            cross_entropy(y_hat, y), expected, rtol=1e-14, atol=0
        )
