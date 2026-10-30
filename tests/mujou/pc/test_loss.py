import torch

from mujou.pc.functional import cross_entropy, mse
from mujou.pc.loss import CrossEntropyLoss, MSELoss


def _generator() -> torch.Generator:
    return torch.Generator().manual_seed(0)


class TestMSELoss:
    def test_is_the_function_with_its_batch(self):
        g = _generator()
        y_hat = torch.randn(2, 3, 4, generator=g, dtype=torch.float64)
        y = torch.randn(2, 3, 4, generator=g, dtype=torch.float64)
        for batch in [(), 2, (2, 3)]:
            assert torch.equal(MSELoss(batch)(y_hat, y), mse(y_hat, y, batch))

    def test_without_a_batch_by_default(self):
        y_hat, y = torch.zeros(3), torch.ones(3)
        assert MSELoss()(y_hat, y).shape == ()


class TestCrossEntropyLoss:
    def test_is_the_function_with_its_batch(self):
        g = _generator()
        y_hat = torch.randn(2, 3, 4, generator=g, dtype=torch.float64)
        y = torch.randint(0, 4, (2, 3), generator=g)  # 4 classes after (2, 3)
        assert torch.equal(
            CrossEntropyLoss((2, 3))(y_hat, y), cross_entropy(y_hat, y, (2, 3))
        )
        y = torch.randint(0, 3, (2, 4), generator=g)  # 3 classes, 4 per sample
        assert torch.equal(CrossEntropyLoss(2)(y_hat, y), cross_entropy(y_hat, y, 2))

    def test_without_a_batch_by_default(self):
        y_hat, y = torch.zeros(4), torch.tensor(1)
        assert CrossEntropyLoss()(y_hat, y).shape == ()
