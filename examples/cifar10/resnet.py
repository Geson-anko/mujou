"""ResNet-20 for CIFAR-10 (He et al., 2016), cut into one module per node.

The network is a stem, 9 residual blocks in three stages of 16, 32 and 64
channels, and a head that pools and gives the 10 logits. ``resnet()`` cuts it
into ten modules, one per node: the stem with the first block, each of the
other 8 blocks, and the head. Inside a node, autograd differentiates the
module as in plain PyTorch; between the nodes, only the messages of PC-ALM
flow.

A node relaxes its activity ``h`` with what the next node sends back, through
the Jacobian ``J`` of the next node's module. The singular values of ``J`` of a
block grow while the network learns (2 to 9 after a few epochs, measured with
BP and with PC-ALM), and an explicit step of ``h`` of ``period / tau.h`` is
stable only while ``(1 + |J|^2) * period / tau.h < 2``: below about 2.6 with
the period ``tau.h / 4`` of ``pc.Chain``, which the blocks go past. With
``tau.h / 4``, the ``lam`` of the weights BP learned in 3 epochs did not settle
on the credit of backpropagation (a cosine of 0.1 or less); with ``tau.h / 16``
it did (0.95 in the first nodes), as with ``tau.h / 64``. This is why
``main.py`` gives the nodes a period of ``tau.h / 16``.

GroupNorm replaces BatchNorm: it normalizes each sample on its own, so it has no
running statistics and no train or eval mode to switch, and a batch padded with
zeros does not change the other samples. Each block starts as the identity (the
scale of its last GroupNorm starts at 0), and where a stage starts, the shortcut
has no weights (the option A of He et al.).
"""

from typing import override

from torch import Tensor, nn


def norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with groups of 2 channels."""
    return nn.GroupNorm(channels // 2, channels)


class Block(nn.Module):
    """``relu(shortcut(x) + F(x))``, ``F`` two 3x3 convolutions. With ``stride``
    2, the block halves the resolution, and the shortcut keeps every other
    pixel; the channels beyond ``cin`` of the shortcut are zero."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            norm(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            last := norm(cout),
        )
        nn.init.zeros_(last.weight)  # the block starts as the identity
        self.stride = stride
        self.extra = cout - cin  # the channels the shortcut pads

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x[:, :, :: self.stride, :: self.stride]
        shortcut = nn.functional.pad(shortcut, (0, 0, 0, 0, 0, self.extra))
        return (shortcut + self.f(x)).relu()


def resnet() -> list[nn.Module]:
    """The modules of the ten nodes: ``(N, 3, 32, 32) -> (N, 16, 32, 32)`` (the
    stem and the first block), the other 8 blocks (the 4th and the 7th halve
    the resolution and double the channels), then
    ``(N, 64, 8, 8) -> (N, 10)``."""
    stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), norm(16), nn.ReLU())
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    channels = [16] * 4 + [32] * 3 + [64] * 3  # the input of each block, then 64
    blocks = [
        Block(cin, cout, stride=1 if cin == cout else 2)
        for cin, cout in zip(channels[:-1], channels[1:], strict=True)
    ]
    return [nn.Sequential(stem, blocks[0]), *blocks[1:], head]
