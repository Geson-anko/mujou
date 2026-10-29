"""ResNet-20 for CIFAR-10 (He et al., 2016), cut into one module per node.

The network is a stem, 9 residual blocks in three stages of 16, 32 and 64
channels, and a head that pools and gives the 10 logits. ``resnet()`` cuts it
into four modules, one per node: the stem with the first stage, the second
stage, the third stage, and the head. Inside a node, autograd differentiates
the module as in plain PyTorch; between the nodes, only the messages of PC-ALM
flow.

The credit moves one node down in each period of the nodes, so a sample needs
about twice as many periods as there are connections from the input to the
output before every node has its credit. On one GPU, a period of all the nodes
costs about the same however the network is cut (launching the kernels from
the CPU is the limit), so fewer nodes learn faster: cut into one node per
block (11 nodes), a batch needed 24 periods and could not keep up with a period
of 2.5 ms of real time; cut into stages, 12 periods are enough.

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
    """The modules of the four nodes: ``(N, 3, 32, 32) -> (N, 16, 32, 32)`` (the
    stem and the first stage), ``-> (N, 32, 16, 16)`` and ``-> (N, 64, 8, 8)``
    (the other stages, each starting with a block that halves the resolution
    and doubles the channels), then ``-> (N, 10)``."""
    stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), norm(16), nn.ReLU())
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    channels = [16] * 4 + [32] * 3 + [64] * 3  # the input of each block, then 64
    blocks = [
        Block(cin, cout, stride=1 if cin == cout else 2)
        for cin, cout in zip(channels[:-1], channels[1:], strict=True)
    ]
    return [
        nn.Sequential(stem, *blocks[:3]),
        nn.Sequential(*blocks[3:6]),
        nn.Sequential(*blocks[6:]),
        head,
    ]
