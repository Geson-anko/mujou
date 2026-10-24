"""ResNet-20 for CIFAR-10 (He et al., 2016), cut into the modules of two nodes.

The network is a stem, 9 residual blocks in three stages of 16, 32 and 64
channels, and a head that pools and gives the 10 logits. ``resnet()`` cuts it
into two modules, one per node: the stem and the first two stages, then the
third stage and the head. Inside a node, autograd differentiates the module as
in plain PyTorch; between the two nodes, only the messages of PC-ALM flow.

Why cut there: a node relaxes its activity ``h`` with what the next node sends
back, through the Jacobian ``J`` of the next node's module. With the period
``tau.h / 4`` of ``pc.Chain``, that relaxation is stable only while the largest
singular value of ``J`` is below about 2.6 (``1 + |J|^2 < 8``), and below
about 1.2 to 1.8 when updates come late (in a linear chain). As the network
learns, the Jacobian of a stage grows to 5 to 12 and that of a single block to
3 to 6, while the third stage with the head (where the pooling averages 64
pixels) is about 1.3 after 3 epochs and 2.2 after 10. As the next node is the
output, the curvature of the cross entropy (at most 1/2) multiplies its square,
so the relaxation stays stable. With a node per stage, the relaxation
diverges; a node per block would be past the bound too.

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


def stage(cin: int, cout: int) -> nn.Sequential:
    """3 blocks from ``cin`` to ``cout`` channels; the first halves the
    resolution when the channels grow."""
    return nn.Sequential(
        Block(cin, cout, stride=1 if cin == cout else 2),
        Block(cout, cout),
        Block(cout, cout),
    )


def resnet() -> list[nn.Module]:
    """The modules of the two nodes: ``(N, 3, 32, 32) -> (N, 32, 16, 16)``, then
    ``-> (N, 10)``."""
    stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), norm(16), nn.ReLU())
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))
    return [
        nn.Sequential(stem, stage(16, 16), stage(16, 32)),
        nn.Sequential(stage(32, 64), head),
    ]
