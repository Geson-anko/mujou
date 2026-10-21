"""PC-ALM in plain PyTorch, from the equations of ``specs/pc-module.md``: the
oracles that the tests of the nodes of ``mujou.pc`` compare with."""

import copy
import math
from collections.abc import Callable, Sequence
from typing import override

import torch
from torch import nn

import mujou.pc as pc
from tests.mujou.nodes import F64

H = 1e-3  # seconds per step

B = 3
N_OUT = 2
TAU = pc.TimeConstants(h=4e-3, lam=8e-3, theta=0.05)
NO_LEARNING = pc.TimeConstants(h=TAU.h, lam=TAU.lam, theta=math.inf)
MASK = torch.tensor([True, False, True])


class ResBlock(nn.Module):
    """``h + fc2(act(fc1(h)))``: a residual block as the ``f`` of one node."""

    def __init__(self, n: int, act: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n, n), nn.Linear(n, n)
        self.act = act

    @override
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(self.act(self.fc1(h)))


class Act(nn.Module):
    def __init__(self, act: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.act = act

    @override
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.act(h)


def identity(h: torch.Tensor) -> torch.Tensor:
    return h


ACTS = {"linear": identity, "tanh": torch.tanh}


def make_modules(
    block: str, act: str, widths: Sequence[int], seed: int = 0
) -> tuple[list[nn.Module], nn.Module]:
    """The modules of the layers and the head, float64: ``Linear`` then the
    activation, or a residual block (of constant width)."""
    layers: list[nn.Module] = []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for n_in, n_out in zip(widths[:-1], widths[1:], strict=True):
            if block == "linear":
                layers.append(nn.Sequential(nn.Linear(n_in, n_out), Act(ACTS[act])))
            else:
                assert n_in == n_out
                layers.append(ResBlock(n_in, ACTS[act]))
        head = nn.Linear(widths[-1], N_OUT)
    return [m.double() for m in layers], head.double()


def sample(widths: Sequence[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(B, widths[0], dtype=F64, generator=generator)
    y = torch.randn(B, N_OUT, dtype=F64, generator=generator)
    return x, y


def masked_mse(
    y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The loss of the output node, written out: sum_b mask_b 0.5 ||y_b - y_hat_b||^2."""
    return (0.5 * (y - y_hat) ** 2).sum(dim=1)[mask].sum()


def forward(
    layers: Sequence[nn.Module], head: nn.Module, x: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """``[h_1, ..., h_L]`` and ``y_hat`` of the forward pass, in the graph."""
    hs = [x]
    for module in layers:
        hs.append(module(hs[-1]))
    return hs[1:], head(hs[-1])


def credits(
    layers: Sequence[nn.Module],
    head: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> list[torch.Tensor]:
    """``-d loss / d h_i`` of backpropagation through the whole network."""
    hs, y_hat = forward(layers, head, x)
    grads = torch.autograd.grad(masked_mse(y_hat, y, mask), hs)
    return [-g for g in grads]


class Clock:
    """When a node learns, as ``specs/pc-module.md`` states it: from the time
    ``time`` (ns) the host wrote the input, every ``interval`` (ns) after it,
    never in the update that first sees the input; ``elapsed`` is the time since
    the last update, dropped when the input changes before the node learned."""

    def __init__(self, interval: int | None) -> None:
        self.interval = interval
        self.seen, self.learn_at, self.elapsed, self.learned_input = -1, 0, 0.0, False

    def tick(self, dt: float, time: int, now: int) -> tuple[bool, bool]:
        new = time != self.seen
        if new:
            self.seen = time
            if self.interval is not None:
                self.learn_at = time + self.interval
            if not self.learned_input:
                self.elapsed = 0.0
            self.learned_input = False
        self.elapsed += dt
        return new, self.interval is None or (not new and now >= self.learn_at)

    def learned(self, now: int) -> None:
        self.elapsed, self.learned_input = 0.0, True
        if self.interval is not None:
            behind = now - self.learn_at
            self.learn_at += (behind // self.interval + 1) * self.interval


class Reference:
    """PC-ALM in plain PyTorch: every node moves at once from the values of the
    previous step (a Jacobi update).

    ``h[0]`` is the ``x`` the input sent and ``h[i + 1]`` the activity of layer
    ``i`` (its state, which it sent), each with the count of jumps, the target, mask and
    time it carries; all start at zero (mask False, time 0), as the ports do. A layer
    moves its ``h`` with its prediction, ``h <- mu + (h - mu_last)``, when the
    carried time or the jumps of ``h_in`` changes, and then advances its jumps.
    When it learns, it keeps the jump ``f(h_in; theta_new) - f(h_in; theta)``
    and adds it to ``h`` in its next step, advancing its jumps for it only when
    it learns at intervals. The input counts the writes as its jumps.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        head: nn.Module,
        widths: Sequence[int],
        learn_interval: float | None = None,
    ) -> None:
        self.layers = [copy.deepcopy(m) for m in layers]
        self.head = copy.deepcopy(head)
        interval = None if learn_interval is None else round(learn_interval * 1e9)
        self.clocks = [Clock(interval) for _ in widths]  # the layers, then the head
        self.lam = [torch.zeros(B, n, dtype=F64) for n in widths[1:]]
        self.mu_last = [torch.zeros(B, n, dtype=F64) for n in widths[1:]]
        self.jump: list[torch.Tensor | None] = [None for _ in widths[1:]]
        self.seen_jumps = [0 for _ in widths[1:]]
        self.input_seen = -1
        self.jumps = [0 for _ in widths]  # what each sender sent
        self.h = [torch.zeros(B, n, dtype=F64) for n in widths]
        self.carried = [
            (torch.zeros(B, N_OUT, dtype=F64), torch.zeros(B, dtype=torch.bool), 0)
            for _ in widths
        ]
        self.g_in: list[torch.Tensor] = [torch.zeros(B, n, dtype=F64) for n in widths]

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        time: int,
        dt: float,
        now: int,
    ) -> None:
        tau = TAU
        # The output: the gradient of the masked loss.
        y_c, mask_c, time_c = self.carried[-1]
        _, learn = self.clocks[-1].tick(dt, time_c, now)
        h_in = self.h[-1].clone().requires_grad_()
        params = list(self.head.parameters())
        self.y_hat = self.head(h_in)
        g_next, *g_theta = torch.autograd.grad(
            masked_mse(self.y_hat, y_c, mask_c), [h_in, *params]
        )
        self.y_hat = self.y_hat.detach()
        self.prediction_carried = (y_c, mask_c)
        self.g_in[-1] = g_next
        if learn:
            self._learn(params, g_theta, mask_c, self.clocks[-1], now)
        # The layers, from the last: each with the g the next one made from the
        # previous values.
        h_new, jumps_new = list(self.h), list(self.jumps)
        for i in reversed(range(len(self.layers))):
            _, mask_i, time_i = self.carried[i]
            clock = self.clocks[i]
            new, learn = clock.tick(dt, time_i, now)
            if learn and not mask_i.any():
                clock.learned(now)
                learn = False
            h = self.h[i + 1]
            jump = self.jump[i]
            if jump is not None:  # theta moved in the step before
                h = h + jump
                self.jump[i] = None
            reanchor = new or self.jumps[i] != self.seen_jumps[i]
            self.seen_jumps[i] = self.jumps[i]
            module = self.layers[i]
            h_in = self.h[i].clone().requires_grad_()
            params = list(module.parameters())
            mu = module(h_in)
            # r = h - mu, which is h - mu_last when h moves with mu below.
            r = h - (self.mu_last[i] if reanchor else mu.detach())
            delta = torch.where(mask_i[:, None], r + self.lam[i], 0.0)
            g_in, *g_theta = torch.autograd.grad(mu, [h_in, *params], -delta)
            mu = mu.detach()
            if reanchor:
                h = mu + (h - self.mu_last[i])
            h = h - (dt / tau.h) * (h - mu + self.lam[i] + g_next)
            self.lam[i] = self.lam[i] + (dt / tau.lam) * (h - mu)
            self.mu_last[i] = mu
            h_new[i + 1] = h
            if reanchor or (jump is not None and clock.interval is not None):
                jumps_new[i + 1] = self.jumps[i + 1] + 1
            if learn:
                self._learn(params, g_theta, mask_i, clock, now)
                with torch.no_grad():
                    moved = module(self.h[i])
                self.jump[i] = moved - mu
                self.mu_last[i] = moved
            self.g_in[i] = g_in
            g_next = g_in
        h_new[0] = x
        if time != self.input_seen:  # a write the input has not seen
            self.input_seen = time
            jumps_new[0] = self.jumps[0] + 1
        self.h, self.jumps = h_new, jumps_new
        self.carried = [(y, mask, time), *self.carried[:-1]]

    def _learn(
        self,
        params: Sequence[torch.Tensor],
        g_theta: list[torch.Tensor],
        mask: torch.Tensor,
        clock: Clock,
        now: int,
    ) -> None:
        if mask.any():
            with torch.no_grad():
                for p, g in zip(params, g_theta, strict=True):
                    p -= (clock.elapsed / TAU.theta) * g / B
        clock.learned(now)
