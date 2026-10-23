import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

import mujou.pc as pc
from mujou import DictSpec, Group, TensorSpec, TimestampSpec
from mujou.testing import ManualRun
from tests.mujou.nodes import F64, close
from tests.mujou.pc._reference import ADAMW, SGD, TAU_THETA

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual

H = 1e-3  # seconds per step
B, N_IN, N_OUT = 3, 4, 2
TAU = pc.TimeConstants(h=4e-3, lam=8e-3)
X, Y = TensorSpec((B, N_IN), F64), TensorSpec((B, N_OUT), F64)
# The activity of float32 h_in, for the nodes that are only made.
ACT = pc.activity_spec(TensorSpec((B, N_IN)), TensorSpec((B, N_OUT)), batch=B)


def head(seed: int = 0) -> nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Linear(N_IN, N_OUT).double()


class Direct:
    """``Input -> Output`` stepped from the output: the output reads the input
    sent in the previous step."""

    def __init__(
        self, start_manual: StartManual, module: nn.Module, **kwargs: Any
    ) -> None:
        self.inp = pc.Input(x=X, y=Y, period=H, batch=B)
        learning: dict[str, Any] = {"optimizer": SGD, **kwargs}
        self.out = pc.Output(
            module,
            act_spec=self.inp.act_spec,
            loss=pc.MSELoss(B),
            tau=TAU,
            **learning,
        )
        self.inp.next.connect(self.out.prev)
        self.run: ManualRun = start_manual(Group(inp=self.inp, out=self.out))

    def weights(self) -> dict[str, torch.Tensor]:
        return self.run.state_dict()["nodes"]["out"]["state"]["module"]

    def step(
        self, sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ) -> int:
        """Advance, write ``(x, y, mask)`` unless ``None``, step; return the time
        of the write (or of the step)."""
        timer = self.run.timer
        timer.advance(H)
        t = timer.now().ns
        if sample is not None:
            x, y, mask = sample
            t = self.run.write({self.inp.prev: {"x": x, "y": y, "mask": mask}}).ns
        self.run.step(0.0, self.out, self.inp)
        return t


def samples(n: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    masks = [
        torch.tensor([True, False, True]),
        torch.tensor([False, False, False]),
        torch.tensor([True, True, True]),
    ]
    return [
        (
            torch.randn(B, N_IN, dtype=F64, generator=generator),
            torch.randn(B, N_OUT, dtype=F64, generator=generator),
            masks[k % len(masks)],
        )
        for k in range(n)
    ]


def gradients(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """``y_hat``, ``d loss / d x`` and ``d loss / d theta`` of
    ``sum_b mask_b mean_j (y_bj - y_hat_bj)^2``, written out."""
    x = x.clone().requires_grad_()
    y_hat = module(x)
    loss = ((y - y_hat) ** 2).mean(dim=1)[mask].sum()
    g_in, *g_theta = torch.autograd.grad(loss, [x, *module.parameters()])
    return y_hat.detach(), g_in, g_theta


class TestConstruction:
    def test_a_loss_that_averages_is_refused(self):
        with pytest.raises(ValueError, match="pc.MSELoss"):
            pc.Output(
                nn.Linear(N_IN, N_OUT),
                loss=nn.MSELoss(),
                act_spec=ACT,
                tau=TAU,
                optimizer=SGD,
                learn_interval=None,
            )

    def test_the_prediction_has_the_target(self):
        act = pc.activity_spec(
            TensorSpec((B, N_IN)), TensorSpec((B,), torch.int64), batch=B
        )
        out = pc.Output(
            nn.Linear(N_IN, 10),
            act_spec=act,
            loss=pc.CrossEntropyLoss(B),
            tau=TAU,
            optimizer=SGD,
            learn_interval=None,
        )
        assert out.next.send_spec == DictSpec(
            pc.Prediction,
            y_hat=TensorSpec((B, 10)),
            y=TensorSpec((B,), torch.int64),
            mask=TensorSpec((B,), torch.bool),
            time=TimestampSpec(),
        )

    def test_the_period_defaults_to_a_quarter_of_tau_h(self):
        out = pc.Output(
            nn.Linear(N_IN, N_OUT),
            loss=pc.MSELoss(B),
            act_spec=ACT,
            tau=TAU,
            optimizer=SGD,
            learn_interval=None,
        )
        assert out.period == pytest.approx(TAU.h / 4)

    def test_the_period_must_not_be_longer_than_max_dt(self):
        with pytest.raises(ValueError, match="max_dt"):
            pc.Output(
                nn.Linear(N_IN, N_OUT),
                loss=pc.MSELoss(B),
                act_spec=ACT,
                tau=TAU,
                optimizer=SGD,
                learn_interval=None,
                period=TAU.h,  # max_dt is tau.h / 2
            )


class TestPairing:
    def test_each_prediction_meets_the_target_of_its_own_input(
        self, start_manual: StartManual
    ):
        # A new sample every step, faster than it reaches the output: the
        # output still pairs the input of the previous step with its target,
        # and learns from that pair only.
        module = head()
        reference = copy.deepcopy(module)
        direct = Direct(start_manual, module, learn_interval=None)
        data = samples(7)
        times: list[int] = []
        for k, (x, y, mask) in enumerate(data):
            times.append(direct.step((x, y, mask)))
            if k == 0:
                continue
            x_, y_, mask_ = data[k - 1]
            y_hat, g_in, g_theta = gradients(reference, x_, y_, mask_)
            prediction = direct.run.read(direct.out.next)
            close(prediction["y_hat"], y_hat)
            assert torch.equal(prediction["y"], y_)
            assert torch.equal(prediction["mask"], mask_)
            assert prediction["time"].ns == times[k - 1]
            g_sent = direct.run.read(direct.out.prev)
            close(g_sent, g_in)
            assert torch.equal(g_sent[~mask_], torch.zeros_like(g_sent[~mask_]))
            if mask_.any():
                with torch.no_grad():
                    for p, g in zip(reference.parameters(), g_theta, strict=True):
                        p -= (H / TAU_THETA) * g / B
            for key, value in reference.state_dict().items():
                close(direct.weights()[key], value)


class TestUnlabeled:
    def test_no_target_skips_the_update_and_drops_the_time(
        self, start_manual: StartManual
    ):
        # learn_interval = 2 steps. Step 1 writes A without a target: the output
        # sees it from step 2 and skips its updates at steps 3 and 5 (2 and 4
        # steps after the write). Step 6 writes B, which the output sees at step
        # 7 and learns at step 8, by the time since the skip at step 5 only
        # (steps 6 to 8).
        module = head()
        reference = copy.deepcopy(module)
        direct = Direct(start_manual, module, learn_interval=2 * H)
        (x_a, y_a, _), (x_b, y_b, mask_b) = samples(3)[0], samples(3)[2]
        no_target = torch.zeros(B, dtype=torch.bool)
        initial = copy.deepcopy(module.state_dict())
        for s in range(1, 8):
            new = {1: (x_a, y_a, no_target), 6: (x_b, y_b, mask_b)}.get(s)
            direct.step(new)
            for key, value in initial.items():
                assert torch.equal(direct.weights()[key], value), s
        direct.step(None)  # step 8
        _, _, g_theta = gradients(reference, x_b, y_b, mask_b)
        with torch.no_grad():
            for p, g in zip(reference.parameters(), g_theta, strict=True):
                p -= (3 * H / TAU_THETA) * g / B
        for key, value in reference.state_dict().items():
            close(direct.weights()[key], value)


class TestMaxDt:
    def test_a_long_dt_weighs_the_update_by_max_dt(self, start_manual: StartManual):
        # Learning in every update: a step of 4 H weighs the update of theta by
        # max_dt = 2 H only.
        module = head()
        reference = copy.deepcopy(module)
        direct = Direct(start_manual, module, learn_interval=None, max_dt=2 * H)
        x, y, mask = samples(1)[0]
        direct.step((x, y, mask))  # the output reads zeros: nothing to learn
        direct.run.timer.advance(3 * H)
        direct.step(None)
        _, _, g_theta = gradients(reference, x, y, mask)
        with torch.no_grad():
            for p, g in zip(reference.parameters(), g_theta, strict=True):
                p -= (2 * H / TAU_THETA) * g / B
        for key, value in reference.state_dict().items():
            close(direct.weights()[key], value)


class TestOptimizer:
    def test_the_first_update_with_adamw_is_its_closed_form(
        self, start_manual: StartManual
    ):
        # Learning in every update: the output first learns at step 2, by H.
        # The first step of AdamW has m_hat = g and v_hat = g^2 whatever tau_m
        # and tau_v: theta (1 - eta w) - eta g / (|g| + eps), eta = H / tau,
        # with the mean gradient g = g_theta / B.
        module = head()
        reference = copy.deepcopy(module)
        direct = Direct(start_manual, module, optimizer=ADAMW, learn_interval=None)
        x, y, mask = samples(1)[0]
        direct.step((x, y, mask))  # the output reads zeros: nothing to learn
        direct.step(None)
        _, _, g_theta = gradients(reference, x, y, mask)
        eta = H / ADAMW.tau
        decay = 1 - eta * ADAMW.weight_decay
        with torch.no_grad():
            for p, g_sum in zip(reference.parameters(), g_theta, strict=True):
                g = g_sum / B
                p.copy_(p * decay - eta * g / (g.abs() + ADAMW.eps))
        for key, value in reference.state_dict().items():
            close(direct.weights()[key], value, 1e-12)
