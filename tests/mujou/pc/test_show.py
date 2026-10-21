import copy
from typing import TYPE_CHECKING

import torch

import mujou.pc as pc
from mujou import Network, TensorSpec
from mujou.time import Timer
from tests.mujou.nodes import F64, assert_same_tree, close
from tests.mujou.pc._reference import N_OUT, TAU, B, forward, make_modules, sample

if TYPE_CHECKING:
    from tests.mujou.conftest import Start

WIDTHS = [4, 5]
X, Y = TensorSpec((B, WIDTHS[0]), F64), TensorSpec((B, N_OUT), F64)


class TestShow:
    def test_returns_the_prediction_before_learning_once_the_time_has_passed(
        self, start: Start
    ):
        # The output reads x itself (no layer), so every prediction of the
        # sample is f(x) until the output learns, learn_interval after the
        # write; seconds is three times that.
        head = make_modules("linear", "tanh", WIDTHS[:1])[1]
        initial = copy.deepcopy(head)
        net = pc.Chain(head, x=X, y=Y, loss=pc.mse, tau=TAU, learn_interval=0.05)
        x, y = sample(WIDTHS, 0)
        labeled = torch.ones(B, dtype=torch.bool)
        run = start(net, timer=Timer())
        before = run.timer.now()
        prediction = pc.show(run, net, pc.Sample(x=x, y=y, mask=labeled), seconds=0.15)
        assert run.timer.now() - before >= 0.15
        run.shutdown()
        with torch.no_grad():
            close(prediction["y_hat"], initial(x), 1e-12)
        assert torch.equal(prediction["y"], y)
        assert torch.equal(prediction["mask"], labeled)
        learned = net.output.module.state_dict()
        assert not torch.equal(learned["weight"], initial.state_dict()["weight"])

    def test_an_unlabeled_sample_is_the_forward_pass_and_learns_nothing(
        self, start: Start
    ):
        layers, head = make_modules("linear", "tanh", WIDTHS)
        before = [copy.deepcopy(m.state_dict()) for m in [*layers, head]]
        net = pc.Chain(
            *layers, head, x=X, y=Y, loss=pc.mse, tau=TAU, learn_interval=None
        )
        x, y = sample(WIDTHS, 0)
        unlabeled = torch.zeros(B, dtype=torch.bool)
        run = start(net, timer=Timer())
        prediction = pc.show(
            run, net, pc.Sample(x=x, y=y, mask=unlabeled), seconds=0.02
        )
        run.shutdown()
        with torch.no_grad():
            _, y_hat = forward(layers, head, x)
        close(prediction["y_hat"], y_hat, 1e-12)
        for module, state in zip([*layers, head], before, strict=True):
            assert_same_tree(module.state_dict(), state)

    def test_shows_a_network_built_by_hand(self, start: Start):
        class ByHand(Network):
            def __init__(self) -> None:
                super().__init__()
                self.input = pc.Input(x=X, y=Y, period=TAU.period)
                head = make_modules("linear", "tanh", WIDTHS[:1])[1]
                self.output = pc.Output(
                    head,
                    act_spec=self.input.act_spec,
                    loss=pc.mse,
                    tau=TAU,
                    learn_interval=None,
                    period=TAU.period / 2,
                )
                self.input.next.connect(self.output.prev)
                self.prev, self.next = self.input.prev, self.output.next

        net = ByHand()
        x, y = sample(WIDTHS, 0)
        unlabeled = torch.zeros(B, dtype=torch.bool)
        run = start(net, timer=Timer())
        prediction = pc.show(
            run, net, pc.Sample(x=x, y=y, mask=unlabeled), seconds=0.02
        )
        run.shutdown()
        with torch.no_grad():
            close(prediction["y_hat"], net.output.module(x), 1e-12)
