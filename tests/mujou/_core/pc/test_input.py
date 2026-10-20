from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

import mujou.pc as pc
from mujou import Group, TensorSpec
from mujou.testing import ManualRun

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual, Step

B = 3
CARRY = pc.carry_spec(TensorSpec((B,), torch.int64), batch=B)


class TestInput:
    def _run(self, start_manual: StartManual) -> tuple[pc.Input, ManualRun[Any]]:
        inp = pc.Input((B, 4), carry=CARRY, period=1e-3)
        return inp, start_manual(inp)

    def test_sends_x_carrying_the_target_and_the_time_of_the_write(
        self, start_manual: StartManual, step: Step
    ):
        inp, run = self._run(start_manual)
        generator = torch.Generator().manual_seed(0)
        for _ in range(2):  # a second write carries its own time
            run.timer.advance(1e-3)
            x = torch.randn(B, 4, generator=generator)
            y = torch.randint(10, (B,), generator=generator)
            mask = torch.tensor([True, False, True])
            written_at = run.write({inp.prev: {"x": x, "y": y, "mask": mask}})
            step(run, inp)
            sent, _, _, carried = run.state_dict()["slots"]["next/out"]
            assert torch.equal(sent["h"], x)
            assert torch.equal(carried["y"], y)
            assert torch.equal(carried["mask"], mask)
            assert int(carried["time"]) == written_at.ns

    def test_sends_zeros_without_a_target_before_the_host_writes(
        self, start_manual: StartManual, step: Step
    ):
        inp, run = self._run(start_manual)
        step(run, inp)
        sent, _, _, carried = run.state_dict()["slots"]["next/out"]
        assert torch.equal(sent["h"], torch.zeros(B, 4))
        assert int(sent["epoch"]) == 0
        assert torch.equal(carried["y"], torch.zeros(B, dtype=torch.int64))
        assert torch.equal(carried["mask"], torch.zeros(B, dtype=torch.bool))
        assert int(carried["time"]) == 0

    def test_the_epoch_counts_the_writes(self, start_manual: StartManual, step: Step):
        # A write at time 0, as the zeros before it carry, still counts; an
        # update without a new write does not.
        inp, run = self._run(start_manual)
        sample = {
            "x": torch.ones(B, 4),
            "y": torch.zeros(B, dtype=torch.int64),
            "mask": torch.ones(B, dtype=torch.bool),
        }
        epochs: list[int] = []
        for s in range(5):
            if s in (0, 3):
                run.write({inp.prev: sample})  # at times 0 and 3 ms
            step(run, inp)
            epochs.append(int(run.state_dict()["slots"]["next/out"][0]["epoch"]))
        assert epochs == [1, 1, 1, 2, 2]
        assert run.state_dict()["nodes"][""]["_seen_write"] == 3_000_000

    def test_connects_to_a_layer(self):
        inp = pc.Input((B, 4), carry=CARRY, period=1e-3)
        tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=1.0)
        layer = pc.Layer(nn.Linear(4, 5), input_shape=(B, 4), carry=CARRY, tau=tau)
        inp.next.connect(layer.prev)
        Group(inp=inp, layer=layer).build()

    def test_the_shape_must_start_with_the_batch(self):
        with pytest.raises(ValueError, match="batch"):
            pc.Input((B + 1, 4), carry=CARRY, period=1e-3)

    def test_carry_must_come_from_carry_spec(self):
        with pytest.raises(TypeError, match="carry_spec"):
            pc.Input((B, 4), carry=TensorSpec((B,)), period=1e-3)  # pyright: ignore[reportArgumentType]
