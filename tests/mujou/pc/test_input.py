from typing import TYPE_CHECKING

import pytest
import torch

import mujou.pc as pc
from mujou import ManualRun, TensorSpec
from mujou.time import Timestamp, VirtualTimer

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual

H = 1e-3  # seconds per step

B = 3
X, Y = TensorSpec((B, 4)), TensorSpec((B,), torch.int64)


class TestInput:
    def _run(self, start_manual: StartManual) -> tuple[pc.Input, ManualRun]:
        inp = pc.Input(x=X, y=Y, period=1e-3, batch=B)
        return inp, start_manual(inp)

    def test_sends_x_with_the_target_the_reset_and_the_time_of_the_write(
        self, start_manual: StartManual
    ):
        inp, run = self._run(start_manual)
        generator = torch.Generator().manual_seed(0)
        for _ in range(2):  # a second write has its own time
            run.timer.advance(1e-3)
            x = torch.randn(B, 4, generator=generator)
            y = torch.randint(10, (B,), generator=generator)
            mask = torch.tensor([True, False, True])
            reset = torch.tensor([True, True, False])
            written_at = run.write(
                {inp.prev: {"x": x, "y": y, "mask": mask, "reset": reset}}
            )
            run.step(H, inp)
            sent = run.state_dict()["nodes"][""]["ports"]["next"]["out"]["value"]
            assert torch.equal(sent["h"], x)
            assert torch.equal(sent["y"], y)
            assert torch.equal(sent["mask"], mask)
            assert torch.equal(sent["reset"], reset)
            assert sent["time"] == written_at.ns  # the tree holds a time as its ns
        # What it sends is the last write, so a restored run needs nothing more.
        assert set(run.state_dict()["nodes"][""]["state"]) == {"generator"}

    def test_sends_zeros_without_a_target_before_the_host_writes(
        self, start_manual: StartManual
    ):
        # At time 0 whatever the time of the run's timer.
        inp = pc.Input(x=X, y=Y, period=1e-3, batch=B)
        run = start_manual(inp, timer=VirtualTimer(start=Timestamp(ns=5_000_000)))
        run.step(H, inp)
        sent = run.state_dict()["nodes"][""]["ports"]["next"]["out"]["value"]
        assert torch.equal(sent["h"], torch.zeros(B, 4))
        assert torch.equal(sent["y"], torch.zeros(B, dtype=torch.int64))
        assert torch.equal(sent["mask"], torch.zeros(B, dtype=torch.bool))
        assert torch.equal(sent["reset"], torch.zeros(B, dtype=torch.bool))
        assert sent["time"] == 0

    def test_x_and_y_must_start_with_the_batch(self):
        with pytest.raises(ValueError, match="start with the batch"):
            pc.Input(x=TensorSpec((B + 1, 4)), y=Y, period=1e-3, batch=B)

    def test_without_a_batch_takes_one_sample(self, start_manual: StartManual):
        inp = pc.Input(x=TensorSpec((4,)), y=TensorSpec((), torch.int64), period=1e-3)
        assert inp.act_spec.specs["mask"] == TensorSpec((), torch.bool)
        run = start_manual(inp)
        x = torch.arange(4.0)
        y, mask = torch.tensor(7), torch.tensor(True)
        run.write({inp.prev: {"x": x, "y": y, "mask": mask, "reset": mask}})
        run.step(H, inp)
        sent = run.state_dict()["nodes"][""]["ports"]["next"]["out"]["value"]
        assert torch.equal(sent["h"], x)
        assert torch.equal(sent["y"], y)
        assert torch.equal(sent["mask"], mask)
