import inspect

import pytest

import mujou
import mujou.testing
from mujou import Group
from mujou.time import Timestamp, VirtualTimer
from tests.mujou.nodes import Calls


class TestPublicNames:
    def test_testing_exports(self):
        assert sorted(mujou.testing.__all__) == ["ManualRun", "run"]

    def test_run_takes_the_arguments_of_mujou_run_but_workers(self):
        # A manual run updates one node at a time, so the default workers do.
        ours = inspect.signature(mujou.testing.run).parameters
        theirs = inspect.signature(mujou.run).parameters
        assert list(ours) == [name for name in theirs if name != "workers"]
        assert "workers" in theirs

    def test_no_private_parameter_is_shown(self):
        callables = [
            mujou.run,
            mujou.testing.run,
            mujou.Network.build,
            mujou.Group.__init__,
            *(
                getattr(mujou.Run, n)
                for n in vars(mujou.Run)
                if not n.startswith("_") and callable(getattr(mujou.Run, n))
            ),
            mujou.testing.ManualRun.step,
        ]
        for f in callables:
            names = inspect.signature(f).parameters
            assert not [n for n in names if n.startswith("_")], f


class TestStep:
    def test_advances_the_timer_by_dt_then_updates_in_the_given_order(self):
        order: list[str] = []
        a = Calls(lambda: order.append("a"))
        b = Calls(lambda: order.append("b"))
        with mujou.testing.run(Group(a=a, b=b)) as run:
            run.step(0.25, b, a)
            assert run.timer.now() == Timestamp(ns=250_000_000)
            run.step(0.5)  # every node, in the order of the tree
        assert order == ["b", "a", "a", "b"]
        assert a.dts == b.dts == [0.25, 0.5]

    def test_ignores_the_period(self):
        node = Calls()  # period 0.01
        with mujou.testing.run(node) as run:
            run.step(0.001)
            run.step(1.0)
        assert node.dts == [0.001, 1.0]

    def test_a_node_given_twice_updates_once(self):
        node = Calls()
        with mujou.testing.run(node) as run:
            run.step(0.001, node, node)
        assert node.dts == [0.001]

    def test_the_first_dt_is_from_the_start(self):
        timer = VirtualTimer(start=Timestamp(ns=3_000_000_000))
        node = Calls()
        with mujou.testing.run(node, timer=timer) as run:
            run.step(0.5)
        assert node.dts == [0.5]

    def test_a_node_of_another_run_moves_no_time(self):
        with mujou.testing.run(Calls()) as run:
            with pytest.raises(ValueError, match="not a node of this run"):
                run.step(1.0, Calls())
            assert run.timer.now() == Timestamp(ns=0)

    def test_after_a_normal_end_moves_no_time(self):
        run = mujou.testing.run(Calls())
        run.shutdown()
        with pytest.raises(RuntimeError, match="stopped"):
            run.step(1.0)
        assert run.timer.now() == Timestamp(ns=0)

    def test_a_negative_dt_moves_no_time(self):
        node = Calls()
        with mujou.testing.run(node) as run:
            with pytest.raises(ValueError):
                run.step(-1.0)
            assert run.timer.now() == Timestamp(ns=0)
        assert node.dts == []
