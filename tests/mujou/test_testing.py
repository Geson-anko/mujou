import inspect

import mujou
import mujou.testing


class TestPublicNames:
    def test_testing_exports(self):
        assert sorted(mujou.testing.__all__) == ["ManualRun", "VirtualTimer", "run"]

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
