import subprocess
import sys

import mujou


class TestPublicNames:
    def test_exports(self):
        # The top-level names of specs/network.md; a change here is a change of the
        # public API.
        assert sorted(mujou.__all__) == [
            "Channel",
            "DictSpec",
            "Group",
            "MessageSpec",
            "MultiPort",
            "Network",
            "NetworkError",
            "Node",
            "Port",
            "Run",
            "TensorSpec",
            "__version__",
            "run",
        ]

    def test_imports_no_submodule(self):
        # Each is imported by name (import mujou.time), in a new interpreter: this
        # one has imported them all.
        script = (
            "import sys, mujou; "
            "print([m for m in ('mujou.time', 'mujou.testing', 'mujou.pc') "
            "if m in sys.modules])"
        )
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert done.stdout.strip() == "[]"


class TestPcPublicNames:
    def test_exports(self):
        # The names of mujou.pc; a change here is a change of the
        # public API.
        import mujou.pc

        assert sorted(mujou.pc.__all__) == [
            "Activity",
            "Chain",
            "CrossEntropyLoss",
            "Input",
            "Layer",
            "LearnClock",
            "MSELoss",
            "Output",
            "Prediction",
            "Sample",
            "Size",
            "Tick",
            "TimeConstants",
            "activity_spec",
            "functional",
            "show",
        ]
        assert sorted(mujou.pc.functional.__all__) == [
            "cross_entropy",
            "learn_",
            "loss_vjp",
            "mse",
            "relax_",
            "vjp",
        ]

    def test_pc_is_a_submodule_outside_the_top_level(self):
        assert "pc" not in mujou.__all__
