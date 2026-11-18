import importlib
import os
import subprocess
import sys
import sysconfig

import pytest

import mujou


class TestPublicNames:
    def test_exports(self):
        # The top-level names of specs/network.md; a change here is a change of the
        # public API.
        assert sorted(mujou.__all__) == [
            "BoolSpec",
            "Channel",
            "DictSpec",
            "FieldSpec",
            "FloatSpec",
            "Group",
            "IntSpec",
            "ManualRun",
            "MessageSpec",
            "MultiPort",
            "Network",
            "NetworkError",
            "Node",
            "Port",
            "Run",
            "Stateful",
            "TensorSpec",
            "TimestampSpec",
            "__version__",
            "run",
            "run_manually",
        ]

    def test_import_loads_no_submodule_and_keeps_the_gil_disabled(self):
        # One new interpreter for both (this one has imported every submodule, and
        # another extension may have re-enabled the GIL). Each submodule is
        # imported by name (import mujou.time). mujou.types is not one of them: the
        # core annotates with its types, so it comes with mujou. ``import mujou``
        # loads the extension mujou._native, torch and every module of the core.
        script = (
            "import sys, mujou; "
            "print([m for m in ('mujou.time', 'mujou.pc', 'mujou.viz') "
            "if m in sys.modules]); "
            "print(sys._is_gil_enabled())"
        )
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            # PYTHON_GIL=0 would keep the GIL off whatever the modules declare.
            env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
        )
        submodules, gil_enabled = done.stdout.split()
        assert submodules == "[]"
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert gil_enabled == "False"

    def test_the_manual_run_is_no_longer_in_mujou_testing(self):
        # mujou.run_manually replaced it, with no name kept for compatibility.
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("mujou.testing")


class TestPcPublicNames:
    def test_exports(self):
        # The names of mujou.pc; a change here is a change of the
        # public API.
        import mujou.pc

        assert sorted(mujou.pc.__all__) == [
            "Chain",
            "Input",
            "Layer",
            "Output",
            "Shown",
            "TimeConstants",
            "clock",
            "functional",
            "loss",
            "msg",
            "optim",
            "relax",
            "show",
        ]
        assert sorted(mujou.pc.optim.__all__) == [
            "AdamW",
            "BoundOptimizer",
            "Optimizer",
            "SGD",
        ]
        assert sorted(mujou.pc.relax.__all__) == [
            "BoundRelaxer",
            "Euler",
            "Relaxer",
        ]
        assert sorted(mujou.pc.loss.__all__) == ["CrossEntropyLoss", "MSELoss"]
        assert sorted(mujou.pc.msg.__all__) == [
            "Activity",
            "Prediction",
            "Sample",
            "activity_spec",
        ]
        assert sorted(mujou.pc.clock.__all__) == ["LearnClock", "Tick"]
        assert sorted(mujou.pc.functional.__all__) == [
            "cross_entropy",
            "loss_vjp",
            "mse",
            "relax_",
            "vjp",
        ]

    def test_pc_is_a_submodule_outside_the_top_level(self):
        assert "pc" not in mujou.__all__


class TestTypesPublicNames:
    def test_exports(self):
        # The names of mujou.types; a change here is a change of the public API.
        import mujou.types

        assert sorted(mujou.types.__all__) == ["Device", "Message", "Size"]


class TestVizPublicNames:
    def test_exports(self):
        # The names of mujou.viz; a change here is a change of the public API.
        import mujou.viz

        assert mujou.viz.__all__ == ["to_html"]
