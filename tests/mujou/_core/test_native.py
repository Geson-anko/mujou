import os
import subprocess
import sys
import sysconfig

import pytest

from mujou._core import _native


class TestCoreModule:
    @pytest.mark.skipif(
        not sysconfig.get_config_var("Py_GIL_DISABLED"),
        reason="requires a free-threaded build",
    )
    def test_import_keeps_gil_disabled(self):
        # A fresh interpreter, so that no other extension has re-enabled the GIL.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, mujou._core._native; print(sys._is_gil_enabled())",
            ],
            capture_output=True,
            text=True,
            check=True,
            # PYTHON_GIL=0 would keep the GIL off whatever the module declares.
            env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
        )
        assert result.stdout.strip() == "False"

    @pytest.mark.parametrize(
        "name",
        [n for n in _native.__all__ if isinstance(getattr(_native, n), type)],
    )
    def test_every_class_belongs_to_this_module(self, name: str):
        assert getattr(_native, name).__module__ == _native.__name__

    def test_exceptions_keep_their_builtin_bases(self):
        assert issubclass(_native.NetworkError, RuntimeError)
        assert issubclass(_native.PauseTimeoutError, TimeoutError)
        assert str(_native.NetworkError("message")) == "message"
