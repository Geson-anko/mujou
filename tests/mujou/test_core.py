import subprocess
import sys
import sysconfig

import pytest


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
                "import sys, mujou._core; print(sys._is_gil_enabled())",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "False"
