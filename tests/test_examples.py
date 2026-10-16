"""Run the programs in examples/, so that the code the documentation shows keeps
working."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
# A program is examples/<name>.py or examples/<name>/main.py (examples/README.md).
PROGRAMS = sorted(EXAMPLES_DIR.glob("*.py")) + sorted(EXAMPLES_DIR.glob("*/main.py"))
TIMEOUT_SECONDS = 60


class TestExamples:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "program", PROGRAMS, ids=lambda p: str(p.relative_to(EXAMPLES_DIR))
    )
    def test_program_exits_successfully(self, program: Path) -> None:
        result = subprocess.run(
            [sys.executable, program.name],
            cwd=program.parent,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        assert result.returncode == 0, result.stderr
