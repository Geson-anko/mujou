import mujou


class TestPublicNames:
    def test_exports(self):
        # The top-level names of specs/network.md; a change here is a change of the
        # public API.
        assert sorted(mujou.__all__) == [
            "Channel",
            "DictSpec",
            "Graph",
            "Group",
            "Network",
            "NetworkError",
            "NextPort",
            "Node",
            "Port",
            "PrevPort",
            "Run",
            "StateDict",
            "TensorSpec",
            "__version__",
            "run",
        ]

    def test_time_is_a_submodule_outside_all(self):
        # A star import must not shadow the standard `time`.
        assert "time" not in mujou.__all__
        assert mujou.time.__name__ == "mujou.time"

    def test_typing_is_a_submodule_outside_all(self):
        # A star import must not shadow the standard `typing`.
        assert "typing" not in mujou.__all__
        assert mujou.typing.__name__ == "mujou.typing"


class TestPcPublicNames:
    def test_exports(self):
        # The names of specs/pc-module.md; a change here is a change of the
        # public API.
        import mujou.pc

        assert sorted(mujou.pc.__all__) == [
            "Activity",
            "Input",
            "Layer",
            "LearnClock",
            "Output",
            "Prediction",
            "Sample",
            "Tick",
            "TimeConstants",
            "activity_spec",
            "cross_entropy",
            "functional",
            "mse",
        ]
        assert sorted(mujou.pc.functional.__all__) == [
            "learn_",
            "loss_vjp",
            "relax_",
            "vjp",
        ]

    def test_pc_is_a_submodule_outside_the_top_level(self):
        assert "pc" not in mujou.__all__
