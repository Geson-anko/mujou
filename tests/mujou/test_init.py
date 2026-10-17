import mujou


class TestPublicNames:
    def test_exports(self):
        # The top-level names of specs/network.md; a change here is a change of the
        # public API.
        assert sorted(mujou.__all__) == [
            "Channel",
            "Compose",
            "DictSpec",
            "Graph",
            "Network",
            "NetworkError",
            "NextPort",
            "Node",
            "PauseTimeoutError",
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
