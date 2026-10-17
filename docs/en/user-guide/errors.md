# Errors and Shutdown

!!! note "To be written"

    What happens when a hook raises: every node gets `on_exception` and then `on_stop`, and the host sees `NetworkError`. `mujou.run` itself raises the exceptions of `on_start`. The timeouts of stopping (30 seconds of real time by default).
