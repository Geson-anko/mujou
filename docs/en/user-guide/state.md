# Saving and Restoring State

!!! note "To be written"

    Saving with `h.state_dict()` and restoring with `mujou.run(..., state=...)`. The default `load_state_dict` copies in place; override `state_dict` and `load_state_dict` for state that is neither a tensor nor a module. Why Mujou does not save the state after an exception.
