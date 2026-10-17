"""The implementation of mujou. Private: ``mujou`` re-exports the public names.

``_native`` is the extension module built from the Rust core (``crates/``). It is
the only module here that is promised not to depend on torch; the others may use
it.
"""
