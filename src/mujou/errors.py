"""The exceptions of mujou. A leaf: the native core raises them too."""


class NetworkError(RuntimeError):
    """The network stopped because a node raised.

    ``__cause__`` is the first exception a node raised. Once a run has stopped
    this way, the host's calls on it (``write``, ``read``, ``state_dict``,
    ``shutdown``, ...) raise ``NetworkError`` too.
    """

    __module__ = "mujou"
