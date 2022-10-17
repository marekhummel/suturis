class BaseDebuggingHandler:
    """Base class which just extends a class by a flag to enable debug outputs"""

    _debugging_enabled: bool
    _path: str

    def __init__(self) -> None:
        """Create new instance, flag set to False."""
        self._debugging_enabled = False
        self._path = "data/out/debug/"

    def enable_debug_outputs(self):
        """Enable the flag."""
        self._debugging_enabled = True
