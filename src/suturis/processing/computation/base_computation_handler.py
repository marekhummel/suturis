from typing import TypeVar, Generic


T = TypeVar("T")


class BaseComputationHandler(Generic[T]):
    """Base class which just extends a class by a flag to enable debug outputs and caching"""

    _caching_enabled: bool
    _cache: T | None
    _debugging_enabled: bool
    _debug_path: str

    def __init__(self, *, caching_enabled: bool) -> None:
        """Create new instance, flag set to False."""
        self._caching_enabled = caching_enabled
        self._cache = None
        self._debugging_enabled = False
        self._debug_path = "data/out/debug/"

    def enable_debug_outputs(self) -> None:
        """Enable the flag."""
        self._debugging_enabled = True
