import logging as log
from threading import Event, Lock, Thread
from typing import Literal, Union

from suturis.typing import Image

_ReadImageType = Union[tuple[Literal[True], Image], tuple[Literal[False], None]]


class BaseReader:
    """Abstract base class for any type of image input (aka a reader)."""

    index: int
    _current: Image | None
    _lock: Lock
    _cancellation_token: Event

    def __init__(self, index: int, /) -> None:
        """Creates new reader instance, should not be called explicitly only from subclasses.

        Parameters
        ----------
        index : int
            0-based index of this reader. Given implicitly by list index in config
        """
        log.debug(f"Init reader #{index}")
        self.index = index
        self._current = None

    def start(self, cancellation_token: Event) -> None:
        """Starts this reader, meaning it fetches frames as they're available.

        Parameters
        ----------
        cancellation_token : Event
            Threading event needed to finish thread
        """
        self._lock = Lock()
        self._cancellation_token = cancellation_token
        thread = Thread(target=self._fetch_images, args=(), daemon=True)
        thread.start()

    def get(self) -> Image | None:
        """Method used by executor to access current frame. Blocks until frame is available.

        Returns
        -------
        Image | None
            Frame if available or None if the thread finished.
        """
        # Stall while no image is preset or exit if event is triggered
        while self._current is None:
            if self._cancellation_token.is_set():
                return None

        # Return data and set memory to None
        with self._lock:
            if self._current is not None:
                log.debug(f"Frame in reader #{self.index} was overriden and not used for stitching.")

            data = self._current
            self._current = None
        return data

    def _fetch_images(self) -> None:
        """Main threaded loop to read images"""
        # Loop while event is not set
        while not self._cancellation_token.is_set():
            # Retrieve image
            success, frame = self._read_image()

            # Break if reader fails
            if not success:
                break

            # Update memory with new frame
            with self._lock:
                self._current = frame

        self._cancellation_token.set()

    def _read_image(self) -> _ReadImageType:
        """Abstract method for reading the next image.

        Raises
        ------
        NotImplementedError
            Unless overriden, this method will raise an error.
        """
        raise NotImplementedError("Abstract method needs to be overriden")
