import logging as log
import time
from typing import Any

import cv2
from suturis.io.reader.basereader import BaseReader, _ReadImageType
from suturis.typing import Image


class FileReader(BaseReader):
    """Reader class that reads its images from a video file."""

    _capture: cv2.VideoCapture
    _last_read: float | None
    _frame_time: float
    _single_frame: Image | None

    def __init__(
        self, *args: Any, path: str, skip: float = 0.0, speed_up: float = 1.0, single_frame: bool = False, **kwargs: Any
    ) -> None:
        """Creates new file reader.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        path : str
            Path to the video file
        skip : float, optional
            Seconds to skip at start of this video (to sync with other input), by default 0
        speed_up : float, optional
            Factor to multiply fps by, by default 1
        single_frame : bool, optional
            Flag to freeze video, if set, the first frame will always be returned, by default False
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}

        Raises
        ------
        FileNotFoundError
            Raised if path not existing or not readable by cv2
        """
        log.debug(f"Init file reader from {path} skipping {skip} seconds and accelerate fps by {speed_up}")
        super().__init__(*args, **kwargs)

        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            log.error("Opening capture failed, probably due to a invalid path")
            raise FileNotFoundError(path)

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frame_time = 1.0 / (fps * speed_up)
        for _ in range(int(fps * skip)):
            self._capture.read()

        self._last_read = None
        self._single_frame = Image(self._capture.read()[1]) if single_frame else None

    def _read_image(self) -> _ReadImageType:
        """Returns next frame in video capture or first frame if single_frame was set.

        Returns
        -------
        _ReadImageType
            Either True and the frame or False and None
        """
        log.debug(f"Reading image from reader #{self.index}")

        # Check availablity of capture
        if not self._capture.isOpened():
            log.info(f"Trying to read from closed capture in reader #{self.index}, return")
            return False, None

        # Return first frame if single_frame is set
        if self._single_frame is not None:
            return True, self._single_frame

        # Read frame
        success, frame = self._capture.read()
        if not success:
            log.info(f"Reading image failed in reader #{self.index}, return")
            return False, None

        # Sleep to match fps
        now = time.perf_counter()
        if self._last_read and (now - self._last_read) < self._frame_time:
            delay = self._frame_time - (now - self._last_read)
            log.debug(f"Delay read of reader #{self.index} by {delay:.6f}s to match fps")
            time.sleep(delay)

        # Return
        log.debug(f"Reading image from reader #{self.index} successful")
        self._last_read = time.perf_counter()
        return True, Image(frame)
