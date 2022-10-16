import logging as log
from typing import Any

import cv2
from suturis.io.reader.basereader import BaseReader, _ReadImageType
from suturis.typing import Image


class RtspReader(BaseReader):
    """Reader class to read from RTSP streams. Note: Untested so far."""

    _capture: cv2.VideoCapture

    def __init__(self, *args: Any, uri: str, **kwargs: Any) -> None:
        """Creates new RTSP reader.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        uri : str
            URI to stream
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init rtsp reader from {uri}")
        super().__init__(*args, **kwargs)
        self._capture = cv2.VideoCapture(uri)

    def read_image(self) -> _ReadImageType:
        """Returns next image sent in the stream.

        Returns
        -------
        _ReadImageType
            Either true and the frame or false and None
        """
        log.debug("Reading image")
        if not self._capture.isOpened():
            log.info("Trying to read from closed capture, return")
            return False, None

        success, frame = self._capture.read()
        if not success:
            log.info("Reading image failed, return")
            return False, None

        log.debug("Reading image successful")
        return True, Image(frame)
