import logging as log

import cv2
from suturis.io.reader.basereader import BaseReader, _ReadImageType
from suturis.typing import Image


class RtspReader(BaseReader):
    _capture: cv2.VideoCapture

    def __init__(self, index: int, /, uri: str) -> None:
        log.debug("Init rtsp reader from {uri}")
        super().__init__(index)
        self._capture = cv2.VideoCapture(uri)

    def read_image(self) -> _ReadImageType:
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
