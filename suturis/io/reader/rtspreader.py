import logging as log
from typing import Tuple

import cv2
from suturis.io.reader.basereader import BaseReader
import numpy as np


class RtspReader(BaseReader):
    capture: cv2.VideoCapture

    def __init__(self, uri: str) -> None:
        log.debug("Init rtsp reader from {uri}")
        super().__init__()
        self.capture = cv2.VideoCapture(uri)

    async def read_image(self) -> Tuple[bool, np.ndarray]:
        log.debug("Reading image")
        if not self.capture.isOpened():
            log.info("Trying to read from closed capture, return")
            return False, None

        success, frame = self.capture.read()
        if not success:
            log.info("Reading image failed, return")
            return False, None

        log.debug("Reading image successful")
        return True, frame
