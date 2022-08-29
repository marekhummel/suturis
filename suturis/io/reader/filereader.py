from typing import Tuple
from time import time

import cv2
import asyncio
from numpy.typing import ArrayLike
from suturis.io.reader.basereader import BaseReader
import logging as log


class FileReader(BaseReader):
    capture: cv2.VideoCapture
    last_read: float

    def __init__(self, path: str, *, skip=0, speed_up=1, single_frame=False) -> None:
        log.debug(
            "Init file reader from %s skipping %f seconds and accelerate fps by %f",
            path,
            skip,
            speed_up,
        )
        super().__init__()
        self.capture = cv2.VideoCapture(path)

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / (fps * speed_up)
        for _ in range(int(fps * skip)):
            self.capture.read()

        self.last_read = None
        self.single_frame = self.capture.read()[1] if single_frame else None

    async def read_image(self) -> Tuple[bool, ArrayLike]:
        log.debug("Reading image")
        if not self.capture.isOpened():
            log.info("Trying to read from closed capture, return")
            return False, None

        if self.single_frame is not None:
            return True, self.single_frame

        now = time()
        if self.last_read and (now - self.last_read) < self.frame_time:
            await asyncio.sleep(self.last_read + self.frame_time - now)

        success, frame = self.capture.read()
        if not success:
            log.info("Reading image failed, return")
            return False, None

        log.debug("Reading image successful")
        self.last_read = time()
        return True, frame
