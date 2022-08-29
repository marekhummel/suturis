from typing import Tuple
from time import time

import cv2
import asyncio
from numpy.typing import ArrayLike
from suturis.io.reader.basereader import BaseReader


class FileReader(BaseReader):
    capture: cv2.VideoCapture
    last_read: float

    def __init__(self, path: str, skip=0, single_frame=False) -> None:
        super().__init__()
        self.capture = cv2.VideoCapture(path)

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / fps
        for _ in range(int(fps * skip)):
            self.capture.read()

        self.last_read = None
        self.single_frame = self.capture.read()[1] if single_frame else None

    async def read_image(self) -> Tuple[bool, ArrayLike]:
        if not self.capture.isOpened():
            return False, None

        if self.single_frame is not None:
            return True, self.single_frame

        now = time()
        if self.last_read and (now - self.last_read) < self.frame_time:
            await asyncio.sleep(self.last_read + self.frame_time - now)

        success, frame = self.capture.read()
        if not success:
            return False, None

        self.last_read = time()
        return True, frame
