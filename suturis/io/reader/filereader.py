import logging as log
import time
from typing import Tuple

import cv2
import numpy as np
from suturis.io.reader.basereader import BaseReader


class FileReader(BaseReader):
    capture: cv2.VideoCapture
    last_read: float

    def __init__(
        self, index: int, /, path: str, *, skip=0, speed_up=1, single_frame=False
    ) -> None:
        log.debug(
            f"Init file reader #{index} from {path} skipping {skip} seconds and accelerate fps by {speed_up}"
        )
        super().__init__(index)
        self.capture = cv2.VideoCapture(path)

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / (fps * speed_up)
        for _ in range(int(fps * skip)):
            self.capture.read()

        self.last_read = None
        self.single_frame = self.capture.read()[1] if single_frame else None

    def read_image(self) -> Tuple[bool, np.ndarray]:
        log.debug(f"Reading image from reader #{self.index}")
        if not self.capture.isOpened():
            log.info(
                f"Trying to read from closed capture in reader #{self.index}, return"
            )
            return False, None

        if self.single_frame is not None:
            return True, self.single_frame

        success, frame = self.capture.read()
        if not success:
            log.info(f"Reading image failed in reader #{self.index}, return")
            return False, None

        now = time.perf_counter()
        if self.last_read and (now - self.last_read) < self.frame_time:
            delay = self.frame_time - (now - self.last_read)
            log.debug(f"Delay read of reader {self.index} by {delay:.6f}s to match fps")
            time.sleep(delay)

        log.debug(f"Reading image from reader #{self.index} successful")
        self.last_read = time.perf_counter()
        return True, frame
