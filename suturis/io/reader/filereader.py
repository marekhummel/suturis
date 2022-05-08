from typing import Tuple

import cv2
import numpy as np
from numpy.typing import ArrayLike
from suturis.io.reader.basereader import BaseReader


class FileReader(BaseReader):
    capture: cv2.VideoCapture

    def __init__(self, path: str, delay=0) -> None:
        super().__init__()
        self.capture = cv2.VideoCapture(path)

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.delay = int(fps * delay)
        self.backup = np.zeros((height, width, 3), np.uint8)

    async def read_image(self) -> Tuple[bool, ArrayLike]:
        if not self.capture.isOpened():
            return False, None

        if self.delay > 0:
            self.delay -= 1
            return True, self.backup

        success, frame = self.capture.read()
        if not success:
            return False, None

        return True, frame
