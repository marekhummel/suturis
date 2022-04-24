from typing import Tuple

import cv2
from numpy.typing import ArrayLike
from suturis.io.reader.basereader import BaseReader


class FileReader(BaseReader):
    capture: cv2.VideoCapture

    def __init__(self, path: str) -> None:
        super().__init__()
        self.capture = cv2.VideoCapture(path)

    def read_image(self) -> Tuple[bool, ArrayLike]:
        if not self.capture.isOpened():
            return False, None

        success, frame = self.capture.read()
        if not success:
            return False, None

        return True, frame
