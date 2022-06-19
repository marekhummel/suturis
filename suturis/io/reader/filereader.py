from typing import Tuple

import cv2
from numpy.typing import ArrayLike
from suturis.io.reader.basereader import BaseReader


class FileReader(BaseReader):
    capture: cv2.VideoCapture

    def __init__(self, path: str, skip=0, single_frame=False) -> None:
        super().__init__()
        self.capture = cv2.VideoCapture(path)

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        for _ in range(int(fps * skip)):
            self.capture.read()

        self.single_frame = self.capture.read()[1] if single_frame else None

    async def read_image(self) -> Tuple[bool, ArrayLike]:
        if not self.capture.isOpened():
            return False, None

        if self.single_frame is not None:
            return True, self.single_frame

        success, frame = self.capture.read()
        if not success:
            return False, None

        return True, frame

