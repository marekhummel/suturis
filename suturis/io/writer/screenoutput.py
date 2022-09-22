import logging as log

import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter


class ScreenOutput(BaseWriter):
    title: str

    def __init__(self, index: int, /, title: str | None = None) -> None:
        super().__init__(index)
        self.title = title or "Current Frame"

    def write_image(self, image: np.ndarray) -> None:
        log.debug("Display image in window '%s'", self.title)
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, image.astype(np.uint8))
