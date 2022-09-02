import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter

import logging as log


class ScreenOutput(BaseWriter):
    def __init__(self, index, /, title: str = None) -> None:
        super().__init__(index)
        self.title = title or "Current Frame"

    def write_image(self, image: np.ndarray) -> None:
        log.debug("Display image in window '%s'", self.title)
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, image.astype(np.uint8))
