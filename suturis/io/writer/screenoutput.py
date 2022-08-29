import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter

import logging as log


class ScreenOutput(BaseWriter):
    def __init__(self, title: str = None) -> None:
        self.title = title if title else "Current Frame"

    def write_image(self, image: np.ndarray) -> None:
        log.debug("Display image in window '%s'", self.title)
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, image.astype(np.uint8))
