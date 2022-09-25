import logging as log

import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter, SourceImage


class ScreenOutput(BaseWriter):
    title: str

    def __init__(self, index: int, /, source: str = SourceImage.OUTPUT.name, *, title: str | None = None) -> None:
        log.debug(f"Init screen output to display {source} images")
        super().__init__(index, source)
        self.title = title or "Current Frame"

    def write_image(self, image: np.ndarray) -> None:
        log.debug("Display image in window '%s'", self.title)
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, image.astype(np.uint8))
