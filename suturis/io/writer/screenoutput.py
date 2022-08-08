import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter


class ScreenOutput(BaseWriter):
    def __init__(self, title: str = None) -> None:
        self.title = title if title else "Current Frame"

    async def write_image(self, image: np.ndarray) -> None:
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, image.astype(np.uint8))
