import cv2
from numpy.typing import ArrayLike
from suturis.io.writer.basewriter import BaseWriter


class ScreenOutput(BaseWriter):
    def write_image(self, image: ArrayLike) -> None:
        cv2.imshow('Frame', image)
