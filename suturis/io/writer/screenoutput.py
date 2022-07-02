import cv2
from numpy.typing import ArrayLike
from suturis.io.writer.basewriter import BaseWriter


class ScreenOutput(BaseWriter):
    def __init__(self, title: str = None) -> None:
        self.title = title if title else 'Current Frame'

    async def write_image(self, image: ArrayLike) -> None:
        cv2.imshow(self.title, image)
