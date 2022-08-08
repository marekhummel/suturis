from os import makedirs
from os.path import isdir
from suturis.io.writer.basewriter import BaseWriter
import cv2


class FileWriter(BaseWriter):
    def __init__(self, subdir: str, name: str) -> None:
        super().__init__()
        self.dir = subdir
        self.name = name
        self.counter = 0

        if not isdir(f"out/{self.dir}/"):
            makedirs(f"out/{self.dir}/")

    async def write_image(self, image) -> None:
        cv2.imwrite(f"out/{self.dir}/{self.name}_{self.counter}.png", image)
        self.counter += 1
