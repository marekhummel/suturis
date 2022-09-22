from os import makedirs
from os.path import isdir

import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter


class FileWriter(BaseWriter):
    subdir: str
    name: str
    _counter: int

    def __init__(self, index: int, /, subdir: str, name: str) -> None:
        super().__init__(index)
        self.name = name
        self.subdir = subdir
        self._counter = 0

        if not isdir(f"out/{self.subdir}/"):
            makedirs(f"out/{self.subdir}/")

    def write_image(self, image: np.ndarray) -> None:
        cv2.imwrite(f"out/{self.subdir}/{self.name}_{self._counter}.png", image)
        self._counter += 1
