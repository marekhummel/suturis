import numpy as np
import enum


class SourceImage(enum.IntEnum):
    INPUT_ONE = 0
    INPUT_TWO = 1
    OUTPUT = 2


class BaseWriter:
    index: int
    source: SourceImage

    def __init__(self, index: int, /, source: str) -> None:
        self.index = index
        self.source = SourceImage[source]

    def write_image(self, _: np.ndarray) -> None:
        raise NotImplementedError("Abstract method needs to be overriden")
