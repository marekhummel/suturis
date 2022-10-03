import enum

from suturis.typing import Image


class SourceImage(enum.IntEnum):
    OUTPUT = 0
    INPUT_ONE = 1
    INPUT_TWO = 2


class BaseWriter:
    index: int
    source: SourceImage

    def __init__(self, index: int, /, source: str) -> None:
        self.index = index
        self.source = SourceImage[source]

    def write_image(self, _: Image) -> None:
        raise NotImplementedError("Abstract method needs to be overriden")
