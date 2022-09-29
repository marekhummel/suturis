import enum

from suturis.typing import Image


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

    def write_image(self, _: Image) -> None:
        raise NotImplementedError("Abstract method needs to be overriden")
