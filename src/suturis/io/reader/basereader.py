from typing import Literal, Union

from suturis.typing import Image


_ReadImageType = Union[tuple[Literal[True], Image], tuple[Literal[False], None]]


class BaseReader:
    index: int

    def __init__(self, index: int, /) -> None:
        self.index = index

    def read_image(self) -> _ReadImageType:
        raise NotImplementedError("Abstract method needs to be overriden")
