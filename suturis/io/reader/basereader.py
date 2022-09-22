from typing import Literal, Tuple, Union
import numpy as np


_ReadImageType = Union[Tuple[Literal[True], np.ndarray], Tuple[Literal[False], None]]


class BaseReader:
    index: int

    def __init__(self, index: int, /) -> None:
        self.index = index

    def read_image(self) -> _ReadImageType:
        raise NotImplementedError("Abstract method needs to be overriden")
