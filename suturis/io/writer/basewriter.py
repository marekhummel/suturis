import numpy as np


class BaseWriter:
    index: int

    def __init__(self, index: int, /) -> None:
        self.index = index

    def write_image(self, image: np.ndarray) -> None:
        raise NotImplementedError("Abstract method needs to be overriden")
