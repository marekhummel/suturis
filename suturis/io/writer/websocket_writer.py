import numpy as np
from suturis.io.writer.basewriter import BaseWriter


class WebSocketWriter(BaseWriter):
    def __init__(self, index: int, /) -> None:
        super().__init__(index)

    def write_image(self, image: np.ndarray) -> None:
        return super().write_image(image)
