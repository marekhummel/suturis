from suturis.io.writer.basewriter import BaseWriter
from suturis.typing import Image


class WebSocketWriter(BaseWriter):
    def __init__(self, index: int, /, source: str) -> None:
        super().__init__(index, source)

    def write_image(self, image: Image) -> None:
        return super().write_image(image)
