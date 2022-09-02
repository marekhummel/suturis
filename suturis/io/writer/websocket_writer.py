from suturis.io.writer.basewriter import BaseWriter


class WebSocketWriter(BaseWriter):
    def __init__(self, index, /) -> None:
        super().__init__(index)

    def write_image(self, image) -> None:
        return super().write_image(image)
