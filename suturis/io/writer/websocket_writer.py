from suturis.io.writer.basewriter import BaseWriter


class WebSocketWriter(BaseWriter):
    def write_image(self, image) -> None:
        return super().write_image(image)
