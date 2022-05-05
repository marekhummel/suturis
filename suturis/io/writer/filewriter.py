from suturis.io.writer.basewriter import BaseWriter


class FileWriter(BaseWriter):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    async def write_image(self, image) -> None:
        return super().write_image(image)
