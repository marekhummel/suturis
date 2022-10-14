from suturis.io.writer.basewriter import BaseWriter
from suturis.typing import Image


class WebSocketWriter(BaseWriter):
    """Writer class to write into websockets. Note: Not implemented so far."""

    def __init__(self, index: int, /, source: str) -> None:
        """Creates new websocket writer instance.

        Parameters
        ----------
        index : int
            0-based index of this writer. Given implicitly by list index in config
        source : str
            Describes what image to write, has to be member of SourceImage
        """
        super().__init__(index, source)

    def write_image(self, image: Image) -> None:
        """Writes image to websocket.

        Parameters
        ----------
        image : Image
            The image to write
        """
        return super().write_image(image)
