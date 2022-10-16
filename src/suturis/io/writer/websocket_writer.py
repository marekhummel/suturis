import logging as log
from typing import Any

from suturis.io.writer.basewriter import BaseWriter
from suturis.typing import Image


class WebSocketWriter(BaseWriter):
    """Writer class to write into websockets. Note: Not implemented so far."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates new websocket writer instance.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init websocket output")
        super().__init__(*args, **kwargs)

    def write_image(self, image: Image) -> None:
        """Writes image to websocket.

        Parameters
        ----------
        image : Image
            The image to write
        """
        return super().write_image(image)
