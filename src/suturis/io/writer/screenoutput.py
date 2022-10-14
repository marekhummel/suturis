import logging as log

import cv2
from suturis.io.writer.basewriter import BaseWriter, SourceImage
from suturis.typing import Image


class ScreenOutput(BaseWriter):
    """Writer class to output images to the screen in a opencv window"""

    title: str

    def __init__(self, index: int, /, source: str = SourceImage.OUTPUT.name, *, title: str = "Current Frame") -> None:
        """Creates new screen output writer.

        Parameters
        ----------
        index : int
            0-based index of this writer. Given implicitly by list index in config
        source : str, optional
            Describes what image to write, has to be member of SourceImage, by default "OUTPUT"
        title : str | None, optional
            Title of the window to be displayed, by default "Current Frame"
        """
        log.debug(f"Init screen output to display {source} images")
        super().__init__(index, source)
        self.title = title

    def write_image(self, image: Image) -> None:
        """Display image in opencv window.

        Parameters
        ----------
        image : Image
            The image to display
        """
        log.debug("Display image in window '%s'", self.title)
        cv2.namedWindow(self.title)
        cv2.imshow(self.title, image)
