import logging as log

import cv2
from suturis.io.writer.basewriter import BaseWriter, SourceImage
from suturis.typing import CvSize, Image


class ScreenOutput(BaseWriter):
    """Writer class to output images to the screen in a opencv window"""

    title: str
    window_size: CvSize | None

    def __init__(
        self,
        index: int,
        /,
        source: str = SourceImage.OUTPUT.name,
        *,
        title: str = "Current Frame",
        window_size: CvSize | None = None,
    ) -> None:
        """Creates new screen output writer.

        Parameters
        ----------
        index : int
            0-based index of this writer. Given implicitly by list index in config
        source : str, optional
            Describes what image to write, has to be member of SourceImage, by default "OUTPUT"
        title : str | None, optional
            Title of the window to be displayed, by default "Current Frame"
        window_size : CvSize | None, optional
            Size of the output window. If not given, size is dependent on stitching result, by default None
        """
        log.debug(f"Init screen output '{title}' to display {source} images")
        super().__init__(index, source)
        self.title = title
        self.window_size = window_size

    def write_image(self, image: Image) -> None:
        """Display image in opencv window.

        Parameters
        ----------
        image : Image
            The image to display
        """
        log.debug("Display image in window '%s'", self.title)
        cv2.namedWindow(self.title)

        if self.window_size:
            image = Image(cv2.resize(image, dsize=self.window_size, interpolation=cv2.INTER_CUBIC))

        cv2.imshow(self.title, image)
