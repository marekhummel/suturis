import logging as log
from datetime import datetime
from os import makedirs
from os.path import isdir, join
from time import time
from typing import Any

import cv2
from suturis.io.writer.basewriter import BaseWriter
from suturis.typing import CvSize, Image


class FileWriter(BaseWriter):
    """Write class that writes its images to a mp4 file."""

    _writer: cv2.VideoWriter
    _dimensions: CvSize
    _last_frame: Image | None
    _last_write_time: float
    _fps: int = 30

    def __init__(
        self,
        *args: Any,
        dimensions: CvSize,
        target_dir: str = "data/out/",
        filename: str = "{date}_stitching.mp4",
        **kwargs: Any,
    ) -> None:
        """Creates new file writer.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        dimensions : CvSize
            Dimensions of the video
        target_dir : str, optional
            Directory in which to save the file, by default "data/out/"
        filename : str, optional
            Filename to use, the substring "{date}" will be replaced with the current date and time,
            by default "{date}_stitching.mp4"
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug(f"Init file writer with dimensions {dimensions}")
        super().__init__(*args, **kwargs)

        # Create target dir
        if not isdir(target_dir):
            makedirs(target_dir)

        # Prepare filename
        if not filename.endswith(".mp4"):
            log.warning("File writer got invalid filename (no mp4), using default name now")
            filename = "stitching_{date}.mp4"
        filename = filename.replace("{date}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        # Create writer and relevant fields
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        target = join(target_dir, filename)
        self._dimensions = dimensions
        self._writer = cv2.VideoWriter(target, fourcc, self._fps, dimensions)
        self._last_write_time = 0
        log.info(f"Target file of file writer is at '{target}'")

    def write_image(self, image: Image) -> None:
        """Writes images to file

        Parameters
        ----------
        image : Image
            The frame to append.
        """
        log.debug(f"Writing image with writer #{self.index}")

        # Prepare image
        image_save = image
        if image.shape[1::-1] != self._dimensions:
            image_save = Image(cv2.resize(image_save, dsize=self._dimensions, interpolation=cv2.INTER_CUBIC))

        # Write last image repeatedly to match fps of input / processing
        if self._last_write_time != 0:
            assert self._last_frame is not None
            frame_time = 1 / self._fps
            current = self._last_write_time + frame_time
            while current < time():
                self._writer.write(self._last_frame)
                current += frame_time

        # Write and update refs
        self._writer.write(image)
        self._last_write_time = time()
        self._last_frame = Image(image_save)
