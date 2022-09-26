from datetime import datetime
from os import makedirs
from os.path import isdir, join
from time import time
import cv2
import numpy as np
from suturis.io.writer.basewriter import BaseWriter, SourceImage

import logging as log


class FileWriter(BaseWriter):
    _writer: cv2.VideoWriter
    _dimensions: tuple[int, int]
    _last_frame: np.ndarray | None
    _last_write_time: int
    _fps: int = 30

    def __init__(
        self,
        index: int,
        /,
        source: str = SourceImage.OUTPUT.name,
        *,
        dimensions: tuple[int, int],
        target_dir: str = "data/out/",
        filename: str = "{date}_stitching.mp4",
    ) -> None:
        log.debug(f"Init file writer #{index} with dimensions {dimensions} to save {source} images")
        super().__init__(index, source)

        if not isdir(target_dir):
            makedirs(target_dir)

        if not filename.endswith(".mp4"):
            log.warning(f"File writer #{index} got invalid filename (no mp4), using default name now")
            filename = "stitching_{date}.mp4"
        filename = filename.replace("{date}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        target = join(target_dir, filename)
        self._dimensions = dimensions
        self._writer = cv2.VideoWriter(target, fourcc, self._fps, dimensions)
        self._last_write_time = 0
        log.info(f"Target file of file writer #{index} is at '{target}'")

    def write_image(self, image: np.ndarray) -> None:
        log.debug(f"Writing image with writer #{self.index}")
        image = image.astype(np.uint8)
        if image.shape[1::-1] != self._dimensions:
            image = cv2.resize(image, dsize=self._dimensions, interpolation=cv2.INTER_CUBIC)

        # Write last image repeatedly to match fps of input / processing
        if self._last_write_time != 0:
            frame_time = 1 / self._fps
            current = self._last_write_time + frame_time
            while current < time():
                self._writer.write(self._last_frame)
                current += frame_time

        # Write and update refs
        self._writer.write(image)
        self._last_write_time = time()
        self._last_frame = image
