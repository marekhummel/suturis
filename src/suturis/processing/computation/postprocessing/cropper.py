import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.postprocessing.base_postprocessor import BasePostprocessor
from suturis.typing import Image


class Cropper(BasePostprocessor[tuple[int, int, int, int]]):
    """Postprocessor which crops back areas from the image"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates new cropping postprocessor.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Cropping postprocessor")
        super().__init__(*args, **kwargs)

    def process(self, image: Image) -> Image:
        """Crop image.

        Parameters
        ----------
        image: Image
            Stitched image, may be modified by previous postprocessors

        Returns
        -------
        Image
            Modified image.
        """
        log.debug("Crop image")

        height, width = image.shape[:2]
        if not self._caching_enabled or self._cache is None:
            # TODO: Refactor
            xmin, xmax = 0, width - 1
            ymin, ymax = 0, height - 1

            is_black = np.all(image[:, :] == [0, 0, 0], axis=2)
            black_ratio_rows = np.average(is_black.astype(np.float64), axis=1)
            black_ratio_cols = np.average(is_black.astype(np.float64), axis=0)

            borders = [[xmin, 1, 1], [xmax, 1, -1], [ymin, 0, 1], [ymax, 0, -1]]

            ratios = [black_ratio_rows, black_ratio_cols]
            while True:
                best = None
                for border_index, (index, ratio_index, _) in enumerate(borders):
                    curr_ratio = ratios[ratio_index][index]
                    if best is None or curr_ratio > best[1]:
                        best = border_index, curr_ratio

                if not best or best[1] < 0.1:
                    break

                borders[best[0]][0] += borders[best[0]][2]

            xmin, xmax, ymin, ymax = [b[0] for b in borders]
            self._cache = (xmin, xmax, ymin, ymax)

        xmin, xmax, ymin, ymax = self._cache
        cropped = image[ymin : ymax + 1, xmin : xmax + 1]
        log.debug(f"Cropped image from {(0, width - 1)} to {(xmin, xmax)} and {(0, height - 1)} to {(ymin, ymax)}")
        return Image(cropped.astype(np.uint8))
