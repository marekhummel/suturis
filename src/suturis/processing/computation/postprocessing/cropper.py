import logging as log
from typing import Any

import numpy as np
from suturis.processing.computation.postprocessing.base_postprocessor import BasePostprocessor
from suturis.typing import Image


class Cropper(BasePostprocessor[tuple[int, int, int, int]]):
    """Postprocessor which crops back areas from the image"""

    threshold: float

    def __init__(self, *args: Any, threshold: float = 0.02, **kwargs: Any) -> None:
        """Creates new cropping postprocessor.

        Parameters
        ----------
        *args : Any, optional
            Positional arguments passed to base class, by default []
        threshold : float, optional
            Threshold to use for cropping. Means line has to have at least threshold * 100% black pixels, to be removed
        **kwargs : dict, optional
            Keyword params passed to base class, by default {}
        """
        log.debug("Init Cropping postprocessor")
        super().__init__(*args, **kwargs)
        self.threshold = threshold

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
            # Repeatedly cut off one of the outer edges until there's almost no black left

            # Define bool array which marks all black pixels
            is_black = np.all(image[:, :] == [0, 0, 0], axis=2).astype(np.float64)

            # Starting point are the four outer edges (line index, axis in image, increment / decrement)
            edges = [[0, 1, 1], [width - 1, 1, -1], [0, 0, 1], [height - 1, 0, -1]]

            # Iterate while edges can be removed
            while True:
                # Find edge with highest black pixel ratio
                best = None
                for edge_index, (index, axis, _) in enumerate(edges):
                    # Take line, crop of pixels that are already removed, compute ratio
                    other_dims = [e[0] for e in edges if e[1] != axis]
                    line_ratio = np.average(np.take(is_black, index, axis)[other_dims[0] : other_dims[1] + 1])

                    # Find arg max
                    if best is None or line_ratio > best[1]:
                        best = edge_index, line_ratio

                # If best line doesn't exceed threshold, abort
                if not best or best[1] < self.threshold:
                    break

                # Move edge according to increment
                edges[best[0]][0] += edges[best[0]][2]

            xmin, xmax, ymin, ymax = [b[0] for b in edges]
            self._cache = (xmin, xmax, ymin, ymax)

        xmin, xmax, ymin, ymax = self._cache
        cropped = image[ymin : ymax + 1, xmin : xmax + 1]
        log.debug(f"Cropped image from {(0, width - 1)} to {(xmin, xmax)} and {(0, height - 1)} to {(ymin, ymax)}")
        return Image(cropped.astype(np.uint8))
