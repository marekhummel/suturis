import logging as log

import numpy as np
from suturis.processing.computation.homography.base_homography_handler import BaseHomographyHandler
import suturis.processing.computation.manager as mgr
from suturis.processing.computation.masking.base_masking_handler import BaseMaskingHandler
from suturis.timer import track_timings
from suturis.typing import Image


_homography_delegate: BaseHomographyHandler | None = None
_masking_delegate: BaseMaskingHandler | None = None


@track_timings(name="Stitching")
def compute(*images: Image) -> Image:
    assert _homography_delegate and _masking_delegate
    assert len(images) == 2
    image1, image2 = images

    # ** Get data
    log.debug("Fetch current stitching params")
    params = mgr.get_params(image1, image2, _homography_delegate, _masking_delegate)

    if params is None:
        log.debug("Initial computation hasn't finished yet, return black image")
        return np.zeros_like(image1)

    # ** Stitch
    log.debug("Stitch images")
    img1_translated, img2_warped = _homography_delegate.apply_transformations(image1, image2, *params[0])
    masked_img = _masking_delegate.apply_mask(img1_translated, img2_warped, params[1])

    return masked_img


def set_delegates(homography: BaseHomographyHandler, masking: BaseMaskingHandler):
    global _homography_delegate, _masking_delegate
    _homography_delegate = homography
    _masking_delegate = masking
