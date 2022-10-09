import logging as log

import numpy as np
from suturis.processing.computation.homography.base_homography_handler import BaseHomographyHandler
import suturis.processing.computation.manager as mgr
from suturis.processing.computation.masking.base_masking_handler import BaseMaskingHandler
from suturis.timer import track_timings
from suturis.typing import Image


_homography_delegate: BaseHomographyHandler | None = None
_masking_delegate: BaseMaskingHandler | None = None
_default_image: Image = Image(np.zeros(shape=(720, 1280, 3)))


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
        return _default_image

    # ** Stitch
    log.debug("Stitch images")
    transformation, crop, mask = params
    img1_tf, img2_tf = _homography_delegate.apply_transformations(image1, image2, *transformation)
    img1_tf_crop, img2_tf_crop = _homography_delegate.apply_crop(img1_tf, img2_tf, *crop)
    masked_img = _masking_delegate.apply_mask(img1_tf_crop, img2_tf_crop, mask)

    return masked_img


def set_delegates(homography: BaseHomographyHandler, masking: BaseMaskingHandler):
    global _homography_delegate, _masking_delegate
    _homography_delegate = homography
    _masking_delegate = masking
