import logging as log

import cv2
import numpy as np
import suturis.processing.computation.manager as mgr
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.processing.computation.preprocessing import BasePreprocessor
from suturis.timer import track_timings
from suturis.typing import Image

_debug_outputs: bool = False
_preprocessor_delegates: list[BasePreprocessor] | None = None
_homography_delegate: BaseHomographyHandler | None = None
_masking_delegate: BaseMaskingHandler | None = None
_default_image: Image = Image(np.zeros(shape=(720, 1280, 3), dtype=np.uint8))


@track_timings(name="Stitching")
def compute(*images: Image) -> Image:
    """Stiches two images with the params given by the manager.

    Parameters
    ----------
    images : list[Image]
        List of images to stitch. Right now it has to be two images

    Returns
    -------
    Image
        Stitched image
    """
    assert _preprocessor_delegates is not None and _homography_delegate and _masking_delegate
    assert len(images) == 2
    image1, image2 = images

    # Get data
    log.debug("Fetch current stitching params")
    params = mgr.get_params(image1, image2, _preprocessor_delegates, _homography_delegate, _masking_delegate)

    if params is None:
        log.debug("Initial computation hasn't finished yet, return black image")
        return _default_image

    # Stitch
    log.debug("Stitch images")
    transformation, crop, mask = params

    for preprocessor in _preprocessor_delegates:
        image1, image2 = preprocessor.process(image1, image2)

    img1_tf, img2_tf = _homography_delegate.apply_transformations(image1, image2, *transformation)
    img1_tf_crop, img2_tf_crop = _homography_delegate.apply_crop(img1_tf, img2_tf, *crop)
    masked_img = _masking_delegate.apply_mask(img1_tf_crop, img2_tf_crop, mask)

    # Debug
    if _debug_outputs:
        log.debug("Write debug images of stitching")
        cv2.imwrite("data/out/debug/img1.jpg", image1)
        cv2.imwrite("data/out/debug/img2.jpg", image2)
        cv2.imwrite("data/out/debug/img1_transformed.jpg", img1_tf)
        cv2.imwrite("data/out/debug/img2_transformed.jpg", img2_tf)
        cv2.imwrite("data/out/debug/img1_cropped.jpg", img1_tf_crop)
        cv2.imwrite("data/out/debug/img2_cropped.jpg", img2_tf_crop)

    # Return
    return masked_img


def set_delegates(
    preprocessors: list[BasePreprocessor], homography: BaseHomographyHandler, masking: BaseMaskingHandler
) -> None:
    """Sets delegates at beginning of runtime.

    Parameters
    ----------
    preprocessors : list[BasePreprocessor]
        List of preprocessors
    homography : BaseHomographyHandler
        Homography handler
    masking : BaseMaskingHandler
        Masking handler
    """
    global _preprocessor_delegates, _homography_delegate, _masking_delegate
    _preprocessor_delegates = preprocessors
    _homography_delegate = homography
    _masking_delegate = masking


def enable_debug_outputs() -> None:
    """Enable debug outputs (various intermediate images for debugging)"""
    global _debug_outputs
    log.info("Debug outputs are enabled")
    _debug_outputs = True

    assert _preprocessor_delegates is not None and _homography_delegate and _masking_delegate
    for preprocessor in _preprocessor_delegates:
        preprocessor.enable_debug_outputs()
    _homography_delegate.enable_debug_outputs()
    _masking_delegate.enable_debug_outputs()
