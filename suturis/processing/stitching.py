import logging as log

import cv2
import numpy as np
import suturis.processing.computation._homography as hg
import suturis.processing.computation.manager as mgr
from suturis.timer import track_timings


@track_timings(name="Stitching")
def compute(*images):
    assert len(images) == 2
    image1, image2 = images

    # ** Get data
    log.debug("Fetch current stitching params")
    params = mgr.get_params(image1, image2)

    if params is None:
        log.debug("Initial computation hasn't finished yet, return black image")
        return np.zeros_like(image1)

    # ** Stitch
    log.debug("Stitch images")
    img1_translated, img2_warped = hg.translate_and_warp(image1, image2, *(params[0]))
    stitch_mask = params[1]
    (seam_start_y, seam_start_x), (seam_end_y, seam_end_x) = params[2]

    masked_img1 = stitch_mask * img1_translated.astype(np.float64)
    masked_img2 = (1 - stitch_mask) * img2_warped.astype(np.float64)
    masked_img = masked_img1 + masked_img2

    # Seam
    log.debug("Draw seam")
    seammat = params[3]
    seam_line = np.zeros(shape=masked_img.shape[:2], dtype=bool)
    seam_line[seam_start_y:seam_end_y, seam_start_x:seam_end_x] = seammat
    masked_img[seam_line] = [0, 255, 0]

    # Crop
    log.debug("Crop image to relevant frame")
    cv2.circle(masked_img, (seam_start_x, seam_start_y), 4, [255, 0, 0], -1)
    cv2.circle(masked_img, (seam_end_x, seam_end_y), 4, [255, 0, 0], -1)
    masked_img = masked_img[seam_start_y:seam_end_y, seam_start_x:seam_end_x]

    return masked_img
