import numpy as np
import cv2

import suturis.processing.computation.manager as mgr


def compute(*images):
    assert len(images) == 2
    image1, image2 = images

    # ** Get data
    params = mgr.get_params(image1, image2)

    if params is None:
        return np.zeros_like(image1)

    # Unpack
    img1_translated = params.translated_base
    img2_warped = params.warped_query
    stitch_mask = params.stitch_mask
    seam_start_x, seam_start_y, seam_end_x, seam_end_y = params.seam_corners

    # ** Stitch
    masked_img1 = stitch_mask * img1_translated.astype(np.float64)
    masked_img2 = (1 - stitch_mask) * img2_warped.astype(np.float64)
    masked_img = masked_img1 + masked_img2

    # Seam

    seammat = params.seam_matrix
    seam_line = np.zeros(shape=masked_img.shape[:2], dtype=bool)
    seam_line[seam_start_y:seam_end_y, seam_start_x:seam_end_x] = seammat
    masked_img[seam_line] = [0, 255, 0]

    # Crop
    cv2.circle(masked_img, (seam_start_x, seam_start_y), 4, [255, 0, 0], -1)
    cv2.circle(masked_img, (seam_end_x, seam_end_y), 4, [255, 0, 0], -1)
    masked_img = masked_img[seam_start_y:seam_end_y, seam_start_x:seam_end_x]

    return masked_img
