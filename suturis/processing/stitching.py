import suturis.processing._homography as hg
import suturis.processing._seaming as seam
import suturis.processing._masking as mask
import numpy as np


async def compute(*images):
    assert len(images) == 2
    image1, image2 = images

    # Warping
    homography, _, _, _ = hg.find_homography_matrix(image1, image2)
    (
        h_translation,
        x_max,
        x_min,
        y_max,
        y_min,
        translation_dist,
        rows1,
        cols1,
    ) = hg.find_transformation(image1, image2, homography)
    img1_translated, img2_warped = hg.translate_and_warp(
        image1,
        image2,
        h_translation,
        x_max,
        x_min,
        y_max,
        y_min,
        homography,
        translation_dist,
        rows1,
        cols1,
    )

    # Seam calculation
    seam_start, seam_end = seam.find_important_pixels(image1, homography, h_translation)
    preferred_seam = [
        (x, img2_warped.shape[0] // 2) for x in range(seam_start[1], seam_end[1])
    ]
    modified_img = seam.prepare_img_for_seam_finding(
        img1_translated, img2_warped, preferred_seam, seam_start
    )
    seammat = seam.find_seam_dynamically(
        modified_img, img2_warped, seam_start, seam_end
    )

    # # Create mask
    x_trans = h_translation[0][2]
    y_trans = h_translation[1][2]
    stitch_mask = mask.create_binary_mask(
        seammat.copy(),
        img1_translated,
        img2_warped,
        seam_start[1],
        seam_start[0],
        x_trans,
        y_trans,
        image1.shape[1],
        image1.shape[0],
    )

    # Stitch
    masked_img1 = stitch_mask * img1_translated.astype(np.float64)
    masked_img2 = (1 - stitch_mask) * img2_warped.astype(np.float64)
    masked_img = masked_img1 + masked_img2

    seam_line = np.zeros(shape=masked_img.shape[:2], dtype=bool)
    seam_line[seam_start[0] : seam_end[0], seam_start[1] : seam_end[1]] = seammat
    masked_img[seam_line] = [0, 255, 0]

    return masked_img
