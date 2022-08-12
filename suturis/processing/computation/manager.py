import suturis.processing.computation._homography as hg
import suturis.processing.computation._seaming as seam
import suturis.processing.computation._masking as mask
import multiprocessing as mp
import threading
from suturis.processing.computation.memory import LocalParams, Param, SharedParams
from typing import Optional

background_running = False
local_params = None


def get_params(image1, image2) -> Optional[LocalParams]:
    global background_running, local_params

    if not background_running:
        shared_params = SharedParams()
        proc = mp.Process(
            target=_compute, args=(image1, image2, shared_params), daemon=True
        )
        background_running = True
        proc.start()

        watcher = threading.Thread(
            target=_computation_watcher, args=(proc, shared_params), daemon=True
        )
        watcher.start()

    # Return (is None until first computation has been completed)
    return local_params


def _computation_watcher(process, shared_params):
    global background_running, local_params
    process.join()

    local_params = LocalParams(shared_params)
    background_running = False


def _compute(image1, image2, shared_params):
    # ** Warping
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

    # ** Seam calculation
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
    # seammat = np.zeros(
    #     (seam_end[0] - seam_start[0], seam_end[1] - seam_start[1]), dtype=bool
    # )
    # seammat[seammat.shape[0] // 2, :] = True

    # ** Create mask
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

    # Set computed values in shared memory
    shared_params.set_value(Param.TranslatedBase, img1_translated)
    shared_params.set_value(Param.WarpedQuery, img2_warped)
    shared_params.set_value(Param.StitchingMask, stitch_mask)
    shared_params.set_value(Param.SeamCorners, (seam_start, seam_end))
    shared_params.set_value(Param.SeamMatrix, seammat)
    shared_params.close()
