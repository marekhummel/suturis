import suturis.processing.computation._homography as hg
import suturis.processing.computation._seaming as seam
import suturis.processing.computation._masking as mask
import multiprocessing as mp
import threading
import logging as log
import sys
from suturis.processing.computation.memory import LocalParams, Param, SharedParams
from typing import Optional

background_running = False
local_params = None


def get_params(image1, image2) -> Optional[LocalParams]:
    global background_running, local_params
    log.info("Getting params")

    if not background_running:
        log.info("Recompute stitching params")
        local, fork = mp.Pipe(duplex=True)
        proc = mp.Process(target=_compute, args=(fork,), daemon=True)
        background_running = True
        proc.start()
        log.info("Process started")

        watcher = threading.Thread(
            target=_computation_watcher, args=(proc, local, image1, image2), daemon=True
        )
        watcher.start()
        log.info("Watcher started")

    # Return (is None until first computation has been completed)
    log.info("Return params")
    return local_params


def _computation_watcher(process, pipe_conn, image1, image2):
    global background_running, local_params

    # Send data
    log.info("Watcher: Send images")
    pipe_conn.send(image1)
    pipe_conn.send(image2)

    # Block until receiving data
    log.info("Watcher: Receive params")
    img1_translated = pipe_conn.recv()
    img2_warped = pipe_conn.recv()
    stitch_mask = pipe_conn.recv()
    (seam_start, seam_end) = pipe_conn.recv()
    seammat = pipe_conn.recv()

    # Finish
    log.info("Watcher: Update params")
    process.join()
    local_params = LocalParams(
        img1_translated, img2_warped, stitch_mask, [*seam_start, *seam_end], seammat
    )
    background_running = False


def _compute(pipe_conn):
    log.basicConfig(level=log.INFO, handlers=[log.StreamHandler(sys.stdout)])
    log.info("Process: Go")

    # ** Get images
    log.info("Process: Receive images")
    image1 = pipe_conn.recv()
    image2 = pipe_conn.recv()

    # ** Warping
    log.info("Process: Compute warping")
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
    log.info("Process: Compute seam")
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
    log.info("Process: Compute mask")
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
    log.info("Process: Return params")
    pipe_conn.send(img1_translated)
    pipe_conn.send(img2_warped)
    pipe_conn.send(stitch_mask)
    pipe_conn.send((seam_start, seam_end))
    pipe_conn.send(seammat)
