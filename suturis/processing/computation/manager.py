import logging as log
import logging.handlers
import multiprocessing as mp
import sys
import threading
from typing import Optional, Any

import suturis.processing.computation._homography as hg
import suturis.processing.computation._masking as mask
import suturis.processing.computation._seaming as seam

background_running = False
local_params = None


def get_params(image1, image2) -> Optional[Any]:
    global background_running, local_params

    if not background_running:
        log.debug("Recompute stitching params")

        # Prepare logging
        log_queue = mp.Queue()
        local, fork = mp.Pipe(duplex=True)
        proc = mp.Process(target=_compute_params, args=(fork, log_queue), daemon=True)

        watcher = threading.Thread(
            target=_computation_watcher,
            args=(proc, image1, image2, local, fork, log_queue),
            daemon=True,
        )
        watcher.start()
        background_running = True
        log.debug("Watcher started")

    # Return (is None until first computation has been completed)
    log.debug("Return params")
    return local_params


def _computation_watcher(process, image1, image2, pipe_local, pipe_fork, log_queue):
    global background_running, local_params

    try:
        # Logging
        log.debug("Setup logging")
        ql = logging.handlers.QueueListener(
            log_queue, *log.getLogger().handlers, respect_handler_level=True
        )
        ql.start()

        # Start and send data
        log.debug("Start process and send data.")
        process.start()
        pipe_local.send(image1)
        pipe_local.send(image2)

        # Idle until results
        log.debug("Wait until results received.")
        warping_info = pipe_local.recv()
        seam_corners = pipe_local.recv()
        seammat = pipe_local.recv()
        stitch_mask = pipe_local.recv()
        process.join()

        # Update param object
        log.debug("Connection completed, update data.")
        pipe_local.close()
        pipe_fork.close()
        local_params = warping_info, stitch_mask, seam_corners, seammat

    except EOFError:
        log.error("Pipe error, update aborted.")
        return
    finally:
        background_running = False
        log.debug("Update complete.")
        ql.stop()


def _compute_params(pipe_conn, logging_queue):
    # ** Set logging
    qh = logging.handlers.QueueHandler(logging_queue)
    proc_logger = log.getLogger()
    proc_logger.setLevel(logging.DEBUG)
    proc_logger.addHandler(qh)
    proc_logger.debug("Updating process started")

    # ** Get images
    proc_logger.debug("Receive images")
    image1 = pipe_conn.recv()
    image2 = pipe_conn.recv()

    # ** Warping
    proc_logger.debug("Compute warping")
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
        x_max - x_min,
        y_max - y_min,
        homography,
        translation_dist,
        rows1,
        cols1,
    )
    pipe_conn.send(
        (
            h_translation,
            x_max - x_min,
            y_max - y_min,
            homography,
            translation_dist,
            rows1,
            cols1,
        )
    )

    # ** Seam calculation
    proc_logger.debug("Compute seam")
    seam_start, seam_end = seam.find_important_pixels(image1, homography, h_translation)
    preferred_seam = [
        # (x, img2_warped.shape[0] // 2) for x in range(seam_start[1], seam_end[1])
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
    pipe_conn.send((seam_start, seam_end))
    pipe_conn.send(seammat)

    # ** Create mask
    proc_logger.debug("Compute mask")
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
    pipe_conn.send(stitch_mask)
