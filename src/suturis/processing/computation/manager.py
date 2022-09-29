import logging as log
import logging.handlers
import multiprocessing as mp
import multiprocessing.connection as mpc
import threading
from multiprocessing.synchronize import Event as EventType

import numpy as np
from suturis.processing.computation.homography.base_homography_handler import BaseHomographyHandler
from suturis.processing.computation.masking.base_masking_handler import BaseMaskingHandler
from suturis.timer import track_timings
from suturis.typing import ComputationParams, Image

_computation_running: bool = False
_shutdown_event: EventType = mp.Event()
_local_params: ComputationParams | None = None


def get_params(
    image1: Image,
    image2: Image,
    homography_handler: BaseHomographyHandler,
    masking_handler: BaseMaskingHandler,
) -> ComputationParams | None:
    global _computation_running, _local_params, _shutdown_event

    # Recompute params when possible
    if not _computation_running:
        log.debug("Recompute stitching params")

        # Prepare logging
        log_queue: mp.Queue = mp.Queue()
        local, fork = mp.Pipe(duplex=True)
        proc = mp.Process(target=_compute_params, args=(fork, _shutdown_event, log_queue), daemon=True)

        watcher = threading.Thread(
            target=_computation_watcher,
            args=(proc, image1, image2, homography_handler, masking_handler, local, fork, log_queue),
            daemon=True,
        )
        watcher.start()
        _computation_running = True
        log.debug("Watcher started")

    # Return
    log.debug("Return params")
    return _local_params


def shutdown() -> None:
    global _shutdown_event
    _shutdown_event.set()


@track_timings(name="Computation")
def _computation_watcher(
    process: mp.Process,
    image1: Image,
    image2: Image,
    homography_handler: BaseHomographyHandler,
    masking_handler: BaseMaskingHandler,
    pipe_local: mpc.Connection,
    pipe_fork: mpc.Connection,
    log_queue: mp.Queue,
) -> None:
    global _computation_running, _shutdown_event, _local_params

    try:
        # Logging
        log.debug("Setup logging")
        ql = logging.handlers.QueueListener(log_queue, *log.getLogger().handlers, respect_handler_level=True)
        ql.start()

        # Start and send data
        log.debug("Start process and send data")
        process.start()
        pipe_local.send(image1)
        pipe_local.send(image2)
        pipe_local.send(homography_handler)
        pipe_local.send(masking_handler)

        # Idle until results
        log.debug("Wait until results received")
        warping_info = pipe_local.recv()
        mask = pipe_local.recv()
        process.join()

        # Update param object
        log.debug("Connection completed, update data")
        pipe_local.close()
        pipe_fork.close()

        if _local_params is None:
            log.info("Initial computation of params completed")
        _local_params = warping_info, mask

    except EOFError:
        if not _shutdown_event.is_set():
            log.error("Pipe error, update aborted")
        return
    finally:
        _computation_running = False
        log.debug("Update complete.")
        ql.stop()


def _compute_params(pipe_conn: mpc.Connection, shutdown_event: EventType, logging_queue: mp.Queue) -> None:
    # ** Set logging
    qh = logging.handlers.QueueHandler(logging_queue)
    proc_logger = log.getLogger()
    proc_logger.setLevel(logging.DEBUG)
    proc_logger.addHandler(qh)
    proc_logger.debug("Updating process started")

    try:
        # ** Get images
        proc_logger.debug("Receive images")
        image1 = pipe_conn.recv()
        image2 = pipe_conn.recv()
        homography_delegate = pipe_conn.recv()
        masking_delegate = pipe_conn.recv()

        # ** Warping
        proc_logger.debug("Compute warping")
        translation, target_size, homography = homography_delegate.find_homography(image1, image2)
        img1_translated, img2_warped = homography_delegate.apply_transformations(
            image1, image2, target_size, translation, homography
        )
        pipe_conn.send((target_size, translation, homography))

        # ** Mask calculation
        proc_logger.debug("Compute mask")
        seam_area = homography_delegate.find_crop(image1, homography, translation)
        mask = masking_delegate.compute_mask(img1_translated, img2_warped, target_size, translation, seam_area)
        pipe_conn.send(mask)

    except BrokenPipeError:
        if not shutdown_event.is_set():
            log.error("Pipe error in daemon process, computation aborted.")
