import logging as log
import logging.handlers
import multiprocessing as mp
import multiprocessing.connection as mpc
from multiprocessing.synchronize import Event as EventType

from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.timer import track_timings, finalize_timings
from suturis.typing import ComputationParams, Image


proc_logger = log.getLogger()
homography_delegate: BaseHomographyHandler
masking_delegate: BaseMaskingHandler


def main(pipe_conn: mpc.Connection, shutdown_event: EventType, logging_queue: mp.Queue) -> None:
    global proc_logger, homography_delegate, masking_delegate
    # Set logging
    qh = logging.handlers.QueueHandler(logging_queue)
    proc_logger.setLevel(log.DEBUG)
    proc_logger.addHandler(qh)
    proc_logger.debug("Subprocess started")

    # Receive delegate classes
    homography_delegate = pipe_conn.recv()
    masking_delegate = pipe_conn.recv()

    while not shutdown_event.is_set():
        try:
            # Check if data is available
            if not pipe_conn.poll(0.25):
                continue

            # Get images and delegates
            proc_logger.debug("Receive images")
            image1: Image = pipe_conn.recv()
            image2: Image = pipe_conn.recv()

            # Compute
            proc_logger.debug("Compute params")
            warping_info, crop_area, mask = _compute_params(image1, image2)

            # Return
            proc_logger.debug("Return params")
            pipe_conn.send(warping_info)
            pipe_conn.send(crop_area)
            pipe_conn.send(mask)

        except (BrokenPipeError, EOFError):
            if not shutdown_event.is_set():
                log.error("Pipe error in daemon process, computation aborted.")
                continue
            break

    finalize_timings()


@track_timings(name="Raw computation")
def _compute_params(image1: Image, image2: Image) -> ComputationParams:
    global proc_logger, homography_delegate, masking_delegate

    # Warping
    proc_logger.debug("Compute warping")
    translation, canvas_size, homography = homography_delegate.find_homography(image1, image2)
    img1_translated, img2_warped = homography_delegate.apply_transformations(
        image1, image2, translation, canvas_size, homography
    )

    # Crop
    proc_logger.debug("Compute crop area")
    crop_area, crop_size = homography_delegate.find_crop(image1.shape, homography, translation)
    img1_translated_crop, img2_warped_crop = homography_delegate.apply_crop(img1_translated, img2_warped, *crop_area)

    # Mask calculation
    proc_logger.debug("Compute mask")
    mask = masking_delegate.compute_mask(img1_translated_crop, img2_warped_crop, crop_size)

    return ((translation, canvas_size, homography), crop_area, mask)
