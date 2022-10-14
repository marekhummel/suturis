import logging as log
import logging.handlers
import multiprocessing as mp
import multiprocessing.connection as mpc
from multiprocessing.synchronize import Event as EventType


from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.processing.computation.preprocessing import BasePreprocessor
from suturis.timer import timings, track_timings
from suturis.typing import ComputationParams, Image

proc_logger = log.getLogger()
preprocessors: list[BasePreprocessor]
homography_delegate: BaseHomographyHandler
masking_delegate: BaseMaskingHandler


def main(pipe_conn: mpc.PipeConnection, shutdown_event: EventType, logging_queue: mp.Queue) -> None:
    """Main loop for the subprocess. Initially receives handlers, then continously receive images and compute params.

    Parameters
    ----------
    pipe_conn : mpc.PipeConnection
        Connection used to communicate with main process
    shutdown_event : EventType
        Shutdown event used by the main process to signalize when to exit this process
    logging_queue : mp.Queue
        Queue needed for logging, so that the configured loggers can be used
    """
    global proc_logger, preprocessors, homography_delegate, masking_delegate
    # Set logging
    qh = logging.handlers.QueueHandler(logging_queue)
    proc_logger.setLevel(log.DEBUG)
    proc_logger.addHandler(qh)
    proc_logger.debug("Subprocess started")

    # Receive delegate classes
    proc_logger.debug("Receive handlers")
    preprocessors = pipe_conn.recv()
    homography_delegate = pipe_conn.recv()
    masking_delegate = pipe_conn.recv()

    # Start main loop
    proc_logger.debug("Start main loop")
    while not shutdown_event.is_set():
        try:
            # Check if data is available
            if not pipe_conn.poll(0.25):
                continue

            # Get images
            proc_logger.debug("Receive images")
            image1: Image = pipe_conn.recv()
            image2: Image = pipe_conn.recv()

            # Preprocessing
            proc_logger.debug("Preprocess images")
            for preprocessor in preprocessors:
                if preprocessor.needed_for_computation:
                    image1, image2 = preprocessor.process(image1, image2)

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

    # Return recorded timings to main process
    pipe_conn.send(timings)


@track_timings(name="Raw computation")
def _compute_params(image1: Image, image2: Image) -> ComputationParams:
    """Function to compute the parameters from the images

    Parameters
    ----------
    image1 : Image
        First input image
    image2 : Image
        Second input image

    Returns
    -------
    ComputationParams
        Compute parameters needed for stitching
    """
    global proc_logger, homography_delegate, masking_delegate
    assert image1.shape == image2.shape

    # Compute transformation and canvas params
    proc_logger.debug("Compute warping and target canvas")
    homography = homography_delegate.find_homography(image1, image2)
    canvas_size, translation, crop_area = homography_delegate.analyze_transformed_canvas(image1.shape, homography)

    # Apply transformation
    proc_logger.debug("Warp to target space")
    img1_translated, img2_warped = homography_delegate.apply_transformations(
        image1, image2, canvas_size, translation, homography
    )
    img1_translated_crop, img2_warped_crop = homography_delegate.apply_crop(img1_translated, img2_warped, *crop_area)

    # Mask calculation
    proc_logger.debug("Compute mask")
    mask = masking_delegate.compute_mask(img1_translated_crop, img2_warped_crop)
    return ((canvas_size, translation, homography), crop_area, mask)
