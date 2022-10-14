import logging as log
import logging.handlers
import multiprocessing as mp
import multiprocessing.connection as mpc
import threading
from multiprocessing.synchronize import Event as EventType

import suturis.processing.computation.subprocess as subprc
from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
from suturis.processing.computation.preprocessing import BasePreprocessor
from suturis.timer import track_timings, update_timings
from suturis.typing import ComputationParams, CropArea, Image, Mask, TransformationInfo

_local_params: ComputationParams | None = None
_computation_running: bool = False
_process: mp.Process | None = None
_local_pipe: mpc.PipeConnection | None = None
_queue_listener: logging.handlers.QueueListener | None = None
_shutdown_event: EventType = mp.Event()


def get_params(
    image1: Image,
    image2: Image,
    preprocessing_handlers: list[BasePreprocessor],
    homography_handler: BaseHomographyHandler,
    masking_handler: BaseMaskingHandler,
) -> ComputationParams | None:
    """Return params needed for stitching (start new computation in seperate process if possible).

    Parameters
    ----------
    image1 : Image
        First input image
    image2 : Image
        Second input image
    preprocessing_handlers : list[BasePreprocessor]
        List of preprocessors
    homography_handler : BaseHomographyHandler
        Homography handler
    masking_handler : BaseMaskingHandler
        Masking handler

    Returns
    -------
    ComputationParams | None
        Set of computation params
    """
    global _computation_running, _process

    # Create subprocess if needed
    if _process is None:
        log.debug("Creating daemon process")
        _create_subprocess(preprocessing_handlers, homography_handler, masking_handler)

    # Recompute params when possible
    if not _computation_running:
        log.debug("Recompute stitching params")
        _computation_running = True
        watcher = threading.Thread(target=_computation_watcher, args=(image1, image2), daemon=True)
        watcher.start()
        log.debug("Watcher started")

    # Return params
    log.debug("Return params")
    return _local_params


def shutdown() -> None:
    """Finalize the application by closing the subprocess and clean up."""
    global _shutdown_event, _computation_running, _process, _local_pipe, _queue_listener

    log.debug("Cleanly close subprocess")
    _shutdown_event.set()

    # This might take some time if a new computation just started
    if _process:
        log.debug("Requested termination of subprocess")
        if _local_pipe:
            while _computation_running:
                pass
            subproc_timings = _local_pipe.recv()
            update_timings(subproc_timings)
        _process.join()

    if _local_pipe:
        _local_pipe.close()

    if _queue_listener:
        _queue_listener.stop()


def _create_subprocess(
    preprocessing_handlers: list[BasePreprocessor],
    homography_handler: BaseHomographyHandler,
    masking_handler: BaseMaskingHandler,
) -> None:
    """Create a new subprocess at the beginning of the application lifetime.

    Parameters
    ----------
    preprocessing_handlers : list[BasePreprocessor]
        List of preprocessors
    homography_handler : BaseHomographyHandler
        Homography handler
    masking_handler : BaseMaskingHandler
        Masking handler
    """
    global _process, _local_pipe, _queue_listener

    # Setup logging
    log_queue: mp.Queue = mp.Queue()
    _queue_listener = logging.handlers.QueueListener(log_queue, *log.getLogger().handlers, respect_handler_level=True)
    _queue_listener.start()

    # Start process and pipes
    _local_pipe, fork = mp.Pipe(duplex=True)
    _process = mp.Process(target=subprc.main, args=(fork, _shutdown_event, log_queue), daemon=True)
    _process.start()

    # Send handlers once
    log.debug("Pass handlers to subprocess")
    _local_pipe.send(preprocessing_handlers)
    _local_pipe.send(homography_handler)
    _local_pipe.send(masking_handler)


@track_timings(name="Update call")
def _computation_watcher(image1: Image, image2: Image) -> None:
    """Daemon thread which communicates with the subprocess and updates params when completed.

    Parameters
    ----------
    image1 : Image
        First input image
    image2 : Image
        Second input image
    """
    global _computation_running, _local_params, _shutdown_event, _local_pipe
    assert _local_pipe is not None

    try:
        # Start and send data
        log.debug("Start watcher and send data")
        _local_pipe.send(image1)
        _local_pipe.send(image2)

        # Idle until results
        log.debug("Wait until results received")
        warping_info: TransformationInfo = _local_pipe.recv()
        crop_area: CropArea = _local_pipe.recv()
        mask: Mask = _local_pipe.recv()

        # Update local params
        if _local_params is None:
            log.info("Initial computation of params completed")
        _local_params = warping_info, crop_area, mask

    except (BrokenPipeError, EOFError):
        if not _shutdown_event.is_set():
            log.error("Pipe error, update aborted")
        return
    finally:
        _computation_running = False
        log.debug("Update complete.")
