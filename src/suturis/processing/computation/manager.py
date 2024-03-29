import contextlib
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
from suturis.typing import ComputationParams, Image, Mask, TransformationInfo

_local_params: ComputationParams | None = None
_computation_running: bool = False
_no_recomputation_needed: bool = False
_process: mp.Process | None = None
_local_pipe: mpc._ConnectionBase | None = None
_queue_listener: logging.handlers.QueueListener | None = None
_shutdown_event: EventType = mp.Event()


def get_params(image1: Image, image2: Image) -> ComputationParams | None:
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
    global _computation_running
    assert _process

    # Recompute params when possible
    if not _computation_running and (_local_params is None or not _no_recomputation_needed):
        log.debug("Recompute stitching params")
        _computation_running = True
        watcher = threading.Thread(target=_computation_watcher, args=(image1, image2), daemon=True)
        watcher.start()
        log.debug("Watcher started")

    # Return params
    log.debug("Return params")
    return _local_params


def setup(
    preprocessing_handlers: list[BasePreprocessor],
    homography_handler: BaseHomographyHandler,
    masking_handler: BaseMaskingHandler,
) -> None:
    """Setups the manager by creating the subprocess.

    Parameters
    ----------
    preprocessing_handlers : list[BasePreprocessor]
        List of preprocessors
    homography_handler : BaseHomographyHandler
        Homography handler
    masking_handler : BaseMaskingHandler
        Masking handler
    """
    global _process, _local_pipe, _queue_listener, _no_recomputation_needed

    assert _process is None
    log.debug("Creating daemon process")

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

    # Check if process needs to operate more than once
    _no_recomputation_needed = homography_handler._caching_enabled and masking_handler._caching_enabled
    if _no_recomputation_needed:
        log.info("Both main delegates have caching enabled, thus the subprocess will only be contacted once")


def shutdown() -> None:
    """Finalize the application by closing the subprocess and clean up."""
    log.debug("Cleanly close subprocess")
    _shutdown_event.set()

    # This might take some time if a new computation just started
    if _process:
        log.info("Requested termination of subprocess (this might take a few seconds)")
        if _local_pipe:
            while _computation_running:
                pass
            with contextlib.suppress(Exception):
                subproc_timings = _local_pipe.recv()
                update_timings(subproc_timings)
        _process.join()

    if _local_pipe:
        _local_pipe.close()

    if _queue_listener:
        _queue_listener.stop()


@track_timings(name="Manager Watcher")
def _computation_watcher(image1: Image, image2: Image) -> None:
    """Daemon thread which communicates with the subprocess and updates params when completed.

    Parameters
    ----------
    image1 : Image
        First input image
    image2 : Image
        Second input image
    """
    global _computation_running, _local_params
    assert _local_pipe is not None

    try:
        # Start and send data
        log.debug("Start watcher and send data")
        _local_pipe.send(image1)
        _local_pipe.send(image2)

        # Idle until results
        log.debug("Wait until results received")
        warping_info: TransformationInfo = _local_pipe.recv()
        mask: Mask = _local_pipe.recv()

        # Update local params
        if _local_params is None:
            log.info("Initial computation of params completed")
        _local_params = warping_info, mask

    except (BrokenPipeError, EOFError):
        if not _shutdown_event.is_set():
            log.error("Pipe error, update aborted")
        return
    finally:
        _computation_running = False
        log.debug("Update complete")
