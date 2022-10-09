import logging as log
import logging.handlers
import multiprocessing as mp
import multiprocessing.connection as mpc
import threading
from multiprocessing.synchronize import Event as EventType

from suturis.processing.computation.homography import BaseHomographyHandler
from suturis.processing.computation.masking import BaseMaskingHandler
import suturis.processing.computation.subprocess as subprc
from suturis.timer import track_timings
from suturis.typing import ComputationParams, Image, Mask, WarpingInfo, CropArea

_local_params: ComputationParams | None = None
_computation_running: bool = False
_process: mp.Process | None = None
_local_pipe: mpc.PipeConnection | None = None
_queue_listener: logging.handlers.QueueListener | None = None
_shutdown_event: EventType = mp.Event()


def get_params(
    image1: Image,
    image2: Image,
    homography_handler: BaseHomographyHandler,
    masking_handler: BaseMaskingHandler,
) -> ComputationParams | None:
    global _computation_running, _process

    # Create subprocess if needed
    if _process is None:
        _create_subprocess(homography_handler, masking_handler)

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
    global _shutdown_event, _process, _local_pipe, _queue_listener
    log.debug("Cleanly close subprocess")
    _shutdown_event.set()

    if _local_pipe:
        _local_pipe.close()

    # This might take some time if a new computation just started
    if _process:
        _process.join()

    if _queue_listener:
        _queue_listener.stop()


def _create_subprocess(homography_handler: BaseHomographyHandler, masking_handler: BaseMaskingHandler):
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
    _local_pipe.send(homography_handler)
    _local_pipe.send(masking_handler)


@track_timings(name="Update call")
def _computation_watcher(image1: Image, image2: Image) -> None:
    global _computation_running, _local_params, _shutdown_event, _local_pipe
    assert _local_pipe is not None

    try:
        # Start and send data
        log.debug("Start watcher and send data")
        _local_pipe.send(image1)
        _local_pipe.send(image2)

        # Idle until results
        log.debug("Wait until results received")
        warping_info: WarpingInfo = _local_pipe.recv()
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
