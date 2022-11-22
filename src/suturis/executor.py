import logging as log
from threading import Event

import cv2

import suturis.processing.computation.manager as mgr
from suturis.config_parser import IOConfig, MiscConfig, StichingConfig
from suturis.io.reader.basereader import BaseReader
from suturis.io.writer.basewriter import BaseWriter
from suturis.processing import stitching
from suturis.timer import finalize_timings, track_timings
from suturis.typing import Image


def run(io: IOConfig, delegates: StichingConfig, misc: MiscConfig) -> None:
    """Main application loop.

    Parameters
    ----------
    io : IOConfig
        Configuration of IO
    delegates : StichingConfig
        Configuration of stitching
    misc : MiscConfig
        Miscellaneous configuration
    """
    # Setup
    readers, writers = io
    if misc.get("enable_debug_outputs", False):
        _enable_debug_outputs(delegates)
        stitching.enable_debug_outputs()
    stitching.set_delegates(*delegates)

    # Set shared event for each reader to signal cancellation when one reader fails (EOF most likely)
    assert len(readers) == 2
    cancellation_token = Event()
    for r in readers:
        r.start(cancellation_token)

    # Run while all readers live
    log.debug("Starting main loop")
    debug_keyinputs = misc.get("enable_debug_keyinputs", False)
    while not cancellation_token.is_set():
        _run_iteration(readers, writers, cancellation_token, debug_keyinputs)


def shutdown() -> None:
    """Finialize the application."""
    mgr.shutdown()
    finalize_timings()


@track_timings(name="Iteration")
def _run_iteration(
    readers: list[BaseReader], writers: list[BaseWriter], cancellation_token: Event, debug_keyinputs: bool
) -> None:
    """Single stitching iteration. Starts with receiving images, ends by returning the stitched image.

    Parameters
    ----------
    readers : list[BaseReader]
        List of defined readers to get the images from.
    writers : list[BaseWriter]
        List of defined writers to write the output (or the input) to.
    cancellation_token : Event
        Threading event to signal end of loop

    Returns
    -------
    bool
        True, as long as the application didn't fail.
    """
    # ** Read (reading might block)
    log.debug("Read images")
    image1, image2 = _read_image(readers)

    # One reader failed
    if image1 is None or image2 is None:
        log.error("Readers failed")
        cancellation_token.set()
        return

    # ** Process
    log.debug("Compute stitched image")
    output = stitching.compute(image1, image2)

    # ** Write output
    log.debug("Write stitched image to outputs")
    possibles = (output, image1, image2)
    for w in writers:
        w.write_image(possibles[w.source])

    # ** Debug
    if debug_keyinputs:
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            log.info("Manually aborting")
            cancellation_token.set()
        if key == ord("e"):
            log.info("Manually raising error")
            raise ZeroDivisionError
        if key == ord("p"):
            log.info("Manually pausing")
            while cv2.waitKey(25) & 0xFF != ord("p"):
                pass


@track_timings(name="Readers")
def _read_image(readers: list[BaseReader]) -> tuple[Image | None, Image | None]:
    """Returns the current images of both readers.

    Parameters
    ----------
    readers : list[BaseReader]
        List (pair) of readers

    Returns
    -------
    tuple[Image | None, Image | None]
        Most recent images.
    """
    return readers[0].get(), readers[1].get()


def _enable_debug_outputs(delegates: StichingConfig) -> None:
    """Enable debug outputs (various intermediate images for debugging)"""
    log.info("Debug outputs are enabled")

    preprocessors, homography, masking, postprocessors = delegates

    for preprocessor in preprocessors:
        preprocessor.enable_debug_outputs()
    homography.enable_debug_outputs()
    masking.enable_debug_outputs()
    for postprocessor in postprocessors:
        postprocessor.enable_debug_outputs()
