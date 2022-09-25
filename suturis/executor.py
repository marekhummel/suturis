import logging as log
from concurrent.futures import ThreadPoolExecutor

import cv2

import suturis.processing.computation.manager as mgr
from suturis.io.reader.basereader import BaseReader
from suturis.io.writer.basewriter import BaseWriter
from suturis.processing import stitching
from suturis.timer import finalize_timings, track_timings


def run(io: tuple[list[BaseReader], list[BaseWriter]]):
    readers, writers = io
    assert len(readers) == 2

    # Loop
    log.debug("Starting main loop")
    running = True
    while running:
        running = _run_iteration(readers, writers)


def shutdown() -> None:
    mgr.shutdown()
    finalize_timings()


@track_timings(name="Iteration")
def _run_iteration(readers: list[BaseReader], writers: list[BaseWriter]) -> bool:
    # ** Read (reading might block hence the threads)
    log.debug("Read images")
    with ThreadPoolExecutor(max_workers=len(readers)) as tpe:
        results = list(tpe.map(lambda r: r.read_image(), readers))

    if len(results) != len(readers):
        log.error("Readers failed")
        return False

    success1, image1 = results[0]
    success2, image2 = results[1]

    if not success1 or not success2:
        log.error("Readers failed")
        return False

    # ** Process
    log.debug("Compute stitched image")
    output = stitching.compute(image1, image2)

    # ** Write output
    log.debug("Write stitched image to outputs")
    possibles = (image1, image2, output)
    for w in writers:
        w.write_image(possibles[w.source])

    # ** Debug
    key = cv2.waitKey(25) & 0xFF
    if key == ord("q"):
        log.info("Manually aborting")
        return False
    if key == ord("e"):
        log.info("Manually raising error")
        raise ZeroDivisionError
    if key == ord("p") or False:
        log.info("Manually pausing")
        while cv2.waitKey(25) & 0xFF != ord("p"):
            pass

    return True
