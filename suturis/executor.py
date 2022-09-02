import logging as log
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Tuple

import cv2

import suturis.processing.computation.manager as mgr
from suturis.io.reader.basereader import BaseReader
from suturis.io.writer.basewriter import BaseWriter
from suturis.processing import stitching

STEPWISE = False


def run(io: Tuple[List[BaseReader], List[BaseWriter]]):
    readers, writers = io
    assert len(readers) == 2

    # Loop
    log.debug("Starting main loop")
    while True:
        # ** Read (reading might block thus the threads)
        log.debug("Read images")
        with ThreadPoolExecutor(max_workers=len(readers)) as tpe:
            results = list(tpe.map(lambda r: r.read_image(), readers))

        if len(results) != len(readers):
            log.error("Readers failed")
            break

        success1, image1 = results[0]
        success2, image2 = results[1]

        if not success1 or not success2:
            log.error("Readers failed")
            break

        # ** Process
        log.debug("Compute stitched image")
        image = stitching.compute(image1, image2)

        # ** Write output
        log.debug("Write stitched image to outputs")
        for w in writers:
            w.write_image(image)

        # ** Debug
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            log.info("Manually aborting")
            break
        if key == ord("e"):
            log.info("Manually raising error")
            raise ZeroDivisionError
        if key == ord("p") or STEPWISE:
            log.info("Manually pausing")
            while cv2.waitKey(25) & 0xFF != ord("p"):
                pass


def shutdown():
    mgr.shutdown()
