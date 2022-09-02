from typing import List, Tuple

import cv2
import asyncio

from suturis.io.reader import BaseReader, FileReader
from suturis.io.writer import BaseWriter, ScreenOutput, FileWriter
from suturis.processing import stitching
import logging as log

STEPWISE = False


async def run(io):
    readers, writers = io

    # Loop
    log.debug("Starting main loop")
    while True:
        # Read
        log.debug("Read images")
        results = await asyncio.gather(readers[0].read_image(), readers[1].read_image())
        success1, image1 = results[0]
        success2, image2 = results[1]

        if not success1 or not success2:
            log.error("Readers failed")
            break

        # Process
        log.debug("Compute stitched image")
        image = stitching.compute(image1, image2)

        # Write output
        log.debug("Write stitched image to outputs")
        for w in writers:
            w.write_image(image)

        # Debug
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break
        if key == ord("e"):
            raise ZeroDivisionError
        if key == ord("p") or STEPWISE:
            while cv2.waitKey(25) & 0xFF != ord("p"):
                pass
