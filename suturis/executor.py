import cv2
import asyncio
from suturis.processing import stitching
import suturis.processing.computation.manager as mgr
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
