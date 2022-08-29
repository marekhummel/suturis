from typing import List, Tuple

import cv2
import asyncio

from suturis.io.reader import BaseReader, FileReader
from suturis.io.writer import BaseWriter, ScreenOutput, FileWriter
from suturis.processing import stitching

STEPWISE = False


async def run():
    # Define readers / writers
    readers: Tuple[BaseReader, BaseReader] = (
        FileReader("./data/lr/starboard_0120220423163949.mp4", skip=9.2, speed_up=4),
        FileReader("./data/lr/port_0120220423162959.mp4", speed_up=4)
        # FakeRtspReader('./data/lr/img/first'),
        # FakeRtspReader('./data/lr/img/second'),
    )
    writers: List[BaseWriter] = [
        ScreenOutput("Stitched"),
        # FileWriter('Final', 'Test')
    ]

    # Loop
    ctr = 0
    while True:
        # Read
        results = await asyncio.gather(readers[0].read_image(), readers[1].read_image())
        success1, image1 = results[0]
        success2, image2 = results[1]

        if not success1 or not success2:
            print("readers failed")
            break

        # Process
        image = stitching.compute(image1, image2)

        # Write output
        for w in writers:
            w.write_image(image)

        # Misc
        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            cv2.imwrite(f"./data/lr/img/first/{ctr}.jpg", image1)
            cv2.imwrite(f"./data/lr/img/second/{ctr}.jpg", image2)
            ctr += 1
        if key == ord("p") or STEPWISE:
            while cv2.waitKey(25) & 0xFF != ord("p"):
                pass
