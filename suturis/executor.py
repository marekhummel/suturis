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
        FileReader("./data/lr/starboard_0120220423163949.mp4", 9.2, single_frame=True),
        FileReader("./data/lr/port_0120220423162959.mp4", single_frame=True)
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
        image = await stitching.compute(image1, image2)

        # Write output
        await asyncio.gather(*[w.write_image(image) for w in writers])

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
