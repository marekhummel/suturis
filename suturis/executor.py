from typing import List, Tuple

import cv2

from suturis.io.reader import BaseReader, FileReader
from suturis.io.reader.fakertspreader import FakeRtspReader
from suturis.io.writer import BaseWriter, ScreenOutput
from suturis.processing import stitcher, blender


def run():
    # Define readers / writers
    readers: Tuple[BaseReader, BaseReader] = (
        # FileReader('./data/Boot1-Vorne-0120210705165153.mp4'),
        # FileReader('./data/Boot1-Achtern-0120210705165038.mp4')
        FakeRtspReader('./data/img/first'),
        FakeRtspReader('./data/img/second'),
    )
    writers: List[BaseWriter] = [
        ScreenOutput()
    ]

    # Loop
    ctr = 0
    while True:
        # Read
        success1, image1 = readers[0].read_image()
        success2, image2 = readers[1].read_image()

        if not success1 or not success2:
            print('readers failed')
            break

        # Process
        image = stitcher.stitch(image1, image2)
        image = blender.blend(image)

        # Write
        for writer in writers:
            writer.write_image(image)

        # Misc
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(f'./data/img/first/{ctr}.jpg', image1)
            cv2.imwrite(f'./data/img/second/{ctr}.jpg', image2)
            ctr += 1
