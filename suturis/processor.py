from typing import List, Tuple

import cv2
import numpy as np

from suturis.io.reader import BaseReader, FileReader
from suturis.io.writer import BaseWriter
from suturis.io.writer.screenoutput import ScreenOutput


def run():
    # Define readers / writers
    readers: Tuple[BaseReader, BaseReader] = (
        FileReader('./data/Boot1-Vorne-0120210705165153.mp4'),
        FileReader('./data/Boot1-Achtern-0120210705165038.mp4')
    )
    writers: List[BaseWriter] = [
        ScreenOutput()
    ]

    # Loop
    while True:
        # Read
        success1, image1 = readers[0].read_image()
        success2, image2 = readers[1].read_image()

        if not success1 or not success2:
            print('readers failed')
            break

        # Process
        image1 = cv2.resize(image1, (0, 0), None, .5, .5)
        image2 = cv2.resize(image2, (0, 0), None, .5, .5)
        image2 = cv2.rotate(image2, rotateCode=cv2.ROTATE_180)
        image = np.concatenate((image1, image2), axis=0)

        # Write
        for writer in writers:
            writer.write_image(image)

        while cv2.waitKey(25) & 0xFF != ord('q'):
            pass

        break
