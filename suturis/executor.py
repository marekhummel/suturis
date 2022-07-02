from typing import List, Tuple

import cv2
import asyncio
import numpy as np

from suturis.io.reader import BaseReader, FileReader
from suturis.io.reader.fakertspreader import FakeRtspReader
from suturis.io.writer import BaseWriter, ScreenOutput
from suturis.processing import blender
from suturis.processing.stitching import stitcher
from suturis.processing.util import concat_images, draw_matches, highlight_features


async def run():
    # Define readers / writers
    readers: Tuple[BaseReader, BaseReader] = (
        FileReader('./data/lr/port_0120220423162959.mp4', single_frame=True),
        FileReader('./data/lr/starboard_0120220423163949.mp4', 10.3, True)
        # FakeRtspReader('./data/lr/img/first'),
        # FakeRtspReader('./data/lr/img/second'),
    )
    writers: List[BaseWriter] = [
        ScreenOutput('Stitched')
    ]
    input_window = ScreenOutput('Input')
    input_window2 = ScreenOutput('Input2')

    # Loop
    ctr = 0
    while True:
        # Read
        results = await asyncio.gather(
            readers[0].read_image(),
            readers[1].read_image()
        )
        success1, image1 = results[0]
        success2, image2 = results[1]

        if not success1 or not success2:
            print('readers failed')
            break

        # Process
        image = await stitcher.compute(image1, image2)
        image = await blender.blend(image)

        # Show input

        # Params
        (kpsA, kpsB) = stitcher.get_keypoints()
        (matches, status) = stitcher.get_matches_with_status()

        # Show matches
        input_vis = draw_matches(image1, image2, kpsA, kpsB, matches, status)
        resize_vis = cv2.resize(input_vis, (0, 0), None, 0.5, 0.5)
        await input_window2.write_image(resize_vis)

        # Show highlighted input
        highlighted1 = highlight_features(image1, kpsA, matches, status, 1)
        highlighted2 = highlight_features(image2, kpsB, matches, status, 0)
        concat_input = concat_images(highlighted1, highlighted2)
        await input_window.write_image(concat_input)

        # Write
        await asyncio.gather(*[w.write_image(image) for w in writers])

        # Misc
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite(f'./data/lr/img/first/{ctr}.jpg', image1)
            cv2.imwrite(f'./data/lr/img/second/{ctr}.jpg', image2)
            ctr += 1
        if key == ord('p'):
            while cv2.waitKey(25) & 0xFF != ord('p'):
                pass
