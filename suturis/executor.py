from typing import List, Tuple

import cv2
import asyncio

from suturis.io.reader import BaseReader, FileReader
from suturis.io.writer import BaseWriter, ScreenOutput
from suturis.processing import stitching
from suturis.processing.stitcher.video_stitcher import VideoStitcher
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
        image = await stitching.compute(image1, image2)

        # Show input
        stitcher = stitching.get_sticher()
        if isinstance(stitcher, VideoStitcher):
            (kpsA, kpsB) = stitcher.keypoints
            (matches, status) = (stitcher.cachedH[0], stitcher.cachedH[2])

            input_vis = draw_matches(image1, image2, kpsA, kpsB, matches, status)
            resize_vis = cv2.resize(input_vis, (0, 0), None, 0.5, 0.5)
            await input_window.write_image(resize_vis)

        # Write output
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
