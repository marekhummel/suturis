from suturis.io.reader.basereader import BaseReader

from os import listdir
from os.path import isfile, join
import cv2
import asyncio


class FakeRtspReader(BaseReader):
    def __init__(self, path: str) -> None:
        onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        onlyimages = [f for f in onlyfiles if f.endswith('.jpg')]
        self.images = [cv2.imread(i) for i in onlyimages]
        self.counter = 0

    async def read_image(self):
        await asyncio.sleep(1.0)
        img = self.images[self.counter]
        self.counter = (self.counter + 1) % len(self.images)
        return True, img
