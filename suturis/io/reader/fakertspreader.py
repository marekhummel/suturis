from suturis.io.reader.basereader import BaseReader

from os import listdir
from os.path import isfile, join
import cv2
import time


class FakeRtspReader(BaseReader):
    def __init__(self, path: str) -> None:
        onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        onlyimages = [f for f in onlyfiles if f.endswith('.jpg')]
        self.images = [cv2.imread(i) for i in onlyimages]
        self.counter = 0

    def read_image(self):
        time.sleep(0.5)
        img = self.images[self.counter]
        self.counter = (self.counter + 1) % len(self.images)
        return True, img
