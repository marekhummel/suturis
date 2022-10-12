from suturis.typing import Image


class BasePreprocessor:
    index: int
    needed_for_computation: bool

    def __init__(self, index: int, /, needed_for_computation: bool) -> None:
        self.index = index
        self.needed_for_computation = needed_for_computation

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        raise NotImplementedError("Abstract method needs to be overriden")
