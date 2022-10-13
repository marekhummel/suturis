import logging as log

from suturis.typing import Image


class BasePreprocessor:
    index: int
    needed_for_computation: bool

    def __init__(self, index: int, /, needed_for_computation: bool) -> None:
        log.debug(f"Init preprocessing handler #{index}, with needed_for_computation set to {needed_for_computation}")
        self.index = index
        self.needed_for_computation = needed_for_computation

    def process(self, img1: Image, img2: Image) -> tuple[Image, Image]:
        raise NotImplementedError("Abstract method needs to be overriden")
