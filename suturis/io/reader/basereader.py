class BaseReader:
    index: int

    def __init__(self, index: int, /) -> None:
        self.index = index

    def read_image(self):
        raise NotImplementedError("Abstract method needs to be overriden")
