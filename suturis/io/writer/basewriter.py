class BaseWriter:
    def __init__(self, index, /) -> None:
        self.index = index

    def write_image(self, image) -> None:
        raise NotImplementedError("Abstract method needs to be overriden")
