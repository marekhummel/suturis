class BaseWriter:
    def write_image(self, image) -> None:
        raise NotImplementedError('Abstract method needs to be overriden')
