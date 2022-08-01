class BaseReader:
    async def read_image(self):
        raise NotImplementedError("Abstract method needs to be overriden")
