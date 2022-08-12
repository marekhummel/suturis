import asyncio

import suturis.executor as runner
import logging as log
import sys


# async def test():
#     while True:
#         print("Test")
#         await asyncio.sleep(1)


# async def main():
#     task1 = asyncio.create_task(test())
#     task2 = asyncio.create_task(runner.run())
#     await asyncio.wait({task1, task2}, return_when=asyncio.FIRST_COMPLETED)


# # asyncio.run(main())
if __name__ == "__main__":
    log.basicConfig(level=log.INFO, handlers=[log.StreamHandler(sys.stdout)])
    log.info("Lets go")
    asyncio.run(runner.run())
