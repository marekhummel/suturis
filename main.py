import asyncio

import suturis.executor as runner


# async def test():
#     while True:
#         print("Test")
#         await asyncio.sleep(1)


# async def main():
#     task1 = asyncio.create_task(test())
#     task2 = asyncio.create_task(runner.run())
#     await asyncio.wait({task1, task2}, return_when=asyncio.FIRST_COMPLETED)


# # asyncio.run(main())
asyncio.run(runner.run())
