import asyncio
import random as rnd
from time import perf_counter

param = None
updating_task = set()


def printt(text):
    print(f"{perf_counter() - start_time:.3f}s: ", text)


async def get_data():
    printt("Get data")
    await asyncio.sleep(0.5)
    return rnd.choice([2, 3, 4, 5])


async def process(data):
    if len(updating_task) == 0:
        printt("Create updating task")
        new_task = asyncio.create_task(update_params(data))
        updating_task.add(new_task)
        new_task.add_done_callback(updating_task.discard)

    if param is None and len(updating_task) == 1:
        printt("Return default while init")
        return -1

    printt("Process")
    await asyncio.sleep(0.1)
    return data * param


async def update_params(data):
    global param

    printt("Param update start")
    await asyncio.sleep(5)
    if param is None:
        param = 1
    param += 1
    printt("Param update complete")


async def main():
    while True:
        x = await get_data()
        y = await process(x)
        printt(f"Values: {x} {y}")


start_time = perf_counter()
asyncio.run(main())
