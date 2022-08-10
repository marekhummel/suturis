import asyncio
from concurrent.futures import ProcessPoolExecutor
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
    def finish_update(future):
        global param
        updating_task.discard(future)
        param = future.result()

    if len(updating_task) == 0:
        printt("Create updating task")

        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=4) as executor:
            parallel = loop.run_in_executor(executor, update_params, data)
            new_task = asyncio.create_task(parallel)
            updating_task.add(new_task)
            new_task.add_done_callback(finish_update)

    if param is None:
        printt("Return default while init")
        return -1

    printt("Process")
    await asyncio.sleep(0.1)  # What if these are not awaitable
    return data * param


async def update_params(data):
    global param

    printt("Param update start")
    await asyncio.sleep(3)
    # if param is None:
    #     param = 1
    # param += 1
    return data / 2
    printt(f"Param update complete (value: {param})")


async def main():
    while perf_counter() - start_time < 6:
        x = await get_data()
        y = await process(x)
        printt(f"Values: {x} {y}")


start_time = perf_counter()
asyncio.run(main())
