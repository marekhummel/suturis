from asyncio.windows_utils import pipe
import multiprocessing as mp
import random as rnd
from time import perf_counter, sleep
import threading


start_time = perf_counter()
background_running = False
param = None


def printt(text):
    print(f"{perf_counter() - start_time:.3f}s: {text}")


def get_data():
    printt("Get data")
    sleep(0.5)
    return rnd.choice([2, 3, 4, 5])


def process(data):
    global background_running, param

    if not background_running:
        local, fork = mp.Pipe(duplex=True)
        proc = mp.Process(target=update_param, args=(fork,), daemon=True)

        printt(f"Param update start (input: {data})")
        background_running = True
        proc.start()

        watcher = threading.Thread(
            target=update_param_watcher, args=(proc, data, local), daemon=True
        )
        watcher.start()

    if param is None:
        printt("Return default while init")
        return -1

    printt("Process")
    sleep(0.1)
    return data * param


def update_param_watcher(process, data, pipe_conn):
    global param, background_running
    pipe_conn.send(data)

    result = pipe_conn.recv()
    process.join()

    printt(f"Background process done (value: {result})")
    param = result
    background_running = False


def update_param(pipe_conn):
    data = pipe_conn.recv()
    sleep(3)
    result = data // 2
    pipe_conn.send(result)


def main():
    while perf_counter() - start_time < 7:
        x = get_data()
        y = process(x)
        printt(f"Values: {x} {y}")


if __name__ == "__main__":
    main()
