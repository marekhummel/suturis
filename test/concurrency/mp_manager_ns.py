import multiprocessing as mp
import multiprocessing.managers as mpm
import threading
from time import perf_counter, sleep

import numpy as np


class TestNamespace(mpm.Namespace):
    data1: np.ndarray
    data2: np.ndarray
    result1: float
    result2: np.ndarray


start_time = perf_counter()
background_running = False
param = None
shutdown = False


def printt(text):
    print(f"{perf_counter() - start_time:.3f}s: {text}")


def get_data():
    # Create random data (with delay for better overview)
    printt("Get data")
    sleep(0.5)
    data1 = np.random.rand(720, 1280, 3)
    data2 = np.random.rand(720, 1280, 3)
    return data1, data2


def process(data):
    global background_running, pipes, param

    # Start background updater
    if not background_running:
        ns = mp.Manager().Namespace()
        proc = mp.Process(target=update_param, args=(ns,), daemon=True)

        printt(f"Param update start (data: {type(data)})")
        background_running = True

        watcher = threading.Thread(
            target=update_param_watcher, args=(proc, data, ns), daemon=True
        )
        watcher.start()

    # Default return while init
    if param is None:
        printt("Return None while init")
        return None

    # Use params for computation
    printt("Process")
    sleep(0.1)
    return (data[0] * param, data[1] * param)


def update_param_watcher(process, data, ns: TestNamespace):
    global param, background_running, shutdown
    data1, data2 = data

    # Send data in manager
    ns.data1 = data1
    ns.data2 = data2

    # Idle until results
    process.start()
    process.join()
    result = ns.result1
    _ = ns.result2

    # Update param object
    printt(f"Background process done (value: {result})")
    param = result
    background_running = False


def update_param(ns: TestNamespace):
    # Receive data for computation
    data1 = ns.data1
    data2 = ns.data2

    # Compute new params
    sleep(3)
    result = np.average(data1) + np.average(data2)
    result2 = np.random.rand(720, 1280, 3)

    # Return results
    ns.result1 = result
    ns.result2 = result2


def main():
    global shutdown

    while perf_counter() - start_time < 10:
        x = get_data()
        y = process(x)
        printt(f"Values: {type(x)} {type(y)} {param}")

    shutdown = True
    printt("EXIT")


if __name__ == "__main__":
    main()
