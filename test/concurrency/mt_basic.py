import threading
from concurrent.futures import ThreadPoolExecutor, wait
from time import perf_counter, sleep

import numpy as np


start_time = perf_counter()
background_running = False
param = None
shutdown = False


def printt(text):
    print(f"{perf_counter() - start_time:.3f}s: {text}")


def get_data():
    # Create random data (with delay for better overview)
    printt("Get data")
    sleep(0.05)
    data1 = np.random.rand(720, 1280, 3)
    data2 = np.random.rand(720, 1280, 3)
    return data1, data2


def process(data):
    global background_running, pipes, param

    # Start background updater
    if not background_running:
        printt(f"Param update start (data: {type(data)})")
        background_running = True

        watcher = threading.Thread(
            target=update_param_watcher, args=(data,), daemon=True
        )
        watcher.start()

    # Default return while init
    if param is None:
        printt("Return None while init")
        return None

    # Use params for computation
    printt("Process")
    sleep(0.02)
    return (data[0] * param, data[1] * param)


def update_param_watcher(data):
    global param, background_running, shutdown
    data1, data2 = data

    # Send data in manager
    with ThreadPoolExecutor() as executor:
        future = executor.submit(update_param, data1, data2)
        wait([future])
        param, _ = future.result()
        printt(f"Background process done (value: {param})")
        background_running = False


def update_param(data1, data2):
    # Compute new params
    sleep(3)
    result = np.average(data1) + np.average(data2)
    result2 = np.random.rand(720, 1280, 3)

    # Return results
    return result, result2


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
