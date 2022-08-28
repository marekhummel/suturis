import threading
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
from util import compute_param

# Notes:
# GIL

background_running = False


def finalize():
    while background_running:
        pass


def updater(data, delay_update, when_done_callback, start_time):
    global background_running

    # Start background updater
    if background_running:
        return False

    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(data, delay_update, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True
    return True


def _update_param_watcher(data, delay_update, when_done_callback, start_time):
    global param, background_running, shutdown
    data1, data2 = data

    # Send data in manager
    with ThreadPoolExecutor() as executor:
        future = executor.submit(_update_param, data1, data2, delay_update)
        wait([future])
        result1, result2_array = future.result()
        result2 = np.average(result2_array)
        # print((result1, result2))

        when_done_callback((result1, result2), start_time)
        background_running = False


def _update_param(data1, data2, delay_update):
    # Return results
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))
    return result1, result2
