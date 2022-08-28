import multiprocessing as mp
import threading

import numpy as np
from util import compute_param

# Notes:
#

background_running = False


def finalize():
    while background_running:
        pass


def updater(data, delay_update, when_done_callback, start_time):
    global background_running

    # Start background updater
    if background_running:
        return False

    mgr = mp.Manager()
    mpdict = mgr.dict()
    proc = mp.Process(target=_update_param, args=(mpdict, delay_update), daemon=True)
    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(proc, data, mpdict, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True
    return True


def _update_param_watcher(process, data, mpdict, when_done_callback, start_time):
    global background_running
    data1, data2 = data

    # Send data in manager
    mpdict["data1"] = data1
    mpdict["data2"] = data2

    # Idle until results
    process.start()
    process.join()
    result1 = mpdict["result1"]
    result2 = np.average(mpdict["result2"])
    # print((result1, result2))

    # Update param object
    when_done_callback((result1, result2), start_time)
    background_running = False


def _update_param(mpdict, delay_update):
    # Receive data for computation
    data1 = mpdict["data1"]
    data2 = mpdict["data2"]

    # Compute new params
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))

    # Return results
    mpdict["result1"] = result1
    mpdict["result2"] = result2
