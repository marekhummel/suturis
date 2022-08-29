import multiprocessing as mp
import multiprocessing.managers as mpm
import threading

import numpy as np
from util import compute_param


# Notes:
#


class TransferNamespace(mpm.Namespace):
    data1: np.ndarray
    data2: np.ndarray
    result1: float
    result2: np.ndarray


background_running = False


def finalize():
    while background_running:
        pass


def updater(data, delay_update, when_done_callback, start_time):
    global background_running

    # Start background updater
    if background_running:
        return False

    ns = mp.Manager().Namespace()
    proc = mp.Process(target=_update_param, args=(ns, delay_update), daemon=True)
    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(proc, data, ns, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True
    return True


def _update_param_watcher(
    process, data, ns: TransferNamespace, when_done_callback, start_time
):
    global param, background_running
    data1, data2 = data

    # Send data in namespace
    ns.data1 = data1
    ns.data2 = data2

    # Idle until results
    process.start()
    process.join()
    result1 = ns.result1
    result2 = np.average(ns.result2)
    # print((result1, result2))

    # Update param object
    when_done_callback((result1, result2), start_time)
    background_running = False


def _update_param(ns: TransferNamespace, delay_update):
    # Receive data for computation
    data1 = ns.data1
    data2 = ns.data2

    # Compute new params
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))

    # Return results
    ns.result1 = result1
    ns.result2 = result2
