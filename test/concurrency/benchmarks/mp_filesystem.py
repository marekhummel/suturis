import multiprocessing as mp
import os
import threading

import numpy as np
from util import compute_param

# Notes:
# Works, but slow due to file system. Use tempfile.mkstemp and RAM mounted folder with tmpfs


background_running = False
file_dir = "/tmp/"


def finalize():
    while background_running:
        pass


def updater(data, delay_update, when_done_callback, start_time):
    global background_running

    # Start background updater
    if background_running:
        return False

    proc = mp.Process(target=_update_param, args=(delay_update,), daemon=True)
    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(proc, data, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True
    return True


def _update_param_watcher(process, data, when_done_callback, start_time):
    global background_running, file_dir
    # Send data to process via files
    data1, data2 = data
    np.save(file_dir + "data1", data1)
    np.save(file_dir + "data2", data2)

    # Idle until results
    process.start()
    process.join()
    result1 = float(open(file_dir + "result1.txt", mode="r").readline())
    result2 = np.average(np.load(file_dir + "result2.npy"))
    # print((result1, result2))

    # Update param object and cleanup
    for f in os.listdir(file_dir):
        os.remove(os.path.join(file_dir, f))

    when_done_callback((result1, result2), start_time)
    background_running = False


def _update_param(delay_update):
    # Extract
    data1 = np.load(file_dir + "data1.npy")
    data2 = np.load(file_dir + "data2.npy")

    # Compute new params
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))

    # Return results
    with open(file_dir + "result1.txt", mode="w") as f:
        f.write(str(result1))
    np.save(file_dir + "result2", result2)
