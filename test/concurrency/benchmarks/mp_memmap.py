import multiprocessing as mp
import os.path as path
import shutil
import threading
from tempfile import mkdtemp

import numpy as np
from util import compute_param

# Notes:
#

background_running = False
tmp_dirs = []


def finalize():
    global background_running, tmp_dirs

    while background_running:
        pass

    for td in tmp_dirs:
        shutil.rmtree(td)


def updater(data, delay_update, when_done_callback, start_time):
    global background_running, tmp_dirs

    # Start background updater
    if background_running:
        return False

    tmp_dir = mkdtemp()
    tmp_dirs.append(tmp_dir)

    proc = mp.Process(target=_update_param, args=(tmp_dir, delay_update), daemon=True)
    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(proc, data, tmp_dir, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True

    return True


def _update_param_watcher(process, data, tmp_dir, when_done_callback, start_time):
    global background_running
    data1, data2 = data
    # Send data to process via temp files
    fptr1 = np.memmap(
        path.join(tmp_dir, "data1"), dtype=np.float64, mode="w+", shape=data1.shape
    )
    fptr2 = np.memmap(
        path.join(tmp_dir, "data2"), dtype=np.float64, mode="w+", shape=data2.shape
    )
    fptr1[:] = data1.astype(np.float64)
    fptr2[:] = data2.astype(np.float64)
    fptr1.flush()
    fptr2.flush()

    # Idle until results
    process.start()
    process.join()
    result1 = float(open(path.join(tmp_dir, "result1.txt"), mode="r").readline())
    fptr_res = np.memmap(
        path.join(tmp_dir, "result2"), dtype=np.float64, mode="r", shape=(720, 1280, 3)
    )
    result2_array = np.empty((720, 1280, 3))
    result2_array[:] = fptr_res[:]
    result2 = np.average(result2_array)
    # print((result1, result2))

    # Update param object and cleanup
    when_done_callback((result1, result2), start_time)
    background_running = False


def _update_param(tmp_dir, delay_update):
    data1 = np.memmap(path.join(tmp_dir, "data1"), dtype=np.float64, mode="r")
    data2 = np.memmap(path.join(tmp_dir, "data2"), dtype=np.float64, mode="r")

    # Compute new params
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))

    # Return results
    with open(path.join(tmp_dir, "result1.txt"), mode="w") as f:
        f.write(str(result1))
    fptr = np.memmap(
        path.join(tmp_dir, "result2"), dtype=np.float64, mode="w+", shape=result2.shape
    )
    fptr[:] = result2.astype(np.float64)
    fptr.flush()
    del fptr
