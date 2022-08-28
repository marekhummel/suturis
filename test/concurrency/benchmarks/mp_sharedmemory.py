import multiprocessing as mp
import multiprocessing.shared_memory as mpsm
import struct
import threading

import numpy as np
from util import compute_param

# Notes:
# Sizes of return arrays need to be preset
# Struct packing of float causes rounding error

background_running = False


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
    global param, background_running
    data1, data2 = data

    # Send data into shared memory
    # Set data
    memsize1 = np.dtype(np.float64).itemsize * data1.size
    memsize2 = np.dtype(np.float64).itemsize * data2.size
    shm1 = mpsm.SharedMemory(create=True, size=memsize1, name="data1")
    shm2 = mpsm.SharedMemory(create=True, size=memsize2, name="data2")
    dst1 = np.ndarray(shape=data1.shape, dtype=np.float64, buffer=shm1.buf)
    dst2 = np.ndarray(shape=data2.shape, dtype=np.float64, buffer=shm2.buf)
    dst1[:] = data1.astype(np.float64)
    dst2[:] = data2.astype(np.float64)

    # Prep result memory
    shm_res1 = mpsm.SharedMemory(create=True, size=struct.calcsize("f"), name="result1")
    shm_res2 = mpsm.SharedMemory(
        create=True,
        size=np.dtype(np.float64).itemsize * (1280 * 720 * 3),
        name="result2",
    )  # down side, pre set size

    # Idle until results
    process.start()
    process.join()

    # Get values
    result1 = struct.unpack("!f", shm_res1.buf)[0]
    result2 = np.average(
        np.ndarray((720, 1280, 3), dtype=np.float64, buffer=shm_res2.buf)
    )
    # print((result1, result2))

    # Clean memory
    shm1.close()
    shm2.close()
    shm_res1.close()
    shm_res2.close()
    shm1.unlink()
    shm2.unlink()
    shm_res1.unlink()
    shm_res2.unlink()

    # Update param object
    when_done_callback((result1, result2), start_time)
    background_running = False


def _update_param(delay_update):
    # Receive data for computation
    shm_data1 = mpsm.SharedMemory(name="data1")
    data1 = np.ndarray((720, 1280, 3), dtype=np.float64, buffer=shm_data1.buf)
    shm_data2 = mpsm.SharedMemory(name="data2")
    data2 = np.ndarray((720, 1280, 3), dtype=np.float64, buffer=shm_data2.buf)

    # Compute new params
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))

    # Return results
    shm_res1 = mp.shared_memory.SharedMemory(name="result1")
    struct.pack_into("!f", shm_res1.buf, 0, result1)
    shm_res2 = mpsm.SharedMemory(name="result2")
    dst_res2 = np.ndarray(shape=result2.shape, dtype=np.float64, buffer=shm_res2.buf)
    dst_res2[:] = result2.astype(np.float64)

    # Clear refs
    shm_data1.close()
    shm_data2.close()
    shm_res1.close()
    shm_res2.close()
