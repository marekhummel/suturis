import multiprocessing as mp
import multiprocessing.managers as mpm
import multiprocessing.shared_memory as mpsm
import struct
import threading
from time import perf_counter, sleep

import numpy as np

start_time = perf_counter()
background_running = False
memory = None
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
    global background_running, memory, param

    # Start background updater
    if not background_running:
        proc = mp.Process(target=update_param, daemon=True)

        printt(f"Param update start (data: {type(data)})")
        background_running = True

        watcher = threading.Thread(
            target=update_param_watcher, args=(proc, data), daemon=True
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


def update_param_watcher(process, data):
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

    # # Prep result memory
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
    result = struct.unpack("f", shm_res1.buf)
    result2 = np.ndarray((720, 1280, 3), dtype=np.float64, buffer=shm_res2.buf)

    # Update param object
    printt(f"Background process done (values: {result} {np.average(result2)})")
    param = result
    background_running = False


def update_param():
    # Receive data for computation
    shm_data1 = mpsm.SharedMemory(name="data1")
    data1 = np.ndarray((720, 1280, 3), dtype=np.float64, buffer=shm_data1.buf)
    shm_data2 = mpsm.SharedMemory(name="data2")
    data2 = np.ndarray((720, 1280, 3), dtype=np.float64, buffer=shm_data2.buf)

    # Compute new params
    sleep(3)
    result = np.average(data1) + np.average(data2)
    print(result)
    result2 = np.random.rand(720, 1280, 3)

    # Return results
    shm_res1 = mp.shared_memory.SharedMemory(name="result1")
    result = struct.pack_into("f", shm_res1.buf, 0, result)
    shm_res2 = mpsm.SharedMemory(name="result2")
    dst_res2 = np.ndarray(shape=result2.shape, dtype=np.float64, buffer=shm_res2.buf)
    dst_res2[:] = result2.astype(np.float64)


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
