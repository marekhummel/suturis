from dataclasses import dataclass
import multiprocessing as mp
import threading
from time import perf_counter, sleep
import ctypes
import numpy as np


start_time = perf_counter()
background_running = False
memory = None
param = None
shutdown = False


@dataclass
class RawNumpyArray:
    data: mp.RawArray
    shape: mp.RawArray
    ndim: mp.RawValue


class SharedParams:
    data1: RawNumpyArray
    data2: RawNumpyArray
    result1: mp.RawValue
    result2: RawNumpyArray


def printt(text):
    print(f"{perf_counter() - start_time:.3f}s: {text}")


def to_shared_array(array: np.ndarray):
    raw_data = mp.RawArray(ctypes.c_double, array.size)
    np_data_wrapper = np.ndarray(array.shape, dtype=np.float64, buffer=raw_data)
    np.copyto(np_data_wrapper, array.astype(np.float64))

    raw_shape = mp.RawArray(ctypes.c_int, array.shape)
    raw_ndim = mp.RawValue(ctypes.c_int, array.ndim)

    return RawNumpyArray(raw_data, raw_shape, raw_ndim)


def from_shared_array(shared_array: RawNumpyArray):
    shape = tuple(shared_array.shape)
    np_array = np.frombuffer(shared_array.data).reshape(shape)
    return np_array


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
        memory = SharedParams()
        proc = mp.Process(target=update_param, args=(memory,), daemon=True)

        printt(f"Param update start (data: {type(data)})")
        background_running = True

        watcher = threading.Thread(
            target=update_param_watcher, args=(proc, data, memory), daemon=True
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


def update_param_watcher(process, data, memory):
    global param, background_running
    data1, data2 = data

    # Send data in manager
    memory.data1 = to_shared_array(data1)
    memory.data2 = to_shared_array(data2)
    memory.result1 = mp.RawValue(ctypes.c_float, 0)
    memory.result2 = RawNumpyArray(
        mp.RawArray(ctypes.c_double, 720 * 1280 * 3), (720, 1280, 3), 3
    )

    # Idle until results
    process.start()
    process.join()
    result = memory.result1.value
    _ = from_shared_array(memory.result2)

    # Update param object
    printt(f"Background process done (value: {result})")
    param = result
    background_running = False


def update_param(memory):
    # Receive data for computation
    data1 = from_shared_array(memory.data1)
    data2 = from_shared_array(memory.data2)

    # Compute new params
    sleep(3)
    result = np.average(data1) + np.average(data2)
    result2 = np.random.rand(720, 1280, 3)

    # Return results
    memory.result1.value = result
    buffered_result = np.ndarray(
        memory.result2.shape, dtype=np.float64, buffer=memory.result2.data
    )
    np.copyto(buffered_result, result2.astype(np.float64))


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
    # data1 = np.random.rand(720, 1280, 3)
    # cdata, cdata_ptr = CNumpyArray.from_ndarray_to_ptr(data1)
    # pdata = cdata_ptr.contents.to_ndarray()
    # print(pdata.sum())
