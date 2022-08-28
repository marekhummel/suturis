import ctypes
import multiprocessing as mp
import threading
from dataclasses import dataclass

import numpy as np
from util import compute_param

# Notes:
# Sizes of return arrays need to be preset

background_running = False


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


def finalize():
    while background_running:
        pass


def updater(data, delay_update, when_done_callback, start_time):
    global background_running

    # Start background updater
    if background_running:
        return False

    memory = SharedParams()
    proc = mp.Process(target=_update_param, args=(memory, delay_update), daemon=True)
    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(proc, data, memory, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True
    return True


def _update_param_watcher(process, data, memory, when_done_callback, start_time):
    global background_running
    data1, data2 = data

    # Send data in manager
    memory.data1 = _to_shared_array(data1)
    memory.data2 = _to_shared_array(data2)
    memory.result1 = mp.RawValue(ctypes.c_float, 0)
    memory.result2 = RawNumpyArray(
        mp.RawArray(ctypes.c_double, 720 * 1280 * 3), (720, 1280, 3), 3
    )

    # Idle until results
    process.start()
    process.join()
    result1 = memory.result1.value
    result2 = np.average(_from_shared_array(memory.result2))
    # print((result1, result2))

    # Update param object
    when_done_callback((result1, result2), start_time)
    background_running = False


def _update_param(memory, delay_update):
    # Receive data for computation
    data1 = _from_shared_array(memory.data1)
    data2 = _from_shared_array(memory.data2)

    # Compute new params
    result1, result2 = compute_param(data1, data2, delay_update)
    # print((result1, np.average(result2)))

    # Return results
    memory.result1.value = result1
    buffered_result = np.ndarray(
        memory.result2.shape, dtype=np.float64, buffer=memory.result2.data
    )
    np.copyto(buffered_result, result2.astype(np.float64))


def _to_shared_array(array: np.ndarray):
    raw_data = mp.RawArray(ctypes.c_double, array.size)
    np_data_wrapper = np.ndarray(array.shape, dtype=np.float64, buffer=raw_data)
    np.copyto(np_data_wrapper, array.astype(np.float64))

    raw_shape = mp.RawArray(ctypes.c_int, array.shape)
    raw_ndim = mp.RawValue(ctypes.c_int, array.ndim)

    return RawNumpyArray(raw_data, raw_shape, raw_ndim)


def _from_shared_array(shared_array: RawNumpyArray):
    shape = tuple(shared_array.shape)
    np_array = np.frombuffer(shared_array.data).reshape(shape)
    return np_array
