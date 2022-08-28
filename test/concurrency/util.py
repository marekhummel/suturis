from time import perf_counter, sleep
import numpy as np


def printt(text, start_time):
    return
    print(f"{perf_counter() - start_time:.3f}s: {text}")


def get_data(delay):
    # Create random data (with delay for better overview)
    start = perf_counter()
    data1 = np.random.rand(720, 1280, 3)
    data2 = np.random.rand(720, 1280, 3)
    end = perf_counter()
    _cpu_sleep(delay - (end - start))
    return data1, data2


def process(data, param, delay):
    # Sleep even if param is missing, for consistency

    # Use params for computation
    start = perf_counter()
    result = (data[0] * param[0], data[1] * param[1]) if param else None
    end = perf_counter()
    _cpu_sleep(delay - (end - start))
    return result


def compute_param(data1, data2, delay):
    start = perf_counter()
    result = np.average(data1) + np.average(data2)
    result2 = np.random.rand(720, 1280, 3)
    end = perf_counter()
    _cpu_sleep(delay - (end - start))
    return result, result2


def average_delta(step_delta):
    return sum(step_delta) / len(step_delta)


def _cpu_sleep(duration):
    now = perf_counter()
    x = 0
    while perf_counter() - now < duration:
        for _ in range(20):
            x *= 1
