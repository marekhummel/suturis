import logging as log
from functools import wraps
from time import perf_counter_ns

import numpy as np

timings = {}


def track_timings(*, name):
    assert name not in timings, "Function with same name already tracked."
    timings[name] = []

    def decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            begin = perf_counter_ns()
            result = func(*args, **kwargs)
            end = perf_counter_ns()
            timings[name].append(end - begin)
            return result

        return _wrapper

    return decorator


def finalize_timings():
    for name, times in timings.items():
        if len(times) > 0:
            avg_time = np.mean(times) / 1e9
            std_time = np.std(times) / 1e9
            log.info(
                f"Timings of '{name}' (mean ± std): {avg_time:.5f} ± {std_time:.5f} secs"
            )
        else:
            log.info(f"Timings of '{name}' unknown, method has not finished once")
