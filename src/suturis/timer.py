import logging as log
from functools import wraps
from time import perf_counter_ns
from typing import Any, Callable

import numpy as np

timings: dict[str, list[float]] = {}


def track_timings(*, name: str) -> Callable[[Callable], Any]:
    """Decorator which can be used to track runtime of single methods.
    Average times will be logged when the application is closing.

    Parameters
    ----------
    name : str
        Name of this entry, to be used in logs.

    Returns
    -------
    Callable[[Callable], Any]
        Decorated functions
    """
    global timings
    assert name not in timings, "Function with same name already tracked."
    timings[name] = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            begin = perf_counter_ns()
            result = func(*args, **kwargs)
            end = perf_counter_ns()
            timings[name].append(end - begin)
            return result

        return wrapper

    return decorator


def update_timings(other_timings: dict):
    """Updates this modules timing dictionary with another one. Needed for time tracking in subprocess.

    Parameters
    ----------
    other_timings : dict
        Other dictionary
    """
    global timings
    for name, times in other_timings.items():
        timings[name].extend(times)


def finalize_timings() -> None:
    """Computes average and standard deviation for all timings and logs them."""
    global timings
    log.debug("Calculate final timing results")

    timing_results = []
    timing_missing = []
    for name, times in timings.items():
        if len(times) > 0:
            avg_time = np.mean(times) / 1e9
            std_time = np.std(times) / 1e9
            timing_results.append((name, avg_time, std_time))
        else:
            timing_missing.append(name)

    timing_results.sort(key=lambda tpl: tpl[1], reverse=True)
    for name, avg, std in timing_results:
        log.info(f"Timings of '{name}' (mean ± std): {avg:.5f} ± {std:.5f} secs")

    for name in sorted(timing_missing):
        log.info(f"Timings of '{name}' unknown, method has not finished once")
