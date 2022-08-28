from importlib import reload
from itertools import product
from time import perf_counter

import benchmarks.mp_filesystem as mpfile
import benchmarks.mp_futures as mpfuture
import benchmarks.mp_manager_dict as mpmdict
import benchmarks.mp_manager_ns as mpmns
import benchmarks.mp_memmap as mpmmap
import benchmarks.mp_pipe as mppipe
import benchmarks.mp_rawvalue as mpraw
import benchmarks.mp_sharedmemory as mpshm
import benchmarks.mt_basic as mtbas
from util import average_delta, get_data, printt, process

DELAY_DATA_RETREIVE = [0.5, 0.1, 0.05]
DELAY_PROCESSING = [0.1, 0.02, 0.01]
DELAY_UPDATE_PARAM = [3]
MAX_RUNTIME = 10
MODULES = [mpfile, mpfuture, mpmdict, mpmns, mpmmap, mppipe, mpraw, mpshm, mtbas]


param = None
shutdown = False
step_delta = []
update_times = []


def main(module, delay_data, delay_proc, delay_update):
    global param, shutdown

    # Set methods
    update = module.updater
    finalize = module.finalize if hasattr(module, "finalize") else lambda: None

    # Go
    start_time = perf_counter()
    while perf_counter() - start_time < MAX_RUNTIME:
        step_start = perf_counter()
        printt("Step", start_time)
        x = get_data(delay_data)
        if update(x, delay_update, update_done_callback, start_time):
            printt("Background process start", start_time)
            update_times.append((perf_counter(), False))
        _ = process(x, param, delay_proc)
        step_delta.append(perf_counter() - step_start)

    shutdown = True
    finalize()
    printt("EXIT", start_time)


def update_done_callback(result, start_time):
    global param, update_times, shutdown
    if not shutdown:
        param = result
        printt("Background process done", start_time)
        update_times.append((perf_counter() - update_times.pop()[0], True))


if __name__ == "__main__":
    counter = 1
    with open("test/concurrency/results.csv", mode="w") as f:
        cases = list(
            product(
                MODULES, zip(DELAY_DATA_RETREIVE, DELAY_PROCESSING), DELAY_UPDATE_PARAM
            )
        )

        f.write(
            "MODULE;DELAY_DATA_RETREIVE;DELAY_PROCESSING;STEP_DELTA_AVG;DELAY_UPDATE_PARAM;UPDATE_DELTA_AVG;OVERHEAD_STEP;OVERHEAD_UPDATE\n"
        )
        for module, (delay_data, delay_proc), delay_update in cases:
            print(
                f"{counter:02d}/{len(cases)}:",
                module.__name__,
                delay_data,
                delay_proc,
                delay_update,
            )

            module = reload(module)
            main(module, delay_data, delay_proc, delay_update)

            average_step = average_delta(step_delta)
            average_update = average_delta([t for t, b in update_times if b])

            f.write(f"{module.__name__};")
            f.write(f"{delay_data:.3f};")
            f.write(f"{delay_proc:.3f};")
            f.write(f"{average_step:.6f};")
            f.write(f"{delay_update:.3f};")
            f.write(f"{average_update:.6f};")
            f.write(f"{average_step - delay_data - delay_proc:.6f};")
            f.write(f"{average_update - delay_update:.6f}")
            f.write("\n")

            step_delta.clear()
            update_times.clear()
            param = None
            shutdown = False
            counter += 1
