import multiprocessing as mp
from time import perf_counter, sleep
import threading
import numpy as np
import os
from tempfile import mkdtemp
import os.path as path
import shutil

start_time = perf_counter()
background_running = False
tmp_dirs = []
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
    global background_running, pipes, param, tmp_dirs

    # Start background updater
    if not background_running:
        tmp_dir = mkdtemp()
        tmp_dirs.append(tmp_dir)
        proc = mp.Process(target=update_param, args=(tmp_dir,), daemon=True)

        printt(f"Param update start (data: {type(data)})")
        background_running = True

        watcher = threading.Thread(
            target=update_param_watcher, args=(proc, tmp_dir, data), daemon=True
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


def update_param_watcher(process, tmp_dir, data):
    global param, background_running, shutdown
    data1, data2 = data
    # Send data to process via temp files
    fptr1 = np.memmap(
        path.join(tmp_dir, "data1"), dtype=np.float64, mode="w+", shape=data1.shape
    )
    fptr2 = np.memmap(
        path.join(tmp_dir, "data2"), dtype=np.float64, mode="w+", shape=data2.shape
    )
    fptr1[:] = data1.astype(np.float64)[:]
    fptr2[:] = data2.astype(np.float64)[:]
    fptr1.flush()
    fptr2.flush()

    # Idle until results
    process.start()
    process.join()
    result = float(open(path.join(tmp_dir, "result1.txt"), mode="r").readline())
    fptr_res = np.memmap(path.join(tmp_dir, "data1"), dtype=np.float64, mode="r")
    result2 = fptr_res[:]

    # Update param object and cleanup
    printt(f"Background process done (values: {result, np.average(result2)})")
    param = result
    background_running = False

def update_param(tmp_dir):
    data1 = np.memmap(path.join(tmp_dir, "data1"), dtype=np.float64, mode="r")
    data2 = np.memmap(path.join(tmp_dir, "data2"), dtype=np.float64, mode="r")

    # Compute new params
    sleep(3)
    result = np.average(data1) + np.average(data2)
    result2 = np.random.rand(720, 1280, 3)

    # Return results
    with open(path.join(tmp_dir, "result1.txt"), mode="w") as f:
        f.write(str(result))
    fptr = np.memmap(
        path.join(tmp_dir, "result2"), dtype=np.float64, mode="w+", shape=result2.shape
    )
    fptr[:] = result2.astype(np.float64)[:]
    fptr.flush()


def main():
    global shutdown, tmp_dirs

    while perf_counter() - start_time < 10:
        x = get_data()
        y = process(x)
        printt(f"Values: {type(x)} {type(y)} {param}")

    for td in tmp_dirs:
        shutil.rmtree(td)
    shutdown = True
    printt("EXIT")


if __name__ == "__main__":
    main()
