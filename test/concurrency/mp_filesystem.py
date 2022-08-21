import multiprocessing as mp
from time import perf_counter, sleep
import threading
import numpy as np
import os


start_time = perf_counter()
background_running = False
file_dir = "/tmp/"
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
    global background_running, pipes, param

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
    global param, background_running, shutdown, file_dir
    data1, data2 = data
    # Send data to process via files
    np.save(file_dir + "data1", data1)
    np.save(file_dir + "data2", data2)

    # Idle until results
    process.start()
    process.join()
    result = float(open(file_dir + "result1.txt", mode="r").readline())
    _ = np.load(file_dir + "result2.npy")

    # Update param object and cleanup
    printt(f"Background process done (value: {result})")
    for f in os.listdir(file_dir):
        os.remove(os.path.join(file_dir, f))
    param = result
    background_running = False


def update_param():
    data1 = np.load(file_dir + "data1.npy")
    data2 = np.load(file_dir + "data2.npy")

    # Compute new params
    sleep(3)
    result = np.average(data1) + np.average(data2)
    result2 = np.random.rand(720, 1280, 3)

    # Return results
    with open(file_dir + "result1.txt", mode="w") as f:
        f.write(str(result))
    np.save(file_dir + "result2", result2)


def main():
    global shutdown

    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)

    while perf_counter() - start_time < 7:
        x = get_data()
        y = process(x)
        printt(f"Values: {type(x)} {type(y)}")

    shutdown = True
    printt("EXIT")


if __name__ == "__main__":
    main()
