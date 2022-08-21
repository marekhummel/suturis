from csv import excel
import multiprocessing as mp
from time import perf_counter, sleep
import threading
import numpy as np


start_time = perf_counter()
background_running = False
pipes = None
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
        local, fork = mp.Pipe(duplex=True)
        pipes = local, fork
        proc = mp.Process(target=update_param, args=(fork,), daemon=True)

        printt(f"Param update start (data: {type(data)})")
        background_running = True
        proc.start()

        watcher = threading.Thread(
            target=update_param_watcher, args=(proc, data, local, fork), daemon=True
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


def update_param_watcher(process, data, pipe_local, pipe_fork):
    global param, background_running, shutdown
    data1, data2 = data
    # Send data to process via pipe

    try:
        pipe_local.send(data1)
        pipe_local.send(data2)

        # Idle until results
        result = pipe_local.recv()
        _ = pipe_local.recv()
        process.join()

        # Update param object
        printt(f"Background process done (value: {result})")
        pipe_local.close()
        pipe_fork.close()
        param = result
        background_running = False
    except EOFError:
        # If error occured due to application shutdown, fine, otherwise re-throw
        if shutdown:
            return
        raise


def update_param(pipe_conn):
    # Receive data for computation
    try:
        data1 = pipe_conn.recv()
        data2 = pipe_conn.recv()

        # Compute new params
        sleep(3)
        result = np.average(data1) + np.average(data2)
        result2 = np.random.rand(720, 1280, 3)

        # Return results
        pipe_conn.send(result)
        pipe_conn.send(result2)
    except EOFError:
        # Pipe has been closed
        return


def main():
    global shutdown

    while perf_counter() - start_time < 7:
        x = get_data()
        y = process(x)
        printt(f"Values: {type(x)} {type(y)}")

    shutdown = True
    printt("EXIT")


if __name__ == "__main__":
    main()
