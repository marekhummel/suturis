import multiprocessing as mp
import threading

import numpy as np
from util import compute_param

# Notes:
# Pipe typing

background_running = False
pipes = None


def finalize():
    while background_running:
        pass


def updater(data, delay_update, when_done_callback, start_time):
    global background_running, pipes, param

    # Start background updater
    if background_running:
        return False

    local, fork = mp.Pipe(duplex=True)
    pipes = local, fork
    proc = mp.Process(target=_update_param, args=(fork, delay_update), daemon=True)
    watcher = threading.Thread(
        target=_update_param_watcher,
        args=(proc, data, local, fork, when_done_callback, start_time),
        daemon=True,
    )
    watcher.start()
    background_running = True
    return True


def _update_param_watcher(
    process, data, pipe_local, pipe_fork, when_done_callback, start_time
):
    global background_running
    data1, data2 = data
    # Send data to process via pipe

    try:
        process.start()
        pipe_local.send(data1)
        pipe_local.send(data2)

        # Idle until results
        result1 = pipe_local.recv()
        result2 = np.average(pipe_local.recv())
        # print((result1, result2))
        process.join()

        # Update param object
        pipe_local.close()
        pipe_fork.close()
        when_done_callback((result1, result2), start_time)
    except EOFError:
        return
    finally:
        background_running = False


def _update_param(pipe_conn, delay_update):
    # Receive data for computation
    try:
        data1 = pipe_conn.recv()
        data2 = pipe_conn.recv()

        # Compute new params
        result1, result2 = compute_param(data1, data2, delay_update)
        # print((result1, np.average(result2)))

        # Return results
        pipe_conn.send(result1)
        pipe_conn.send(result2)
    except EOFError:
        # Pipe has been closed
        return
