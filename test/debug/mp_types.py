import multiprocessing as mp
import multiprocessing.connection as mpc


def test(x: mpc.Connection):
    return None


pipes: tuple[mpc.Connection, mpc.Connection] = mp.Pipe(duplex=True)
y, z = pipes
test(y)
