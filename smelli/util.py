from flavio.util import get_datapath
from collections import defaultdict
import multiprocessing
# Use explicit 'fork' context for compatibility with Python 3.14+
# where the default start method changed to 'forkserver'
Pool = multiprocessing.get_context('fork').Pool
import numpy as np


def tree():
    """Tree data structure.

    See https://gist.github.com/hrldcpr/2012250
    """
    return defaultdict(tree)


def multithreading_map(func, iterable, threads=1, pool=None):
    if threads > 1 or pool is not None:
        if pool is None:
            pool_instance = Pool(threads)
        else:
            pool_instance = pool
        try:
            result = pool_instance.map(func, iterable)
        except:
            pool_instance.close()
            raise
        pool_instance.close()
        if pool is None:
            pool_instance.join()
    else:
        result = map(func, iterable)
    return result


def as_float(x):
    """Convert a number or a size-1 array to a plain float."""
    return float(np.asarray(x).item())
