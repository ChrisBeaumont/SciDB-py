# License: Simplified BSD, 2014
# See LICENSE.txt for more information
from numpy.random import randint
import numpy as np

from .. import connect
from ..schema_utils import rechunk

RTOL = 1e-6

sdb = connect()

# NOTE: this needs to be explicitly imported in each test module,
#      or pytest won't run it


def teardown_function(function):
    sdb.reap()


class TestBase(object):

    def teardown_method(self, method):
        sdb.reap()


"""
The code below randomly rechunks arrays created
by standard factory methods like sdb.zeros()

This tests that SciDBPy functions are robust to details about chunks
"""

unfuzzed = {}

def chunk_fuzz(func):

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        persistent = result.persistent
        result.persistent = False
        chunks = randint(100, 300, (result.ndim,))
        chunk_overlap = randint(0, 3, (result.ndim,))
        result = rechunk(result, chunk_size=chunks, chunk_overlap=chunk_overlap)
        result.eval()
        result.persistent = persistent
        return result

    return wrapper

for fac in ('zeros ones random from_array from_sparse from_dataframe '
            'identity linspace arange random randint').split():
    unfuzzed[fac] = getattr(sdb, fac)
    setattr(sdb, fac, chunk_fuzz(getattr(sdb, fac)))


def randarray(shape, dtypes, names=None):
    recsize = sum([np.dtype(d).itemsize for d in dtypes])
    if names is None:
        names = ['f%i' % i for i in range(len(dtypes))]

    dtype = list(zip(names, dtypes))
    s = [s * recsize for s in shape]
    return np.random.randint(0, 256, s).astype(np.uint8).view(dtype=dtype)
