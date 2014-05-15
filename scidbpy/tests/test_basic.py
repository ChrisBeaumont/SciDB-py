# License: Simplified BSD, 2014
# See LICENSE.txt for more information

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

MISSING_PD = False
MISSING_SP = False
try:
    import pandas as pd
except ImportError:
    MISSING_PD = True
try:
    from scipy import sparse
except ImportError:
    MISSING_SP = True

needs_pandas = pytest.mark.skipif(MISSING_PD, reason='Test requires Pandas')
needs_scipy = pytest.mark.skipif(MISSING_SP, reason='Test requires SciPy')


# In order to run tests, we need to connect to a valid SciDB engine
from scidbpy import SciDBArray, SciDBShimInterface, connect


sdb = connect()

RTOL = 1E-6


def test_copy_rename():
    """Test the copying and renaming of an array"""
    X = sdb.random(10)
    Xcopy = X.copy()
    assert X.name != Xcopy.name

    new_name = 'silly_array1234'
    X.rename(new_name)
    assert X.name == new_name
    assert_array_equal(X.toarray(), Xcopy.toarray())


def test_from_array():
    """Test import of numpy array for various types"""
    def check_from_array(dtype):
        Xnp = np.empty(10, dtype=dtype)
        Xarr = sdb.from_array(Xnp)
        assert_array_equal(Xnp, Xarr.toarray())

    for dtype in [int, float, [('i', int), ('f', float)]]:
        yield check_from_array, dtype


def test_to_array():
    """Test export to a numpy array"""
    X = np.random.random((10, 6))

    Xsdb = sdb.from_array(X)

    def check_toarray(transfer_bytes):
        Xnp = Xsdb.toarray(transfer_bytes=transfer_bytes)
        # set ATOL high because we're translating text
        assert_allclose(Xnp, X, atol=1E-5)

    for transfer_bytes in (True, False):
        yield check_toarray, transfer_bytes


@needs_scipy
def test_from_sparse():
    """Test import from Scipy sparse matrix"""

    X = np.random.random((10, 10))
    X[X < 0.9] = 0
    Xcsr = sparse.csr_matrix(X)
    Xarr = sdb.from_sparse(Xcsr)
    assert_allclose(X, Xarr.toarray())


@needs_scipy
def test_to_sparse():
    """Test export to Scipy Sparse matrix"""

    X = np.random.random((10, 6))
    Xsdb = sdb.from_array(X)
    Xcsr = Xsdb.tosparse('csr')
    assert_allclose(X, Xcsr.toarray())


@needs_pandas
def test_from_dataframe():
    """Test import from Pandas dataframe"""

    X = np.zeros(10, dtype=[('i', int), ('f', float)])
    X['i'] = np.arange(10)
    X['f'] = 0.1 * np.arange(10, 20)
    Xdf = pd.DataFrame(X)
    Xsdb = sdb.from_dataframe(Xdf)
    assert_array_equal(X, Xsdb.toarray())


@needs_pandas
def test_to_dataframe():
    """Test export to Pandas dataframe"""

    d = sdb.random(10, dtype=float)
    i = sdb.randint(10, lower=0, upper=10, dtype=int)
    X = sdb.join(d, i)

    Xnp = X.toarray()
    Xpd = X.todataframe()

    for col in Xpd.columns:
        assert_allclose(Xnp[col], Xpd[col])


def test_nonzero_nonnull():
    # create a matrix with empty, null, and non-null entries
    N = 4
    tridiag = sdb.new_array((N, N), dtype='<v:double null>')
    sdb.query('store(build_sparse({A}, '
              '    iif({A.d0}={A.d1}, 1, null), '
              '    {A.d0} <= {A.d1}+1 and {A.d0} >= {A.d1}-1), '
              '  {A})',
              A=tridiag)

    assert tridiag.contains_nulls()
    assert_array_equal(tridiag.nonempty(), N + 2 * (N - 1))
    assert_array_equal(tridiag.nonnull(), N)


def test_array_creation():
    def check_array_creation(create_array):
        # Create an array with 5x5 elements
        A = create_array((5, 5))
        name = A.name
        assert name in sdb.list_arrays()

        # when A is reaped, its data should be deleted from the engine
        A.reap()

        assert name not in sdb.list_arrays()

    for create_array in [sdb.zeros, sdb.ones, sdb.random, sdb.randint]:
        yield check_array_creation, create_array


def test_constant_creation():
    def check(creator, shp):
        A = getattr(sdb, creator)(shp)
        Anp = getattr(np, creator)(shp)
        assert_allclose(A.toarray(), Anp)
        assert A.toarray().shape == Anp.shape
    for shp in [(5,), (5, 4), (1,), (3, 1, 4)]:
        yield check, 'zeros', shp
        yield check, 'ones', shp


def test_arange():
    def check_arange(args):
        A = sdb.arange(*args)
        Anp = np.arange(*args)
        assert_allclose(A.toarray(), Anp)
    for args in [(10,), (0, 10), (0, 9.9, 0.5)]:
        yield check_arange, args


def test_linspace():
    def check_linspace(args):
        A = sdb.linspace(*args)
        Anp = np.linspace(*args)
        assert_allclose(A.toarray(), Anp)
    for args in [(0.2, 1.5), (0.2, 1.5, 10)]:
        yield check_linspace, args


def test_reshape():
    A = sdb.random(12)

    def check_reshape(shape):
        B = A.reshape(shape)
        Bnp = A.toarray().reshape(shape)
        assert_allclose(B.toarray(), Bnp)

    for shape in [(3, 4), (2, 2, 3), (1, 3, 4)]:
        yield check_reshape, shape


def test_raw_query():
    """Test a more involved raw query: creating a tri-diagonal matrix"""
    arr = sdb.new_array((10, 10))
    sdb.query('store(build({A},iif({A.d0}={A.d1},2,'
              'iif(abs({A.d0}-{A.d1})=1,1,0))),{A})',
              A=arr)

    # Build the numpy equivalent
    np_arr = np.zeros((10, 10))
    np_arr.flat[0::11] = 2  # set diagonal to 2
    np_arr.flat[1::11] = 1  # set upper off-diagonal to 1
    np_arr.flat[10::11] = 1  # set lower off-diagonal to 1

    assert_allclose(arr.toarray(), np_arr, rtol=RTOL)


@needs_scipy
@pytest.mark.parametrize('sparse', (True, False))
def test_identity(sparse):
    n = 6
    I = sdb.identity(n, sparse=sparse)
    assert_allclose(I.toarray(), np.identity(n))


def test_dot():
    def check_dot(Ashape, Bshape):
        A = sdb.random(Ashape)
        B = sdb.random(Bshape)
        C = sdb.dot(A, B)
        Cnp = np.dot(A.toarray(), B.toarray())
        if isinstance(C, SciDBArray):
            assert_allclose(C.toarray(), Cnp, rtol=RTOL)
        else:
            assert_allclose(C, Cnp, rtol=RTOL)

    for Ashape in [(4, 5), 5]:
        for Bshape in [(5, 6), 5]:
            yield check_dot, Ashape, Bshape


def test_dot_nullable():
    """Test the dot product of arrays with nullable attributes"""
    X = sdb.random((5, 5), dtype='<f0:double null>')
    Y = sdb.random((5, 5), dtype='<f0:double null>')

    assert X.sdbtype.nullable[0]
    assert Y.sdbtype.nullable[0]

    Z = sdb.dot(X, Y)
    assert_allclose(Z.toarray(), np.dot(X.toarray(), Y.toarray()))


def test_svd():
    # chunk_size=32 currently required for svd
    A = sdb.random((6, 10), chunk_size=32)
    U, S, VT = sdb.svd(A)
    U2, S2, VT2 = np.linalg.svd(A.toarray(), full_matrices=False)

    assert_allclose(U.toarray(), U2, rtol=RTOL)
    assert_allclose(S.toarray(), S2, rtol=RTOL)
    assert_allclose(VT.toarray(), VT2, rtol=RTOL)


def test_slicing():
    # note that slices must be a divisor of chunk size
    A = sdb.random((10, 10), chunk_size=12)

    def check_subarray(slc):
        Aslc = A[slc]
        if isinstance(Aslc, SciDBArray):
            Aslc = Aslc.toarray()
        assert_allclose(Aslc, A.toarray()[slc], rtol=RTOL)

    for slc in [(slice(None), slice(None)),
                (2, 3),
                1,
                slice(2, 6),
                (slice(None), 2),
                (slice(2, 8), slice(3, 7)),
                (slice(2, 8, 2), slice(None, None, 3))]:
        yield check_subarray, slc


def test_ops():
    from operator import add, sub, mul, truediv, mod, pow
    A = sdb.random((5, 5))
    B = 1.2

    def check_join_op(op):
        C = op(A, B)
        assert_allclose(C.toarray(), op(A.toarray(), B), rtol=RTOL)

    for op in (add, sub, mul, truediv, mod, pow):
        yield check_join_op, op


def test_reverse_ops():
    from operator import add, sub, mul, truediv, mod, pow
    A = 1.2
    B = sdb.random((5, 5))

    def check_join_op(op):
        C = op(A, B)
        assert_allclose(C.toarray(), op(A, B.toarray()), rtol=RTOL)

    for op in (add, sub, mul, truediv, mod, pow):
        yield check_join_op, op


def test_join_ops():
    from operator import add, sub, mul, truediv, mod, pow
    A = sdb.random((5, 5))
    B = sdb.random((5, 5))

    def check_join_op(op):
        C = op(A, B)
        assert_allclose(C.toarray(), op(A.toarray(), B.toarray()), rtol=RTOL)

    for op in (add, sub, mul, truediv, mod, pow):
        yield check_join_op, op


def test_join_ops_same_array():
    from operator import add, sub, mul, truediv, mod, pow
    A = sdb.random((5, 5))

    def check_join_op(op):
        C = op(A, A)
        assert_allclose(C.toarray(), op(A.toarray(), A.toarray()), rtol=RTOL)

    for op in (add, sub, mul, truediv, mod, pow):
        yield check_join_op, op


def test_array_broadcast():
    def check_array_broadcast(shape1, shape2):
        A = sdb.random(shape1)
        B = sdb.random(shape2)
        C = A + B
        assert_allclose(C.toarray(), A.toarray() + B.toarray())

    for shapes in [((5, 1), 4), (4, (5, 1)),
                   ((5, 1), (1, 4)), ((1, 4), (5, 1)),
                   ((5, 1), (5, 5)), ((5, 5), (5, 1)),
                   ((1, 5, 1), 4), (4, (1, 5, 1)),
                   ((5, 1, 3), (4, 1)), ((4, 1), (5, 1, 3))]:
        yield check_array_broadcast, shapes[0], shapes[1]


def test_abs():
    A = sdb.random((5, 5))
    B = abs(A - 1)
    assert_allclose(B.toarray(), abs(A.toarray() - 1))


def test_transcendentals():
    def np_op(op):
        D = dict(asin='arcsin', acos='arccos', atan='arctan')
        return D.get(op, op)

    A = sdb.random((5, 5))

    def check_op(op):
        C = getattr(sdb, op)(A)
        C_np = getattr(np, np_op(op))(A.toarray())
        assert_allclose(C.toarray(), C_np, rtol=RTOL)

    for op in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan',
               'exp', 'log', 'log10']:
        yield check_op, op


def test_substitute():
    # Generate a SciDB array with nullable attribtue
    arr = sdb.new_array()
    sdb.query("store(build(<v:double null>[i=1:5,5,0],null),{0})", arr)
    assert_allclose(np.zeros(5), arr.substitute(0).toarray())


def test_scidb_aggregates():
    A = sdb.random((5, 5))
    ind_dict = {1: 0, 0: 1, (0, 1): (), (): (0, 1), None: None}

    def check_op(op, ind):
        C = getattr(sdb, op)(A, ind, scidb_syntax=True)
        if op in ['var', 'std']:
            C_np = getattr(np, op)(A.toarray(), ind_dict[ind], ddof=1)
        else:
            C_np = getattr(np, op)(A.toarray(), ind_dict[ind])
        assert_allclose(C.toarray(), C_np, rtol=1E-6)

    for op in ['min', 'max', 'sum', 'var', 'std', 'mean']:
        for ind in [None, 0, 1, (0, 1), ()]:
            # some aggregates produce nulls.  We won't test these
            if ind == (0, 1) and op in ['var', 'std', 'mean']:
                continue
            yield check_op, op, ind


def test_numpy_aggregates():
    A = sdb.random((5, 5))

    def check_op(op, ind):
        C = getattr(sdb, op)(A, ind)
        if op in ['var', 'std']:
            C_np = getattr(np, op)(A.toarray(), ind, ddof=1)
        else:
            C_np = getattr(np, op)(A.toarray(), ind)
        assert_allclose(C.toarray(), C_np, rtol=1E-6)

    for op in ['min', 'max', 'sum', 'var', 'std', 'mean']:
        for ind in [None, 0, 1, (0, 1), ()]:
            # some aggregates produce nulls.  We won't test these
            if ind == () and op in ['var', 'std', 'mean']:
                continue
            yield check_op, op, ind


def test_transpose():
    A = sdb.random((5, 4, 3))

    for args in [(1, 0, 2), ((2, 0, 1),), (None,), (2, 1, 0)]:
        AT = A.transpose(*args).toarray()
        npAT = A.toarray().transpose(*args)
        assert_allclose(AT, npAT)

    assert_allclose(A.T.toarray(), A.toarray().T)


def test_join():
    A = sdb.randint(10)
    B = sdb.randint(10)
    C = sdb.join(A, B)

    Cnp = C.toarray()
    names = Cnp.dtype.names
    assert_allclose(Cnp[names[0]], A.toarray())
    assert_allclose(Cnp[names[1]], B.toarray())


def test_cross_join():
    A = sdb.random((10, 5))
    B = sdb.random(10)
    AB = sdb.cross_join(A, B, (0, 0))

    ABnp = AB.toarray()
    names = ABnp.dtype.names
    assert_allclose(ABnp[names[0]],
                    A.toarray())
    assert_allclose(ABnp[names[1]],
                    B.toarray()[:, None] + np.zeros(A.shape[1]))


def test_regrid():
    A = sdb.random((8, 4))
    Ag = A.regrid(2, "sum")

    np_A = A.toarray()
    np_Ag = sum(np_A[i::2, j::2] for i in range(2) for j in range(2))

    assert_allclose(Ag.toarray(), np_Ag)


def test_bad_url():

    with pytest.raises(ValueError) as cm:
        SciDBShimInterface('http://www.google.com')
    assert cm.value.args[0] == ('Could not connect to a SciDB instance at '
                                'http://www.google.com')


def test_random_persistent():
    """Regression test for #38"""
    x = sdb.random((8, 4), persistent=True)
    assert x.persistent
    assert x.shape == (8, 4)
    x.persistent = False
    x.reap()

    x = sdb.randint((8, 4), persistent=True)
    assert x.persistent
    assert x.shape == (8, 4)
    x.persistent = False
    x.reap()


def test_reap():

    A = sdb.random((8, 4))
    name = A.name
    A.reap()
    assert name not in sdb.list_arrays()
    assert A.interface is None
    assert A.name == '__DELETED__'


def test_reap_ignored_if_persistent():

    A = sdb.random((1, 1))
    A.persistent = True
    name = A.name

    A.reap(ignore=True)
    assert name in sdb.list_arrays()
    assert A.name is name
    assert A.interface is sdb

    A.persistent = False
    A.reap()
    assert name not in sdb.list_arrays()


def test_interface_reap():

    A = sdb.random((1, 1))
    B = sdb.random((1, 1))

    aname = A.name
    bname = B.name

    sdb.reap()

    assert aname not in sdb.list_arrays()
    assert bname not in sdb.list_arrays()
    assert A.interface is None
    assert A.name == "__DELETED__"


@pytest.mark.parametrize('shp', ((15,), (10, 10), (3, 3, 3)))
def test_sparse_to_dense(shp):
    x = sdb.random(shp)
    expected = x.toarray()
    expected[expected <= 0.5] = 0

    actual = sdb.afl.filter(x, 'f0 > 0.5').toarray()
    np.testing.assert_allclose(expected, actual)


@pytest.mark.parametrize('shp', ((15,), (10, 10), (3, 3, 3)))
def test_sparse_to_dense_multiatt(shp):
    x = sdb.random(shp)
    y = sdb.afl.apply(x, 'f1', 'f0 + 1').eval()
    expected = y.toarray()
    expected[expected['f0'] <= 0.5] = (0, 0)

    actual = sdb.afl.filter(y, 'f0 > 0.5').toarray()
    np.testing.assert_allclose(expected['f0'], actual['f0'])
    np.testing.assert_allclose(expected['f1'], actual['f1'])


def test_interface_reap_after_manual_reap_is_silent():

    A = sdb.random((1, 1))
    A.reap()
    sdb.reap()


def test_reap_called_on_context_manager():
    with connect() as sdb2:
        X = sdb2.random((1, 1))
        name = X.name
        assert X.name in sdb.list_arrays()

    assert X.name == "__DELETED__"
    assert name not in sdb.list_arrays()
