# License: Simplified BSD, 2014
# See LICENSE.txt for more information
import re

import numpy as np

from ._py3k_compat import string_type

__all__ = ['meshgrid', 'broadcastable', 'iter_record', 'as_list']


# This is back-ported from numpy version 1.7
# The earlier version of meshgrid only accepts two arrays
def meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from two or more coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
         If True a sparse grid is returned in order to conserve memory.
         Default is False.
    copy : bool, optional
        If False, a view into the original arrays are returned in
        order to conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous arrays.
        Furthermore, more than one element of a broadcast array may refer to
        a single memory location.  If you need to write to the arrays, make
        copies first.

    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.
    Giving the string 'ij' returns a meshgrid with matrix indexing,
    while 'xy' returns a meshgrid with Cartesian indexing.  In the 2-D case
    with inputs of length M and N, the outputs are of shape (N, M) for 'xy'
    indexing and (M, N) for 'ij' indexing.  In the 3-D case with inputs of
    length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M,
    N, P) for 'ij' indexing.  The difference is illustrated by the following
    code snippet::

        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.

    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)

    """
    if len(xi) < 2:
        msg = ('meshgrid() takes 2 or more arguments (%d given)'
               % int(len(xi) > 0))
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)

    copy_ = kwargs.get('copy', True)
    sparse = kwargs.get('sparse', False)
    indexing = kwargs.get('indexing', 'xy')
    if not indexing in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::])
              for i, x in enumerate(args)]

    shape = [x.size for x in output]

    if indexing == 'xy':
        # switch first and second axis
        output[0].shape = (1, -1) + (1,) * (ndim - 2)
        output[1].shape = (-1, 1) + (1,) * (ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)


def broadcastable(shape1, shape2):
    """Check whether two array shapes are broadcastable in NumPy

    Parameters
    ----------
    shape1 : list or tuple
        shape of the first array
    shape2 : list or tuple
        shape of the second array

    Returns
    -------
    broadcastable : boolean
        True if an array of shape1 and an array of shape2 are broadcastable
    """
    return all((i1 == 1 or i2 == 1 or i1 == i2)
               for (i1, i2) in zip(reversed(shape1), reversed(shape2)))


def slice_syntax(f):
    """
    This decorator wraps a function that accepts a tuple of slices.

    After wrapping, the function acts like a property that accepts
    bracket syntax (e.g., p[1:3, :, :])

    Parameters
    ----------
    f : function
    """

    def wrapper(self):
        result = SliceIndexer(f, self)
        result.__doc__ = f.__doc__
        return result
    result = property(wrapper)
    return result


class SliceIndexer(object):

    def __init__(self, func, _other):
        self._func = func
        self._other = _other

    def __getitem__(self, view):
        return self._func(self._other, view)


def _is_query(name_or_query):
    """
    Returns True if input syntax matches an AFL query
    """
    if '(' in name_or_query:
        return True


def iter_record(item):
    """
    Iterator over items in a single array record, or yield a scalar.

    This provides a uniform way to iterate over each scalar item
    in an array element, regardless of whether the array dtype
    is a scalar or record
    """
    if item.dtype.fields is not None:
        for i in item:
            yield i
    else:
        yield item


def new_attribute_label(suggestion='val', *arrays):
    """Return a new attribute label

    The label will not clash with any attribute or dimension labels in the given arrays
    """
    label_list = sum([[dim[0] for dim in arr.sdbtype.full_rep]
                      for arr in arrays], [])
    label_list += sum([a.dim_names for a in arrays], [])

    if suggestion not in label_list:
        return suggestion
    else:
        # find all labels of the form val_0, val_1, val_2 ... etc.
        # where `val` is replaced by suggestion
        R = re.compile(r'^{0}_(\d+)$'.format(suggestion))
        nums = sum([list(map(int, R.findall(label))) for label in label_list], [])

        nums.append(-1)  # in case it's empty
        return '{0}_{1}'.format(suggestion, max(nums) + 1)

_new_attribute_label = new_attribute_label


def as_list(x):
    if isinstance(x, string_type):
        return [x]
    try:
        return list(x)
    except TypeError: # not iterable
        return [x]
