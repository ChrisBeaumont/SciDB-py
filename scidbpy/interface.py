"""
Low-level Interface to SciDB
============================
These interfaces are designed to be extensible and allow various interfaces
to the SciDB engine.

The following interfaces are currently available:

- SciDBShimInterface : interface via HTTP using Shim [1]_

[1] https://github.com/Paradigm4/shim
"""
# License: Simplified BSD, 2013
# See LICENSE.txt for more information

from __future__ import print_function, unicode_literals
import warnings
import abc
import os
import atexit
import logging
import csv
from time import time
from itertools import product
from fnmatch import fnmatch

import requests

from ._py3k_compat import (quote,
                           iteritems, string_type, reduce)

import re
import numpy as np
from .scidbarray import SciDBArray, SciDBDataShape, ArrayAlias, SDB_IND_TYPE
from .errors import SHIM_ERROR_DICT, SciDBQueryError, SciDBInvalidSession
from .utils import broadcastable, _is_query, iter_record, _new_attribute_label, as_list
from .schema_utils import (disambiguate, as_row_vector, as_column_vector,
                           zero_indexed, match_dimensions,
                           assert_single_attribute,
                           boundify, match_chunks, redimension,
                           match_chunk_permuted, assert_schema,
                           right_dimension_pad, left_dimension_pad)
from .robust import (join, merge, gemm, uniq, gesvd)

from . import arithmetic

__all__ = ['SciDBInterface', 'SciDBShimInterface', 'connect']

SCIDB_RAND_MAX = 2147483647  # 2 ** 31 - 1
UNESCAPED_QUOTE = re.compile(r"(?<!\\)'")


def _df(arr, ind):
    if not isinstance(arr, ArrayAlias):
        arr = ArrayAlias(arr)
    return "{{a.d{i}f}}".format(i=ind).format(a=arr)


def _af(arr, ind):
    if not isinstance(arr, ArrayAlias):
        arr = ArrayAlias(arr)
    return "{{a.a{i}f}}".format(i=ind).format(a=arr)


def _scidb_serialize(arr, chunk_size):
    """
    Serialize a multidimensional numpy array into a 1D array,
    with an interleaving scheme that matches SciDB.

    Such a scheme first interleaves chunks in C-contiguous order.
    Then it interleaves cells in the chunk in C-contiguous order.
    """
    chunk_size = as_list(chunk_size)
    if len(chunk_size) == 1:
        chunk_size = chunk_size * arr.ndim
    result = []
    for start in product(*(range(0, s, c)
                           for s, c in zip(arr.shape, chunk_size))):
        slices = tuple(slice(i, i + s)
                       for i, s in zip(start, chunk_size))
        result.append(arr[slices].ravel())
    return np.hstack(result)


def _to_bytes(arr, chunk_size=1000):
    """
    Convert a numpy array to a bytestring in SciDB's binary format
    """
    arr = _scidb_serialize(arr, chunk_size)

    # easy case: no strings, SciDB format matches numpy format
    if not any(np.issubdtype(t, np.character) for l, t in arr.dtype.descr):
        return arr.tostring(order='C')

    # some attributes are strings
    # very inefficient like this.
    result = []
    for item in arr.ravel():
        for datum in iter_record(item):
            dtype = datum.dtype
            if np.issubdtype(dtype, np.character):
                datum = datum.astype('U').tostring().decode('utf32').encode('utf8')
                sz = len(datum) + 1
                prefix = np.int32(sz).newbyteorder('<').tostring()
                result.append(prefix + datum + b'\x00')
            else:
                result.append(np.array(datum, dtype=dtype).tostring())
    result = b''.join(result)
    return result


class SciDBInterface(object):

    def __init__(self):
        self._created = []
        self._persistent = set()
        atexit.register(self.reap)

    """SciDBInterface Abstract Base Class.

    This class provides a wrapper to the low-level interface to sciDB.  The
    actual communication with the database should be implemented in
    subclasses.

    Subclasses should implement the following methods, with descriptions given
    below:

    - ``_execute_query``
    - ``_upload_bytes``
    - ``_release_session``
    """
    __metaclass__ = abc.ABCMeta

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.reap()

    @abc.abstractmethod
    def _execute_query(self, query, response=False, n=0, fmt='auto'):
        """Execute a query on the SciDB engine

        Parameters
        ----------
        query : string
            A string representing a SciDB query.  This string will be
            executed on the engine.
        response : boolean (optional)
            Indicate whether the query returns a response.  If True, then
            the response is formatted according to the ``n`` and ``fmt``
            keywords.  Default is False.
        n : integer
            number of lines to return.  If n=0 (default), then return all
            lines.  Only accessed if response=True.
        fmt : str
            format code for the returned values.  Options are:
            =================== ===============================================
            Format Code         Description
            =================== ===============================================
            auto (default)      SciDB array format.
            csv	                Comma-separated values (CSV)
            csv+                CSV with dimension indices.
            dcsv                CSV, distinguishing between dims and attrs
            dense               Ideal for viewing 2-dimensional matrices.
            lcsv+               Like csv+, with a flag indicating empty cells.
            sparse              Sparse SciDB array format.
            lsparse             Sparse format with flag indicating empty cells.
            (type1, type2...)   Raw binary format
            =================== ===============================================
        Returns
        -------
        results : string, bytes, or None
            If response is False, None is returned.  If response is True, then
            either a string or a byte string is returned, depending on the
            value of ``fmt``.
        """
        logging.getLogger(__name__).debug(query)
        if not hasattr(self, '_query_log'):
            self._query_log = []
        self._query_log.append(query)

    @abc.abstractmethod
    def _upload_bytes(self, data):
        """Upload binary data to the SciDB engine

        Parameters
        ----------
        data : bytestring
            The raw byte data to upload to a file on the SciDB server.
        Returns
        -------
        (filename, session_id)

        filename : string
            The name of the resulting file on the SciDB server.
        session : object
            The session ID associated with the file

        It is the responsibility of the caller of _upload_bytes
        to call _free_session with the returned session object
        once finished with the uploaded file.
        """
        pass

    @abc.abstractmethod
    def _get_uid(self):
        """Get a unique query ID from the database"""
        pass

    @abc.abstractmethod
    def _release_session(self, session):
        """
        Close a session with the database.
        """
        pass

    def reap(self):
        """
        Reap all arrays created via new_array
        """
        for array in list(self._created):
            if array in self._persistent:
                continue
            try:
                self.query("remove({0})", array)
            except SciDBQueryError:  # array does not exist
                pass

        self._created = []

    def _db_array_name(self):
        """Return a unique array name for a new array on the database"""
        arr_key = 'py'

        if not hasattr(self, 'uid'):
            self.uid = self._get_uid()

        if not hasattr(self, 'array_count'):
            self.array_count = 1
        else:
            # on subsequent calls, increment the array count
            self.array_count += 1

        result = "{0}{1}_{2:05}".format(arr_key, self.uid, self.array_count)
        self._created.append(result)
        return result

    def _scan_array(self, name, **kwargs):
        """Return the contents of the given array"""
        if 'response' not in kwargs:
            kwargs['response'] = True
        if _is_query(name):
            return self._execute_query(name, **kwargs)
        return self._execute_query("scan({0})".format(name), **kwargs)

    def _show_array(self, name, **kwargs):
        """Show the schema of the given array"""
        if 'response' not in kwargs:
            kwargs['response'] = True
        name = UNESCAPED_QUOTE.sub(r"\'", name)
        if _is_query(name):
            # need to add a fake store command to trigger
            # att/dim disambiguation
            tmp = self._db_array_name()
            query = "show('store({0}, {1})', 'afl')".format(name, tmp)
        else:
            query = "show({0})".format(name)

        return self._execute_query(query, **kwargs)

    def _array_dimensions(self, name, **kwargs):
        """Show the dimensions of the given array"""
        if 'response' not in kwargs:
            kwargs['response'] = True
        return self._execute_query("dimensions({0})".format(name), **kwargs)

    @property
    def afl(self):
        if not hasattr(self, '_afl'):
            from .afl import AFLNamespace
            self._afl = AFLNamespace(self)
        return self._afl

    def wrap_array(self, scidbname, persistent=True):
        """
        Create a new SciDBArray object that references an existing SciDB
        array

        Parameters
        ----------
        scidbname : string
            Wrap an existing scidb array referred to by `scidbname`. The
            SciDB array object persistent value will be set to True, and
            the object shape, datashape and data type values will be
            determined by the SciDB array.
        persistent : boolean
            If True (default) then array will not be deleted when this
            variable goes out of scope. Warning: if persistent is set to
            False, data could be lost!
        """
        # TODO: use SciDBArray.wrap_array() here; test that it works
        schema = self._show_array(scidbname, fmt='csv')
        datashape = SciDBDataShape.from_schema(schema)
        return SciDBArray(datashape, self, scidbname, persistent=persistent)

    # TODO: give the option to pass a user-defined array name
    #       (use this in copy(), rename(), and others)
    def new_array(self, shape=None, dtype='double', persistent=False,
                  **kwargs):
        """
        Create a new array, either instantiating it in SciDB or simply
        reserving the name for use in a later query.

        Parameters
        ----------
        shape : int or tuple (optional)
            The shape of the array to create.  If not specified, no array
            will be created and a name will simply be reserved for later use.
            WARNING: if shape=None and persistent=False, an error will result
            when the array goes out of scope, unless the name is used to
            create an array on the server.
        dtype : string (optional)
            the datatype of the array.  This is only referenced if `shape`
            is specified.  Default is 'double'.
        persistent : boolean (optional)
            whether the created array should be persistent, i.e. survive
            in SciDB past when the object wrapper goes out of scope.  Default
            is False.
        **kwargs : (optional)
            If `shape` is specified, additional keyword arguments are passed
            to SciDBDataShape.  Otherwise, these will not be referenced.
        Returns
        -------
        arr : SciDBArray
            wrapper of the new SciDB array instance.
        """
        name = self._db_array_name()
        if shape is not None or ('dim_low' in kwargs and 'dim_high' in kwargs):
            datashape = SciDBDataShape(shape, dtype, **kwargs)
            query = "CREATE ARRAY {0} {1}".format(name, datashape.schema)
            self._execute_query(query)
        else:
            datashape = None
        result = SciDBArray(datashape, self, name, persistent=persistent)
        return result

    def _format_query_string(self, query, *args, **kwargs):
        """Format query string.

        See query() documentation for more information
        """
        parse = lambda x: ArrayAlias(x) if isinstance(x, SciDBArray) else x
        args = (parse(v) for v in args)
        kwargs = dict((k, parse(v)) for k, v in iteritems(kwargs))
        query = query.format(*args, **kwargs)
        return query

    # TODO: add query_and_store() convenience routine.  Many of the
    #       uses of query are wrapped in a call to store(...).
    def query(self, query, *args, **kwargs):
        """Perform a query on the database.

        This wraps a query constructor which allows the creation of
        sophisticated SciDB queries which act on arrays wrapped by SciDBArray
        objects.  See Notes below for details.

        Parameters
        ----------
        query : string
            The query string, with curly-braces to indicate insertions
        *args, **kwargs :
            Values to be inserted (see below).

        Details
        -------
        The query string uses the python string formatting convention, with
        appropriate substitutions made for arguments or keyword arguments that
        are subclasses of SciDBAttribute, such as SciDBArrays or their indices.

        For example, a 3x3 identity matrix can be created using the query
        function as follows:

        >>> sdb = SciDBShimInterface('http://localhost:8080')
        >>> A = sdb.new_array((3, 3), dtype, **kwargs)
        >>> sdb.query('store(build({A}, iif({A.d0}={A.d1}, 1, 0)), {A})', A=A)
        >>> A.toarray()
        array([[ 1.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        The query string follows the python string formatting conventions,
        where variables to be replaced are indicated by curly braces
        ('{' and '}'). The contents of the braces should refer to either an
        argument index (e.g. ``{0}`` references the first non-keyword
        argument) or the name of a keyword argument (e.g. ``{A}`` references
        the keyword argument ``A``).  Arguments of type SciDBAttribute (such
        as SciDBArrays, SciDBArray indices, etc.) have their ``name`` attribute
        inserted.  Dimension and Attribute names can be inserted using the
        following syntax.  As an example, we'll use an array with the following
        schema:

            myarray<val:double,label:int64> [i=0:4,5,0,j=0:4,5,0]

        The dimension/attribute shortcuts are as follows:

        - A.d0, A.d1, A.d2... : insert array dimension names.
          For the above case, '{A.d0}' will be translated to 'i','
          and '{A.d1}' will be translated to 'j'.
        - A.d0f, A.d1f, A.d2f... : insert fully-qualified dimension names.
          For the above case, '{A.d0f}' will be translated to 'myarray.i',
          and '{A.d1f}' will be translated to 'myarray.j'.
        - A.a0, A.a1, A.a2... : insert array attribute names.
          For the above case, '{A.a0}' will be translated to 'val',
          and '{A.a1}' will be translated to 'label'.
        - A.a0f, A.a1f, A.a2f... : insert fully-qualified attribute names.
          For the above case, '{A.a0f}' will be translated to 'myarray.val',
          and '{A.a1f}' will be translated to 'myarray.label'.

        All other argument types are inserted as-is, i.e. with their string
        representation.
        """
        qstring = self._format_query_string(query, *args, **kwargs)
        return self._execute_query(qstring)

    def list_arrays(self, parsed=True, n=0):
        """List the arrays currently in the database

        Parameters
        ----------
        parsed : boolean
            If True (default), then parse the results into a dictionary of
            array names as keys, schema as values
        n : integer
            the maximum number of arrays to list.  If n=0, then list all

        Returns
        -------
        array_list : string or dictionary
            The list of arrays.  If parsed=True, then the result is returned
            as a dictionary.
        """
        # TODO: more stable way to do this than string parsing?
        arr_list = self._execute_query("list('arrays')", n=n, response=True)
        if parsed:
            if arr_list.strip() == '[]':
                arr_list = {}
            else:
                # find the correct quote char for the output.
                # Depending on the SciDB version, it could be " or '
                quotechar = arr_list[2]
                if quotechar not in ["'", '"']:
                    raise ValueError("output of list('arrays') "
                                     "is not in the expected format")
                R = re.compile(r'\(([^\(\)]*)\)')
                splits = R.findall(arr_list)
                arr_list = dict((a[0], a[1:])
                                for a in csv.reader(splits,
                                                    quotechar=quotechar))
        return arr_list

    def ls(self, pattern='*'):
        """
        List the arrays in the database, optionally matching to a pattern

        Parameters
        ----------
        pattern : String (optional)
            A glob-style pattern string. If present, only arrays whose names
            match the pattern are displayed. '*' matches any string,
            '?' matches any character

        Returns
        -------
        result : list
            A list of SciDB array names
        """
        result = self.list_arrays()
        return [r for r in result if fnmatch(r, pattern)]


    def remove(self, array):
        """
        Remove an array from the database

        This removes the array even if its persistent property is True!

        Parameters
        ----------
        array : str or SciDBArray
            The array (or name of array) to remove

        See Also
        --------
        reap(), SciDBArray.reap()
        """
        self.query("remove({0})", array)

    def ones(self, shape, dtype='double', **kwargs):
        """Return an array of ones

        Parameters
        ----------
        shape : tuple or int
            The shape of the array
        dtype : string or list
            The data type of the array
        **kwargs :
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of all ones.
        """
        arr = self.new_array(shape, dtype, **kwargs)
        self.afl.build(arr, 1).eval(out=arr)
        return arr

    def zeros(self, shape, dtype='double', **kwargs):
        """Return an array of zeros

        Parameters
        ----------
        shape : tuple or int
            The shape of the array
        dtype : string or list
            The data type of the array
        **kwargs :
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of all zeros.
        """
        schema = SciDBDataShape(shape, dtype, **kwargs).schema
        return self.afl.build(schema, 0).eval()

    def random(self, shape, dtype='double', lower=0, upper=1, persistent=False,
               **kwargs):
        """Return an array of random floats between lower and upper

        Parameters
        ----------
        shape : tuple or int
            The shape of the array
        dtype : string or list
            The data type of the array
        lower : float
            The lower bound of the random sample (default=0)
        upper : float
            The upper bound of the random sample (default=1)
        persistent : bool
            Whether the new array is persistent (default=False)
        **kwargs :
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of random floating point numbers,
            uniformly distributed between `lower` and `upper`.
        """
        # TODO: can be done more efficiently
        #       if lower is 0 or upper - lower is 1
        array = self.new_array(persistent=persistent)
        schema = SciDBDataShape(shape, dtype, **kwargs).schema
        rng = (upper - lower) / float(SCIDB_RAND_MAX)
        fill_value = 'random()*{0}+{1}'.format(rng, lower)
        return self.afl.build(schema, fill_value).eval(out=array)

    def randint(self, shape, dtype='uint32', lower=0, upper=SCIDB_RAND_MAX,
                persistent=False, **kwargs):
        """Return an array of random integers between lower and upper

        Parameters
        ----------
        shape : tuple or int
            The shape of the array
        dtype : string or list
            The data type of the array
        lower : float
            The lower bound of the random sample (default=0)
        upper : float
            The upper bound of the random sample (default=2147483647)
        persistent : bool
            Whether the array is persistent (default=False)
        **kwargs :
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr: SciDBArray
            A SciDBArray consisting of random integers, uniformly distributed
            between `lower` and `upper`.
        """
        array = self.new_array(persistent=persistent)
        schema = SciDBDataShape(shape, dtype, **kwargs).schema
        fill_value = 'random() % {0} + {1}'.format(upper - lower, lower)
        return self.afl.build(schema, fill_value).eval(out=array)

    def arange(self, start, stop=None, step=1, dtype=None, **kwargs):
        """arange([start,] stop[, step,], dtype=None, **kwargs)

        Return evenly spaced values within a given interval.

        Values are generated within the half-open interval ``[start, stop)``
        (in other words, the interval including `start` but excluding `stop`).
        For integer arguments the behavior is equivalent to the Python
        `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
        but returns an ndarray rather than a list.

        When using a non-integer step, such as 0.1, the results will often not
        be consistent.  It is better to use ``linspace`` for these cases.

        Parameters
        ----------
        start : number, optional
            Start of interval.  The interval includes this value.  The default
            start value is 0.
        stop : number
            End of interval.  The interval does not include this value, except
            in some cases where `step` is not an integer and floating point
            round-off affects the length of `out`.
        step : number, optional
            Spacing between values.  For any output `out`, this is the distance
            between two adjacent values, ``out[i+1] - out[i]``.  The default
            step size is 1.  If `step` is specified, `start` must also be
            given.
        dtype : dtype
            The type of the output array.  If `dtype` is not given, it is
            inferred from the type of the input arguments.
        **kwargs :
            Additional arguments are passed to SciDBDatashape when creating
            the output array.

        Returns
        -------
        arange : SciDBArray
            Array of evenly spaced values.

            For floating point arguments, the length of the result is
            ``ceil((stop - start)/step)``.  Because of floating point overflow,
            this rule may result in the last element of `out` being greater
            than `stop`.
        """
        if stop is None:
            stop = start
            start = 0

        if dtype is None:
            dtype = np.array(start + stop + step).dtype

        size = int(np.ceil((stop - start) * 1. / step))

        arr = self.new_array(size, dtype, **kwargs)

        fill_value = '{0} + {1} * {2}'.format(start, step,
                                              arr.dim_names[0])
        self.afl.build(arr, fill_value).eval(out=arr)
        return arr

    def linspace(self, start, stop, num=50,
                 endpoint=True, retstep=False, **kwargs):
        """
        Return evenly spaced numbers over a specified interval.

        Returns `num` evenly spaced samples, calculated over the
        interval [`start`, `stop` ].

        The endpoint of the interval can optionally be excluded.

        Parameters
        ----------
        start : scalar
            The starting value of the sequence.
        stop : scalar
            The end value of the sequence, unless `endpoint` is set to False.
            In that case, the sequence consists of all but the last of
            ``num + 1`` evenly spaced samples, so that `stop` is excluded.
            Note that the step size changes when `endpoint` is False.
        num : int, optional
            Number of samples to generate. Default is 50.
        endpoint : bool, optional
            If True, `stop` is the last sample. Otherwise, it is not included.
            Default is True.
        retstep : bool, optional
            If True, return (`samples`, `step`), where `step` is the spacing
            between samples.
        **kwargs :
            additional keyword arguments are passed to SciDBDataShape

        Returns
        -------
        samples : SciDBArray
            There are `num` equally spaced samples in the closed interval
            ``[start, stop]`` or the half-open interval ``[start, stop)``
            (depending on whether `endpoint` is True or False).
        step : float (only if `retstep` is True)
            Size of spacing between samples.
        """
        num = int(num)

        if endpoint:
            step = (stop - start) * 1. / (num - 1)
        else:
            step = (stop - start) * 1. / num

        arr = self.new_array(num, **kwargs)
        fill_value = '{0} + {1} * {2}'.format(start, step, arr.dim_names[0])
        self.afl.build(arr, fill_value).eval(out=arr)

        if retstep:
            return arr, step
        else:
            return arr

    def identity(self, n, dtype='double', sparse=False, **kwargs):
        """Return a 2-dimensional square identity matrix of size n

        Parameters
        ----------
        n : integer
            the number of rows and columns in the matrix
        dtype : string or list
            The data type of the array
        sparse : boolean
            specify whether to create a sparse array (default=False)
        **kwargs :
            Additional keyword arguments are passed to SciDBDataShape.

        Returns
        -------
        arr : SciDBArray
            A SciDBArray containint an [n x n] identity matrix
        """
        if len(as_list(dtype)) > 1:
            raise NotImplementedError("Identity matrices must have 1 attribute")
        dtype = as_list(dtype)[0]

        query = self.afl.build('<x:%s>[i0=0:%i,1000,0,i1=0:%i,1000,0]' % (dtype, n - 1, n - 1),
                               'iif(i0=i1,1,0)')

        if sparse:
            # redimension converts NULL to empty
            query = query.apply('i', 'iif(x=1, i0, NULL)', 'j', 'iif(x=1, i1, NULL)')
            query = query.redimension('<x:%s>[i=0:%i,1000,0,j=0:%i,1000,0]' % (dtype, n - 1, n - 1))
        return query.eval()

    def dot(self, A, B):
        """Compute the matrix product of A and B

        Parameters
        ----------
        A : SciDBArray
            A must be a two-dimensional matrix of shape (n, p)
        B : SciDBArray
            B must be a two-dimensional matrix of shape (p, m)

        Returns
        -------
        C : SciDBArray
            The wrapper of the SciDB Array, of shape (n, m), consisting of the
            matrix product of A and B
        """

        # TODO: make matrix-vector and vector-vector cases more efficient.
        #       Currently they involve creating copies of the arrays, but this
        #       is just a place-holder for a more efficient implementation.
        if A.ndim not in (1, 2) or B.ndim not in (1, 2):
            raise ValueError("dot requires 1 or 2-dimensional arrays")

        A = boundify(A)
        B = boundify(B)

        # TODO: the following four transformations should be done by building
        #       a single query rather than executing separate queries.
        #       The following should be considered a place-holder for right
        #       now.
        if A.ndim == 1:
            A = as_row_vector(A)

        if B.ndim == 1:
            B = as_column_vector(B)

        A = zero_indexed(A)
        B = zero_indexed(B)

        A, B = match_dimensions(A, B, ((1, 0),))

        if A.sdbtype.nullable[0]:
            A = A.substitute(0)

        if B.sdbtype.nullable[0]:
            B = B.substitute(0)

        # TODO: is there a more efficient way to do this than instantiating
        #       an array of zeros?
        # TODO: use spgemm() when the matrices are sparse
        kwargs = {'dtype': 'double'}
        if A.dim_names[0] != B.dim_names[1]:
            kwargs['dim_names'] = [A.dim_names[0], B.dim_names[1]]
        C = self.zeros((A.shape[0], B.shape[-1]), **kwargs)
        C = gemm(A, B, C).eval()

        if C.size == 1:
            return C[0, 0]
        if C.shape[0] == 1:
            return C[0, :]
        elif C.shape[1] == 1:
            return C[:, 0]
        else:
            return C

    def svd(self, A, return_U=True, return_S=True, return_VT=True):
        """Compute the Singular Value Decomposition of the array A:

        A = U.S.V^T

        Parameters
        ----------
        A : SciDBArray
            The array for which the SVD will be computed.  It should be a
            2-dimensional array with a single value per cell.  Currently, the
            svd routine requires non-overlapping chunks of size 32.
        return_U, return_S, return_VT : boolean
            if any is True, then return the associated array.  All are True
            by default

        Returns
        -------
        [U], [S], [VT] : SciDBArrays
            Arrays storing the singular values and vectors of A.
        """
        if (A.ndim != 2):
            raise ValueError("svd requires 2-dimensional arrays")
        self.afl.load_library("'dense_linear_algebra'")

        out_dict = dict(U=return_U, S=return_S, VT=return_VT)

        # TODO: check that data type is double and chunk size is 32
        ret = []
        for output in ['U', 'S', 'VT']:
            if out_dict[output]:
                ret.append(gesvd(A, "'%s'" % output))
        return tuple(ret)

    def from_array(self, A, instance_id=0, chunk_size=1000, **kwargs):
        """Initialize a scidb array from a numpy array

        Parameters
        ----------
        A : array_like (numpy array or sparse array)
            input array from which the scidb array will be created
        instance_id : integer
            the instance ID used in loading
            (default=0; see SciDB documentation)
        chunk_size : integer or list of integers
            The chunk size of the uploaded SciDBArray. Default=1000
        **kwargs :
            Additional keyword arguments are passed to new_array()

        Returns
        -------
        arr : SciDBArray
            SciDB Array object built from the input array
        """
        if 'chunk_overlap' in kwargs:
            raise ValueError("chunk_overlap not supported by from_array. "
                             "Use schema_utils.rechunk instead")
        q = self.afl.quote
        A = np.asarray(A)
        instance_id = int(instance_id)

        filename, session_id = self._upload_bytes(_to_bytes(A, chunk_size=chunk_size))
        filename = q(filename)

        arr = self.new_array(A.shape, A.dtype, chunk_size=chunk_size, **kwargs)
        self.afl.load(arr, filename, instance_id,
                      q(arr.sdbtype.bytes_fmt)).eval(store=False)
        self._release_session(session_id)

        return arr

    def from_dataframe(self, A, instance_id=0, **kwargs):
        """Initialize a scidb array from a pandas dataframe

        Parameters
        ----------
        A : pandas dataframe
            data from which the scidb array will be created.
        instance_id : integer
            the instance ID used in loading
            (default=0; see SciDB documentation)
        **kwargs :
            Additional keyword arguments are passed to new_array()

        Returns
        -------
        arr : SciDBArray
            SciDB Array object built from the input array
        """
        # load to SciDB via a record array (with index in 'index' field)
        A_rec = A.to_records(index=True)
        A_sdb = self.from_array(A_rec)

        # TODO: rename index if value is passed?
        if 'dim_names' in kwargs:
            warnings.warn("from_dataframe: ignoring 'dim_names' argument")
        kwargs['dim_names'] = [A_rec.dtype.names[0]]

        # redimension the array on the index
        try:
            dim_low = (m for m in A.index.min())
            dim_high = (m for m in A.index.max())
        except TypeError:
            dim_low = (A.index.min(),)
            dim_high = (A.index.max(),)

        arr = self.new_array(dim_low=dim_low, dim_high=dim_high,
                             dtype=A_rec.dtype.descr[1:],
                             **kwargs)
        self.afl.redimension_store(A_sdb, arr).eval(store=False)
        return arr

    def from_sparse(self, A, instance_id=0, **kwargs):
        """Initialize a scidb array from a sparse array

        Parameters
        ----------
        A : sparse array
            sparse input array from which the scidb array will be created.
            Note that this array will internally be converted to COO format.
        instance_id : integer
            the instance ID used in loading
            (default=0; see SciDB documentation)
        **kwargs :
            Additional keyword arguments are passed to new_array()

        Returns
        -------
        arr : SciDBArray
            SciDB Array object built from the input array
        """
        try:
            A = A.tocoo()
        except:
            raise ValueError("input must be a scipy.sparse matrix")
        instance_id = int(instance_id)

        if 'dim_names' not in kwargs:
            kwargs['dim_names'] = ['i0', 'i1']

        if len(kwargs['dim_names']) != 2:
            raise ValueError("dim_names must have two dimensions")
        d1, d2 = kwargs['dim_names']

        # first flatten the array & indices
        # We'll treat the general case where A.data can be a record array,
        # though this will be very uncommon.
        flat_dtype = [(str(d1), SDB_IND_TYPE),
                      (str(d2), SDB_IND_TYPE)]
        if A.dtype.names is None:
            flat_dtype += [(str('f0'), A.dtype)]
        else:
            flat_dtype += A.descr

        M = np.empty(len(A.data), dtype=flat_dtype)
        M[d1] = A.row
        M[d2] = A.col
        M['f0'] = A.data
        arr_flat = self.from_array(M)

        # redimension the flat array to a sparse array
        arr = self.new_array(A.shape, A.dtype, **kwargs)
        self.afl.redimension_store(arr_flat, arr).eval(store=False)
        return arr

    def toarray(self, A, transfer_bytes=True):
        """Convert a SciDB array to a numpy array"""
        return A.toarray(transfer_bytes=transfer_bytes)

    def tosparse(self, A, sparse_fmt='recarray', transfer_bytes=True):
        """Convert a SciDB array to a sparse representation"""
        return A.tosparse(sparse_fmt=sparse_fmt, transfer_bytes=transfer_bytes)

    def todataframe(self, A, transfer_bytes=True):
        """Convert a SciDB array to a pandas dataframe"""
        return A.todataframe(transfer_bytes=transfer_bytes)

    def _from_file(self, filename, **kwargs):
        # TODO: allow creation of arrays from uploaded files
        # TODO: allow creation of arrays from pre-existing files within the
        #       database
        raise NotImplementedError()

    def unique(self, x, is_sorted=False):
        """
        Store the unique elements of an array in a new array

        Parameters
        ----------
        x : SciDBArray
            The array to compute unique elements of.
        is_sorted : bool
            Whether the array is pre-sorted. If True,
            x must be a 1D array.

        Returns
        -------
        u : SciDBArray
           The unique elements of x
        """
        return uniq(x, is_sorted=is_sorted)

    def _apply_func(self, A, func):
        # TODO: new value name could conflict.  How to generate a unique one?
        # TODO: add optional ``out`` argument as in numpy
        att = A.att(0)
        newatt = "{0}_{1}".format(func, att)
        expr = "{0}({1})".format(func, att)
        return self.afl.papply(A, newatt, expr)

    def sin(self, A):
        """Element-wise trigonometric sine"""
        return self._apply_func(A, 'sin')

    def cos(self, A):
        """Element-wise trigonometric cosine"""
        return self._apply_func(A, 'cos')

    def tan(self, A):
        """Element-wise trigonometric tangent"""
        return self._apply_func(A, 'tan')

    def asin(self, A):
        """Element-wise trigonometric inverse sine"""
        return self._apply_func(A, 'asin')

    def acos(self, A):
        """Element-wise trigonometric inverse cosine"""
        return self._apply_func(A, 'acos')

    def atan(self, A):
        """Element-wise trigonometric inverse tangent"""
        return self._apply_func(A, 'atan')

    def exp(self, A):
        """Element-wise natural exponent"""
        return self._apply_func(A, 'exp')

    def log(self, A):
        """Element-wise natural logarithm"""
        return self._apply_func(A, 'log')

    def log10(self, A):
        """Element-wise base-10 logarithm"""
        return self._apply_func(A, 'log10')

    def sqrt(self, A):
        """Element-wise square root"""
        return self._apply_func(A, 'sqrt')

    def ceil(self, A):
        """Element-wise ceiling function"""
        return self._apply_func(A, 'ceil')

    def floor(self, A):
        """Element-wise floor function"""
        return self._apply_func(A, 'floor')

    def isnan(self, A):
        """Element-wise nan test function"""
        return self._apply_func(A, 'is_nan')

    def min(self, A, index=None, scidb_syntax=False):
        """
        Array or axis minimum.

        see :meth:`SciDBArray.min` """
        return A.min(index, scidb_syntax)

    def max(self, A, index=None, scidb_syntax=False):
        """
        Array or axis maximum.

        see :meth:`SciDBArray.max` """
        return A.max(index, scidb_syntax)

    def sum(self, A, index=None, scidb_syntax=False):
        """
        Array or axis sum.

        see :meth:`SciDBArray.sum` """
        return A.sum(index, scidb_syntax)

    def var(self, A, index=None, scidb_syntax=False):
        """
        Array or axis variance.

        see :meth:`SciDBArray.var` """
        return A.var(index, scidb_syntax)

    def stdev(self, A, index=None, scidb_syntax=False):
        """
        Array or axis standard deviation.

        see :meth:`SciDBArray.stdev` """
        return A.stdev(index, scidb_syntax)

    def std(self, A, index=None, scidb_syntax=False):
        """
        Array or axis standard deviation.

        see :meth:`SciDBArray.std` """
        return A.std(index, scidb_syntax)

    def avg(self, A, index=None, scidb_syntax=False):
        """
        Array or axis average.

        see :meth:`SciDBArray.avg` """
        return A.avg(index, scidb_syntax)

    def mean(self, A, index=None, scidb_syntax=False):
        """
        Array or axis mean.

        see :meth:`SciDBArray.mean` """
        return A.mean(index, scidb_syntax)

    def count(self, A, index=None, scidb_syntax=False):
        """
        Array or axis count.

        see :meth:`SciDBArray.count` """
        return A.count(index, scidb_syntax)

    def approxdc(self, A, index=None, scidb_syntax=False):
        """
        Array or axis unique element estimate.

        see :meth:`SciDBArray.approxdc` """
        return A.approxdc(index, scidb_syntax)

    def substitute(self, A, value):
        """
        Replace null values in an array

        See :meth:`SciDBArray.substitute`
        """
        return A.substitute(value)

    def merge(self, A, B):
        """Merge two arrays"""
        return merge(A, B)

    def join(self, *args):
        """
        Perform a series of array joins on the arguments
        and return the result.
        """
        return reduce(join, args)

    def cross_join(self, A, B, *dims):
        """Perform a cross-join on arrays A and B.

        Parameters
        ----------
        A, B : SciDBArray
        *dims : tuples
            The remaining arguments are tuples of dimension indices which
            should be joined.
        """
        B = match_chunk_permuted(B, A, dims).eval()
        dims = [_df(arr, index)
                for dim in dims
                for arr, index in zip([A, B], dim)]
        return self.afl.cross_join(A, B, *dims)

    def _join_operation(self, left, right, op):
        """Perform a join operation across arrays or values.

        See e.g. SciDBArray.__add__ for an example usage.
        """

        f = self.afl
        left, right = disambiguate(left, right)

        if isinstance(left, SciDBArray):
            assert_single_attribute(left)
            left = left.eval()
            left_name = left.name
            left_fmt = _af(left, 0)
            left_is_sdb = True
            left_is_sparse = left.issparse()
        else:
            left_name = None
            left_fmt = left
            left_is_sdb = False
            left_is_sparse = False

        if isinstance(right, SciDBArray):
            assert_single_attribute(right)
            right = right.eval()
            right_name = right.name
            right_fmt = _af(right, 0)
            right_is_sdb = True
            right_is_sparse = right.issparse()
        else:
            right_name = None
            right_fmt = right
            right_is_sdb = False
            right_is_sparse = False

        # some common names needed below
        _op = op
        op = op(left_fmt, right_fmt)

        aL = aR = None
        permutation = None

        # Neither entry is a SciDBArray
        if not (left_is_sdb or right_is_sdb):
            raise ValueError("One of left/right needs to be a SciDBArray")

        # Both entries are SciDBArrays
        elif (left_is_sdb and right_is_sdb):
            # array shapes match: use a join
            if left.shape == right.shape:
                attr = _new_attribute_label('x', left, right)
                if left_name == right_name:
                    # same array: we can do this without a join
                    return f.papply(left, attr, op)
                else:
                    if left_is_sparse or right_is_sparse:
                        return arithmetic.sparse_join(left, right, _op)
                    result = join(left, right)
                    return result.papply(attr, _op(result.att_names[0], result.att_names[1]))

            # array shapes are broadcastable: use a cross_join
            elif broadcastable(left.shape, right.shape):
                join_indices = []
                left_slices = []
                right_slices = []

                # cross join requires matched chunks
                left, right = match_chunks(left, right)
                left_fmt = _af(left.eval(), 0)
                right_fmt = _af(right.eval(), 0)
                op = _op(left_fmt, right_fmt)

                for tup in zip(reversed(list(enumerate(left.shape))),
                               reversed(list(enumerate(right.shape)))):
                    (i1, s1), (i2, s2) = tup
                    if (s1 == s2):
                        join_indices.append((i1, i2))
                        if (left.chunk_size[i1] != right.chunk_size[i2] or
                            left.chunk_overlap[i1] != right.chunk_overlap[i2]):
                            raise ValueError("join operations require chunk_"
                                             "size/chunk_overlap to match.")
                    elif s1 == 1:
                        left_slices.append(i1)
                    elif s2 == 1:
                        right_slices.append(i2)
                    else:
                        # should never get here, but just in case...
                        raise ValueError("shapes cannot be broadcast")

                # build the left slice query if needed
                if left_slices:
                    aL = ArrayAlias(left, "alias_left")
                    dims = [item for sl in left_slices
                            for item in [left.dim_names[sl], 0]]
                    left_query = f.as_(f.slice(left, *dims), aL)
                else:
                    left_query = left
                    aL = left

                # build the right slice query if needed
                if right_slices:
                    aR = ArrayAlias(right, "alias_right")
                    dims = [item for sl in right_slices
                            for item in [right.dim_names[sl], 0]]
                    right_query = f.as_(f.slice(right, *dims), aR)
                else:
                    right_query = right
                    aR = right

                # build the cross_join query
                dims = [_df(arr, ind)
                        for inds in join_indices
                        for arr, ind in zip([aL, aR], inds)]

                attr = _new_attribute_label('x', left, right)
                query = f.papply(f.cross_join(
                                 left_query, right_query, *dims),
                                 attr, op)
                result = query

                # determine the dimension permutation
                # Here's the problem: cross_join puts all the left array dims
                # first, and the right array dims second.  This is different
                # than numpy's broadcast behavior.  It's also difficult to do
                # in a single operation because conflicting dimension names
                # have a rename scheme that might be difficult to duplicate.
                # So we compromise, and perform a dimension permutation on
                # the result if needed.
                left_shape = list(left.shape)
                right_shape = list(right.shape)
                i_left = 0
                i_right = len(join_indices) + len(right_slices)
                permutation = [-1] * max(left.ndim, right.ndim)

                # first pad the shapes so they're the same length
                if left.ndim > right.ndim:
                    i_right += left.ndim - right.ndim
                    right_shape = [-1] * (left.ndim - right.ndim) + right_shape
                else:
                    left_shape = [-1] * (right.ndim - left.ndim) + left_shape

                # now loop through dimensions and build permutation
                for i, (L, R) in enumerate(zip(left_shape, right_shape)):
                    if L == R or R == -1 or (R == 1 and L >= 0):
                        permutation[i] = i_left
                        i_left += 1
                    elif L == -1 or (L == 1 and R >= 0):
                        permutation[i] = i_right
                        i_right += 1
                    else:
                        # This should never happen, but just to be sure...
                        raise ValueError("shapes are not compatible")

                if permutation == range(len(permutation)):
                    permutation = None

                if permutation is not None:
                    result = result.transpose(permutation)
                return result

            else:
                raise ValueError("Array of shape {0} can not be "
                                 "broadcasted with array of shape "
                                 "{1}".format(left.shape, right.shape))

        # only left entry is a SciDBArray
        elif left_is_sdb:
            try:
                float(right)
            except:
                raise ValueError("rhs must be a scalar or SciDBArray")
            if left_is_sparse:
                return arithmetic.sparse_scalar_join(left, right, _op)

            attr = _new_attribute_label('x', left)
            return f.papply(left, attr, op)

        # only right entry is a SciDBArray
        elif right_is_sdb:
            try:
                float(left)
            except:
                raise ValueError("lhs must be a scalar or SciDBArray")
            if right_is_sparse:
                return arithmetic.scalar_sparse_join(left, right, _op)

            attr = _new_attribute_label('x', right)
            return f.papply(right, attr, op)

        # reorder the dimensions if needed (for cross_join)
        if permutation is not None:
            arr = arr.transpose(permutation)
        return arr

    def concatenate(self, arrays, axis=0):
        """
        Concatenate several arrays along a particular dimension.

        This behaves like numpy's concatenate function when the
        input array dimensions are > the concatenation axis. It
        behaves differently than numpy when the array dimensions
        are less than the concatenation axis, in the following way:

        Concatenating 1D arrays along axis=0 behaves like numpy's vstack.
        Concatenating 1D arrays along axis=1 behaves like numpy's hstack.
        Concatenating 1D or 2D arrays along axis=2 behaves like dstack.

        Parameters
        ----------
        arrays : Sequence of SciDBArrays
            The arrays to concatenate
        axis : int, optional (default 0)
            The dimension to join on. Array shapes must match along
            all dimensions except this axis.

        Returns
        -------
        stacked : SciDBArray
            A stacked array

        See Also
        --------
        hstack(), vstack(), dstack()
        """

        assert_schema(arrays, bounded=True, same_attributes=True)
        slice_at_end = arrays[0].ndim == 1 and axis == 1

        # pad with extra dimensions as needed
        arrays = [right_dimension_pad(left_dimension_pad(a, 2), axis + 1)
                  for a in arrays]
        nd = arrays[0].ndim

        # check for proper shape
        if any(a.ndim != nd for a in arrays):
            raise ValueError("All the input arrays must have the same number of dimensions")

        for i, s in enumerate(arrays[0].shape):
            if i != axis and any(a.shape[i] != s for a in arrays):
                raise ValueError("all the input array dimensions except "
                                 "for the concatenation axis must match exactly")

        # compute the proper position along the concatenation dimension
        # be careful about nonzero origins here
        joinidx = 0 if (nd == 1 and axis == 1) else axis
        join = arrays[0].dim_names[joinidx]

        offset = 0
        result = list(arrays)
        idx = _new_attribute_label('idx', *arrays)

        for i in range(len(result)):
            dims = result[i].dim_names + [idx]
            atts = result[i].att_names

            zero = result[i].datashape.dim_low[joinidx]
            result[i] = result[i].apply(idx, '{0}-{1}+{2}'.format(join, zero, offset))
            result[i] = redimension(result[i], dims, atts)
            offset += result[i].datashape.dim_high[joinidx] - zero + 1

        # merge into one array
        r = result[0]
        for x in result[1:]:
            r = merge(r, x)

        # redimension into proper shape
        dim_names = [d for d in r.dim_names if d != idx]
        dim_names[joinidx] = idx
        r = redimension(r, dim_names, r.att_names)

        # special case for hstack on 1D arrays
        if slice_at_end:
            r = r[0]

        return r

    def hstack(self, arrays):
        """
        Stack arrays in sequence horizontally (column wise).

        Parameters
        ----------
        arrays : Sequence of SciDBArrays
            The arrays to join. All arrays must have the same
            shape along all but the second dimension.

        Returns
        -------
        stacked : SciDBArray
            The array formed by stacking the given arrays.

        See Also
        --------
        vstack(), dstack(), concatenate()
        """
        return self.concatenate(arrays, axis=1)

    def vstack(self, arrays):
        """
        Stack arrays in sequence vertically (column wise).

        Parameters
        ----------
        arrays : Sequence of SciDBArrays
            The arrays to join. All arrays must have the same
            shape along all but the first dimension.

        Returns
        -------
        stacked : SciDBArray
            The array formed by stacking the given arrays.

        See Also
        --------
        hstack(), dstack(), concatenate()
        """
        return self.concatenate(arrays, axis=0)

    def dstack(self, arrays):
        """
        Stack arrays in sequence depth wise (along the third axis).

        Parameters
        ----------
        arrays : Sequence of SciDBArrays
            The arrays to join. All arrays must have the same
            shape along all but the third dimension.

        Returns
        -------
        stacked : SciDBArray
            The array formed by stacking the given arrays.

        See Also
        --------
        hstack(), vstack(), concatenate()
        """
        return self.concatenate(arrays, axis=2)

    def percentile(self, a, q, att=None):
        """
        Compute the qth percentile of the data along the specified axis

        Parameters
        ----------
        a : SciDBArray
           Input array
        q : float in the range [0, 100] or a sequence of floats
           The percentiles to compute
        att : str, optional
           The array attribute to compute percentiles for. Defaults to the first
           attribute

        Returns
        -------
        qs : SciDBArray
           An array with as many elements as q, listing the data value
           at each percentile
        """
        att = att or a.att_names[0]

        q = np.atleast_1d(q)
        if q.min() < 0 or q.max() > 100:
            raise ValueError("Percentiles must be in the range [0, 100]")

        f = self.afl
        sorted = f.sort(a.project(att))
        sz = sorted.count()[0].astype(np.int)

        # use linear interpolation
        inds = (q * (sz - 1) / 100.)
        lo = np.clip(inds.astype(np.int), 0, (sz - 1))
        hi = np.clip(lo + 1, 0, (sz - 1))
        lohi = np.hstack((lo, hi))

        w = 1 - (inds - lo)
        result = sorted._integer_index(lohi).toarray()

        result = (result[:q.size] * w + result[q.size:] * (1 - w))
        return result


class SciDBShimInterface(SciDBInterface):

    """HTTP interface to SciDB via shim [1]_

    Parameters
    ----------
    hostname : string
        A URL pointing to a running shim/SciDB session
    user : string (optional)
        A username, for authentication
    password : string (optional)
        A password, for authentication
    pam : bool (optional)
        Whether to use PAM authentication. If  `True`,
        then user and password are required. If `None`,
        will be guessed based on hostname and password values
    digest : bool (optional)
        Whether to use Digest authentication. If `True`,
        then user and password are required. If `None`,
        will be guessed based on hostname and password values.

    [1] https://github.com/Paradigm4/shim
    """

    def __init__(self, hostname, user=None, password=None,
                 pam=None, digest=None):
        super(SciDBShimInterface, self).__init__()
        self.hostname = hostname.rstrip('/')

        https = self.hostname.startswith('https')
        authenticate = password is not None

        if pam is None:
            pam = https and authenticate

        if digest is None:
            digest = (not https) and authenticate

        if not authenticate and (pam or digest):
            raise ValueError("Must provide username and password "
                             "if using authentication")

        self._pam_auth = None

        # SHIM + digest authentication seems to need
        # the ability to retry, otherwise it throws connection errors
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(max_retries=3)
        s.mount('http://', a)
        self._session = s

        if digest:
            s.auth = requests.auth.HTTPDigestAuth(user, password)

        if pam:
            self.login(user, password)

        try:
            self._get_uid()
        except Exception as e:
            raise ValueError("Could not connect to a SciDB instance at {0}. {1}"
                             .format(self.hostname, e))

    def login(self, user, password):
        """
        Login using PAM authentication (e.g., over HTTPS)
        """
        url = self._shim_url('login', username=user, password=password)
        result = self._shim_urlopen(url)
        self._pam_auth = result.read()

    def logout(self):
        """
        Logout from PAM authentication (e.g., over HTTPS)
        """
        url = self._shim_url('logout')
        self._pam_auth = None
        self._shim_urlopen(url)

    def _get_uid(self):
        # load a library to get a query id
        session = self._shim_new_session()
        return self._shim_execute_query(session,
                                        "load_library('dense_linear_algebra')",
                                        release=True)

    def _execute_query(self, query, response=False, n=0, fmt='auto'):
        # log the query
        SciDBInterface._execute_query(self, query, response, n, fmt)

        session_id = self._shim_new_session()
        if response:
            self._shim_execute_query(session_id, query, save=fmt,
                                     release=False)

            if fmt.startswith('(') and fmt.endswith(')'):
                # binary format
                result = self._shim_read_bytes(session_id, n)
            else:
                # text format
                result = self._shim_read_lines(session_id, n)
            self._shim_release_session(session_id)
        else:
            self._shim_execute_query(session_id, query, release=True)
            result = None
        return result

    def _upload_bytes(self, data):
        session_id = self._shim_new_session()
        return self._shim_upload_file(session_id, data), session_id

    def _release_session(self, session):
        self._shim_release_session(session)

    def _shim_url(self, keyword, **kwargs):

        # add authentication token if needed
        if self._pam_auth is not None:
            kwargs['auth'] = self._pam_auth

        url = self.hostname + '/' + keyword
        if kwargs:
            url += '?' + '&'.join(['{0}={1}'.format(key, val)
                                   for key, val in iteritems(kwargs)])
        return url

    def _shim_urlopen(self, url):
        try:
            r = self._session.get(url, verify=False)
            r.raise_for_status()
        except requests.HTTPError as e:
            Error = SHIM_ERROR_DICT[r.status_code]
            raise Error(r.text)

        def read():
            return r.content

        r.read = read
        return r

    def _shim_new_session(self):
        """Request a new HTTP session from the service"""
        url = self._shim_url('new_session')
        result = self._shim_urlopen(url)
        session_id = int(result.read())
        return session_id

    def _shim_release_session(self, session_id, ignore_invalid=False):
        url = self._shim_url('release_session', id=session_id)
        if ignore_invalid:
            try:
                self._shim_urlopen(url)
            except SciDBInvalidSession:
                pass
        else:
            self._shim_urlopen(url)

    def _shim_execute_query(self, session_id, query, save=None, release=False):
        url = self._shim_url('execute_query',
                             id=session_id,
                             query=quote(query.encode('utf-8')),
                             release=int(bool(release)))
        if save is not None:
            url += "&save={0}".format(quote(save))

        try:
            result = self._shim_urlopen(url)
            query_id = result.read()
        except KeyboardInterrupt:
            self._shim_cancel(session_id)
            self._shim_release_session(session_id, ignore_invalid=True)
            raise KeyboardInterrupt("Query cancelled")

        return query_id.decode('UTF-8')

    def _shim_cancel(self, session_id):
        url = self._shim_url('cancel', id=session_id)
        self._shim_urlopen(url)

    def _shim_read_lines(self, session_id, n):
        url = self._shim_url('read_lines', id=session_id, n=n)
        t0 = time()
        result = self._shim_urlopen(url)
        text_result = result.read()
        dt = time() - t0
        pl = len(text_result) / 1048576
        logging.getLogger(__name__).debug("Transfer time: %0.1f sec", dt)
        logging.getLogger(__name__).debug("Payload:       %0.2f MB", pl)

        # the following check is for Py3K compatibility
        if not isinstance(text_result, string_type):
            text_result = text_result.decode('UTF-8')
        return text_result

    def _shim_read_bytes(self, session_id, n):
        url = self._shim_url('read_bytes', id=session_id, n=n)
        t0 = time()
        result = self._shim_urlopen(url)
        bytes_result = result.read()
        dt = time() - t0
        pl = len(bytes_result) / 1048576
        logging.getLogger(__name__).debug("Transfer time: %0.1f sec", dt)
        logging.getLogger(__name__).debug("Payload:       %0.2f MB", pl)

        return bytes_result

    def _shim_upload_file(self, session_id, data):
        # TODO: can this be implemented in urllib to remove dependency?
        url = self._shim_url('upload_file', id=session_id)
        result = self._session.post(url, files=dict(fileupload=data), verify=False)
        scidb_filename = result.text.strip()
        return scidb_filename


def connect(url=None, username=None, password=None):
    """
    Connect to a SciDB instance.

    Parameters
    ----------
    url : str (optional)
        Connection URL. If not provided, will fall back to
        the SCIDB_URL environment variable (if present),
        or http://127.0.0.1:8080. MUST begin with http or
        https. Username and password are mandatory with https.

    username : str (optional)
        SciDB username, for authenticated communication. Defaults to the value
        of the SCIDB_USER environment variable. If that doesn't exist,
        unauthetnicated communication is used.

    password : str (optional)
        SciDB password, for authenticated communication. Defaults to the value
        of the SCIDB_PASSWORD environment variable. If that doesn't exist,
        unauthetnicated communication is used

    Returns
    -------
    A SciDBShimInterface connection to the database.
    """
    url = url or os.environ.get('SCIDB_URL', 'http://127.0.0.1:8080')

    if username is None:
        username = os.environ.get('SCIDB_USER', None)

    if password is None:
        password = os.environ.get('SCIDB_PASSWORD', None)

    return SciDBShimInterface(url, user=username, password=password)
