"""SciDB Array Wrapper"""

# License: Simplified BSD, 2014
# See LICENSE.txt for more information

from __future__ import print_function

import numpy as np
import re

from copy import copy
from .errors import SciDBError, SciDBForbidden

# Numpy 1.7 meshgrid backport
from .utils import meshgrid, slice_syntax
from ._py3k_compat import genfromstr, iteritems
from .schema_utils import change_axis_schema

__all__ = ["sdbtype", "SciDBArray", "SciDBDataShape"]


# Create mappings between scidb and numpy string representations
_np_typename = lambda s: np.dtype(s).descr[0][1]
SDB_NP_TYPE_MAP = {'bool': _np_typename('bool'),
                   'float': _np_typename('float32'),
                   'double': _np_typename('float64'),
                   'int8': _np_typename('int8'),
                   'int16': _np_typename('int16'),
                   'int32': _np_typename('int32'),
                   'int64': _np_typename('int64'),
                   'uint8': _np_typename('uint8'),
                   'uint16': _np_typename('uint16'),
                   'uint32': _np_typename('uint32'),
                   'uint64': _np_typename('uint64'),
                   'char': _np_typename('c'),
                   'string': '|S100'}

NP_SDB_TYPE_MAP = dict((val, key)
                       for key, val in iteritems(SDB_NP_TYPE_MAP))

SDB_IND_TYPE = 'int64'


class sdbtype(object):

    """SciDB data type class.

    This class encapsulates the information about the datatype of SciDB
    arrays, with tools to convert to and from numpy dtypes.

    Parameters
    ----------
    typecode : string, list, sdbtype, or dtype
        An object representing a datatype.
    """

    def __init__(self, typecode):
        if isinstance(typecode, sdbtype):
            self.schema = typecode.schema
            self.dtype = typecode.dtype
            self.full_rep = [copy(t) for t in typecode.full_rep]

        else:
            try:
                self.dtype = np.dtype(typecode)
                self.schema = None
            except:
                self.schema = self._regularize(typecode)
                self.dtype = None

            if self.schema is None:
                self.schema = self._dtype_to_schema(self.dtype)

            if self.dtype is None:
                self.dtype = self._schema_to_dtype(self.schema)

            self.full_rep = self._schema_to_list(self.schema)

    def __repr__(self):
        return "sdbtype('{0}')".format(self.schema)

    def __str__(self):
        return self.schema

    @property
    def names(self):
        return [f[0] for f in self.full_rep]

    @property
    def nullable(self):
        return np.asarray([f[2] for f in self.full_rep])

    @property
    def bytes_fmt(self):
        """The format used for transfering raw bytes to and from scidb"""
        return '({0})'.format(','.join(rep[1] for rep in self.full_rep))

    @classmethod
    def _regularize(cls, schema):
        # if a full schema is given, we need to split out the type info
        R = re.compile(r'\<(?P<schema>[\S\s]*?)\>')
        results = R.search(schema)
        try:
            schema = results.groupdict()['schema']
        except AttributeError:
            pass
        return '<{0}>'.format(schema)

    @classmethod
    def _schema_to_list(cls, schema):
        """Convert a scidb type schema to a list representation

        Parameters
        ----------
        schema : string
            a SciDB type schema, for example
            "<val:double,rank:int32>"

        Returns
        -------
        sdbtype_list : list of tuples
            the corresponding full-representation of the scidbtype
        """
        schema = cls._regularize(schema)
        schema = schema.lstrip('<').rstrip('>')

        # assume that now we just have the dtypes themselves
        # TODO: support default values?
        sdbL = schema.split(',')
        sdbL = [list(map(str.strip, s.split(':'))) for s in sdbL]

        names = [s[0] for s in sdbL]
        dtypes = [s[1].split()[0] for s in sdbL]
        nullable = ['null' in str.lower(''.join(s[1].split()[1:]))
                    for s in sdbL]
        return list(zip(names, dtypes, nullable))

    @classmethod
    def _schema_to_dtype(cls, schema):
        """Convert a scidb type schema to a numpy dtype

        Parameters
        ----------
        schema : string
            a SciDB type schema, for example
            "<val:double,rank:int32>"

        Returns
        -------
        dtype : np.dtype object
            The corresponding numpy dtype descriptor
        """
        sdbL = cls._schema_to_list(schema)

        if len(sdbL) == 1:
            return np.dtype(SDB_NP_TYPE_MAP[sdbL[0][1]])
        else:
            return np.dtype([(s[0], SDB_NP_TYPE_MAP[s[1]]) for s in sdbL])

    @classmethod
    def _dtype_to_schema(cls, dtype):
        """Convert a scidb type schema to a numpy dtype

        Parameters
        ----------
        dtype : np.dtype object
            The corresponding numpy dtype descriptor

        Returns
        -------
        schema : string
            a SciDB type schema, for example "<val:double,rank:int32>"
        """
        dtype = np.dtype(dtype).descr

        # Hack: if we re-encode this as a dtype, then de-encode again, numpy
        #       will add default names where they are missing
        dtype = np.dtype(dtype).descr
        pairs = ["{0}:{1}".format(d[0], NP_SDB_TYPE_MAP[d[1]]) for d in dtype]
        return '<{0}>'.format(','.join(pairs))


class SciDBDataShape(object):

    """Object to store SciDBArray data type and shape"""

    def __init__(self, shape, typecode, dim_names=None,
                 chunk_size=1000, chunk_overlap=0):
        # Process array shape
        try:
            self.shape = tuple(shape)
        except:
            self.shape = (shape,)

        # process array typecode: either a scidb schema or a numpy dtype
        self.sdbtype = sdbtype(typecode)
        self.dtype = self.sdbtype.dtype

        # process array dimension names; define defaults if needed
        # we need to make sure these defaults don't clash with attributes
        if dim_names is None:
            import re
            R = re.compile('^i([0-9]+)$')
            matches = (R.match(name) for name in self.sdbtype.names)
            matches = filter(lambda m: m is not None, matches)
            vals = (int(m.groups()[0]) for m in matches)
            try:
                start = max(vals) + 1
            except ValueError:  # empty sequence
                start = 0
            dim_names = ['i{0}'.format(i)
                         for i in range(start, start + len(self.shape))]

        if len(dim_names) != len(self.shape):
            raise ValueError("length of dim_names should match "
                             "number of dimensions")
        self.dim_names = dim_names

        # process chunk sizes.  Either an integer, or a list of integers
        if not hasattr(chunk_size, '__len__'):
            chunk_size = [chunk_size for s in self.shape]
        if len(chunk_size) != len(self.shape):
            raise ValueError("length of chunk_size should match "
                             "number of dimensions")
        self.chunk_size = chunk_size

        # process chunk overlaps.  Either an integer, or a list of integers
        if not hasattr(chunk_overlap, '__len__'):
            chunk_overlap = [chunk_overlap for s in self.shape]
        if len(chunk_overlap) != len(self.shape):
            raise ValueError("length of chunk_overlap should match "
                             "number of dimensions")
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_schema(cls, schema):
        """Create a DataShape object from a SciDB Schema.

        This function uses a series of regular expressions to take an input
        schema such as::

            schema = "not empty myarray<val:double> [i0=0:3,4,0,i1=0:4,5,0]"

        parse it, and return a SciDBDataShape object.
        """
        # First split out the array name, data types, and shapes.  e.g.
        #
        #   "myarray<val:double,rank:int32> [i=0:4,5,0,j=0:9,5,0]"
        #
        # will become
        #
        #   dict(arrname = "myarray"
        #        dtypes  = "val:double,rank:int32"
        #        dshapes = "i=0:9,5,0,j=0:9,5,0")
        #
        R = re.compile(r'(?P<arrname>[\s\S]+)\<(?P<schema>[\S\s]*?)\>(?:\s*)'
                       '\[(?P<dshapes>\S+)\]')
        schema = schema.lstrip('schema').strip()
        match = R.search(schema)
        try:
            D = match.groupdict()
            arrname = D['arrname']
            schema = D['schema']
            dshapes = D['dshapes']
        except:
            raise ValueError("no match for schema: {0}".format(schema))

        # split dshapes.  TODO: correctly handle '*' dimensions
        #                       handle non-integer dimensions?
        R = re.compile(r'(\S*?)=(\S*?):(\S*?),(\S*?),(\S*?),')
        dshapes = R.findall(dshapes + ',')  # note added trailing comma

        return cls(shape=[int(d[2]) - int(d[1]) + 1 for d in dshapes],
                   typecode=schema,
                   dim_names=[d[0] for d in dshapes],
                   chunk_size=[int(d[3]) for d in dshapes],
                   chunk_overlap=[int(d[4]) for d in dshapes])

    @property
    def schema(self):
        return '{0} {1}'.format(self.sdbtype, self.dim_schema)

    @property
    def dim_schema(self):
        """
        The dimension part of the schema
        """
        result = ','.join(['{0}=0:{1},{2},{3}'.format(d, s - 1, cs, co)
                           for (d, s, cs, co) in zip(self.dim_names,
                                                     self.shape,
                                                     self.chunk_size,
                                                     self.chunk_overlap)])
        return '[%s]' % result

    @property
    def ind_attr_dtype(self):
        """Construct a numpy dtype that can hold indices and values.

        This is useful in downloading sparse array data from SciDB.
        """
        keys = self.dim_names + self.sdbtype.names
        dct = dict(f[:2] for f in self.sdbtype.full_rep)
        types = [SDB_NP_TYPE_MAP[dct.get(key, SDB_IND_TYPE)]
                 for key in keys]
        return np.dtype(list(zip(keys, types)))


class ArrayAlias(object):

    """
    An alias object used for constructing queries
    """

    def __init__(self, arr, name=None):
        self.arr = arr
        if name is None:
            self.name = self.arr.name
        else:
            self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __getattr__(self, attr):
        match = re.search(r'([da])(\d*)(f?)', attr)

        if not match:
            # exception text copied from Python2.6
            raise AttributeError("%r object has no attribute %r" %
                                 (type(self).__name__, attr))

        groups = match.groups()
        i = int(groups[1])

        if groups[2]:
            # seeking a full, qualified name
            ret_str = self.name + '.'
        else:
            ret_str = ''

        if groups[0] == 'd':
            # looking for a dimension name
            try:
                dim_name = self.arr.dimension(i)
            except IndexError:
                raise ValueError("dimension index %i is out of bounds" % i)
            return ret_str + dim_name

        else:
            # looking for an attribute name
            try:
                attr_name = self.arr.attribute(i)
            except IndexError:
                raise ValueError("attribute index %i is out of bounds" % i)
            return ret_str + attr_name


class SciDBArray(object):

    """SciDBArray class

    It is not recommended to instantiate this class directly; use a
    convenience routine from SciDBInterface.
    """

    def __init__(self, datashape, interface, name, persistent=False):
        self._datashape = datashape
        self.interface = interface
        self.name = name
        self.persistent = persistent

    @property
    def afl(self):
        """
        An alias to the AFL namespace
        """
        return self.interface.afl

    def rename(self, new_name, persistent=False):
        """Rename the array in the database, optionally making the new
        array persistent.

        Parameters
        ----------
        new_name : string
            must be a valid array name which does not already
            exist in the database.
        persistent : boolean (optional)
            specify whether the new array is persistent (default=False)

        Returns
        -------
        self : SciDBArray
            return a pointer to self
        """
        new_name = str(new_name)

        if new_name in self.interface.list_arrays():
            raise ValueError("Cannot use name {0}. "
                             "An array with that name "
                             "already exists.".format(new_name))
        self.afl.rename(self, new_name).eval(store=False)
        self.name = new_name
        self.persistent = persistent
        return self

    def copy(self, new_name=None, persistent=False):
        """Make a copy of the array in the database

        Parameters
        ----------
        new_name : string (optional)
            if specifiedmust be a valid array name which does not already
            exist in the database.
        persistent : boolean (optional)
            specify whether the new array is persistent (default=False)

        Returns
        -------
        copy : SciDBArray
            return a copy of the original array
        """
        # TODO: allow new_name to be specified in interface.new_array()
        if new_name is not None:
            new_name = str(new_name)
            if new_name in self.interface.list_arrays():
                raise ValueError("Cannot use name {0}. "
                                 "An array with that name "
                                 "already exists.".format(new_name))

        arr = self.interface.new_array(persistent=persistent)
        if new_name is not None:
            arr.name = new_name

        self.afl.store(self, arr).eval(store=False)
        return arr

    def alias(self, name=None):
        """Return an alias of the array, optionally with a new name"""
        return ArrayAlias(self, name)

    def dimension(self, d):
        """Return the dimension name of the array

        Parameters
        ----------
        d : int
           The index of the dimension to lookup
        """
        return self.datashape.dim_names[d]

    def attribute(self, a):
        """Return the attribute name of the array.

        Parameters
        ----------
        a : int
           Index of the attribute to lookup
        """
        return self.datashape.sdbtype.full_rep[a][0]

    att = attribute

    @property
    def datashape(self):
        if self._datashape is None:
            try:
                schema = self.interface._show_array(self.name, fmt='csv')
                self._datashape = SciDBDataShape.from_schema(schema)
            except SciDBError:
                self._datashape = None
        return self._datashape

    @property
    def shape(self):
        return self.datashape.shape

    @property
    def chunk_size(self):
        return self.datashape.chunk_size

    @property
    def chunk_overlap(self):
        return self.datashape.chunk_overlap

    @property
    def dim_names(self):
        return self.datashape.dim_names

    @property
    def att_names(self):
        return self.sdbtype.names

    @property
    def ndim(self):
        return len(self.datashape.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def sdbtype(self):
        return self.datashape.sdbtype

    @property
    def dtype(self):
        return self.datashape.dtype

    def reap(self, ignore=False):
        """
        Delete this object from the database if it isn't persistent.

        Parameters
        ----------
        ignore : bool (default False)
            If False and the array is persistent, then reap raises an error
            If True and the array is persistent, reap does nothing

        Raises
        ------
        SciDBForbidden if ``persistent=True`` and ``ignore=False`
        """
        if self.persistent:
            if not ignore:
                raise SciDBForbidden("Cannot reap: persistent=True")
            return

        if (self.datashape is not None):
            self.interface.query("remove({0})", self.name)
            self.name = '__DELETED__'
            self.interface = None

    def __repr__(self):
        show = self.interface._show_array(self.name, fmt='csv').split('\n')
        return "SciDBArray({0})".format(show[1])

    def contents(self, **kwargs):
        """
        Return a string representation of the array contents
        """
        return repr(self) + '\n' + self.interface._scan_array(self.name,
                                                              **kwargs)

    def nonempty(self):
        """
        Return the number of nonempty elements in the array.

        Nonempty refers to the sparsity of an array, and thus includes in the
        count elements with values which are set to NULL.

        See Also
        --------
        nonnull()
        """
        query = self.afl.count(self)
        response = query.eval(response=True, fmt='(int64)', store=False)
        return np.fromstring(response, dtype='int64')[0]

    def nonnull(self, attr=0):
        """
        Return the number of non-empty and non-null values.

        This query must be done for each attribute: the default is the first
        attribute.

        Parameters
        ----------
        attr : None, int or array_like
            the attribute or attributes to query.  If None, then query all
            attributes.

        Returns
        -------
        nonnull : array_like
            the nonnull count for each attribute.  The returned value is the
            same shape as the input ``attr``.

        See Also
        --------
        nonempty()
        """
        if attr is None:
            attr = range(len(self.sdbtype.names))
        attr = np.asarray(attr)
        nonnull = np.zeros_like(attr)

        for i in range(attr.size):
            attr_name = self.attribute(attr.flat[i])
            query = self.afl.aggregate(self, 'count(%s)' % attr_name)
            response = query.eval(response=True, store=False, fmt='(int64)')
            nonnull.flat[i] = np.fromstring(response, dtype='int64')[0]
        return nonnull

    def contains_nulls(self, attr=None):
        """Return True if the array contains null values.

        Parameters
        ----------
        attr : None, int, or array_like
            the attribute index/indices to check.  If None, then check all.

        Returns
        -------
        contains_nulls : boolean
        """
        if np.any(self.sdbtype.nullable):
            return np.any(self.nonempty() != self.nonnull(attr))
        else:
            return False

    def issparse(self):
        """Check whether array is sparse."""
        return (self.nonempty() < self.size)

    def _download_data(self, transfer_bytes=True, output='auto'):
        """Utility routine to transfer data from SciDB database.

        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.  This will lead to less approximation of floating point
            data, but for sparse arrays will result in two scan operations,
            one for indices, and one for values.
        output : string
            the output format.  The following are supported:

            - 'auto' : choose the best representation for the data
            - 'dense' : return a dense (numpy) array
            - 'sparse' : return a record array containing the indices and
                         values of non-empty elements.
        """
        if output not in ['auto', 'dense', 'sparse']:
            raise ValueError("unrecognized output: '{0}'".format(output))

        # workaround for strings
        has_strings = False
        if any(s[1] == 'string' for s in self.sdbtype.full_rep):
            has_strings = True
            transfer_bytes = False

        dtype = self.datashape.dtype
        sdbtype = self.sdbtype
        shape = self.datashape.shape
        array_is_sparse = self.issparse()
        full_dtype = self.datashape.ind_attr_dtype

        # check for nulls
        if self.contains_nulls():
            raise NotImplementedError("Arrays with nulls are not supported. "
                                      "Until this is addressed, see the "
                                      "substitute() function to remove nulls "
                                      "from your array.")

        if transfer_bytes:
            # Get byte-string containing array data.
            # This is needed for both sparse and dense outputs
            bytes_rep = self.interface._scan_array(self.name, n=0,
                                                   fmt=self.sdbtype.bytes_fmt)
            bytes_arr = np.atleast_1d(np.fromstring(bytes_rep, dtype=dtype))

        if array_is_sparse:
            # perform a CSV query to find all non-empty index tuples.
            str_rep = self.interface._scan_array(self.name, n=0, fmt='csv+')

            # sanity check: make sure our labels are correct
            str_dtype_names = str_rep.split('\n')[0].strip().split(',')
            if list(full_dtype.names) != str_dtype_names:
                raise ValueError("Fatal: unexpected array labels.")

            # convert the ASCII representation into a numpy record array
            arr = np.atleast_1d(genfromstr(str_rep, delimiter=',',
                                           skip_header=1,
                                           dtype=full_dtype))

            if transfer_bytes:
                # replace parsed ASCII columns with more accurate bytes
                names = self.sdbtype.names
                if len(names) == 1:
                    arr[names[0]] = bytes_arr
                else:
                    for name in self.sdbtype.names:
                        arr[name] = bytes_arr[name]

            if output == 'dense':
                result = np.zeros(self.shape, self.dtype)
                coords = tuple([arr[d] for d in self.dim_names])

                if len(self.att_names) == 1:
                    result[coords] = arr[self.att_names[0]]
                    return result

                for att in self.att_names:
                    result[att][coords] = arr[att]
                return result

        else:
            if transfer_bytes:
                # reshape bytes_array to the correct shape
                arr = bytes_arr.reshape(shape)
            else:
                # transfer via ASCII.  Here we don't need the indices, so we
                # use 'csv' output.
                str_rep = self.interface._scan_array(self.name, n=0, fmt='csv')
                arr = genfromstr(str_rep, delimiter=',', skip_header=1,
                                 dtype=dtype).reshape(shape)

            if output == 'sparse':
                index_arrays = list(map(np.ravel,
                                        meshgrid(*[np.arange(s)
                                                   for s in self.shape],
                                                 indexing='ij')))
                arr = arr.ravel()
                if len(sdbtype.names) == 1:
                    value_arrays = [arr]
                else:
                    value_arrays = [arr[col] for col in sdbtype.names]
                arr = np.rec.fromarrays(index_arrays + value_arrays,
                                        dtype=full_dtype)

        return arr

    def toarray(self, transfer_bytes=True):
        """Transfer data from database and store in a numpy array.

        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.

        Returns
        -------
        arr : np.ndarray
            The dense array containing the data.
        """
        return self._download_data(transfer_bytes=transfer_bytes,
                                   output='dense')

    def todataframe(self, transfer_bytes=True):
        """Transfer array from database and store in a local Pandas dataframe

        This is valid only for a one-dimensional array.

        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.

        Returns
        -------
        arr : pd.DataFrame
            The dataframe object containing the data in the array.
        """
        if self.ndim != 1:
            raise ValueError("Dataframe export is valid only for 1D arrays")
        from pandas import DataFrame
        return DataFrame(self.toarray())

    def tosparse(self, sparse_fmt='recarray', transfer_bytes=True):
        """Transfer array from database and store in a local sparse array.

        Parameters
        ----------
        transfer_bytes : boolean
            if True (default), then transfer data as bytes rather than as
            ASCII.  This is more accurate, but requires two passes over
            the data (one for indices, one for values).
        sparse_format : string or None
            Specify the sparse format to use.  Available formats are:
            - 'recarray' : a record array containing the indices and
              values for each data point.  This is valid for arrays of
              any dimension and with any number of attributes.
            - ['coo'|'csc'|'csr'|'dok'|'lil'] : a scipy sparse matrix.
              These are valid only for 2-dimensional arrays with a single
              attribute.

        Returns
        -------
        arr : ndarray or sparse matrix
            The sparse representation of the data
        """
        if sparse_fmt == 'recarray':
            spmat = None
        else:
            from scipy import sparse
            try:
                spmat = getattr(sparse, sparse_fmt + "_matrix")
            except AttributeError:
                raise ValueError("Invalid matrix format: "
                                 "'{0}'".format(sparse_fmt))

        columns = self._download_data(transfer_bytes=transfer_bytes,
                                      output='sparse')

        if sparse_fmt == 'recarray':
            return columns
        else:
            from scipy import sparse
            full_dtype = self.datashape.ind_attr_dtype
            if self.ndim != 2:
                raise ValueError("Only recarray format is valid for arrays "
                                 "with ndim != 2.")
            if len(full_dtype.names) > 3:
                raise ValueError("Only recarray format is valid for arrays "
                                 "with multiple attributes.")

            labels = full_dtype.names
            data = columns[labels[2]]
            ij = (columns[labels[0]], columns[labels[1]])
            arr = sparse.coo_matrix((data, ij), shape=self.shape)
            return spmat(arr)

    def __getitem__(self, indices):
        # The goal of getitem is to make a numpy-style interface perform
        # the correct operations on a SciDB array.  The corresponding
        # SciDB operations are:
        #
        #  subarray: produces an array of the same number of dimensions,
        #            but only a certain range in each dimension
        #  thin: produces an array of the same number of dimensions, but
        #        uses only every 2nd, 3rd, 4th... value in each dimension
        #  slice: produces an array of fewer dimensions by slicing at the
        #         specified place in each dimension.
        #  [reshape: This applies if a slice argument is newaxis.]

        # TODO: make this more efficient by using a single query
        # TODO: allow newaxis to be passed

        # slice can be either a tuple/iterable or a single integer/slice
        try:
            indices = tuple(indices)
        except TypeError:
            indices = (indices,)

        if len(indices) > self.ndim:
            raise ValueError("too many indices")

        if any(i is None for i in indices):
            raise NotImplementedError("newaxis in slicing")

        # if num indices is less than num dimensions, right-fill them
        indices = list(indices) + [slice(None)] * (self.ndim - len(indices))

        # special case, indexing with N 1D SciDB arrays, return
        # row/column/etc subset where these arrays are nonempty
        if all(isinstance(i, SciDBArray) for i in indices):
            return _subarray(self, *indices)

        # special case: accessing a single element (no slices)
        if all(not isinstance(i, slice) for i in indices):
            limits = list(map(int, indices + indices))
            q = self.afl.subarray(self, *limits)
            return q.toarray().flat[0]

        # if any of the slices are integers, we'll first use SciDB's
        # slice() operation on these
        slices = [item for (d, s) in zip(self.dim_names, indices)
                  if not isinstance(s, slice)
                  for item in (d, s)]
        if slices:
            arr1 = self.afl.slice(self, *slices).eval()
        else:
            arr1 = self

        # pull out the indices from the remaining slices
        shape = arr1.shape
        indices = [i for i in indices if isinstance(i, slice)]
        indices = [sl.indices(sh) for sl, sh in zip(indices, shape)]

        # if a subarray is required, then call the subarray() command
        if any(i[0] != 0 or i[1] != s for (i, s) in zip(indices, shape)):
            limits = [i[0] for i in indices] + [i[1] - 1 for i in indices]
            arr2 = self.afl.subarray(arr1, *limits).eval()
        else:
            arr2 = arr1

        # if thinning is required, then call the thin() command
        if any(i[2] != 1 for i in indices):
            steps = sum([[0, i[2]] for i in indices], [])
            arr3 = self.afl.thin(arr2, *steps).eval()
        else:
            arr3 = arr2

        return arr3

    @slice_syntax
    def sdbslice(self, slices):
        args = [s.start for s in slices] + [s.stop for s in slices]
        return self.afl.subarray(self, *args).eval()

    # join operations: note that these ignore all but the first attribute
    # of each array.
    def __add__(self, other):
        return self.interface._join_operation(self, other, self.afl.add)

    def __radd__(self, other):
        return self.interface._join_operation(other, self, self.afl.add)

    def __sub__(self, other):
        return self.interface._join_operation(self, other, self.afl.sub)

    def __rsub__(self, other):
        return self.interface._join_operation(other, self, self.afl.sub)

    def __mul__(self, other):
        return self.interface._join_operation(self, other, self.afl.mul)

    def __rmul__(self, other):
        return self.interface._join_operation(other, self, self.afl.mul)

    def __div__(self, other):
        return self.interface._join_operation(self, other, self.afl.div)

    def __truediv__(self, other):
        return self.interface._join_operation(self, other, self.afl.div)

    def __rdiv__(self, other):
        return self.interface._join_operation(other, self, self.afl.div)

    def __rtruediv__(self, other):
        return self.interface._join_operation(other, self, self.afl.div)

    def __mod__(self, other):
        return self.interface._join_operation(self, other, self.afl.mod)

    def __rmod__(self, other):
        return self.interface._join_operation(other, self, self.afl.mod)

    def __pow__(self, other):
        return self.interface._join_operation(self, other,
                                              self.afl.pow)

    def __rpow__(self, other):
        return self.interface._join_operation(other, self,
                                              self.afl.pow)

    def __abs__(self):
        return self.interface._apply_func(self, 'abs')

    def transpose(self, *axes):
        """Permute the dimensions of an array.

        Parameters
        ----------
        axes : None, tuple of ints, or `n` ints

         * None or no argument: reverses the order of the axes.

         * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
           `i`-th axis becomes `a.transpose()`'s `j`-th axis.

         * `n` ints: same as an n-tuple of the same ints (this form is
           intended simply as a "convenience" alternative to the tuple form)

        Returns
        -------
        out : ndarray
            Copy of `a`, with axes suitably permuted.
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if axes[0] is None:
                axes = None
            else:
                try:
                    axes = tuple(axes[0])
                except:
                    pass

        if not axes:
            arr = self.afl.transpose(self).eval()
        else:
            # first validate the axes
            axes = [(a + self.ndim if a < 0 else a) for a in axes]
            if any([(a < 0 or a >= self.ndim) for a in axes]):
                raise ValueError("invalid axis for this array")
            if (len(axes) != self.ndim or len(set(axes)) != self.ndim):
                raise ValueError("axes don't match array")

            # set up the new array
            shape = [self.shape[a] for a in axes]
            chunk_size = [self.chunk_size[a] for a in axes]
            chunk_overlap = [self.chunk_overlap[a] for a in axes]
            dim_names = [self.dim_names[a] for a in axes]
            arr = self.interface.new_array(shape=shape,
                                           dtype=self.sdbtype,
                                           chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap,
                                           dim_names=dim_names)
            self.afl.redimension_store(self, arr).eval(store=False)
        return arr

    # This allows the transpose of A to be computed via A.T
    T = property(transpose)

    def reshape(self, shape, **kwargs):
        """Reshape data into a new array

        Parameters
        ----------
        shape : tuple or int
            The shape of the new array.  Must be compatible with the current
            shape
        **kwargs :
            additional keyword arguments will be passed to SciDBDatashape

        Returns
        -------
        arr : SciDBArray
            new array of the specified shape
        """

        # handle -1s (see :meth:`numpy.reshape`)
        if any(s == 1 for s in shape):
            n = np.prod([s for s in shape if s != -1])
            ntot = np.prod(self.shape)
            if (ntot / n) * n != ntot:
                raise ValueError("total size of new array "
                                 "must remain unchanged")
            shape = list(shape)
            shape[shape.index(-1)] = ntot / n

        if np.prod(shape) != np.prod(self.shape):
            raise ValueError("new shape is incompatible")
        arr = self.interface.new_array(shape=shape,
                                       dtype=self.sdbtype,
                                       **kwargs)
        return self.afl.reshape(self, arr).eval(out=arr)

    def substitute(self, value):
        """Reshape data into a new array, substituting a default for any nulls.

        Parameters
        ----------
        value : value to replace nulls (required)

        Returns
        -------
        arr : SciDBArray
            new non-nullable array
        """
        b = self.afl.build('%s[i=0:0,1,0]' % self.sdbtype, value)
        q = self.afl.substitute(self, b)
        return q.eval()

    def _aggregate_operation(self, agg, index=None, scidb_syntax=True):
        """Perform an aggregation query

        Parameters
        ----------
        agg : string
            The aggregation function to apply
        index : int, tuple, or enumeratable
            The set of indices to aggregate over (default=None)
        scidb_syntax : boolean
            If true, use scidb-style index syntax.  Otherwise, use numpy-style
            index syntax.  Default is False (use numpy syntax).

        Notes
        -----
        Numpy and SciDB have different syntax for specifying aggregates over
        indices:

        >>> A = sdb.random((2, 3, 4))
        >>> A.max(1, scidb_syntax=False).shape
        (2, 4)
        >>> A.max(1, scidb_syntax=True).shape
        (3,)

        As we see, in SciDB we specify the index we want returned, and the
        aggregate is performed over all others.  In Numpy we specify the
        indices we want aggregated.

        By default, we use the numpy-like behavior which is more familiar
        to Python users, but keep a flag which allows SciDB-like behavior.
        """
        # TODO: add optional ``out`` argument, as in numpy
        idx_args = []
        if index is not None:
            try:
                ind = tuple(index)
            except:
                ind = (index,)

            # use numpy-style negative indices
            ind = [i + self.ndim if i < 0 else i
                   for i in map(int, ind)]

            # check that indices are in range
            if any(i < 0 or i > self.ndim for i in ind):
                raise ValueError("index out of range")

            # check for duplicates
            if len(set(ind)) != len(ind):
                raise ValueError("duplicate indices specified")

            # numpy syntax is opposite SciDB syntax
            if not scidb_syntax:
                ind = tuple(i for i in range(self.ndim) if i not in ind)

            # corner case where indices are an empty tuple
            if len(ind) > 0:
                idx_args = [self.dim_names[i] for i in ind]

        agg = "{agg}({att})".format(agg=agg, att=self.att(0))
        q = self.afl.aggregate(self, agg, *idx_args)
        return q.eval()

    def min(self, index=None, scidb_syntax=False):
        """
        Return the minimum of the array or the minimum along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('min', index, scidb_syntax)

    def max(self, index=None, scidb_syntax=False):
        """
        Return the maximum of the array or the maximum along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('max', index, scidb_syntax)

    def sum(self, index=None, scidb_syntax=False):
        """
        Return the sum of the array or the sum along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('sum', index, scidb_syntax)

    def var(self, index=None, scidb_syntax=False):
        """
        Return the variance of the array or the variance along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('var', index, scidb_syntax)

    def stdev(self, index=None, scidb_syntax=False):
        """
        Return the standard deviation of the array or along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('stdev', index, scidb_syntax)

    def std(self, index=None, scidb_syntax=False):
        """
        Return the standard deviation of the array or along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array

        Notes
        -----
        Identical to :meth:`SciDBArray.stdev`

        """
        return self._aggregate_operation('stdev', index, scidb_syntax)

    def avg(self, index=None, scidb_syntax=False):
        """
        Return the average of the array or the average along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('avg', index, scidb_syntax)

    def mean(self, index=None, scidb_syntax=False):
        """
        Return the average of the array or the average along an axis.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array

        Notes
        -----
        Identical to :meth:`SciDBArray.avg`
        """
        return self._aggregate_operation('avg', index, scidb_syntax)

    def count(self, index=None, scidb_syntax=False):
        """
        Return the count of the array or the count along an axis.

        The count is equal to the number of nonnull elements.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('count', index, scidb_syntax)

    def approxdc(self, index=None, scidb_syntax=False):
        """
        Return the number of distinct values of the array or along an axis.

        The distinct count is an estimate only.

        Parameters
        ----------
        index : int, optional
            Axis along which to operate. By default, flattened input is used.
        scidb_syntax : bool, optional (default=False)
            If False, index follows the numpy convention
            (i.e., the array is collapsed over the index'th axis).
            If True, index follows the SciDB convention
            (i.e., the array is collapsed over all axes *except* index)

        Returns
        -------
        A SciDB array
        """
        return self._aggregate_operation('approxdc', index, scidb_syntax)

    def regrid(self, size, aggregate="avg"):
        """Regrid the array using the specified aggregate

        Parameters
        ----------
        size : int or tuple of ints
            Specify the size of the regridding along each dimension.  If a
            single integer, then use the same regridding along each dimension.
        aggregate : string
            specify the aggregation function to use when creating the new
            grid.  Default is 'avg'.  Possible values are:
            ['avg', 'sum', 'min', 'max', 'count', 'stdev', 'var', 'approxdc']

        Returns
        -------
        A : scidbarray
            The re-gridded version of the array.  The size of dimension i
            is ceil(self.shape[i] / size[i])
        """
        if hasattr(size, '__len__'):
            if len(size) != self.ndim:
                raise ValueError("grid sizes must match array shape")
        else:
            size = [size for s in self.shape]
        sizes = map(int, size)

        if aggregate not in ['avg', 'sum', 'min', 'max', 'count',
                             'stdev', 'var', 'approxdc']:
            raise ValueError("aggregate='{0}' "
                             "not recognized".format(aggregate))

        agg = "{agg}({att})".format(agg=aggregate, att=self.att(0))
        args = list(map(str, sizes)) + [agg]
        q = self.afl.regrid(self, *args)
        return q.eval()


def _axis_filter(array, mask, axis):
    """
    Extract a subset of entries along a given mask,
    where an input mask array is non-null

    Parameters
    ----------
    array : SciDBArray
        The array to filter
    mask : SciDBArray
        A 1-dimensional SciDBArray, whose non-null values indicate
        the entries to retain
    axis : int
        The axis of array along which to apply the mask. The shape
        of array along this axis must be the length of mask
    """
    from .interface import _new_attribute_label

    f = array.interface.afl

    # for now, force evaulation of AFL
    # XXX remove this once we support deferred arrays
    if hasattr(mask, 'eval'):
        mask = mask.eval()

    dim = array.dim_names[axis]
    chunk = array.datashape.chunk_size[axis]
    overlap = array.datashape.chunk_overlap[axis]
    sz = array.shape[axis]
    ct = int(f.aggregate(mask, 'count(*)').eval()[0])

    new_att = _new_attribute_label(dim, mask)
    idx_att = _new_attribute_label(new_att + '_0', mask)

    # copy the dimension to a new attribute, sort moves nulls to end
    q = f.papply(mask, new_att, mask.dim_names[0])
    q = f.sort(q, '%s asc' % new_att)

    # rename attributes, and swap sorted new_att to a dimension
    # after this step, idx_att contains the final location for
    # each original location
    q = f.cast(q, '<%s:int64>[%s=0:*,1000,0]' % (new_att, idx_att))
    q = f.redimension(q, '<{idx_att}:int64>[{new_att}=0:{stop},{chunk},{overlap}]'
                      .format(new_att=new_att, stop=sz - 1,
                              chunk=chunk, overlap=overlap,
                              idx_att=idx_att))

    # tack on new location for each element
    q = f.cross_join(f.as_(array, 'xj1'),
                     f.as_(q, 'xj2'),
                     'xj1.%s' % dim, 'xj2.%s' % new_att)

    # promote idx_att to a dimension, which rearranges + truncates
    schema = array.sdbtype.schema
    schema += change_axis_schema(array.datashape,
                                 axis, name=idx_att, stop=ct - 1).dim_schema
    q = f.redimension(q, schema)
    return q


def _subarray(array, *masks):
    """
    Return a row/column/etc subset of an Nd array, by dropping
    null locations in N 1D masks

    Parameters
    ----------
    array : SciDBArray
        The array to filter

    masks : tuple of 1D SciDBArrays
       The axis filters

    Returns
    -------
    out : SciDBArray
       The filtered subarray
    """

    # XXX perform shape checking here

    result = array
    for i, mask in enumerate(masks):
        result = _axis_filter(result, mask, i).eval()
    return result
