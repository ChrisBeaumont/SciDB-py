# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ..parse import toarray, NULLS
from . import sdb, TestBase, teardown_function


class TestScalar(TestBase):

    def check(self, array, expected):
        result = toarray(array)
        expected = np.asarray(expected)
        try:
            expected = expected.astype(result.dtype)
        except ValueError:
            pass

        try:
            assert_allclose(result, expected)
        except TypeError:
            assert_array_equal(result, expected)

    def scalar(self, typ, value, null):
        # build a single-element array
        if null:
            typ = typ + ' NULL'
        return sdb.afl.build('<x: %s>[i=0:0,10,0]' % typ, value)

    def check_scalar(self, typ, value, null, expected=None):
        if expected is None:
            expected = [value]
        self.check(self.scalar(typ, value, null), expected)

    def check_null(self, typ):
        x = sdb.afl.build('<x: %s NULL>[i=0:0,10,0]' % typ, 'null')
        self.check(x, [NULLS[typ]])

    def test_scalar_numbers(self):
        for typ, cases in [('float', (-1.0, 0.0, 1.0, 1e100)),
                           ('double', (-1, 0, 1, 1e100, 1e900)),
                           ('int8', (0, 17, -5)),
                           ('uint8', (0, 17, 255)),
                           ('int16', (0, 17, 300, -5)),
                           ('uint16', (0, 17, 300, 2 * 16 - 1)),
                           ('int32', (0, 256, -2 ** 29)),
                           ('uint32', (0, 256, 2 ** 32 - 1)),
                           ('int64', (0, 2 ** 35, -300)),
                           ('uint64', (0, 2 ** 63 - 1)),
                           ('bool', (True, False)),
                           ]:
            for case in cases:
                for null in [False, True]:
                    yield self.check_scalar, typ, case, null

    def test_null(self):
        for dtyp in ('float double int8 uint8 int16 uint16 '
                     'int32 uint32 int64 uint64 bool datetime '
                     'datetimetz char string').split():
            yield self.check_null, dtyp

    def test_datetime(self):
        for val in ['1970-01-02 00:00:00',
                    '1950-01-01 00:00:00',
                    '2520-12-31 11:59:59']:
            expected = [np.datetime64(val + '+0000')]
            for null in [False, True]:
                yield self.check_scalar, 'datetime', "'%s'" % val, null, expected

    def test_datetimetz(self):
        for val, exp in [('1970-01-02 00:00:00 +00:30', '1970-01-02 00:00:00+0030'),
                         ('1970-01-02 00:00:00 -01:30', '1970-01-02 00:00:00-0130')]:
            for null in [False, True]:
                yield self.check_scalar, 'datetimetz', "'%s'" % val, null, [np.datetime64(exp)]

    def test_char(self):
        for char in 'abAB12_ ':
            for null in [False, True]:
                yield self.check_scalar, 'char', "'%s'" % char, null, [char]

    def test_string(self):
        for val in ['a', 'hi', 'hi ho', 'å∫√∂']:
            for null in [False, True]:
                yield self.check_scalar, 'string', "'%s'" % val, null, [val]

    def test_parse_std(self):
        # regression test
        x = sdb.ones(5).std().toarray()
        assert x == 0


class TestMultiAttributeArrays(object):

    def teardown_method(self, method):
        sdb.reap()

    def test_composite_numbers(self):
        x = sdb.afl.join(sdb.afl.build('<x:float>[i=0:3,10,0]', 'i'),
                         sdb.afl.build('<y:uint8>[i=0:3,10,0]', '2*i'))
        y = toarray(x)

        rng = np.arange(4)

        assert_allclose(y['x'], rng)
        assert_allclose(y['y'], rng * 2)

    def test_numbers_with_nulls(self):
        x = sdb.afl.join(sdb.afl.build('<x:float NULL>[i=0:3,10,0]', 'iif(i>0, i, null)'),
                         sdb.afl.build('<y:uint8>[i=0:3,10,0]', '2*i'))
        y = toarray(x)
        assert_allclose(y['x'], [np.nan, 1., 2., 3.])
        assert_array_equal(y['y'], [0, 2, 4, 6])

    def test_numbers_string(self):
        x = sdb.afl.join(sdb.afl.build('<x:float>[i=0:3,10,0]', 'i'),
                         sdb.afl.build('<y:string>[i=0:3,10,0]', "'abc'"))

        y = toarray(x)
        assert_allclose(y['x'], [0, 1, 2, 3])
        assert_array_equal(y['y'], ['abc', 'abc', 'abc', 'abc'])

    def test_double_string(self):
        x = sdb.afl.join(sdb.afl.build('<x:string>[i=0:3,10,0]', "'aaa'"),
                         sdb.afl.build('<y:string NULL>[i=0:3,10,0]', "iif(i>0, 'b', null)"))
        y = toarray(x)
        assert_array_equal(y['x'], ['aaa', 'aaa', 'aaa', 'aaa'])
        assert_array_equal(y['y'], [None, 'b', 'b', 'b'])


def test_array():

    a = sdb.afl.build('<a: int32>[i=0:2,10,0]', 'i*5')
    b = sdb.afl.build('<b: string>[i=0:2,10,0]', "'b'")
    c = sdb.afl.build('<c: float NULL>[i=0:2,10,0]', 'iif(i>0, i/2.0, null)')
    d = sdb.join(a, b, c)

    result = toarray(d)
    assert_array_equal(result['a'], [0, 5, 10])
    assert_array_equal(result['b'], ['b', 'b', 'b'])
    assert_allclose(result['c'], [np.nan, 0.5, 1.0])


def test_nonzero_origin():

    a = sdb.afl.build('<a:int8>[i=1:5,10,0]', 'i')
    assert_array_equal(toarray(a), [1, 2, 3, 4, 5])


def test_unbounded():

    a = sdb.afl.build('<a:int8>[i=0:4,10,0]', 'i')
    a = a.redimension('<a:int8>[i=0:*,10,0]')
    assert_array_equal(toarray(a), [0, 1, 2, 3, 4])
    assert toarray(a).dtype == np.int8


def test_sparse():
    x = sdb.afl.build('<a:int8>[i=0:1,10,0]', 10)
    x = x.redimension('<a:int8>[i=0:2,10,0]')
    assert_array_equal(toarray(x), [10, 10, 0])


def test_fromarray_chunksize():
    from . import unfuzzed
    from_array = unfuzzed['from_array']

    def check(chunk):
        x = np.arange(24).reshape(2, 3, 4)
        y = from_array(x, chunk_size=chunk)
        assert_array_equal(x, y.toarray())
        assert chunk == tuple(y.datashape.chunk_size)

    for chunk in [(2, 3, 4), (1, 2, 2), (1, 1, 1), (5, 5, 5), (5, 1, 5)]:
        yield check, chunk
