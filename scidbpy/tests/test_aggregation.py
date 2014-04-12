from nose.tools import assert_raises

from .. import connect, histogram
import numpy as np

sdb = connect()


class TestHistogram(object):

    def setup_method(self, method):
        np.random.seed(42)

    def test_bad_input(self):
        with assert_raises(TypeError):
            histogram(1)

    def check(self, x, **kwargs):
        s = sdb.from_array(x)
        counts, bins = histogram(s, **kwargs)
        excounts, exbins = np.histogram(x, **kwargs)

        np.testing.assert_array_almost_equal(bins, exbins)
        np.testing.assert_array_equal(counts, excounts)

    def check_multi(self, x, att, **kwargs):
        s = sdb.from_array(x)
        counts, bins = histogram(s, att=att, **kwargs)
        excounts, exbins = np.histogram(x[att], **kwargs)

        np.testing.assert_array_almost_equal(bins, exbins)
        np.testing.assert_array_equal(counts, excounts)

    def test_defaults(self):
        x = np.random.random(100)
        self.check(x)

    def test_nbins(self):
        x = np.random.random(50)
        self.check(x, bins=17)

    def test_range(self):
        x = np.random.random(50)
        self.check(x, range=[0, 3])

    def test_multidimensional(self):
        x = np.random.random((5, 8))
        self.check(x)

    def test_multiatribute(self):
        x = np.zeros((3, 4),
                     dtype=[('x', '<f8'), ('y', '<f8')])
        x['x'] = np.random.random((3, 4))
        x['y'] = np.random.random((3, 4))
        self.check_multi(x, 'x')

    # XXX forbid list bins
