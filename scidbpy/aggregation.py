from . import SciDBArray

__all__ = ['histogram']


def histogram(X, bins=10, att=None, range=None,
              weights=False, density=False):
    """
    Build a 1D histogram from a SciDBArray

    Paramters
    ---------
    X : SciDBArray
       The array to compute a histogram for
    att : str (optional)
       The attribute of the array to consider. Defaults to the first attribute.
    bins : int (optional)
       The number of bins
    range : [min, max] (optional)
       The lower and upper limits of the histogram. Defaults to data limits.

    Returns
    -------
    (counts, locs): A tuple of SciDBArrays
        ``locs`` is an array of *bins* elements, giving the array edges
        ``counts[i]`` gives the number of datapoints
        where locs[i] < counts < locs[i+1]. It has *bins-1* elements.

    """
    if not isinstance(X, SciDBArray):
        raise TypeError("Input must be a SciDBArray: %s" % type(X))
    if not isinstance(bins, int):
        raise NotImplementedError("Only integer bin arguments "
                                  "currently supported")
    if weights or density:
        raise NotImplementedError("weights and density not yet implemented")

    f = X.afl
    a = X.att(0) if att is None else att
    binid = 'bin'  # XXX make unique
    t = 'double'  # XXX lookup from dtype

    # store bounds
    if range is None:
        M = f.aggregate(X, 'min({a}) as min, max({a}) as max'.format(a=a))
        M = M.eval()
    else:
        lo = f.build('<min:%s NULL DEFAULT null>[i=0:0,1,0]' % t, min(range))
        hi = f.build('<max:%s NULL DEFAULT null>[j=0:0,1,0]' % t, max(range))
        M = f.cross_join(lo, hi, 'i', 'j')
        M = M.eval()

    val2bin = 'floor({bins} * ({a}-min)/(.0000001+max-min))'.format(bins=bins,
                                                                    a=a)
    bin2val = '{binid}*(0.0000001+max-min)/{bins} + min'.format(binid=binid,
                                                                bins=bins)

    schema = '<counts: uint64 null>[{0}=0:{1},1000000,0]'.format(binid, bins)
    s2 = ('<counts:uint64 null, min:{t} null, max:{t} null>'
          '[{binid}=0:{bins},1000000,0]'.format(binid=binid, t=t, bins=bins))

    # 0, min, max for each bin
    fill = f.slice(f.cross_join(f.build(schema, 0), M), 'i', 0).eval()
    fill2 = f.build('<v:int64>[i=0:0,1,0]', 0)  # single 0

    q = f.cross_join(X, M)  # val, min, max (Ndata)
    q = f.apply(q, binid, val2bin)  # val, min, max, binid
    q = f.substitute(q, fill2, binid)  # nulls to bin 0
    q = f.redimension(q, s2, 'count(%s) as counts' % binid)  # group bins
    q = f.merge(q, fill)  # replace nulls with 0
    q = f.apply(q, 'bins', bin2val)    # compute bin edges
    q = f.project(q, 'bins', 'counts')  # drop min, max

    result = q.toarray()
    assert result['counts'][-1] == 0
    return result['counts'][:-1], result['bins']
