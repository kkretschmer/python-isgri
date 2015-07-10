import glob
import os

import numpy as np
from scipy.optimize import minimize

from integral.isgri import bgcube
from integral.isgri import cube

e_min, e_max = 25, 80

def backgrounds():
    bg_path = '/Integral/data/resources/bg_model/' \
              'rate_model_below_1_sigma/rev_mod_100'
    files = sorted(glob.glob( \
        '/Integral/data/ic/ibis/bkg/isgr_back_bkg_????.fits'))
    files += sorted(glob.glob(os.path.join(bg_path,
        'rate_model_below_1_sigma_??/bkg_map*.fits')))
    br, bs = [], []
    return [bgcube.Cube(file).rate_shadowgram(e_min, e_max) for file in files]

def chi2(x, bs, cs, csig):
    """
    chi-squared of a shadowgram relative to a background shadowgram

    x: 2-tuple (offset, scale)
    bs: background shadowgram
    cs: cube shadowgram
    csig: cube shadowgram sigma
    """
    return (((x[0] + cs * x[1] - bs) / (csig * x[1]))**2).sum()

def chi2_allbg(bgs, cs, csig):
    """
    chi-squared of a shadowgram relative to all backgrounds
    """
    methods = ('constant', 'proportional', 'linear')
    n_bg, n_met = len(bgs), len(methods)
    c2 = np.zeros((n_met, n_bg))
    for i_met in range(n_met):
        for i_bg in range(n_bg):
            method = methods[i_met]
            bs = bgs[i_bg]
            if method == 'constant':
                x = (0, 1)
            elif method == 'proportional':
                x = (0, bs.mean() / cs.mean())
            elif method == 'linear':
                minres = minimize(chi2, [0, 1], (bs, cs, csig), method='Powell')
                x = minres.x
            else:
                x = (0, 0)
            c2[i_met, i_bg] = chi2(x, bs, cs, csig)
    return c2

def chi2_per_rev():
    bgs = backgrounds()
    byscw = '/Integral/data/reduced/ddcache/byscw'
    avail = cube.osacubes_avail()
    revs = sorted(avail.keys())
    cubes = []
    for i_rev in range(0, len(revs)):
        rev = revs[i_rev]
        ids = sorted(avail[rev])
        scw = ids[len(ids) // 2]
        cubes.append(os.path.join(byscw, list(avail[rev][scw].values())[0]))
    for path in cubes:
        oc = cube.Cube(path)
        cs, csig = oc.rate_shadowgram(e_min, e_max, sigma=True)
        print(chi2_allbg(bgs, cs, csig))
