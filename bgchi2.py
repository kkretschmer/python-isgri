# -*- coding: utf-8 -*-
#
# Copyright (C) 2015  Karsten Kretschmer <kkretsch@apc.univ-paris7.fr>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import glob
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

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
def ra_constant(bs, cts, exp):
    rate = bs
    return rate

def ra_proportional(bs, cts, exp):
    rate = bs * cts.sum() / (bs * exp).sum()
    return rate

def ra_mdu_proportional(bs, cts, exp):
    rate = np.ma.zeros(cts.shape)
    for mdu in cube.Cube.mdu_slices:
        rate[mdu] = bs[mdu] * cts[mdu].sum() / \
                    (bs[mdu] * exp[mdu]).sum()
    return rate

def ra_linear(bs, cts, exp):
    def rate(x):
        return x[0] + bs * x[1]

    def func(x):
        return -qa_logl(rate(x), cts, exp)

    minres = minimize(func, [0, 1], method='Powell')
    x = minres.x
    return rate(x)

def qa_chi2(rate, cts, exp):
    """
    chi-squared of a shadowgram relative to a background shadowgram

    rate: expected rate shadowgram
    cts: cube counts shadowgram
    exp: cube exposure shadowgram
    """
    pixels = ((cts - (rate * exp))**2 / cts).sum()
    return (pixels.sum(), pixels.count())

def chi2_allbg(bgs, cs, csig):
def qa_logl(rate, cts, exp):
    """
    log-likelihood of a shadowgram relative to a background shadowgram

    rate: expected rate shadowgram
    cts: cube counts shadowgram
    exp: cube exposure shadowgram
    """
    idx = np.logical_not(np.logical_or(cts.mask, rate.mask))
    return (poisson.logpmf(cts[idx], rate[idx] * exp[idx]).sum(),
            np.nount_nonzero(idx))

    """
    chi-squared of a shadowgram relative to all backgrounds
    """
def chi2_per_rev(predictable=False):
    rate_algs = [
        ('constant', ra_constant),
        ('proportional', ra_proportional),
        ('mdu-proportional', ra_mdu_proportional),
        ('linear', ra_linear),
    ]
    qual_algs = [
        ('chi2', qa_chi2),
        ('logl', qa_logl)
    ]
    for bg in bgs:
        bs = bg['rsg']

        # The algorithms to adjust the model background rate to the
        # current shadowgram
        #
        for rate_alg, rate_fn in rate_algs:
            rate = rate_fn(bgs, cts, exp)
            #
            # The background quality measures
            #
            for qual_alg, qual_fn in qual_algs:
                cursor.execute(
                    'INSERT OR IGNORE INTO tests'
                    '  (background, rate, method, e_min, e_max)'
                    '  VALUES (?, ?, ?, ?, ?)',
                    (bg['file'], rate_alg, qual_alg, e_min, e_max)
                )
                cursor.execute(
                    'SELECT test_id FROM tests WHERE'
                    '  background = ? AND'
                    '  rate = ? AND'
                    '  method = ? AND'
                    '  e_min = ? AND'
                    '  e_max = ?',
                    (bg['file'], rate_alg, qual_alg, e_min, e_max)
                )
                test_id = cursor.fetchone()[0]
                if test_id:
                    cursor.execute(
                        'INSERT INTO quality'
                        ' (test_id, scwid, npix, value)'
                        ' VALUES (?, ?, ?, ?)',
                        (test_id, scwid, cts.count(),
                         qual_fn(rate, cts, exp)))

    bgs = backgrounds()
    byscw = '/Integral/data/reduced/ddcache/byscw'
    avail = cube.osacubes_avail()
    if predictable:
        predictable_scwlist = \
            '/data/integral/isgri_background/rate_model_below_1_sigma.dat'
        scw_pre = np.loadtxt(predictable_scwlist, dtype=np.uint64)[:, 0]

    revs = sorted(avail.keys())
    cubes = []
    for i_rev in range(0, len(revs)):
        rev = revs[i_rev]
        ids = sorted(avail[rev])
        if predictable:
            ids = np.intersect1d(
                np.array([i[0:12] for i in ids], dtype=np.uint64), scw_pre)
            ids = ['{0:012d}.001'.format(i) for i in ids]
            if len(ids) == 0: continue
        scw = ids[len(ids) // 2]
        cubes.append(
            { 'scw': scw,
              'path': os.path.join(byscw, list(avail[rev][scw].values())[0]),
            }
        )
    for meta in cubes:
        oc = cube.Cube(meta['path'])
        if oc.empty: continue
        cts, exp = oc.cts_exp_shadowgram(e_min, e_max)
        print(meta['scw'])
        print(chi2_allbg(bgs, cs, csig))
        with open('/data/integral/bgchi2.chi2_per_rev', 'a') as out:
            print(meta['scw'], file=out, end=" ")
            chi2_allbg(bgs, cs, csig).flatten().tofile(out, sep=" ")
            print(file=out)
