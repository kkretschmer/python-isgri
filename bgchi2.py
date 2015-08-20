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
import sqlite3

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from integral.isgri import bgcube
from integral.isgri import cube

e_min, e_max = 25, 80

conn = sqlite3.connect('bgchi2.sqlite')
cursor = conn.cursor()
cursor.execute('PRAGMA foreign_keys=ON')
cursor.execute(
    'CREATE TABLE IF NOT EXISTS tests'
    '  (test_id INTEGER PRIMARY KEY,'
    '   background TEXT, rate TEXT, method TEXT,'
    '   e_min NUMERIC, e_max NUMERIC, '
    '   UNIQUE (background , rate, method, e_min, e_max))'
)
cursor.execute(
    'CREATE TABLE IF NOT EXISTS quality'
    '  (test_id INTEGER REFERENCES tests (test_id),'
    '   scwid TEXT, npix INTEGER, value NUMERIC,'
    '   UNIQUE (test_id, scwid))'
)

def backgrounds():
    bg_path = '/data/integral/bgcube/2015-08-10_multi-200'
    files = sorted(glob.glob( \
        '/Integral/data/ic/ibis/bkg/isgr_back_bkg_????.fits'))
    result = [{'file': file,
               'rsg': bgcube.BGCube(file).rate_shadowgram(
                   e_min, e_max, per_keV=False)}
              for file in files]
    def read_bgcube(path):
        hdulist = fits.open(path)
        c = cube.Cube()
        c.counts, c.efficiency = [hdulist[i].data for i in [1, 2]]
        c.tmean = hdulist[1].header['tmean']
        hdulist.close()
        return bgcube.BGCube(c)
    files = sorted(glob.glob(os.path.join(bg_path, 'bgcube????.fits')))
    result += [{'file': file,
                'rsg': read_bgcube(file).rate_shadowgram(
                    e_min, e_max, per_keV=False)}
              for file in files]
    return result

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

def scw_tests(bgs, cts, exp, scwid):
    """
    chi-squared of a shadowgram relative to all backgrounds

    For all backgrounds we predict a rate using a set of algorithms,
    then test the match to the shadowgram using a set of quality measures.
    """
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
        bs = np.ma.asarray(bg['rsg'])

        # The algorithms to adjust the model background rate to the
        # current shadowgram
        #
        for rate_alg, rate_fn in rate_algs:
            rate = rate_fn(bs, cts, exp)
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
                    quality, npix = qual_fn(rate, cts, exp)
                    cursor.execute(
                        'INSERT OR REPLACE INTO quality'
                        ' (test_id, scwid, npix, value)'
                        ' VALUES (?, ?, ?, ?)',
                        (test_id, scwid, int(npix), quality))
                    conn.commit()

def tests_per_rev(predictable=False):
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
        scw_tests(bgs, cts, exp, meta['scw'])

if __name__ == "__main__":
    tests_per_rev(predictable=True)
