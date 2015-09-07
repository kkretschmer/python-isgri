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
import itertools
import os
import sqlite3

import numpy as np
from scipy.optimize import minimize
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from integral.isgri import bgcube
from integral.isgri import cube
from integral.isgri.shadowgram import fitquality

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

outlier_count = fits.open('/data/integral/pixels/outlier_count.fits')
outlier = outlier_count[0].data >= 4
outlier_count.close()

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
    x = cts.sum() / (bs * exp).sum()
    rate = bs * x
    print('ra_proportional: ', x)
    return rate

def ra_mdu_proportional(bs, cts, exp):
    rate = np.ma.zeros(cts.shape)
    for mdu in cube.Cube.mdu_slices:
        rate[mdu] = ra_proportional(bs[mdu], cts[mdu], exp[mdu])
    return rate

def ra_linear(bs, cts, exp):
    bs_mean = bs.mean()
    def rate(x):
        return (x[0] * bs_mean + x[1] * bs)

    def fun(x):
        return qa_chi2(rate(x), cts, exp)[0]

    x0 = np.array([0.1, 0.9])
    minres = minimize(fun, x0, method='Powell')
    x = minres.x
    print('ra_linear: ', x)
    return rate(x)

def ra_mdu_linear(bs, cts, exp):
    rate = np.ma.zeros(cts.shape)
    for mdu in cube.Cube.mdu_slices:
        rate[mdu] = ra_linear(bs[mdu], cts[mdu], exp[mdu])
    return rate

def sum_asic(sg):
    return sg.reshape(64, 2, 64, 2).sum(axis=3).sum(axis=1)

def sum_polycell(sg):
    return sg.reshape(32, 4, 32, 4).sum(axis=3).sum(axis=1)

def ma_outlier(cts):
    masked_cts = cts.copy()
    masked_cts[outlier] = np.ma.masked
    return masked_cts

def scw_tests(bgs, cts, exp, scwid):
    """
    chi-squared of a shadowgram relative to all backgrounds

    For all backgrounds we predict a rate using a set of algorithms,
    then test the match to the shadowgram using a set of quality measures.
    """
    mask_algs = [
        ('nomask', lambda cts: cts),
        ('outlier', ma_outlier),
    ]
    rate_algs = [
        ('constant', ra_constant),
        ('proportional', ra_proportional),
        ('mdu-proportional', ra_mdu_proportional),
        ('linear', ra_linear),
        ('mdu-linear', ra_mdu_linear),
    ]
    qual_algs = [
        ('chi2', fitquality.chi2),
        ('logl', fitquality.logl),
        ('chi2-asic', fitquality.fq_summed(fitquality.chi2,
                                           fitquality.sum_asic)),
        ('logl-asic', fitquality.fq_summed(fitquality.logl,
                                           fitquality.sum_asic)),
        ('chi2-polycell', fitquality.fq_summed(fitquality.chi2,
                                               fitquality.sum_polycell)),
        ('logl-polycell', fitquality.fq_summed(fitquality.logl,
                                               fitquality.sum_polycell)),
    ]

    # I want to iterate over most combinations of background, pixel mask,
    # rate adjustment and quality measure. To keep the nesting depth small
    # a cartesian product is useful and I just accept the overhead of
    # also testing unnecessary combinations as well.
    #
    # The quality algorithm does not need to be in the cartesian product
    # because they all work on the same rate shadowgram.
    #
    for bg, mask_alg, rate_alg in itertools.product(
            bgs, mask_algs, rate_algs):

        bs = np.ma.asarray(bg['rsg'])
        mask_name, mask_fn = mask_alg
        rate_name, rate_fn = rate_alg
        mask_rate_name = ','.join((mask_name, rate_name))

        cts_masked = mask_fn(cts)
        rate = rate_fn(bs, cts_masked, exp)

        for qual_alg in qual_algs:
            qual_name, qual_fn = qual_alg
            cursor.execute(
                'INSERT OR IGNORE INTO tests'
                '  (background, rate, method, e_min, e_max)'
                '  VALUES (?, ?, ?, ?, ?)',
                (bg['file'], mask_rate_name, qual_name, e_min, e_max)
            )
            cursor.execute(
                'SELECT test_id FROM tests WHERE'
                '  background = ? AND'
                '  rate = ? AND'
                '  method = ? AND'
                '  e_min = ? AND'
                '  e_max = ?',
                (bg['file'], mask_rate_name, qual_name, e_min, e_max)
            )
            test_id = cursor.fetchone()[0]
            if test_id:
                quality, npix = qual_fn(rate, cts_masked, exp)
                cursor.execute(
                    'INSERT OR REPLACE INTO quality'
                    ' (test_id, scwid, npix, value)'
                    ' VALUES (?, ?, ?, ?)',
                    (test_id, scwid, int(npix), quality))
                conn.commit()
                print(scwid, bg['file'], mask_rate_name, qual_name, e_min, e_max,
                      quality, npix)

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
