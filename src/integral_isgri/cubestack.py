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
"""
Stacking of INTEGRAL/ISGRI cubes

Combine cubes to build background models.
"""

import io
import logging
import os
import re
import sqlite3

import numpy as np
from scipy.stats import norm
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import cube
from . import shadowgram

class weighted_incremental_variance:
    """
    from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Weighted_incremental_algorithm

    D. H. D. West, Communications of the ACM, Vol. 22 No. 9, Pages 532-535,
    doi: 10.1145/359146.359153
    http://people.xiph.org/~tterribe/tmp/homs/West79-_Updating_Mean_and_Variance_Estimates-_An_Improved_Method.pdf
    """
    def __init__(self, x):
        self.n = 0
        self.sumw = 0.0
        self.m = 0.0
        self.t = 0.0

    def input(self, x, w):
        self.n += 1
        q = x - self.m
        temp = self.sumw + w
        r = q * w / temp
        self.m += r
        self.t += r * self.sumw * q
        self.sumw = temp

    def mean(self):
        return self.m

    def var(self):
        return self.t * self.n / ((self.n - 1) * self.sumw)

class BackgroundBuilder(object):
    """Build an ISGRI background cube from observations <scwids>"""

    def __init__(self, scwids):
        self.scwids = np.array(scwids, dtype=np.uint64)
        self.eff_min = 0.2
        self.sig_e_range = (60, 80)
        self.sig_thresholds = (-2.5, 2.5)
        self.erange_dark = (35, 900)
        self.erange_hot = (35, 51.5)
        self.sigma_max = 4

    def setref(self, bgcube):
        self.ref = bgcube
        self.ref_dark = self.ref.rate_shadowgram(*self.erange_dark)
        self.ref_hot = self.ref.rate_shadowgram(*self.erange_hot)

    def ps_efficiency_threshold(self, cube_in):
        return cube_in.efficiency >= self.eff_min

    def ps_pixel_efficiency(self, cube_in):
        ps = np.ones_like(cube_in.counts.data, dtype=np.bool)
        ps[:, cube_in.efficiency[-1] < self.eff_min] = False
        return ps

    def ps_not_outlier(self, cube_in, write_fits=False):
        ps = np.ones_like(cube_in.counts.data, dtype=np.bool)
        sg, sg_sig = cube_in.rate_shadowgram(self.sig_e_range[0],
                                          self.sig_e_range[-1], True)
        mi, di = sg.mean(), sg.std()
        dark, hot = \
            np.logical_and(sg.data < mi + di * self.sig_thresholds[0],
                           np.logical_not(sg.mask)), \
            np.logical_and(sg.data > mi + di * self.sig_thresholds[1],
                           np.logical_not(sg.mask))
        ps[:, np.logical_or(dark, hot)] = False
        if write_fits:
            cursor = write_fits
            cursor.execute('''CREATE TABLE IF NOT EXISTS ps_not_outlier
                (scwid TEXT PRIMARY KEY, fits BLOB)''')
            pixels = np.zeros(sg.shape, dtype=np.ubyte)
            pixels[sg.mask == True] += 1
            pixels[dark] += 2
            pixels[hot] += 4
            hdu = fits.PrimaryHDU(pixels)
            blob = io.BytesIO()
            hdu.writeto(blob)
            cursor.execute('''INSERT OR REPLACE INTO ps_not_outlier
                (scwid, fits) VALUES (?, ?)''', (cube_in.scwid, blob.getvalue()))
            blob.close()
        return ps

    def ps_not_dark_hot(self, cube_in, write_fits=False):
        name = 'ps_not_dark_hot'
        ps = np.ones_like(cube_in.counts.data, dtype=np.bool)
        cts, exp = cube_in.cts_exp_shadowgram(*self.erange_dark)
        lp_dd, lp_hd = shadowgram.logprob_not_dark_hot(cts, exp, self.ref_dark)
        cts, exp = cube_in.cts_exp_shadowgram(*self.erange_hot)
        lp_dh, lp_hh = shadowgram.logprob_not_dark_hot(cts, exp, self.ref_hot)
        dark, hot = \
            [np.logical_and(lp < norm.logsf(self.sigma_max),
                            np.logical_not(cts.mask))
             for lp in (lp_dd, lp_hh)]
        ps[:, np.logical_or(dark, hot)] = False
        if write_fits:
            cursor = write_fits
            cursor.execute('''CREATE TABLE IF NOT EXISTS {0}
                (scwid TEXT PRIMARY KEY, fits BLOB)'''.format(name))
            pixels = np.zeros(cts.shape, dtype=np.ubyte)
            pixels[cts.mask == True] += 1
            pixels[dark] += 2
            pixels[hot] += 4
            hdu = fits.PrimaryHDU(pixels)
            blob = io.BytesIO()
            hdu.writeto(blob)
            cursor.execute('''INSERT OR REPLACE INTO {0}
                 (scwid, fits) VALUES (?, ?)'''.format(name),
                           (cube_in.scwid, blob.getvalue()))
            blob.close()
        return ps

    def fill_bad_pixels(self, cube_in,
                        module=True, polycell=True,
                        n_modules=5, n_polycells=512):
        """Fill in bad pixels by using the average of the good pixels
        in the same position in module coordinates if more than
        ``num_modules`` of them are available."""

        def cube_mean_mod(cube_in):
            mod = cube.cube2mod(cube_in)
            mod_mean = mod.mean(1)
            mod_mean.mask[mod.count(1) < n_modules] = True
            return cube.mod2cube(np.repeat(
                mod_mean[:, np.newaxis, ...], 8, axis=1))

        def cube_mean_pc(cube_in):
            pc = cube.mod2pc(cube.cube2mod(cube_in))
            pc = pc.reshape((pc.shape[0], 8 * 8 * 16, 4, 4))
            pc_mean = pc.mean(1)
            pc_mean.mask[pc.count(1) < n_polycells] = True
            pc_mean = pc_mean[:, np.newaxis, ...]
            pc_filled = np.repeat(pc_mean, 8 * 8 * 16, axis=1)
            pc_filled = pc_filled.reshape((pc.shape[0], 8, 8, 16, 4, 4))
            return cube.mod2cube(cube.pc2mod(pc_filled))

        runs = []
        if module: runs += [cube_mean_mod]
        if polycell: runs += [cube_mean_pc]

        cube_out = cube.Cube(osacube=True)
        for key in cube_in.__dict__:
            setattr(cube_out, key, getattr(cube_in, key))
        cube_out.counts = np.ma.array(cube_in.counts, dtype=np.float32)
        cube_out.efficiency = np.ma.copy(cube_in.efficiency)
        for run in runs:
            for attr in ['counts', 'efficiency']:
                src, dst = [getattr(i, attr) for i in [cube_in, cube_out]]
                flt = run(src)
                dst[dst.mask] = flt[dst.mask]
        return cube_out

    def read_cubes(self, n_max=None):

        def bgcube_new():
            bgcube = cube.Cube(osacube=True)
            bgcube.counts = np.ma.asarray(bgcube.counts, np.float32)
            setattr(bgcube, 'n_scw', 0)
            for attr in ['tstart', 'tmean', 'tstop']:
                setattr(bgcube, attr, 0.0)
            setattr(bgcube, 'wiv_tmean',
                    weighted_incremental_variance(0.0))
            setattr(bgcube, 'wiv_rate',
                    weighted_incremental_variance(bgcube.efficiency))
            return bgcube

        osacubes, ids = cube.osacubes(self.scwids)

        conn = sqlite3.connect('bgcube.sqlite')
        cursor = conn.cursor()

        selectors = [
            {'fn': self.ps_pixel_efficiency, 'args': (), 'kwargs': {}},
            {'fn': self.ps_not_dark_hot, 'args': (),
             'kwargs': {'write_fits': cursor}}
        ]
        logger = logging.getLogger('read_cubes')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('read_cubes.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(ch)

        tmean = 0
        bgcubes = []
        bgcube = bgcube_new()

        for path in osacubes:
            if n_max is not None and bgcube.n_scw >= n_max:
                # save the current background cube and start a new one
                #
                logger.info('starting new bgcube')
                bgcubes.append(bgcube)
                bgcube = bgcube_new()

            try:
                oc = cube.Cube(path)
                if oc.empty: continue
                if np.count_nonzero(oc.efficiency) == 0: continue
                oc = oc.rebin()
                idx = np.logical_not(oc.counts.mask)
                logger.info('{0}: loaded'.format(oc.scwid))
                logger.debug('{0}: {1} bin*px valid'.format(
                    oc.scwid, np.count_nonzero(idx)))
                for sel in selectors:
                    newidx = sel['fn'](oc, *sel['args'], **sel['kwargs'])
                    idx = np.logical_and(idx, newidx)
                    logger.debug('{0}: {1}, {2} bin*px valid'.format(
                        oc.scwid, repr(sel['fn']), np.count_nonzero(idx)))
            except:
                logger.error('{0}: failed'.format(oc.scwid))
                continue

            oc.counts.mask = np.logical_not(idx)
            oc.efficiency.mask = oc.counts.mask
            fc = self.fill_bad_pixels(oc)

            bgcube.counts += fc.counts
            exposure = fc.efficiency * oc.duration
            bgcube.efficiency += exposure
            bgcube.wiv_rate.input(fc.counts / exposure, oc.duration)

            bgcube.n_scw += 1
            bgcube.duration += oc.duration
            bgcube.ontime += oc.ontime
            if bgcube.tstart == 0: bgcube.tstart = oc.tstart
            tmean = 0.5 * (oc.tstart + oc.tstop)
            bgcube.wiv_tmean.input(tmean, oc.duration)
            bgcube.tmean = bgcube.wiv_tmean.mean()
            bgcube.tstop = oc.tstop

        bgcubes += [bgcube]
        return bgcubes
