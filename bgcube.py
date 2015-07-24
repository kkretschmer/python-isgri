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
Handling of INTEGRAL/ISGRI background maps/cubes

Reading, creating and transforming of cubes is supported
"""

import io
import logging
import os
import sqlite3
import numpy as np
from scipy.stats import norm
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import cube
from . import shadowgram

class BGCube(object):
    """Handles ISGRI background maps
    """
    mdus = range(0, 8)
    mdu_origins = (
        (102, 66), (68, 66), (34, 66), (0, 66),
        (102, 0), (68, 0), (34, 0), (0, 0))

    def __init__(self, path):
        bm_fits = fits.open(path)
        grp = bm_fits['GROUPING']
        self.e_min = grp.data['E_MIN']
        self.e_max = grp.data['E_MAX']
        self.data = np.dstack(
            [bm_fits[pos - 1].data for pos in grp.data['MEMBER_POSITION']])
        bm_fits.close()

    def rate_shadowgram(self, e_min=0, e_max=np.inf):
        """Shadowgram of count rate, optionally for a subset of the energy range

        unit: cts.s-1.keV-1
        """
        e_idx = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        norm = 1 / np.sum(self.e_max[e_idx] - self.e_min[e_idx])
        return np.delete(
            np.delete(
                np.sum(self.data[:, :, e_idx], 2),
                (64, 65), 1),
            (32, 33, 66, 67, 100, 101), 0) * norm

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

    def ps_efficiency_threshold(self, cube):
        return cube.efficiency >= self.eff_min

    def ps_pixel_efficiency(self, cube):
        ps = np.ones_like(cube.counts, dtype=np.bool)
        ps[:, cube.efficiency[-1] < self.eff_min] = False
        return ps

    def ps_not_outlier(self, cube, write_fits=False):
        ps = np.ones_like(cube.counts, dtype=np.bool)
        sg, sg_sig = cube.rate_shadowgram(self.sig_e_range[0],
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
                (scwid, fits) VALUES (?, ?)''', (cube.scwid, blob.getvalue()))
            blob.close()
        return ps

    def ps_not_dark_hot(self, cube, write_fits=False):
        name = 'ps_not_dark_hot'
        ps = np.ones_like(cube.counts, dtype=np.bool)
        cts, exp = cube.cts_exp_shadowgram(*self.erange_dark)
        lp_dd, lp_hd = shadowgram.logprob_not_dark_hot(cts, exp, self.ref_dark)
        cts, exp = cube.cts_exp_shadowgram(*self.erange_hot)
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
                           (cube.scwid, blob.getvalue()))
            blob.close()
        return ps

    def read_cubes(self):
        osacubes, ids = cube.osacubes(self.scwids)
        bgcube = cube.Cube(osacube=True)
        bgcube.counts = np.ma.asarray(bgcube.counts, np.uint32)

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
        for path in osacubes:
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
                    idx = np.logical_and(
                        idx,
                        sel['fn'](oc, *sel['args'], **sel['kwargs'])
                    )
                    logger.debug('{0}: {1}, {2} bin*px valid'.format(
                        oc.scwid, repr(sel['fn']), np.count_nonzero(idx)))
            except:
                logger.error('{0}: failed'.format(oc.scwid))
                continue

            bgcube.counts[idx] += oc.counts[idx]
            bgcube.efficiency[idx] += oc.efficiency[idx] * oc.duration
            bgcube.duration += oc.duration
            bgcube.ontime += oc.ontime

        return bgcube
