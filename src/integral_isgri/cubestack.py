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

import argparse
import io
import logging
import os
import progressbar
import re
import six
import sqlite3

import numpy as np
from scipy.stats import norm
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import cube
from . import bgcube as _bgcube
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

    def __init__(self):
        self.eff_min = 0.2
        self.sig_e_range = (60, 80)
        self.sig_thresholds = (-2.5, 2.5)
        self.erange_dark = (35, 900)
        self.erange_hot = (35, 51.5)
        self.sigma_max = 4.0
        self.outlier_map_output = {}

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

    def write_outlier_map(self, name, scwid, pixels):
        if 'sqlite' in self.outlier_map_output:
            cursor = self.outlier_map_output['sqlite_cursor']
            cursor.execute('''CREATE TABLE IF NOT EXISTS {0}
                (scwid TEXT PRIMARY KEY, fits BLOB)'''.format(name))
            hdu = fits.PrimaryHDU(pixels)
            blob = six.BytesIO()
            try:
                hdu.writeto(blob)
                cursor.execute(
                    'INSERT OR REPLACE INTO {0}'
                    '  (scwid, fits) VALUES (?, ?)'.format(name),
                    (scwid, blob.getvalue())
                )
                blob.close()
            except TypeError:
                self.logger.error('Writing outlier FITS data to SQLite failed.')
        if 'fits' in self.outlier_map_output:
            hdulist = self.outlier_map_output['fits']
            hdu = fits.ImageHDU(pixels)
            hdu.header['EXTNAME'] = scwid
            try:
                hdulist.append(hdu)
            except:
                self.logger.error('Appending outlier FITS extension failed.')

    def ps_not_outlier(self, cube_in, write_fits=False):
        name = 'ps_not_outlier'
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
            pixels = np.zeros(sg.shape, dtype=np.ubyte)
            pixels[sg.mask == True] += 1
            pixels[dark] += 2
            pixels[hot] += 4
            self.write_outlier_map(name, cube_in.scwid, pixels)
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
            pixels = np.zeros(cts.shape, dtype=np.ubyte)
            pixels[cts.mask == True] += 1
            pixels[dark] += 2
            pixels[hot] += 4
            self.write_outlier_map(name, cube_in.scwid, pixels)
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

    def read_cubes(self, cubes, n_max=None):
        """
        Read ``cubes``, clean them and stack them, optionally grouping them
        into batches of maximum size ``n_max``.
        """
        def bgcube_new():
            bgcube = cube.Cube(osacube=True)
            bgcube.counts = np.ma.asarray(bgcube.counts, np.float32)
            setattr(bgcube, 'n_scw', 0)
            setattr(bgcube, 'scwids', [])
            for attr in ['tstart', 'tmean', 'tstop']:
                setattr(bgcube, attr, 0.0)
            setattr(bgcube, 'wiv_tmean',
                    weighted_incremental_variance(0.0))
            setattr(bgcube, 'wiv_rate',
                    weighted_incremental_variance(bgcube.efficiency))
            return bgcube

        if 'sqlite' in self.outlier_map_output:
            conn = sqlite3.connect(self.outlier_map_output['sqlite'])
            self.outlier_map_output['sqlite_cursor'] = conn.cursor()

        selectors = [
            {'fn': self.ps_pixel_efficiency, 'args': (), 'kwargs': {}},
            {'fn': self.ps_not_dark_hot, 'args': (),
             'kwargs': {'write_fits': True}}
        ]
        self.logger = logging.getLogger('read_cubes')
        logger = self.logger
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('read_cubes.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(ch)

        bgcube = bgcube_new()

        bar = progressbar.ProgressBar()
        for input_cube in bar(cubes):
            if n_max is not None and bgcube.n_scw >= n_max:
                # save the current background cube and start a new one
                #
                logger.info('starting new bgcube')
                value = (yield bgcube)
                bgcube = bgcube_new()

            logger.info('input cube: {}'.format(input_cube))
            try:
                oc = cube.Cube(input_cube)
                if oc.empty:
                    logger.info('cube empty!')
                    continue
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
            bgcube.scwids.append(oc.scwid)

        value = (yield bgcube)

def stack_cubes():
    parser = argparse.ArgumentParser(
        description="""Read a list of INTEGRAL science windows and stack the
        corresponding ISGRI data cubes (e.g. in preparation for creating a
        background model). Write the result to a series of FITS files."""
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--paths',
                        help='file with list of paths to input cubes')
    group.add_argument('-s', '--scwids',
                       help='file with list of input SCWIDs used in'
                       ' conjunction with BASE and CUBE')
    parser.add_argument('-b', '--base', help='base path for input cubes')
    parser.add_argument('-c', '--cube',
                        help='path of input cube relative to SCWID')
    parser.add_argument('-g', '--group', type=int, default=1000000,
                        help='maximum number of SCWs per stack')
    parser.add_argument('-r', '--reference',
                        help='path of reference background cube'
                        ' for outlier detection')
    parser.add_argument('--sigma-max', type=float, default=4.0,
                        help='maximum deviation from reference to tolerate')
    parser.add_argument('-o', '--output', default='stack',
                        help='output FITS file (Python format string)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--outlier-sqlite', default='outlier-pixels.sqlite',
                        help='SQLite output file for outlier pixel maps')
    group.add_argument('--no-outlier-sqlite', dest='outlier_sqlite',
                       action='store_false')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--outlier-fits', default='outlier-pixels.fits',
                        help='FITS output file for outlier pixel maps')
    group.add_argument('--no-outlier-fits', dest='outlier_fits',
                       action='store_false')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)

    if args.paths is not None:
        input_cubes = map(lambda x: x.rstrip('\r\n'),
                          io.open(args.paths, 'r'))
    else:
        def scw_path(scwid):
            id_ver = '{0:012d}.001'.format(scwid)
            rev = id_ver[0:4]
            return os.path.join(args.base, rev, id_ver, args. cube)

        scwids = np.loadtxt(args.scwids, usecols=(0,), dtype=np.uint64)
        input_cubes = map(scw_path, scwids)

    bgb = BackgroundBuilder()

    if args.outlier_sqlite:
        bgb.outlier_map_output['sqlite'] = args.outlier_sqlite
    if args.outlier_fits:
        bgb.outlier_map_output['fits'] = fits.HDUList()

    bgb.setref(_bgcube.BGCube(args.reference))
    bgb.sigma_max = args.sigma_max
    bgcubes = bgb.read_cubes(input_cubes, n_max=args.group)

    for i, bc in enumerate(bgcubes):
        hdu_pri = fits.PrimaryHDU()
        hdu_cts = fits.ImageHDU(np.ma.filled(bc.counts, 0))
        hdu_cts.header['EXTNAME'] = 'COUNTS'
        hdu_eff = fits.ImageHDU(np.ma.filled(bc.efficiency, -1))
        hdu_eff.header['EXTNAME'] = 'EXPOSURE'
        hdu_var = fits.ImageHDU(np.ma.filled(bc.wiv_rate.var(), -1))
        hdu_var.header['EXTNAME'] = 'RATE_VAR'
        hdu_scw = fits.BinTableHDU.from_columns([fits.Column(
            name='SCW', format='12A', array=bc.scwids)])
        hdu_scw.header['EXTNAME'] = 'SCWIDS'
        for hdu in [hdu_cts, hdu_eff, hdu_var, hdu_scw]:
            for keyword in ['duration', 'n_scw', 'ontime',
                            'tmean', 'tstart', 'tstop']:
                hdu.header[keyword] = getattr(bc, keyword)
        fitsfile = '{output}{num:04d}.fits'.format(
            output=args.output, num=i)
        hdulist = fits.HDUList([hdu_pri, hdu_cts, hdu_eff, hdu_var, hdu_scw])
        hdulist.writeto(fitsfile, clobber=True)

    if args.outlier_fits:
        try:
            bgb.outlier_map_output['fits'].writeto(
                args.outlier_fits, clobber=True)
        except:
            logging.error('Writing outlier FITS file failed.')
