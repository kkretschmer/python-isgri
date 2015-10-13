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
Model the time evolution of ISGRI background maps using a linear combination
of maps scaled with linearly interpolated light curves.
"""

import argparse
import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import bgcube as _bgcube
from . import cube

class BGLinComb(object):
    def __init__(self, backgrounds):
        self.backgrounds = sorted(backgrounds, key=lambda x: x.tmean)

    def tmean(self):
        return np.array([bc.tmean for bc in self.backgrounds])

    def lightcurve(self, e_min=0, e_max=np.inf):
        def mean_rate(bc):
            return bc.rate_shadowgram(e_min, e_max, per_keV=False).mean()

        return np.array([mean_rate(bc) for bc in self.backgrounds])

    def set_interp(self, t_min=0, t_max=1e5, **kwargs):
        t = np.hstack([t_min, self.t, t_max])

        def extrapolate(i0, i1, t):
            t0, t1 = self.t[[i0, i1]]
            dA = self.A[i1, :] - self.A[i0, :]
            return self.A[i0, :] + dA * (t - t0) / (t1 - t0)

        v_min = extrapolate(0, 1, t_min)
        v_max = extrapolate(-2, -1, t_max)
        A = np.vstack([v_min, self.A, v_max])

        self.f = interp1d(t, A, axis=0, **kwargs)

    def lincomb(self, t, lightcurves):
        """Perform the linear least squares modelling for all pixels
        and store the results in the four preallocated arrays."""

        self.n_ebins, self.n_z, self.n_y, self.n_lc = \
            self.backgrounds[0].data.shape[0], 128, 128, len(lightcurves)
        self.c = np.zeros((self.n_ebins, self.n_z, self.n_y, self.n_lc))
        self.sing = np.zeros((self.n_ebins, self.n_z, self.n_y, self.n_lc))
        self.resid = np.zeros((self.n_ebins, self.n_z, self.n_y))
        self.rank = np.zeros((self.n_ebins, self.n_z, self.n_y))

        self.t = t
        self.A = np.transpose(np.vstack(
            [lc / np.amax(lc) for lc in lightcurves]))

        bc0 = self.backgrounds[0]
        e_ranges = zip(bc0.e_min, bc0.e_max)
        for i_e, e_range in enumerate(e_ranges):
            e_min, e_max = e_range
            logging.info(
                'fitting energy range: [{e_min:0.1f} keV, {e_max:0.1f} keV]'.\
                format(e_min=e_min, e_max=e_max))
            cs = np.vstack(
                [bc.rate_shadowgram(
                    e_min, e_max, per_keV=False)[np.newaxis, Ellipsis]
                 for bc in self.backgrounds])

            for z in range(self.n_z):
                for y in range(self.n_y):
                    rate = cs[:, z, y]
                    p_c, p_resid, p_rank, p_sing = lstsq(self.A, rate)
                    self.c[i_e, z, y, :] = p_c
                    self.resid[i_e, z, y] = p_resid
                    self.rank[i_e, z, y] = p_rank
                    self.sing[i_e, z, y, :] = p_sing

        self.set_interp()

    def writeto(self, out):
        logging.info('writing output')
        list = fits.HDUList()

        ext = fits.BinTableHDU.from_columns(
            [fits.Column(name='time', format='D', unit='d',
                         array=self.t)])
        ext.header['EXTNAME'] = 'TIME'
        list.append(ext)

        ext = fits.BinTableHDU.from_columns(
            [fits.Column(name=attr, format='D', unit='keV',
                         array=getattr(self.backgrounds[0], attr))
             for attr in ['e_min', 'e_gmean', 'e_max', 'bin_width']]
        )
        ext.header['EXTNAME'] = 'ENERGY'
        list.append(ext)

        ext = fits.BinTableHDU.from_columns(
            [fits.Column(name='tracers',
                         format='{}D'.format(self.n_lc),
                         array=self.A)])
        ext.header['EXTNAME'] = 'TRACERS'
        list.append(ext)

        ext = fits.ImageHDU(self.c)
        ext.header['EXTNAME'] = 'CUBES'
        ext.header['BUNIT'] = 's-1'
        list.append(ext)

        ext = fits.ImageHDU(self.resid)
        ext.header['EXTNAME'] = 'RESIDUAL'
        ext.header['BUNIT'] = 's-1'
        list.append(ext)

        list.writeto(out, clobber=True)

    def bgcube(self, t):
        bc = _bgcube.BGCube()
        compact = np.dot(self.c, self.f(t))
        expand_z = np.insert(compact, [32, 32, 64, 64, 96, 96], 0, axis=1)
        bc.data = np.insert(expand_z, [64, 64], 0, axis=2)
        return bc

def mktemplate():
    parser = argparse.ArgumentParser(
        description="""Build a template for an ISGRI background model
        composed of a linear combination of background cubes, each scaled
        by linear interpolation of a light curve over time."""
    )
    parser.add_argument('-o', '--output', help='output FITS file')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('backgrounds', metavar='bg', nargs='+',
                        help='input bgcube FITS file')
    args = parser.parse_args()

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)

    if args.output is None:
        logging.warn('No output file specified, results will not be saved.')

    def read_bgcube(path):
        logging.info('reading "{path}"'.format(path=path))
        hdulist = fits.open(path)
        c = cube.Cube()
        c.counts, c.efficiency = [hdulist[i].data for i in [1, 2]]
        c.tmean = hdulist[1].header['tmean']
        hdulist.close()
        return _bgcube.BGCube(c)

    backgrounds = [read_bgcube(ff) for ff in args.backgrounds]
    logging.info('read {n_cubes} input cubes'.format(
        n_cubes=len(backgrounds)))
    blc = BGLinComb(backgrounds)

    logging.info('extracting light curves')
    t = blc.tmean()
    lightcurves = [
        np.ones_like(t),
        blc.lightcurve(25, 80),
        blc.lightcurve(80, 200),
        blc.lightcurve(600, 900),
        (t - np.amin(t)) / (np.amax(t) - np.amin(t))
    ]

    logging.info('fitting light curves')
    blc.lincomb(t, lightcurves)

    if args.output is not None:
        blc.writeto(args.output)
