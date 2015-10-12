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
of maps sclaed wit light curves
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

    def lincomb(self, t, lightcurves):
        """Perform the linear least squares modelling for all pixels
        and store the results in the four preallocated arrays."""

        n_ebins, n_z, n_y, n_lc = \
            self.backgrounds[0].data.shape[0], 128, 128, len(lightcurves)
        c = np.zeros((n_ebins, n_z, n_y, n_lc))
        sigma = np.zeros((n_ebins, n_z, n_y, n_lc))
        resid = np.zeros((n_ebins, n_z, n_y))
        rank = np.zeros((n_ebins, n_z, n_y))

        A = np.transpose(np.vstack(
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
            # mcs = cs.mean()

            for z in range(n_z):
                for y in range(n_y):
                    rate = cs[:, z, y] # / mcs
                    p_c, p_resid, p_rank, p_sigma = lstsq(A, rate)
                    c[i_e, z, y, :] = p_c
                    resid[i_e, z, y] = p_resid
                    rank[i_e, z, y] = p_rank
                    sigma[i_e, z, y, :] = p_sigma

        self.lc_params = {
            'A': A, 't': t,
            'c': c, 'resid': resid, 'rank': rank, 'sigma': sigma,
            'f': interp1d(t, A, axis=0,
                          bounds_error=False,
                          fill_value=1)
        }

    def bgcube(self, t):
        bc = _bgcube.BGCube()
        bc0 = self.backgrounds[0]
        for attr in bc0.__dict__.keys():
            if attr == 'data': continue
            setattr(bc, attr, getattr(self.backgrounds[0], attr))
        bc.data = np.dot(self.lc_params['c'], self.lc_params['f'](t))
        bc.data = np.insert(bc.data, [32, 32, 64, 64, 96, 96], 0, axis=1)
        bc.data = np.insert(bc.data, [64, 64], 0, axis=2)
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
