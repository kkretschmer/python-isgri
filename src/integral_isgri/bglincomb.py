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

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import bgcube as _bgcube
from . import cube

class BGTimeSeries(object):
    def __init__(self, backgrounds):
        self.backgrounds = backgrounds

    def lightcurve(self, e_min=0, e_max=np.inf):
        return np.array(
            [bc.rate_shadowgram(e_min, e_max, per_keV=False).mean()
             for bc in self.backgrounds]
        )

    def lincomb(self, lightcurves, t):
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
            print('[{e_min:0.1f} keV, {e_max:0.1f} keV]'.\
                  format(e_min=e_min, e_max=e_max), end=' ')
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
