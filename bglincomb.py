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

        e_min, e_max = 25, 80
        cs = np.vstack(
            [bc.rate_shadowgram(
                e_min, e_max, per_keV=False)[np.newaxis, Ellipsis]
             for bc in self.backgrounds])
        # mcs = cs.mean()

        A = np.transpose(np.vstack(
            [lc / np.amax(lc) for lc in lightcurves]))
        
        n_lc = A.shape[1]
        c = np.zeros((128, 128, n_lc))
        sigma = np.zeros((128, 128, n_lc))
        resid = np.zeros((128, 128))
        rank = np.zeros((128, 128))

        for z in range(128):
            for y in range(128):
                rate = cs[:, z, y] # / mcs
                p_c, p_resid, p_rank, p_sigma = lstsq(A, rate)
                c[z, y, :] = p_c
                resid[z, y] = p_resid
                rank[z, y] = p_rank
                sigma[z, y, :] = p_sigma

        self.lc_params = {
            'A': A, 't': t,
            'c': c, 'resid': resid, 'rank': rank, 'sigma': sigma,
            'f': interp1d(t, A, axis=0,
                          bounds_error=False,
                          fill_value=1)
        }

    def bgcube(self, t):
        return np.dot(self.lc_params['c'], self.lc_params['f'](t))
