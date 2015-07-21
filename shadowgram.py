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
"""Manipulates INTEGRAL/ISGRI IDL/OSA detector-space shadowgrams
"""

import numpy as np
from scipy.stats import poisson

def logprob_not_dark_hot(cts, exp, ref, quantile=64):
    """Log-probabilities of pixels being not dark and not hot.

    The cumulative Poisson distribution function (for dark pixels) and its
    inverse, the survival function (for hot pixels) are calculated and
    normalised to account for the number of dead pixels and the additional
    scatter due to mismatch of the reference shape. The pixel ``quantile``
    is used for shape normalisation.

    Arguments
    ----------
    cts : ndarray of ints
        counts per pixel
    exp : ndarray of floats
        exposure per pixel
    ref : ndarray of floats
        reference intensity per pixel

    Keyword arguments
    -----------------
    quantile
        reference pixel for shape normalisation

    Returns
    -------
    (d, h) : log-probabilities of pixels being not (dark, hot)
    """

    if hasattr(cts, 'mask'):
        invalid = cts.mask
        valid = np.logical_not(invalid)
    else:
        invalid = []
        valid = Ellipsis
    ref_cts = ref / ref[valid].mean() * cts[valid].mean()

    logcdf = poisson.logcdf(cts, ref_cts)
    logsf = poisson.logsf(cts, ref_cts)
    logcdf[invalid] = 0
    logsf[invalid] = 0
    logcdf_s = np.sort(logcdf[valid])
    logsf_s = np.sort(logsf[valid])

    logcdf, logsf = \
        [f(cts, ref_cts) for f in (poisson.logcdf, poisson.logsf)]
    for f in logcdf, logsf:
        f[invalid] = 0
    logcdf_s, logsf_s = \
        [np.sort(f[valid]) for f in (logcdf, logsf)]
    norm_cdf, norm_sf = \
        [np.log((cts.count() - quantile) / quantile) / logf_s[quantile]
         for logf_s in (logcdf_s, logsf_s)]
    offset = np.log(1 - ((128**2 - cts.count()) / 128**2))

    return (offset - logcdf * norm_cdf,
            offset - logsf * norm_sf)
