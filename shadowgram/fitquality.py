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

from __future__ import division

import numpy as np
from scipy.stats import poisson

def chi2(rate, cts, exp):
    """
    chi-squared of a shadowgram relative to a background shadowgram

    rate: expected rate shadowgram
    cts: cube counts shadowgram
    exp: cube exposure shadowgram
    """
    pixels = (cts - (rate * exp))**2 / cts
    return (pixels.sum(), pixels.count())

def logl(rate, cts, exp):
    """
    log-likelihood of a shadowgram relative to a background shadowgram

    rate: expected rate shadowgram
    cts: cube counts shadowgram
    exp: cube exposure shadowgram
    """
    idx = np.logical_not(
        np.logical_or(np.logical_or(cts.mask, exp.mask), rate.mask))
    return (poisson.logpmf(
        cts[idx], rate[idx] * exp[idx]).sum(),
            np.count_nonzero(idx))

def sum_asic():
    return lambda sg: sg.reshape(64, 2, 64, 2).sum(axis=3).sum(axis=1), 0.25

def sum_polycell():
    return lambda sg: sg.reshape(32, 4, 32, 4).sum(axis=3).sum(axis=1), 0.0625

def fq_summed(fq, grp):
    def fun_fq_summed(rate, cts, exp):
        grp_fn, scale = grp()
        mask = np.logical_or(cts.mask, rate.mask)
        args = [grp_fn(np.ma.MaskedArray(i, mask))
                for i in (rate * scale, cts, exp)]
        return fq(*args)
    return fun_fq_summed
