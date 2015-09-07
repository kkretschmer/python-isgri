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
from scipy.optimize import minimize

from .. import cube
from . import fitquality

def constant(bs, cts, exp):
    rate = bs
    return rate

def proportional(bs, cts, exp):
    x = cts.sum() / (bs * exp).sum()
    rate = bs * x
    print('proportional: ', x)
    return rate

def mdu_proportional(bs, cts, exp):
    rate = np.ma.zeros(cts.shape)
    for mdu in cube.Cube.mdu_slices:
        rate[mdu] = proportional(bs[mdu], cts[mdu], exp[mdu])
    return rate

def linear(bs, cts, exp):
    bs_mean = bs.mean()
    def rate(x):
        return (x[0] * bs_mean + x[1] * bs)

    def fun(x):
        return fitquality.chi2(rate(x), cts, exp)[0]

    x0 = np.array([0.1, 0.9])
    minres = minimize(fun, x0, method='Powell')
    x = minres.x
    print('linear: ', x)
    return rate(x)

def mdu_linear(bs, cts, exp):
    rate = np.ma.zeros(cts.shape)
    for mdu in cube.Cube.mdu_slices:
        rate[mdu] = linear(bs[mdu], cts[mdu], exp[mdu])
    return rate
