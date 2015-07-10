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
Handles the response of ISGRI to isotropically incoming radiation
"""

import glob
import numpy as np
import os
import re

class isotropic_response(object):

    def __init__(self, path):
        dirs = np.array(glob.glob(os.path.join(path, '*')))
        e_in = np.array([re.sub('keV$', '', os.path.basename(d))
                              for d in dirs], dtype=np.float64)
        idx = np.argsort(e_in)
        self.e_in = e_in[idx]
        ne_in = len(self.e_in)

        r0 = np.loadtxt(os.path.join(dirs[0], 'response.txt'))
        self.e_min = r0[:, 1]
        self.e_max = r0[:, 0]
        self.response = np.zeros((ne_in, r0.shape[0]))
        for i in idx:
            txtfile = os.path.join(dirs[idx[i]], 'response.txt')
            ri = np.loadtxt(txtfile)
            self.response[i, :] = ri[:, 2]
