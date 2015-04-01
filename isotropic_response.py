"""Handles the response of ISGRI to isotropically incoming radiation
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
