"""Accesses INTEGRAL/ISGRI background maps/cubes
"""

import numpy as np
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

class Cube(object):
    """Handles ISGRI background maps
    """
    mdus = range(0, 8)
    mdu_origins = (
        (102, 66), (68, 66), (34, 66), (0, 66),
        (102, 0), (68, 0), (34, 0), (0, 0))

    def __init__(self, path):
        bm_fits = fits.open(path)
        grp = bm_fits['GROUPING']
        self.e_min = grp.data['E_MIN']
        self.e_max = grp.data['E_MAX']
        self.data = np.dstack(
            [bm_fits[pos - 1].data for pos in grp.data['MEMBER_POSITION']])
        bm_fits.close()

    def image(self, e_min=0, e_max=np.inf):
        bins = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        return np.sum(self.data[:, :, bins], 2)
