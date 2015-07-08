"""Accesses INTEGRAL/ISGRI background maps/cubes
"""

import numpy as np
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import cube

class BGCube(object):
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

    def rate_shadowgram(self, e_min=0, e_max=np.inf):
        """Shadowgram of count rate, optionally for a subset of the energy range

        unit: cts.s-1.keV-1
        """
        e_idx = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        norm = 1 / np.sum(self.e_max[e_idx] - self.e_min[e_idx])
        return np.delete(
            np.delete(
                np.sum(self.data[:, :, e_idx], 2),
                (64, 65), 1),
            (32, 33, 66, 67, 100, 101), 0) * norm
