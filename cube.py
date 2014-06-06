"""
access INTEGRAL/ISGRI IDL data cubes
"""

import astropy.io.fits
import numpy as np

class Cube:
    bin_width = 0.5 * (1 + np.round(0.054 * np.arange(0, 257)))
    e_min = 12.0 + np.cumsum(bin_width)
    e_max = e_min[1:]
    e_min = e_min[:-1]
    e_gmean = np.sqrt(e_min * e_max)

    def __init__(self, file=None):
        if file:
            fits = astropy.io.fits.open(file)
            fits.verify('ignore')
            self.counts, self.efficiency, self.low_threshold, self.valid = \
            [fits[i].data for i in range(0, 4)]
            fits.close()
        else:
            self.counts = np.zeros((128, 128, 256), np.int32)
            self.efficiency = np.zeros((128, 128), np.float64)
            self.low_threshold = np.zeros((128, 128, 256), np.float32)
            self.valid = np.zeros((128, 128), np.float32)

