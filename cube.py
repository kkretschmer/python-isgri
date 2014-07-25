"""
access INTEGRAL/ISGRI IDL data cubes
"""

import astropy.io.fits
import numpy as np
import matplotlib.pyplot as plt

class Cube:
    bin_width = 0.5 * (1 + np.round(0.054 * np.arange(0, 257)))
    e_min = 12.0 + np.cumsum(bin_width)
    e_max = e_min[1:]
    e_min = e_min[:-1]
    e_gmean = np.sqrt(e_min * e_max)
    bin_width = bin_width[1:]

    def __init__(self, file=None):
        if file:
            fits = astropy.io.fits.open(file)
            fits.verify('ignore')
            self.counts, self.efficiency, self.low_threshold, self.valid = \
                [fits[i].data for i in range(0, 4)]
            self.duration = fits[0].header['DURATION']
            self.mdu_eff = np.array([fits[0].header['MDU%1i_EFF' % i] \
                for i in range(0, 8)])
            self.deadc = np.array([fits[0].header['DEADC%1i' % i] \
                for i in range(0, 8)])
            fits.close()
        else:
            self.counts = np.zeros((128, 128, 256), np.int32)
            self.efficiency = np.zeros((128, 128), np.float64)
            self.low_threshold = np.zeros((128, 128, 256), np.float32)
            self.valid = np.zeros((128, 128), np.float32)

    def spectrum(self):
        return self.counts.sum((1, 2))

    def corr_shad(self):
        rate = np.array(self.counts, dtype='float64')
        rate /= self.duration * 0.4**2
        mdu_origins = (
            (96, 64), (64, 64), (32, 64), (0, 64),
            (96, 0), (64, 0), (32, 0), (0, 0))
        for i in range(0, 8):
            origin = mdu_origins[i]
            rate[:, origin[0]:origin[0]+32, origin[0]:origin[0]+64] /= \
                self.mdu_eff[i]
        return(rate)

    def show(self, e_min=25.0, e_max = 80.0, *args, **kwargs):
        e = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        print(self.e_min[e])
        shd = self.corr_shad()[e, :, :].sum(0)
        dpi = np.zeros((134, 130))
        for y in (64, 65):
            dpi[:, y] = np.nan
        for z in (32, 33, 66, 67, 100, 101):
            dpi[z, :] = np.nan
        for y in range(0, 2):
            for z in range(0, 4):
                ys0, ys1 = y * 64, y * 64 + 64
                zs0, zs1 = z * 32, z * 32 + 32
                yd0, yd1 = y * 66, y * 66 + 64
                zd0, zd1 = z * 34, z * 34 + 32
                dpi[zd0:zd1, yd0:yd1] = shd[zs0:zs1, ys0:ys1]
        plt.imshow(dpi, aspect='equal', interpolation='nearest',
                   origin='lower', extent=(-0.5, 129.5, -0.5, 133.5),
                   *args, **kwargs)
        # plt.plot([0, 1])
        plt.show()
        plt.close()
