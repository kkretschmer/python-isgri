"""Accesses INTEGRAL/ISGRI IDL/OSA data cubes
"""

try:
    import astropy.io.fits as _fits
except:
    import pyfits as _fits

import matplotlib.pyplot as plt
import numpy as np
import os

class Cube(object):
    """Handles ISGRI cubes in IDL and OSACube formats.
    """

    filenames = { 'dsg': 'isgri_detector_shadowgram_BIN_S.fits.gz',
                  'esg': 'isgri_efficiency_shadowgram_BIN_S.fits.gz' }
    mdus = range(0, 8)
    mdu_origins = (
        (96, 64), (64, 64), (32, 64), (0, 64),
        (96, 0), (64, 0), (32, 0), (0, 0))

    def default_bins(self):
        bin_width = 0.5 * (1 + np.round(0.054 * np.arange(0, 257)))
        e_min = 12.0 + np.cumsum(bin_width)
        self.e_max = e_min[1:]
        self.e_min = e_min[:-1]
        self.e_gmean = np.sqrt(self.e_min * self.e_max)
        self.bin_width = bin_width[1:]

    def __init__(self, path=None):
        # if path is given, load cube from there, otherwise create
        # empty cube
        #
        if path:
            # try to open the provided path as FITS file (IDL cube),
            # otherwise as directory containing OSACube files
            #
            try:
                # IDL cube
                #
                fits = _fits.open(path)
                fits.verify('ignore')
                self.counts, self.pixel_eff, \
                    self.low_threshold, self.valid = \
                        [fits[i].data for i in range(0, 4)]
                self.duration = fits[0].header['DURATION']
                self.mdu_eff = np.array([fits[0].header['MDU%1i_EFF' % i] \
                                         for i in range(0, 8)])
                self.deadc = np.array([fits[0].header['DEADC%1i' % i] \
                                       for i in range(0, 8)])
                fits.close()
                self.default_bins()

                # calculate efficiency shadowgram
                #
                pixel_mod_eff = self.pixel_eff * self.valid
                for i in self.mdus:
                    origin = self.mdu_origins[i]
                    pixel_mod_eff[origin[0]:origin[0]+32,
                                  origin[1]:origin[1]+64] *= \
                        self.mdu_eff[i] * (1 - self.deadc[i])
                self.pixel_mod_eff = pixel_mod_eff
                self.efficiency = np.tile(np.array(pixel_mod_eff,
                                                   dtype=np.float32),
                                          (256, 1, 1))
                self.efficiency *= self.low_threshold

            except IsADirectoryError:
                # OSACube
                #
                dsg = _fits.open(os.path.join(path, self.filenames['dsg']))
                esg = _fits.open(os.path.join(path, self.filenames['esg']))
                grp = dsg['GROUPING']
                self.e_min = grp.data['E_MIN']
                self.e_max = grp.data['E_MAX']
                self.e_gmean = np.sqrt(self.e_min * self.e_max)
                self.bin_width = self.e_max - self.e_min
                self.duration = grp.data['ONTIME'][0]

                self.counts = np.zeros((len(self.e_min), 128, 128), np.int16)
                self.efficiency = np.zeros((len(self.e_min), 128, 128), np.float32)
                dsg.readall()
                esg.readall()
                for ext in range(2, len(esg)):
                    self.counts[ext - 2] = dsg[ext].data
                    self.efficiency[ext - 2] = esg[ext].data
                dsg.close()
                esg.close()

        else:
            self.counts = np.zeros((256, 128, 128), np.int32)
            self.pixel_eff = np.zeros((128, 128), np.float64)
            self.low_threshold = np.zeros((256, 128, 128), np.float32)
            self.valid = np.zeros((128, 128), np.float32)
            self.duration = 0.0
            self.mdu_eff = np.zeros((8,), np.float64)
            self.deadc = np.zeros((8,), np.float64)

    def stack(self, summand):
        self.counts += summand.counts
        self.efficiency = (self.efficiency * self.duration + \
                        summand.efficiency * summand.duration) / \
            (self.duration + summand.duration)
        self.mdu_eff = (self.mdu_eff * self.duration + \
                        summand.mdu_eff * summand.duration) / \
            (self.duration + summand.duration)
        self.valid = (self.valid * self.duration + \
                        summand.valid * summand.duration) / \
            (self.duration + summand.duration)
        self.deadc = 1 - ((1 - self.deadc) * self.duration + \
                          (1 - summand.deadc) * summand.duration) / \
            (self.duration + summand.duration)
        self.duration += summand.duration

    def spectrum(self):
        return self.counts.sum((1, 2))

    def image(self, e_min=0, e_max=np.inf):
        e_idx = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        return self.counts[e_idx, :, :].sum(0)

    def corr_shad(self):
        rate = np.array(self.counts, dtype='float64')
        rate /= self.duration * 0.4**2
        for i in self.mdus:
            origin = self.mdu_origins[i]
            rate[:, origin[0]:origin[0]+32, origin[1]:origin[1]+64] /= \
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
