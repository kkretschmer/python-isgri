"""Accesses INTEGRAL/ISGRI background maps/cubes
"""

import logging
import os
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

class BackgroundBuilder(object):
    def __init__(self, scwids):
        self.scwids = np.array(scwids, dtype=np.uint64)
        self.eff_min = 0.2
        self.sig_e_range = range(60, 80)
        self.sig_thresholds = (-2.5, 2.5)

    def ps_efficiency_threshold(self, cube):
        return cube.efficiency >= self.eff_min

    def ps_pixel_efficiency(self, cube):
        ps = np.ones_like(cube.counts, dtype=np.bool)
        ps[:, cube.efficiency[-1] < self.eff_min] = False
        return ps

    def ps_not_outlier(self, cube, write_fits=False):
        ps = np.ones_like(cube.counts, dtype=np.bool)
        sg, sg_sig = cube.rate_shadowgram(self.sig_e_range.start,
                                          self.sig_e_range.stop, True)
        mi, di = sg.mean(), sg.std()
        dark, hot = \
            np.logical_and(sg.data < mi + di * self.sig_thresholds[0],
                           np.logical_not(sg.mask)), \
            np.logical_and(sg.data > mi + di * self.sig_thresholds[1],
                           np.logical_not(sg.mask))
        ps[:, np.logical_or(dark, hot)] = False
        if write_fits:
            pixels = np.zeros(sg.shape, dtype=np.ubyte)
            pixels[sg.mask == True] += 1
            pixels[dark] += 2
            pixels[hot] += 4
            fitsfile = 'ps_not_outlier/{0}/{1}.fits'.format(
                cube.scwid[0:4], cube.scwid)
            hdu = fits.PrimaryHDU(pixels)
            os.makedirs(os.path.dirname(fitsfile), exist_ok=True)
            hdu.writeto(fitsfile, clobber=True)
        return ps

    def read_cubes(self):
        osacubes, ids = cube.osacubes(self.scwids)
        bgcube = cube.Cube(osacube=True)
        selectors = [
            {'fn': self.ps_pixel_efficiency, 'args': (), 'kwargs': {}},
            {'fn': self.ps_not_outlier, 'args': (), 'kwargs': {'write_fits': True}}
        ]
        logger = logging.getLogger('read_cubes')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('read_cubes.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        for path in osacubes:
            oc = cube.Cube(path).rebin()
            idx = np.logical_not(oc.counts.mask)
            logger.info('{0}: loaded'.format(oc.scwid))
            logger.debug('{0}: {1} px valid'.format(
                oc.scwid, np.count_nonzero(idx)))
            for sel in selectors:
                idx = np.logical_and(
                    idx,
                    sel['fn'](oc, *sel['args'], **sel['kwargs'])
                )
                logger.debug('{0}: loaded, {1}, {2} px valid'.format(
                    oc.scwid, repr(sel['fn']), np.count_nonzero(idx)))
                bgcube.counts[idx] += oc.counts[idx]
                bgcube.efficiency[idx] += oc.efficiency[idx] * oc.duration
                bgcube.duration += oc.duration
                bgcube.ontime += oc.ontime
        return bgcube
