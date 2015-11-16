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
Handling of INTEGRAL/ISGRI background maps/cubes

Reading, creating and transforming of cubes is supported
"""

import argparse
import datetime
import os
import re

import numpy as np
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from . import cube
from . import shadowgram

class BGCube(object):
    """Handles ISGRI background maps
    """
    mdus = range(0, 8)
    mdu_origins_com = (
        (96, 64), (64, 64), (32, 64), (0, 64),
        (96, 0), (64, 0), (32, 0), (0, 0))
    mdu_origins_exp = (
        (102, 66), (68, 66), (34, 66), (0, 66),
        (102, 0), (68, 0), (34, 0), (0, 0))
    n_z, n_y = 134, 130

    def __init__(self, src=None):
        if src is None:
            # create an empty cube
            #
            temp_cube = cube.Cube(osacube=True)
            self.e_min = temp_cube.e_min
            self.e_max = temp_cube.e_max
            self.data = np.zeros((len(self.e_min), self.n_z, self.n_y),
                                 dtype=np.float32)
            self.tstart, self.tstop = 1020.0, 11000.0 # launch - deorbit
        elif hasattr(src, 'counts') and hasattr(src, 'efficiency'):
            # cube-like source (N, 128, 128) where efficiency is
            # actually exposure
            #
            for attr in src.__dict__:
                if attr == 'counts' or attr == 'efficiency': continue
                setattr(self, attr, getattr(src, attr))
            rate = np.zeros_like(src.counts, dtype=np.float32)
            idx = np.logical_and(src.counts > 0,
                                 src.efficiency > 0)
            rate[idx] = src.counts[idx] / src.efficiency[idx]
            rate = np.insert(rate, [32, 32, 64, 64, 96, 96], 0, axis=1)
            rate = np.insert(rate, [64, 64], 0, axis=2)
            self.data = rate
        else:
            # read the OSA background from a FITS file
            #
            bm_fits = fits.open(src)
            grp = bm_fits['GROUPING']
            self.e_min = grp.data['E_MIN']
            self.e_max = grp.data['E_MAX']
            self.data = np.vstack(
                [bm_fits[pos - 1].data[np.newaxis, Ellipsis]
                 for pos in grp.data['MEMBER_POSITION']])
            bm_fits.close()

    @classmethod
    def fromstack(cls, path):
        stack = cube.Cube()
        fits_in = fits.open(path)
        hdr = fits_in['COUNTS'].header
        stack.counts = fits_in['COUNTS'].data
        stack.efficiency = fits_in['EXPOSURE'].data
        for attr in ('tstart', 'tmean', 'tstop'):
            setattr(stack, attr, hdr[attr])
        return cls(stack)

    def rate_shadowgram(self, e_min=0, e_max=np.inf, per_keV=True):
        """Shadowgram of count rate, optionally for a subset of the energy range

        unit: cts.s-1.keV-1 or cts.s-1 depending on the value of per_keV
        """
        e_idx = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        exp_img = self.data[e_idx, Ellipsis].sum(axis=0)
        if per_keV:
            exp_img /= (self.e_max[e_idx] - self.e_min[e_idx]).sum()
        return np.delete(
            np.delete(exp_img, (64, 65), 1),
            (32, 33, 66, 67, 100, 101), 0)

    def writeto(self, out, template=None, **kwargs):
        """Write the background cube to an OSA format FITS file, using
        an existing ``template`` FITS file as source of the DAL structure"""

        tpl = fits.open(template)
        now = datetime.datetime.utcnow()

        origin_utc = datetime.datetime(1999, 12, 31, 23, 58, 55, 816000)
        def utc2ijd(utc):
            return (utc - origin_utc) / datetime.timedelta(days=1)

        def ijd2utc(ijd):
            return origin_utc + datetime.timedelta(days=ijd)

        def timestamp(utc):
            return utc.strftime('%Y-%m-%dT%H:%M:%S')

        grp = tpl['GROUPING']
        gh = grp.header
        updates = [
            ('CREATOR', __name__),
            ('CONFIGUR', 'osa_10.0'),
            ('DATE', timestamp(now)),
            ('STAMP', ' '.join([timestamp(now), __name__])),
            ('RESPONSI', '{user}@{host}'.format(
                user=os.environ['USER'],
                host=os.environ['HOSTNAME'])
            ),
            ('LOCATN', re.match(
                '[^.]*\.(.*)', os.environ['HOSTNAME']).group(1))
        ]
        for key, default in updates:
            if hasattr(self, key):
                value = getattr(self, key)
            else:
                value = default
            gh[key] = value
        grp.data['VSTART'] = self.tstart
        grp.data['VSTOP'] = self.tstop

        images = filter(
            lambda e: ('EXTNAME' in e.header.keys() and
                       e.header['EXTNAME'] == 'ISGR-BACK-BKG'),
            tpl)
        updates += [
            ('STRT_VAL', timestamp(ijd2utc(self.tstart))),
            ('END_VAL', timestamp(ijd2utc(self.tstop))),
            ('VSTART', self.tstart),
            ('VSTOP', self.tstop)
        ]
        for img in images:
            for i_e in np.where(
                    np.logical_and(
                        self.e_min == img.header['E_MIN'],
                        self.e_max == img.header['E_MAX'])
                    )[0]:
                img.data[:, :] = self.data[i_e]
                for key, default in updates:
                    if hasattr(self, key):
                        value = getattr(self, key)
                    else:
                        value = default
                    img.header[key] = value

        tpl.writeto(out, **kwargs)

def stack2osa():
    parser = argparse.ArgumentParser(
        description="""Read a template for an ISGRI background model
        composed of a linear combination of background cubes, each scaled
        by linear interpolation of a light curve over time. Interpolate
        it in time and write it to a background cube."""
    )
    parser.add_argument('-i', '--input', help='input stack FITS file')
    parser.add_argument('-o', '--output', help='output cube FITS file')
    parser.add_argument('-t', '--template', help='template cube FITS file')
    parser.add_argument('-l', '--outlier-map',
                        help='FITS file with outlier counts per pixel')
    parser.add_argument('-c', '--max-outlier-count', type=int, default=0,
                        help='maximum allowed outlier count')
    parser.add_argument('-e', '--mask-module-edges', type=int, default=0,
                        metavar='PIXELS',
                        help='number of pixels to mask around module edges')
    parser.add_argument('--vstart', help='IJD of validity start')
    parser.add_argument('--vstop', help='IJD of validity stop')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)

    if args.output is None:
        args.output = re.sub('\.fits', '_bkgcube.fits', args.input)

    bc = BGCube.fromstack(args.input)
    for attr in ('tstart', 'tstop'):
        arg = getattr(args, re.sub('^t', 'v', attr))
        if arg:
            setattr(bc, attr, float(arg))

    bc.writeto(args.output,
               template=args.template,
               clobber=True)
