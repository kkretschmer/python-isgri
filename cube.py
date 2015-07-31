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
"""Accesses INTEGRAL/ISGRI IDL/OSA data cubes
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
import fitsio

try:
    throng_dir = os.environ['THRONG_DIR']
except KeyError:
    throng_dir = '/Integral/throng'

byscw_dir = '/Integral/data/reduced/ddcache/byscw'

class Cube(object):
    """Handles ISGRI cubes in IDL and OSACube formats.

    Cube.counts = (E, y, x)
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

    def __init__(self, path=None, osacube=False):
        # if path is given, load cube from there, otherwise create
        # empty cube
        #
        if path:
            # if path is a directory, assume it contains OSACube files
            # otherwise try to open the provided path as FITS file (IDL cube)
            #
            if os.path.isdir(path):
                # OSACube
                #
                self.type = 'OSA'
                match = re.compile('/(\d{12})\.\d{3}/').search(path)
                if match:
                    self.scwid = match.groups()[0]
                def open_file(id):
                    return fitsio.FITS(os.path.join(path, self.filenames[id]))

                try:
                    dsg = open_file('dsg')
                except IOError:
                    self.empty = True
                    return
                grp = dsg['GROUPING']
                if grp.get_nrows() == 0:
                    self.empty = True
                    return
                else:
                    self.empty = False

                self.e_min = grp['E_MIN'].read()
                self.e_max = grp['E_MAX'].read()
                self.e_gmean = np.sqrt(self.e_min * self.e_max)
                self.bin_width = self.e_max - self.e_min
                self.duration = grp['ONTIME'].read()[0]
                exts = grp['MEMBER_POSITION'][:] - 1

                # read the specified field from the header of the first
                # detector shadowgram HDU
                #
                self.header_fields = ('TSTART', 'TSTOP', 'TFIRST', 'TLAST', 'TELAPSE',
                          'ONTIME', 'DEADC', 'EXPOSURE', 'LIVETIME',
                          'RISE_MIN', 'RISE_MAX')
                header = dsg['ISGR-DETE-SHD'].read_header()
                for keyword in self.header_fields:
                    setattr(self, keyword.lower(), header[keyword])

                def read_images(sg):
                    return np.vstack(
                        [np.reshape(sg[e].read(), (1, 128, 128)) for e in exts])

                self.counts = read_images(dsg)
                dsg.close()

                try:
                    esg = open_file('esg')
                except IOError:
                    self.empty = True
                    return
                self.efficiency = read_images(esg)
                esg.close()
                invalid = np.all(self.efficiency == 0, 0)
                self.valid = np.logical_not(invalid)
                mask = np.repeat(
                    np.expand_dims(invalid, 0),
                    len(self.e_min), 0)
                for attr in ('counts', 'efficiency'):
                    setattr(self, attr, \
                        np.ma.MaskedArray(getattr(self, attr), mask))
                self.mdu_eff = np.zeros_like(self.mdus, dtype=float)
                for i in self.mdus:
                    origin = self.mdu_origins[i]
                    self.mdu_eff[i] = np.max(
                        self.efficiency[:, origin[0]:origin[0]+32,
                                        origin[1]:origin[1]+64])
                self.mdu_eff /= np.max(self.mdu_eff)

            else:
                # IDL cube
                #
                self.type = 'IDL'
                match = re.compile('/(\d{12})\D').search(path)
                if match:
                    self.scwid = match.groups()[0]
                self.default_bins()
                ff = fitsio.FITS(path)
                self.valid = ff[3].read()
                invalid = self.valid == 0
                self.pixel_eff = np.ma.MaskedArray(ff[1].read(), invalid,
                                                   dtype=np.float32)
                mask = np.repeat(
                    np.expand_dims(invalid, 0),
                    len(self.e_min), 0)
                self.counts, self.low_threshold = \
                        [np.ma.MaskedArray(ff[i].read(), mask) for i in (0, 2)]
                header = ff[0].read_header()
                self.duration = header['DURATION']
                self.mdu_eff = np.array([header['MDU%1i_EFF' % i] \
                                         for i in self.mdus])
                self.deadc = np.array([header['DEADC%1i' % i] \
                                       for i in self.mdus])
                ff.close()

                # calculate efficiency shadowgram
                #
                self.pixel_mod_eff = np.copy(self.pixel_eff)
                for i in self.mdus:
                    origin = self.mdu_origins[i]
                    self.pixel_mod_eff[origin[0]:origin[0]+32,
                                       origin[1]:origin[1]+64] *= \
                        self.mdu_eff[i] * (1 - self.deadc[i])
                self.efficiency = np.repeat(
                    np.expand_dims(self.pixel_mod_eff, 0),
                    len(self.e_min), 0) * self.low_threshold

        else:
            if osacube:
                self.type = 'OSA'
                self.default_bins()
                self.counts = np.ma.zeros((len(self.e_min), 128, 128), np.int16)
                self.efficiency = np.ma.zeros((len(self.e_min), 128, 128), np.float32)
                self.duration = 0.0
                self.ontime = 0.0
            else:
                self.type = 'IDL'
                self.counts = np.ma.zeros((256, 128, 128), np.int32)
                self.pixel_eff = np.ma.zeros((128, 128), np.float64)
                self.low_threshold = np.ma.zeros((256, 128, 128), np.float32)
                self.valid = np.zeros((128, 128), np.float32)
                self.duration = 0.0
                self.mdu_eff = np.zeros((8,), np.float64)
                self.deadc = np.zeros((8,), np.float64)
                self.default_bins()

    def rebin(self, e_min=0, e_max=np.inf):
        rc = Cube(osacube=True)
        if self.empty:
            setattr(rc, 'empty', True)
            return rc
        copy_attrs = ('scwid', 'duration', 'header_fields')
        for keyword in copy_attrs + self.header_fields:
            setattr(rc, keyword.lower(), getattr(self, keyword.lower()))
        e_idx = np.logical_and(rc.e_min >= e_min, rc.e_max <= e_max)
        rc.e_min, rc.e_max = rc.e_min[e_idx], rc.e_max[e_idx]
        rc.counts = np.ma.zeros((len(rc.e_min), 128, 128), np.int16)
        rc.efficiency = np.ma.zeros((len(rc.e_min), 128, 128), np.float32)
        for bin in range(len(rc.e_min)):
            bins = np.logical_and(self.e_min >= rc.e_min[bin],
                                  self.e_max <= rc.e_max[bin])
            rc.counts[bin] = np.sum(self.counts[bins], 0)
            rc.efficiency[bin] = np.sum(self.efficiency[bins], 0) / \
                                 np.count_nonzero(bins)
        return rc

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

    def rate_shadowgram(self, e_min=0, e_max=np.inf, sigma=None):
        """Shadowgram of count rate, optionally for a subset of the energy range

        unit: cts.s-1.keV-1
        """
        e_idx = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        norm = 1 / (self.duration * np.sum(self.bin_width[e_idx]))
        rate = (self.counts[e_idx] / \
                self.efficiency[e_idx]).sum(0) * norm
        if sigma is not None:
            sigma = np.sqrt(self.counts[e_idx].sum(0))
            return (rate, rate / sigma)
        else:
            return rate

    def cts_exp_shadowgram(self, e_min=0, e_max=np.inf, sigma=None):
        """Shadowgrams of counts and efficiency, optionally for a subset
        of the energy range
        units: cts, s
        """
        e_idx = np.logical_and(self.e_min >= e_min, self.e_max <= e_max)
        cts = self.counts[e_idx].sum(0)
        eff = (self.efficiency[e_idx] * \
               np.reshape(self.bin_width[e_idx],
                          (np.count_nonzero(e_idx), 1, 1))).sum(0) / \
                          self.bin_width[e_idx].sum()
        return cts, self.duration * eff

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
        dpi = np.ma.zeros((134, 130))
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

def osacubes_avail():
    """find available OSA cubes from the byscw direcory index file
    """
    dir_index = throng_dir + '/common/scwdb/byscw_find_type-d'
    oc_re = re.compile('.*OSACube.*prod.*')
    hash_re = re.compile('^[0-9a-f]{8}$')
    revs = {}
    with open(dir_index) as find:
        for line in find:
            dir = line.rstrip()
            fields = dir.split('/')
            if len(fields) <= 3: continue
            if not hash_re.match(fields[-1]): continue
            if oc_re.match(fields[2]):
                rev, scwid = fields[0:2]
                id = '/'.join(fields[2:-1])
                if not rev in revs:
                    revs[rev] = {}
                if not scwid in revs[rev]:
                    revs[rev][scwid] = {}
                revs[rev][scwid][id] = dir
    return revs

def osacubes(scwids):
    """Find OSA cubes matching the provided list of science window IDs
    """
    ids_wanted = np.array(scwids, dtype=np.uint64)
    revs = osacubes_avail()
    dirs_avail = []
    for rev in revs.values():
        dirs_avail.extend(rev.keys())
    ids_avail = np.array([id[0:12] for id in dirs_avail], dtype=np.uint64)
    ids = np.intersect1d(ids_wanted, ids_avail)

    def best_cube(id):
        ver_order = (
            'OSACube.prod1_hotfix',
            'OSACube.prod1_arf',
            'OSACube.prod1',
        )

        id2rev = np.array(100000000, dtype=np.uint64)
        rev = '{0:04d}'.format(id // id2rev)
        scw = '{0:012d}.001'.format(id)
        cubes = revs[rev][scw]
        for ver in ver_order:
            if ver in cubes.keys():
                return os.path.join(byscw_dir, cubes[ver])

    cubes = [best_cube(id) for id in ids]
    idx = np.where(cubes)[0]
    ids = ids_wanted[idx]
    return [cubes[i] for i in idx], ids

def cube2mod(cube_in):
    """split a shadow-gram into its constituent modules,
    rotating modules 4-7 by 180° to match 0-3"""
    try:
        return type(cube_in)(cube2mod(cube_in.data), cube2mod(cube_in.mask))
    except AttributeError:
        pass
    if not cube_in.shape[1:] == (128, 128):
        raise IndexError("Shape not as expected. Is this a cube?")
    mod_order = np.array([7, 6, 5, 4, 0, 1, 2, 3])
    cube = np.expand_dims(cube_in, 1)
    halves = np.split(cube, 2, 3)
    halves[0] = halves[0][:, :, ::-1, ::-1]
    half_stack = np.concatenate(halves, axis=2)
    modules = np.split(half_stack, 8, 2)
    mod_stack = np.concatenate(modules, axis=1)
    return mod_stack[:, mod_order]

def mod2pc(modules):
    """split a module stack into its constituent polycells"""
    try:
        return type(modules)(mod2pc(modules.data), mod2pc(modules.mask))
    except AttributeError:
        pass
    if not modules.shape[1:] == (8, 32, 64):
        raise IndexError("Shape not as expected. Is this a module stack?")
    pc_z = np.expand_dims(modules, 2)
    pc_z = np.concatenate(np.split(pc_z, 8, 3), 2)
    pc_y = np.expand_dims(pc_z, 3)
    return np.concatenate(np.split(pc_y, 16, 5), 3)

def pc2mod(pc):
    """concatenate a polycell stack into modules"""
    try:
        return type(pc)(pc2mod(pc.data), pc2mod(pc.mask))
    except AttributeError:
        pass
    if not pc.shape[1:] == (8, 8, 16, 4, 4):
        raise IndexError("Shape not as expected. Is this a polycell stack?")
    pc_y = np.concatenate(np.split(pc, 8, 2), 4)
    mod = np.concatenate(np.split(pc_y, 16, 3), 5)
    print(mod.shape)
    return np.squeeze(np.squeeze(mod, 3), 2)

def mod2cube(mod_stack):
    """concatenate a module stack into a cube"""
    try:
        return type(mod_stack)(mod2cube(mod.data), mod2cube(mod.mask))
    except AttributeError:
        pass
    if not mod_stack.shape[1:] == (8, 32, 64):
        raise IndexError("Shape not as expected. Is this a module stack?")
    mod_order = np.array([4, 5, 6, 7, 3, 2, 1, 0])
    half_stack = np.concatenate(np.split(mod_stack[:, mod_order], 8, 1), 2)
    halves = np.split(half_stack, 2, 2)
    halves[0] = halves[0][:, :, ::-1, ::-1]
    return np.squeeze(np.concatenate(halves, 3), 1)
