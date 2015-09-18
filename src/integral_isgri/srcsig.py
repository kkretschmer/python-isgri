"""Accesses INTEGRAL/ISGRI ii_skyimage source lists
"""

import glob
import os
import re

import numpy as np
try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

try:
    throng_dir = os.environ['THRONG_DIR']
except KeyError:
    throng_dir = '/Integral/throng'

dir_index = throng_dir + '/common/scwdb/byscw_find_type-d'
byscw_dir = '/Integral/data/reduced/ddcache/byscw'

def ii_skyimage_avail():
    """find available ii_skyimage source lists from the byscw direcory index file
    """
    iis_re = re.compile('ii_skyimage.v2')
    hash_re = re.compile('^[0-9a-f]{8}$')
    revs = {}
    with open(dir_index) as find:
        for line in find:
            dir = line.rstrip()
            fields = dir.split('/')
            if len(fields) <= 3: continue
            if not hash_re.match(fields[-1]): continue
            if iis_re.match(fields[2]):
                rev, scwid = fields[0:2]
                task = '/'.join(fields[2:-1])
                if not rev in revs:
                    revs[rev] = {}
                if not scwid in revs[rev]:
                    revs[rev][scwid] = {}
                if task not in revs[rev][scwid]:
                    revs[rev][scwid][task] = []
                revs[rev][scwid][task].append(dir)
    return revs

def isgri_sky_res(scwids):
    """Find isgri_sky_res files matching the provided list of science window IDs
    """
    ids_wanted = np.array(scwids, dtype=np.uint64)
    revs = ii_skyimage_avail()
    dirs_avail = []
    for rev in revs.values():
        dirs_avail.extend(rev.keys())
    ids_avail = np.array([scw[0:12] for scw in dirs_avail], dtype=np.uint64)
    ids = np.intersect1d(ids_wanted, ids_avail)

    result = {}
    id2rev = np.uint64(100000000)
    key = 'ii_skyimage.v2'
    sky_res_glob = 'isgri_sky_res.fits*'
    for i in ids:
        rev = '{0:04d}'.format(i // id2rev)
        scw = '{0:012d}.001'.format(i)
        if key in revs[rev][scw]:
            for run in revs[rev][scw][key]:
                for g in glob.glob(os.path.join(byscw_dir, run, sky_res_glob)):
                    isr = fits.open(g)
                    grp = isr['GROUPING'].data
                    idx = np.logical_and(grp['E_MIN'] == 25,
                                         grp['E_MAX'] == 80)
                    if np.count_nonzero(idx) > 0:
                        ext = grp['MEMBER_POSITION'][idx][0] - 1
                        detsig = isr[ext].data['DETSIG']
                        rms = np.sqrt((detsig**2).sum())
                        sigs = ','.join(['{0:0.1f}'.format(s) for s in detsig])
                        fmt = '{0:012d}\t{1:0.1f}\t{2}'
                        print(fmt.format(i, rms, sigs))
                    isr.close()
