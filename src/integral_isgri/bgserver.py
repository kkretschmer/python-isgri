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
Serve ISGRI background cubes via HTTP.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from future import standard_library
standard_library.install_aliases()

import argparse
import io
import logging
import socketserver
import http.server

try:
    from astropy.io import fits
except ImportError:
    import pyfits as fits

from .bgcube import BGCube, BGCubeSet
from .bglincomb import BGLinComb

class CubeHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """Handle HTTP requests for ISGRI background cubes
    """
    def do_GET(self):
        """Serve a GET request."""
        f = self.send_head()
        if f:
            try:
                self.wfile.write(f.getvalue())
            finally:
                f.close()

    def do_HEAD(self):
        """Serve a HEAD request."""
        f = self.send_head()
        if f:
            f.close()

    def send_head(self):
        """Common code for GET and HEAD commands.

        This sends the response code and MIME headers.

        Return value is a BytesIO object containing the output FITS data.
        """
        components = self.path.split('/')
        if len(components) == 4:
            indexing = components[2]
            if indexing == 'ijd':
                method, ijd = components[1], float(components[3])
                if method in self.server.fromijd.keys():
                    bc = self.server.fromijd[method](ijd)
                else:
                    self.send_error(
                        404, "Generating method '{}' not found.".format(method))
                    return None
            else:
                self.send_error(
                    404, "Indexing method '{}' not found.".format(indexing))
                return None
        else:
            self.send_error(
                404, "URI format not supported.")
            return None

        blob = io.BytesIO()
        bc.writeto(blob, template=self.server.template)

        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(blob.getvalue())))
        self.end_headers()
        return blob

class ForkingHTTPServer(socketserver.ForkingMixIn,
                        http.server.HTTPServer):
    pass

def serve_cubes():
    parser = argparse.ArgumentParser(
        description="""Read a template for an ISGRI background model
        composed of a linear combination of background cubes, each scaled
        by linear interpolation of a light curve over time. Interpolate
        it in time and write it to a background cube."""
    )
    parser.add_argument('inputs', nargs='+',
                        help='input FITS file(s)')
    parser.add_argument('-p', '--http-port', type=int, default=8000,
                        help='port to listen for HTTP requests')
    parser.add_argument('-t', '--template', help='template FITS file')
    parser.add_argument('-l', '--outlier-map',
                        help='FITS file with outlier counts per pixel')
    parser.add_argument('-c', '--max-outlier-count', type=int, default=0,
                        help='maximum allowed outlier count')
    parser.add_argument('-e', '--mask-module-edges', type=int, default=0,
                        metavar='PIXELS',
                        help='number of pixels to mask around module edges')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args()

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)

    server_address = ('', args.http_port)
    httpd = ForkingHTTPServer(server_address, CubeHTTPRequestHandler)

    httpd.fromijd = {}
    stacks = []

    httpd.fromijd['zero'] = lambda ijd: BGCube()
    for infile in args.inputs:
        signature = [t[1] for t in fits.info(infile, output=False)]
        if signature[1:5] == ['TIME', 'ENERGY', 'TRACERS', 'CUBES']:
            # bglincomb template
            #
            blc = BGLinComb(file=infile)
            httpd.fromijd['lincomb'] = blc.bgcube
        elif signature[1:3] == ['COUNTS', 'EXPOSURE']:
            # cube stack
            #
            stacks.append(BGCube.fromstack(infile))

    if len(stacks) > 0:
        bcs = BGCubeSet(stacks)
        httpd.fromijd['nearest'] = bcs.nearest
        httpd.fromijd['linear'] = bcs.linear
    httpd.template = fits.open(args.template, memmap=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
