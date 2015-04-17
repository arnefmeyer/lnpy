#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

import os
from os.path import join, exists

import numpy


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('transform', parent_package, top_path)

    basedir = join(top_path, 'lnpy', 'transform')
    c_file = join(basedir, 'src', 'gammatone', 'Gfb_analyze.c')
    h_file = join(basedir, 'src', 'gammatone', 'Gfb_analyze.h')

    if exists(c_file) and exists(h_file):

        libraries = []
        if os.name == 'posix':
            libraries.append('m')

        # We have to include stdlib.h; otherwise, it might not compile
        with file(c_file, 'r') as f:
            lines = f.read()

        if '#include <stdlib.h>' not in lines:
            print 50 * '-'
            print "Adding #include <stdlib.h> to file"
            print c_file
            with file(c_file, 'w') as f:
                f.write("#include <stdlib.h>\n" + lines)
            print 50 * '-'

        # Further, we have to rename gfb_analyze in the header file
        with file(h_file, 'r') as f:
            lines = f.read()

        if 'void gfb_analyze(' in lines:
            print 50 * '-'
            print "Renaming function 'gfb_analyze' to 'analyze' in"
            print h_file
            lines = lines.replace('void gfb_analyze(', 'void analyze(')
            with file(h_file, 'w') as f:
                f.write(lines)
            print 50 * '-'

        sources = ['wrap_gtfb.c',
                   join('src', 'gammatone', 'Gfb_analyze.c')]
        includes = [join('src', 'gammatone'), numpy.get_include()]

        config.add_extension('wrap_gtfb', sources=sources,
                             libraries=libraries,
                             include_dirs=includes)

    else:
        descr = """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Couldn't find gammatone filterbank code. If you wan't to
compile the package with the gammatone filterbank please download the
code from

    http://medi.uni-oldenburg.de/download/demo/gammatone-filterbank/gammatone_filterbank-1.1.zip

and extract the files Gfb_analyze.c and Gfb_analyze.h into the directory

    lnpy/transform/src/gammatone

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

        print descr

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
