#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

import os
from os.path import join
import numpy


def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('learn', parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    sources = ['pyhelper.cpp',
               join('src', 'code', 'wrap_tron.cpp'),
               join('src', 'code', 'tron.cpp'),
               join('src', 'code', 'helper.cpp'),
               join('src', 'code', 'sgd.cpp'),
               join('src', 'blas', 'ddot.c'),
               join('src', 'blas', 'dnrm2.c'),
               join('src', 'blas', 'daxpy.c'),
               join('src', 'blas', 'dscal.c')]
    includes = [numpy.get_include()]

    compile_args = ['-O3']

    config.add_extension('pyhelper', sources=sources,
                         libraries=libraries,
                         include_dirs=includes,
                         extra_compile_args=compile_args,
                         language='c++')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
