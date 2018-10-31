#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from numpy.distutils.misc_util import Configuration
import os
import numpy


def configuration(parent_package='', top_path=None):

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('lnpy', parent_package, top_path)

    config.add_subpackage('transform')
    config.add_subpackage('lnp')
    config.add_subpackage('linear')
    config.add_subpackage('multilinear')
    config.add_subpackage('learn')
    config.add_subpackage('io')

    # cython file with fast methods
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    sources = ['fast_tools.cpp']
    includes = [numpy.get_include()]

    compile_args = ['-O3']

    config.add_extension('fast_tools', sources=sources,
                         libraries=libraries,
                         include_dirs=includes,
                         extra_compile_args=compile_args,
                         language='c++')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
