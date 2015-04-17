#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from numpy.distutils.misc_util import Configuration
import os


def configuration(parent_package='', top_path=None):

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('lnpy', parent_package, top_path)

    config.add_subpackage('transform')
    config.add_subpackage('lnp')
    config.add_subpackage('learn')
    config.add_subpackage('io')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
