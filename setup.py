#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

import sys
import os
import shutil
from distutils.command.clean import clean as _clean


DESCRIPTION = 'A python module for stimulus-response function estimation'

fpath = os.path.split(__file__)[0]
with open(os.path.join(fpath, 'README')) as f:
    LONG_DESCRIPTION = f.read()


# Custom clean command to remove build artifacts
class CleanCommand(_clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        _clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('lnpy'):
            for filename in filenames:
                if (filename.endswith('.so') or filename.endswith('.pyd')
                        or filename.endswith('.dll')
                        or filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))

cmdclass = {'clean': CleanCommand}


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('lnpy')

    return config


def setup_package():
    metadata = dict(name='lnpy',
                    maintainer='Arne F. Meyer',
                    maintainer_email='arne.f.meyer@gmail.com',
                    description=DESCRIPTION,
                    license='GPLv3',
                    url='http://www.github.com/arnefmeyer/lnpy',
                    version='0.1',
                    download_url='http://www.github.com/arnefmeyer/lnpy',
                    long_description=LONG_DESCRIPTION,
                    classifiers=['Intended Audience :: Science/Research',
                                 'License :: GLMv3',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Topic :: Scientific/Neuroscience',
                                 'Operating System :: Microsoft :: Windows',
                                 'Operating System :: POSIX',
                                 'Operating System :: Unix',
                                 'Operating System :: MacOS',
                                 'Programming Language :: Python :: 2.7',
                                 ],
                    cmdclass=cmdclass)

    if (len(sys.argv) >= 2
            and ('--help' in sys.argv[1:] or sys.argv[1]
                 in ('--help-commands', 'egg_info', '--version', 'clean'))):

        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup

        metadata['version'] = 0.1
    else:
        from numpy.distutils.core import setup

        metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
