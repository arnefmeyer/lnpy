#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

import numpy as np
from ..base import Signal


class BaseTransform(object):
    """Base class for all (time-frequency) transformation classes"""

    def __init__(self):
        pass

    def to_string(self):
        raise NotImplementedError('You should have implemented this!')

    def process(self):
        raise NotImplementedError('You should have implemented this!')

    def get_center_frequencies(self):
        raise NotImplementedError('You should have implemented this!')

    def get_samplerate(self):
        raise NotImplementedError('You should have implemented this!')

    def get_name(self):
        return self.__class__.__name__

    def _parse_arguments(self, arg):
        """get parameters from input argument"""

        if isinstance(arg, np.ndarray):
            x = arg
            fs = None
            t0 = 0

        elif isinstance(arg, Signal):
            x = arg.get_data()
            fs = arg.get_samplerate()
            t = arg.get_time()
            if t is not None:
                t0 = t[0]
            else:
                t0 = 0.

        return x, fs, t0

    def _convert_spectrum_type(self, x, spectype):
        """convert potentially complex spectrum to real part etc."""

        if spectype.lower() == 'complex':
            x = x.astype(np.complex)

        elif spectype.lower() in ['phase', 'angle']:
            x = np.angle(x)

        elif spectype.lower() in ['re', 'real']:
            x = np.real(x)

        elif spectype.lower() in ['im', 'imag']:
            x = np.imag(x)

        elif spectype.lower() in ['abs', 'magnitude', 'envelope']:
            x = np.abs(x)

        else:
            raise ValueError("Unknown spectrum representation:",
                             spectype)

        return x

    def _get_spec_dtype(self, spectype):

        if spectype.lower() == 'complex':
            return np.complex

        elif spectype.lower() in ['phase', 'angle']:
            return np.float

        elif spectype.lower() in ['re', 'real']:
            return np.float

        elif spectype.lower() in ['im', 'imag']:
            return np.float

        elif spectype.lower() in ['abs', 'magnitude', 'envelope']:
            return np.float

        else:
            raise ValueError("Unknown spectrum representation:",
                             spectype)
