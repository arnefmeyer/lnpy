#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Some (hopefully) useful utilities
"""

from __future__ import division

import os
import numpy as np
from scipy import signal
from scipy.stats import wilcoxon

# Try to use scikits' samplerate method
use_scikits_resample = False
try:
    import scikits.samplerate as sks
    use_scikits_resample = True
    max_ratio = 256.
except:
    pass


def makedirs_save(f):
    """simple and save directory generation"""

    if not os.path.exists(f):
        try:
            os.makedirs(f)
        except:
            pass


def resample(x, fs_old, fs_new, axis=0, algorithm='scipy'):
    """Resample signal

    If available resampling is done using scikit samplerate. Otherwise,
    scipy's FFT-based resample function will be used.

    Converters available in scikits.samplerate:
    - sinc_medium
    - linear
    - sinc_fastest
    - zero_order_hold
    - sinc_best

    """

    if fs_old == fs_new:
        return x

    else:

        ratio = float(fs_new) / fs_old
        if use_scikits_resample and algorithm != 'scipy':

            if algorithm == 'scikits':
                algo = 'sinc_medium'
            else:
                algo = algorithm

            if axis == 0:
                tmp = sks.resample(x, ratio, algo)
            else:
                tmp = sks.resample(x.T, ratio, algo)

            if tmp.dtype != x.dtype:
                tmp = tmp.astype(x.dtype, casting='safe')

            return tmp

        else:
            if axis == 0:
                n_samples_new = int(np.round(x.shape[0] * ratio))
                return signal.resample(x, n_samples_new)
            else:
                n_samples_new = int(np.round(x.shape[1] * ratio))
                return signal.resample(x, n_samples_new, axis=axis)


def get_nice_color(name=None, normalize=True):
    """Return some nice colors (RGB values)"""

    cdict = {'white': 3 * [255],
             'black': 3 * [0],
             'blue': [0, 120, 255],
             'orange': [255, 110, 0],
             'green': [35, 140, 45],
             'red': [200, 30, 15],
             'violet': [220, 70, 220],
             'turquoise': [60, 134, 134],
             'gray': [130, 130, 130],
             'lightgray': 3 * [150],
             'darkgray': 3 * [100],
             'yellow': [255, 215, 0]
             }

    # Convert lists to numpy arrays for easier data handling
    for k in cdict.keys():
        cdict[k] = np.array(cdict[k])
        if normalize:
            cdict[k] = cdict[k].astype(np.float) / 255.

    if name is not None:
        if isinstance(name, str):
            return cdict[name.lower()]

        elif isinstance(name, list):
            colors = []
            for n in name:
                colors.append(cdict[n.lower()])
            return colors
    else:
        return cdict


def set_parameters_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                        size_text=8, size_title=8):
    """helper to set parameters of matplotlib axes"""

    ax.title.set_fontsize(size_title + add_size)
    ax.tick_params(axis='both', which='major', labelsize=size_ticks + add_size)
    ax.xaxis.label.set_fontsize(size_labels + add_size)
    ax.xaxis.label.set_fontname('Arial')
    ax.yaxis.label.set_fontsize(size_labels + add_size)
    ax.xaxis.label.set_fontname('Arial')
    for at in ax.texts:
        at.set_fontsize(size_text + add_size)
        at.set_fontname('Arial')


def plot_scatter(ax, x, y, xerr, yerr, xlabel='', ylabel='',
                 calc_wilcoxon=True, color=3*[.5], ecolor=3*[.75],
                 fmt='o', **kwargs):
    """create scatter plot with equally-spaced axes, diagonal line etc."""

    ax.axis('on')
    ax.axis('scaled')
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt,
                color=color, ecolor=ecolor, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    max_err = max([xerr.max(), yerr.max()])
    xmin = min([x.min(), y.min()]) - max_err
    xmax = max([x.max(), y.max()]) + max_err
    zz = np.linspace(xmin, xmax, 100)
    ax.plot(zz, zz, 'k--')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    if calc_wilcoxon:
        _, p_value = wilcoxon(x, y)
        ax.text(.05, .8, 'p = %.3e' % p_value,
                transform=ax.transAxes)
