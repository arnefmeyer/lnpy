#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    some helpers for make nice figures
"""

import numpy as np
from scipy import stats


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
        _, p_value = stats.wilcoxon(x, y)
        ax.text(.05, .8, 'p = %.3e' % p_value,
                transform=ax.transAxes)
