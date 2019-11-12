#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Some helpers for context model parameter analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ..plotting import set_font_axes


def plot_avg_and_profiles_CGF(W_cgfs, shape,
                              dt=0.02,
                              colorbar=True,
                              cmap='RdBu_r',
                              **cb_args):

    fig = plt.figure(figsize=(8, 2.5))

    M = shape[0] - 1
    N = int((shape[1] - 1) / 2)
    extent = (-M*dt*1000, 0, -N - .5, N + .5)

    # Average CGF
    ax = fig.add_subplot(1, 2, 1)

    w_avg = np.reshape(np.mean(W_cgfs, axis=0), shape)
    ax.set_title('Average CGF')
    vmax = np.max(np.abs(w_avg))
    veps = max(1e-12 * vmax, 1e-12)
    im = ax.imshow(w_avg[::-1, :].T,
                   vmin=-vmax - veps,
                   vmax=vmax + veps,
                   cmap=cmap,
                   extent=extent,
                   aspect='auto',
                   interpolation='nearest')
    ax.set_xlabel('Temporal shift (ms)')
    ax.set_ylabel('Frequency shift')

    if colorbar:
        cb = plt.colorbar(im,
                          ax=ax,
                          **cb_args)

    # Temporal profile at zero frequency shift
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Temporal profile')
    tt = np.linspace(-M*dt*1000, 0, shape[0])
    tp_all = W_cgfs[:, N::shape[1]]
    tp_avg = w_avg[::-1, N]
    w_std = np.reshape(np.std(W_cgfs,
                              axis=0) / np.sqrt(W_cgfs.shape[0]), shape)
    tp_std = w_std[::-1, N]

    ax.plot(tt, tp_all[:, ::-1].T, '-',
            color=3*[.8],
            linewidth=1,
            zorder=0)
    ax.plot(tt, tp_avg, '-',
            color=[200/255., 30/255., 15/255.],
            linewidth=2)
    ax.fill_between(tt, tp_avg - tp_std, tp_avg + tp_std,
                    facecolor=3*[.25],
                    edgecolor=[200/255., 30/255., 15/255.],
                    alpha=.5)

    for ax in fig.axes:
        set_font_axes(ax)

    fig.tight_layout()

    if colorbar:
        return fig, cb
    else:
        return fig


def plot_PCs_CGF(W_cgfs, shape, dt=0.02, n_PC=3, colorbar=True, cmap='RdBu_r'):

    fig = plt.figure(figsize=(8, 2.5))

    M = shape[0] - 1
    N = (shape[1] - 1) / 2
    extent = (-M*dt*1000, 0, -N - .5, N + .5)

    # PCA analysis
    pca = PCA(n_components=None, copy=True, whiten=False)
    pca.fit(W_cgfs)
    pcs = pca.components_

    vmax = np.max(np.abs(pcs))
    veps = max(1e-12 * vmax, 1e-12)
    ax1 = None
    for i in range(n_PC):
        ax = fig.add_subplot(1, 1+n_PC, 1+i, sharex=ax1, sharey=ax1)
        ax.set_title('PC %d' % (i+1))
        W = np.reshape(pcs[i, :], shape)
        ax.imshow(W[::-1, :].T, vmin=-vmax - veps, vmax=vmax + veps,
                  cmap=cmap, extent=extent, aspect='auto',
                  interpolation='nearest')

        if ax1 is None:
            ax1 = ax
        else:
            ax.set_xlabel('Time shift (ms)')
            ax.set_ylabel('Frequency shift')

    ax = fig.add_subplot(1, 1+n_PC, 1+n_PC)
    ax.plot(1+np.arange(pca.explained_variance_ratio_.shape[0]),
            pca.explained_variance_ratio_, 'o', color=3*[.5])

    ax.set_xlabel('PC')
    ax.set_ylabel('% variance')
    ax.set_xscale('log')

    for ax in fig.axes:
        set_font_axes(ax)

    fig.tight_layout()

    return fig
