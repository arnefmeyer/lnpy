#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3


"""
    Linear estimators for stimulus-response function estimation

    Note that while there are different linear estimators in the lnp
    subpackage (which find prior hyperparameters using cross-validation)
    the estimators in this subpackage use different schemes, e.g.,
    evidence optimization (see Sahani & Linden NIPS 2003 for an example).
"""

import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

from sklearn.linear_model.base import LinearModel as SKLinearModel


def plot_linear_model(w, shape, dt=0.02, cmap='RdBu_r', timeticks=None,
                      frequencies=None,
                      freqticks=[2000, 4000, 8000, 16000, 32000],
                      windowtitle=None,
                      show_now=False, ax=None, order='C', name=None,
                      colorbar=False, logfscale=True):

    J, K = shape
    W = np.reshape(w, shape, order='C')
    fc = frequencies

    init_fig = False
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        init_fig = True

    else:
        fig = ax.get_figure()

    plt_args = dict(interpolation='nearest', aspect='auto', origin='lower')

    if fc is not None:

        fc = np.asarray(fc) / 1000.
        f_unit = 'kHz'

        if freqticks is not None:
            freqticks = np.asarray(freqticks) / 1000.

        # set ticks
        if logfscale:
            f_extent = [.5, fc.shape[0] + .5]
        else:
            f_extent = [fc.min(), fc.max()]

    else:
        f_extent = [.5, W.shape[1] + .5]
        f_unit = 'channels'

    vmax = np.max(np.abs(w))
    veps = max(1e-3 * vmax, 1e-12)
    extent = (-J*dt*1000, 0, f_extent[0], f_extent[1])
    im = ax.imshow(W.T, vmin=-vmax - veps, vmax=vmax + veps, cmap=cmap,
                   extent=extent, **plt_args)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (%s)' % f_unit)

    if name is not None:
        ax.set_title(name)
    else:
        ax.set_title('STRF')

    # Set time ticks for STRF and PRF
    if timeticks is not None:
        ax.set_xticks(timeticks)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(4))

    if fc is not None:

        if logfscale:
            invfunc = interp1d(fc, np.arange(0, fc.shape[0]),
                               kind='cubic')
            freqticks = freqticks[np.logical_and(freqticks >= np.amin(fc),
                                                 freqticks <= np.amax(fc))]
            fc_loc = invfunc(freqticks)
            ax.yaxis.set_ticks(fc_loc)

        else:
            ax.yaxis.set_ticks(freqticks)

        ax.yaxis.set_ticklabels(freqticks)

    if colorbar:
        cb = plt.colorbar(im, ax=ax)
        cb.locator = plt.MaxNLocator(5)
        cb.update_ticks()

    if init_fig:
        if windowtitle is not None:
            fig.canvas.set_window_title(windowtitle)

        fig.set_size_inches(3, 3)
        fig.tight_layout()

    if show_now:
        plt.show()

    return fig


class LinearModel(SKLinearModel):

    __meta__ = ABCMeta

    def __init__(self, normalize=False, fit_intercept=True, w_shape=None):

        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.w_shape = w_shape

    @abstractmethod
    def fit(self, X, y):
        """Fit model parameters"""

    def predict(self, X):

        z = np.dot(X, self.coef_) + self.intercept_

        return z

    def show(self, shape=None, dt=0.02, cmap='RdBu_r', show_now=True,
             **kwargs):

        if shape is None:
            shape = self.w_shape

        assert shape is not None, "Shape of weights must be provided"

        fig = plot_linear_model(self.coef_, shape, dt=dt, cmap=cmap,
                                show_now=show_now, name=self.get_name(),
                                **kwargs)

        return fig

    def get_weights(self):

        return self.coef_

    def get_intercept(self):

        return self.intercept_

    def get_name(self):

        return self.__class__.__name__


def create_example_data(shape=(15, 15), N=10000, noise_var=.1):
    """create an example data set using a linear-Gaussian model"""

    pass
