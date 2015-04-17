#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    A base class for all receptive field estimation methods
"""

from __future__ import division

import numpy as np
from sklearn.base import BaseEstimator as SKBaseEstimator
import abc

from ..base import Spectrogram


class LNPEstimator(SKBaseEstimator):
    """Base class for receptive field estimation methods

    Parameters
    ----------
    fit_bias : boolean
        Fit intercept?

    n_griditer : int
        Number of grid points used during (cross-validate-based) search
        for model hyperparameters

    n_jobs : int
        The number of workers used during hyperparameter grid search

    verbose : boolean
        Guess what

    param_grid : dict
        Dictionary with hyperparameter names (keys) and grid values (values).
        Example:
            param_grid = {'alpha': 2 ** np.linspace(-10, 5, 10)}

    param_info : dict
        Parameter information; currently only scaling is supported but may
        also be used to bound parameters. Example:
            param_info = {'alpha': {'scaling': 'log2'}}

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, fit_bias=True, n_spikefilt=0, n_griditer=5, n_jobs=1,
                 verbose=True, param_grid=None, param_info=None):
        super(LNPEstimator, self).__init__()

        self.fit_bias = fit_bias
        self.n_spikefilt = n_spikefilt
        self.grid_params = dict(n_griditer=n_griditer, n_jobs=n_jobs,
                                verbose=verbose, param_grid=param_grid,
                                param_info=param_info)

        self.coef_ = None
        self.intercept_ = 0.
        self.spikefilt_ = None

        self.t_fit = 0

    @abc.abstractmethod
    def fit(self, X, Y):
        return

    def predict(self, X):

        w = self.get_weights()
        b = self.get_bias()

        return np.dot(X, w) + b

    @property
    def name(self):
        return self.__class__.__name__

    def to_string(self):
        """Return string representation of method (defaults to name)"""

        return self.name

    def get_weights(self):
        return self.coef_

    def get_bias(self):
        return self.intercept_

    def get_spikefilt(self):
        return self.spikefilt_

    def _split_coef_spikefilt(self, w):
        n_spikefilt = self.n_spikefilt
        if n_spikefilt > 0:
            self.coef_ = w[:-n_spikefilt]
            self.spikefilt_ = w[-n_spikefilt:]
        else:
            self.coef_ = w
            self.spikefilt_ = None


class STRF(Spectrogram):
    """Convenience class to wrap Specrogram as STRF"""

    def __init__(self, *args, **kwargs):
        super(STRF, self).__init__(*args, **kwargs)

    def show(self, **kwargs):
        vmax = np.max(np.abs(self.data))
        return super(STRF, self).show(vmax=vmax, vmin=-vmax, **kwargs)

    @property
    def intercept(self):
        if 'intercept' in self.annotations:
            return self.annotations['intercept']
        else:
            return 0.

    @intercept.setter
    def intercept(self, x):
        if 'intercept' in self.annotations:
            self.annotations['intercept'] = x
        else:
            self.annotations.update(dict(intercept=x))
