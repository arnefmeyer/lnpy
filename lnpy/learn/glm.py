#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Generalized linear models
"""

import numpy as np
from sklearn.linear_model import Ridge as _Ridge
from scipy.linalg import solve

from .base import LinearBaseEstimator
from . import GaussianPrior, SparseProblem, DenseProblem
from .pyhelper import PyBernoulliGLM, PyPoissonGLM, PyGaussianGLM
from .grid_search import (
    scorer_BernoulliLL, scorer_PoissonLL, scorer_ModPoissonLL, scorer_r_squared
)


def create_smooth_prior(wsize):

    p = np.prod(wsize)
    S = np.zeros((p, p))

    x_ind, y_ind = np.unravel_index(np.arange(p), wsize)
    for i in range(p):

        Nn = 0
        for j in range(p):
            dx = np.abs(x_ind[i] - x_ind[j])
            dy = np.abs(y_ind[i] - y_ind[j])
            if dx == 1 and dy == 0:
                S[i, j] = -1
                Nn += 1

            if dy == 1 and dx == 0:
                S[i, j] = -1
                Nn += 1

        S[i, i] = Nn

    return S


def _ridge(X, y, alpha, gamma, mu, D, wsize):

    stim_cov = np.dot(X.T, X)
    sta = np.dot(X.T, y)

    p = X.shape[1]
    if alpha > 0:
        # Ridge prior
        stim_cov += alpha * np.eye(p)

    if mu is not None:
        # Non-zero mean of ridge Gaussian
        sta += alpha * mu

    if gamma > 0 and (D is not None or wsize is not None):
        if D is None and wsize is not None:
            D = create_smooth_prior(wsize)
        stim_cov += gamma * D

    w = solve(stim_cov, sta, sym_pos=False, overwrite_a=False)

    return w


class Ridge(LinearBaseEstimator):
    """Linear-Gaussian GLM with Gaussian prior on weight parameters"""

    def __init__(self, alpha=1., gamma=0., mu=None, D=None, wsize=None,
                 **kwargs):

        super(Ridge, self).__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu
        self.D = D
        self.wsize = wsize

    def fit(self, X, y):

        optimize = self.optimize

        if optimize:
            print "lnpy.glm fit optimize"
            if self.grid_parameters is None:
                self.grid_parameters = dict()

            grid_params = self.grid_parameters
            D = self.D
            wsize = self.wsize

            if 'param_grid' not in grid_params.keys():
                param_grid = dict(alpha=2. ** np.linspace(-10, 10, 7))
                param_info = dict(alpha={'scaling': 'log2'})

                if D is not None or wsize is not None and \
                        'gamma' not in param_grid.keys():
                    dd = dict(gamma=2. ** np.linspace(-10, 10, 7))
                    param_grid.update(dd)
                    param_info.update({'gamma': {'scaling': 'log2'}})

                grid_params.update({'param_grid': param_grid,
                                    'param_info': param_info})

            if 'scorer' not in grid_params.keys():
                grid_params.update({'scorer': self.default_scorer})

            self._fit_optimize(X, y, stratify_folds=False)

        else:
            self._fit(X, y)

    def _fit(self, X, y):

        bias = self.bias_multiplier
        alpha = self.alpha
        gamma = self.gamma
        mu = self.mu
        D = self.D
        wsize = self.wsize

        if gamma > 0 and D is None and wsize is None:
            raise ValueError('Either custom prior or size of w must be given!')

        fit_bias = bias > 0
        if fit_bias:
            X, y, X_mean, y_mean = self._center_data(X, y)

        self.coef_ = _ridge(X, y, alpha, gamma, mu, D, wsize)

        if fit_bias:
            self._set_intercept(X_mean, y_mean)

    @property
    def default_scorer(self):
        return scorer_r_squared

    def __setattr__(self, name, value):
        """Catch prior hyperparameter attributes"""

        self.__dict__[name] = value


class GaussianGLM(LinearBaseEstimator):
    """Linear-Gaussian GLM with canonical link function

    Parameters
    ----------
    prior : Prior
        Prior used for inference of parameters (default: GaussianPrior)

    tolerance : float
        Solver tolerance

    optimizer : str
        Optimizer used to find the model parameters. 'tron' calls the TRON
        solver C implementation and allows for a range of different priors.
        'ridge' uses the closed-form solution with zero-mean Gaussian prior.
        (default: to 'tron').

    alpha, gamma : float
        Prior hyperparameters
    """

    def __init__(self, prior=GaussianPrior(), tolerance=.001,
                 optimizer='tron', alpha=None, gamma=None, **kwargs):

        self.optimizer = optimizer
        self.prior = prior
        self.tolerance = tolerance

        super(GaussianGLM, self).__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, y):

        if self.optimize:
            grid_params = self.grid_parameters
            prior = self.prior
            if prior is not None and 'param_grid' not in grid_params.keys():
                param_grid, param_info = prior.get_default_grid()
                grid_params.update({'param_grid': param_grid,
                                    'param_info': param_info})

            if 'scorer' not in grid_params.keys():
                grid_params.update({'scorer': self.default_scorer})

            self._fit_optimize(X, y, stratify_folds=False)

        else:
            self._fit(X, y)

    def _fit(self, X, y, is_sparse=True):

        bias = self.bias_multiplier
        optim = self.optimizer.lower()

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        fit_bias = bias > 0
        if optim is None or optim == 'tron':

            N, p = X.shape
            prob = SparseProblem(X, y.ravel(), bias=bias, C=None,
                                 binary_targets=False)

            w = np.zeros((p+int(bias > 0),))
            model = PyGaussianGLM(self.prior)
            model.fit(prob, w, verbose=0, tolerance=self.tolerance)

            self._set_coef_intercept_from_w(w)

        elif optim == 'ridge':
            alpha = self.alpha
            model = _Ridge(alpha=alpha, fit_intercept=fit_bias, tol=0.001,
                           normalize=False, copy_X=True, max_iter=None,
                           solver='dense_cholesky')
            model.fit(X, y)
            self.coef_ = np.copy(model.coef_)

            if fit_bias:
                self.intercept_ = model.intercept_
            else:
                self.intercept_ = 0.

    @property
    def default_scorer(self):
        return scorer_r_squared

    def __setattr__(self, name, value):
        """Catch prior hyperparameter attributes"""

        if name == 'optimizer':
            self.__dict__[name] = value

        else:
            optim = self.optimizer.lower()
            if optim in [None, 'tron'] and name in ['alpha', 'gamma']:
                if hasattr(self, 'prior') and self.prior is not None \
                        and hasattr(self.prior, name) and value is not None:
                    setattr(self.prior, name, value)
                else:
                    self.__dict__[name] = value
            else:
                self.__dict__[name] = value


class BernoulliGLM(LinearBaseEstimator):
    """Bernoulli GLM with canonical link function

    Parameters
    ----------
    prior : Prior
        Prior used for inference of parameters (default: GaussianPrior)

    tolerance : float
        Solver tolerance

    alpha, gamma : float
        Prior hyperparameters

    """

    def __init__(self, prior=GaussianPrior(), tolerance=.1,
                 alpha=None, gamma=None, **kwargs):

        super(BernoulliGLM, self).__init__(**kwargs)

        self.prior = prior
        self.tolerance = tolerance

    def fit(self, X, y):
        if self.optimize:
            self._fit_optimize(X, y, stratify_folds=False)
        else:
            self._fit(X, y)

    def _fit(self, X, y, is_sparse=True):


        X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
        y = np.require(y, dtype=np.float64, requirements=['C', 'A'])

        bias = self.bias_multiplier
        N, p = X.shape
        if is_sparse:
            prob = SparseProblem(X, y.ravel(), bias, C=None,
                                 binary_targets=True)
        else:
            prob = DenseProblem(X, y.ravel(), bias, C=None,
                                binary_targets=True)

        w = np.zeros((p+int(bias > 0),))
        model = PyBernoulliGLM(self.prior)
        model.fit(prob, w, tolerance=self.tolerance)

        self._set_coef_intercept_from_w(w)

    @property
    def default_scorer(self):
        return scorer_BernoulliLL


class PoissonGLM(LinearBaseEstimator):
    """Poisson GLM with canonical or modified link function

    Parameters
    ----------
    prior : PyPrior
        Prior used for inference of parameters (default: GaussianPrior)

    canonical : boolean
        Use canonical or modified (cf. Calabrese et al. PLOS ONE 2011)
        link function?

    """

    def __init__(self, prior=GaussianPrior(), canonical=True,
                 tolerance=.1, alpha=1., gamma=1., **kwargs):

        super(PoissonGLM, self).__init__(**kwargs)

        self.prior = prior
        self.canonical = canonical
        self.tolerance = tolerance
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, y, is_sparse=True):

        if self.optimize:
            self._fit_optimize(X, y)
        else:
            self._fit(X, y)

    def _fit(self, X, y, is_sparse=True):

        bias = self.bias_multiplier
        N, p = X.shape

        if is_sparse:
            prob = SparseProblem(X, y.ravel(), bias, None,
                                 binary_targets=False)
        else:
            prob = DenseProblem(X, y.ravel(), bias, None,
                                binary_targets=False)

        w = np.zeros((p+int(bias > 0),))
        model = PyPoissonGLM(self.prior, canonical=self.canonical)
        model.fit(prob, w, verbose=0, tolerance=self.tolerance)

        self._set_coef_intercept_from_w(w)

    @property
    def default_scorer(self):
        if self.canonical:
            return scorer_PoissonLL
        else:
            return scorer_ModPoissonLL
