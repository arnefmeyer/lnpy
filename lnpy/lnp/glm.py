#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Generalized linear models (GLMs) for receptive field estimation
"""

from __future__ import division

import numpy as np
import time

from base import LNPEstimator
from ..learn.grid_search import ParamSearchCV
from ..learn import LogLoss
from ..learn import GaussianPrior
from ..learn.glm import BernoulliGLM as _BernoulliGLM
from ..learn.glm import PoissonGLM as _PoissonGLM
from ..learn.glm import GaussianGLM as _GaussianGLM
from ..learn.sgd import SGD, ASGD

from sklearn.base import BaseEstimator as SKBaseEstimator


class BernoulliGLM(LNPEstimator):
    """Wrapper around the Bernoulli GLM in the learn module

    For a description of the arguments see lnpy.lnp.cbrf.CbRF

    """

    def __init__(self, optimize=True, metric='BernoulliLL', tolerance=0.1,
                 prior=GaussianPrior(), alpha=1., gamma=0., **kwargs):
        super(BernoulliGLM, self).__init__(**kwargs)

        self.optimize = optimize
        self.tolerance = tolerance
        self.metric = metric
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, Y):
        """Train model and extract parameters"""

        prior = self.prior
        n_spikefilt = self.n_spikefilt

        if n_spikefilt > 0:
            nx = X.shape[1]
            nw = nx - n_spikefilt
            prior.set_ndim(nw)

        model = _BernoulliGLM(prior=prior, verbose=0,
                              tolerance=self.tolerance,
                              bias_multiplier=1., alpha=self.alpha,
                              gamma=self.gamma)

        if self.optimize and prior is not None:

            grid_params = self.grid_params
            if 'param_grid' not in grid_params.keys() or \
                    grid_params['param_grid'] is None:
                param_grid, param_info = prior.get_default_grid()

            else:
                param_grid = grid_params['param_grid']
                param_info = grid_params['param_info']

            n_griditer = grid_params['n_griditer']
            verbose = grid_params['verbose']
            n_jobs = grid_params['n_jobs']
            grid = ParamSearchCV(model, param_grid, param_info,
                                 n_griditer=n_griditer,
                                 n_jobs=n_jobs, verbose=verbose,
                                 scorer=self.metric, fit_final=True)

            t0 = time.time()
            grid.fit(X, Y)
            t_fit = time.time() - t0

        else:

            t0 = time.time()
            model.fit(X, Y)
            t_fit = time.time() - t0

        self.t_fit = t_fit

        self._split_coef_spikefilt(model.coef_)
        self.intercept_ = np.copy(model.intercept_)


class PoissonGLM(LNPEstimator):
    """GLM with Poisson distribution of response values

    For a description of the other arguments see lnpy.lnp.cbrf.CbRF

    """

    def __init__(self, optimize=True, metric='PoissonLL',
                 tolerance=1e-5, canonical=True, fit_bias=True,
                 prior=GaussianPrior(), alpha=1, **kwargs):

        super(PoissonGLM, self).__init__(fit_bias=fit_bias, **kwargs)

        self.alpha = alpha
        self.optimize = optimize
        self.metric = metric
        self.tolerance = tolerance
        self.canonical = canonical
        self.prior = prior

        self.t_fit = 0.

    @property
    def algorithm(self):
        return 'BGD'

    def fit(self, X, Y):

        prior = self.prior
        n_spikefilt = self.n_spikefilt

        if Y.ndim > 1:
            Y = np.sum(Y, axis=1)

        if n_spikefilt > 0:
            nx = X.shape[1]
            nw = nx - n_spikefilt
            prior.set_ndim(nw)

        grid_params = self.grid_params
        if 'param_grid' not in grid_params.keys() or \
                grid_params['param_grid'] is None:
            param_grid, param_info = self.prior.get_default_grid()
            grid_params['param_grid'] = param_grid
            grid_params['param_info'] = param_info

        model = _PoissonGLM(alpha=self.alpha, canonical=self.canonical,
                            tolerance=self.tolerance,
                            optimize=self.optimize, prior=prior,
                            scorer=self.metric, **grid_params)

        t0 = time.time()
        model.fit(X, Y)
        self.t_fit = time.time() - t0

        self._split_coef_spikefilt(model.coef_)
        self.intercept_ = model.intercept_


class GaussianGLM(LNPEstimator):
    """GLM with Gausian distribution of response values

    For a description of the other arguments see lnpy.lnp.cbrf.CbRF

    """

    def __init__(self, optimize=True, metric='MSE',
                 verbose=-1, tolerance=.1, fit_bias=True,
                 prior=GaussianPrior(), alpha=.1, **kwargs):

        super(GaussianGLM, self).__init__(fit_bias=fit_bias, **kwargs)

        self.alpha = alpha
        self.optimize = optimize
        self.metric = metric
        self.verbose = verbose
        self.tolerance = tolerance
        self.prior = prior

        self.t_fit = 0.

    @property
    def algorithm(self):
        return 'BGD'

    def fit(self, X, Y):

        prior = self.prior
        n_spikefilt = self.n_spikefilt

        if n_spikefilt > 0:
            nx = X.shape[1]
            nw = nx - n_spikefilt
            prior.set_ndim(nw)

        grid_params = self.grid_params
        if 'param_grid' not in grid_params.keys() or \
                grid_params['param_grid'] is None:
            param_grid, param_info = self.prior.get_default_grid()
            grid_params['param_grid'] = param_grid
            grid_params['param_info'] = param_info

        model = _GaussianGLM(alpha=self.alpha,
                             verbose=self.verbose, tolerance=self.tolerance,
                             optimize=self.optimize, prior=self.prior,
                             scorer=self.metric, **grid_params)

        t0 = time.time()
        model.fit(X, Y)
        self.t_fit = time.time() - t0

        self._split_coef_spikefilt(model.coef_)
        self.intercept_ = model.intercept_


class _SGDGLM(SKBaseEstimator):
    """SGD-based GLM"""

    def __init__(self, metric='logli_poissonexp', family='poisson',
                 link='log', alpha=1.0,
                 n_epochs=1, algorithm='sgd', weighting='permutation',
                 avg_decay=2., warm_start=False, eta0=.1,
                 bias_multiplier=1.):

        self.family = family
        self.metric = metric
        self.link = link
        self.n_epochs = n_epochs
        self.coef_ = None
        self.intercept_ = None
        self.algorithm = algorithm
        self.weighting = weighting
        self.avg_decay = avg_decay
        self.warm_start = warm_start
        self.bias_multiplier = bias_multiplier

        # We define the model here to allow 'warm start' of the classifier
        class_weight = None
        if isinstance(self.weighting, float):
            permutation = self.weighting

        elif isinstance(self.weighting, str) and \
                self.weighting.lower() == 'permutation':
            class_weight = None
            permutation = 'auto'

        elif self.weighting is None or self.weighting.lower() == 'none':
            permutation = None

        bias_multiplier = self.bias_multiplier

        if self.family.lower() == 'poisson':
            if self.link.lower() == 'log':
                loss = LogLoss()
            elif self.link.lower() == 'modified_log':
                loss = LogLoss()
            else:
                raise ValueError("Unknown link:", self.link)
        else:
            raise ValueError("Unknown family:", self.family)

        if self.algorithm.upper() == 'SGD':
            model = SGD(loss=loss, fit_bias=True,
                        n_epochs=self.n_epochs, alpha=alpha,
                        class_weight=class_weight, rand_seed=0,
                        permutation=permutation, warm_start=True,
                        bias_multiplier=bias_multiplier)

        elif self.algorithm.upper() == 'ASGD':
            model = ASGD(loss=loss, n_epochs=self.n_epochs,
                         alpha=alpha, class_weight=class_weight,
                         fit_bias=True, permutation=permutation,
                         start_avg=0, avg_decay=self.avg_decay,
                         avg_method='poly_decay', warm_start=True,
                         bias_multiplier=bias_multiplier)

        self.model = model

    @property
    def iter0(self):
        return self.model.n_iter0

    @iter0.setter
    def iter0(self, n):
        self.model.n_iter0 = n

    @property
    def alpha(self):
        return self.model.alpha

    @alpha.setter
    def alpha(self, value):
        self.model.alpha = value

    def fit(self, X, Y):
        """Wrapper for the fit method to handle multiple spike trains

        """
        if not isinstance(X, np.ndarray) or not X.dtype == np.float64:
            raise ValueError("X must be an ndarray instance (dtype=float64)")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y have incompatible shapes.\n"
                             "X has %d samples, but y has %d." %
                             (X.shape[0], Y.shape[0]))

        model = self.model
        if not self.warm_start:
            model.reset()

        if Y.ndim > 1:
            n_trials = Y.shape[1]
            for trial in range(n_trials):
                model.fit(X, np.asarray(Y[:, trial], dtype=np.int8))

        else:
            model.fit(X, Y.flatten().astype(np.int8))

        self.coef_ = model.get_weights()
        self.intercept_ = model.get_bias()

    def predict(self, X):

        return np.dot(X, self.coef_) + self.intercept_


class StochasticGLM(LNPEstimator):
    """Stochastic gradient descent approximation to GLM

        Approximates the solution to the CbRF method by stochastic gradient
        descent.

               Parameters
        ----------
        optimize : boolean
            Perform automatic model optimization?

        metric : string
            The metric used for model optimization, e.g., 'AUC' or 'MI'

        loss : {'hinge', 'squared_hinge', 'log'}
            Loss function

        alpha : float
            Regularization parameter

        n_epochs : float
            Number of SGD iterations over the data set

        algorithm : {'sgd', 'asgd'}
            Plain SGD ('sgd') or polynomial averaging SGD ('asgd')

        avg_decay : float
            decay factor of asgd algorithm with avg_decay >= 0

        warm_start : boolean
            initialize SGD algorithm with current parameters or reset
            parameters before each training

        suffix : string
            Prepend string to method name for easier result handling

        verbose : boolean
            be verbose

        n_jobs : int
            Number of processes used for optimization

        param_range : list
            Min. and max. initial cost parameters

        weighting : {'permutation', 'inv_prob'}
            Permute samples from spike and no spike classes or use inverse
            class priors?
    """
    def __init__(self, optimize=True, metric='PoissonLL',
                 family='poisson', link='log', alpha=0.01, n_epochs=1,
                 algorithm='sgd',
                 suffix='', verbose=1, weighting='permutation',
                 avg_decay=2., warm_start=False,
                 bias_multiplier=1., **kwargs):

        super(StochasticGLM, self).__init__(**kwargs)

        self.optimize = optimize
        self._alpha = alpha
        self.metric = metric
        self.n_epochs = n_epochs
        self.algorithm = algorithm
        self.suffix = suffix
        self.verbose = verbose
        self.avg_decay = avg_decay

        self.t_total = 0.
        self.t_fit = 0.

        self.__model__ = _SGDGLM(metric=metric, family=family, link=link,
                                 n_epochs=n_epochs, alpha=alpha,
                                 algorithm=algorithm,
                                 weighting=weighting,
                                 avg_decay=self.avg_decay,
                                 warm_start=warm_start,
                                 bias_multiplier=bias_multiplier)

    @property
    def name(self):
        if self.algorithm.lower() == 'sgd':
            return 'StochasticGLM_SGD' + self.suffix

        elif self.algorithm.lower() == 'asgd':
            return 'StochasticGLM_ASGD' + self.suffix

    @property
    def w(self):
        return self.__model__.coef_

    @property
    def b(self):
        return self.__model__.intercept_

    @property
    def alpha(self):
        return self.__model__.alpha

    @alpha.setter
    def alpha(self, value):
        self.__model__.alpha = value

    def get_weights(self):
        return self.w

    def get_bias(self):
        return self.b

    def reset(self):
        """Resets algorithm parameters to initial values"""
        self.__model__.coef_ = None
        self.__model__.intercept = 0.
        self.t_fit = 0.
        self.t_total = 0

    def fit(self, X, Y):
        """Train model and extract parameters"""

        if self.optimize:

            grid_params = self.grid_params

            if 'param_grid' not in grid_params.keys() or \
                    grid_params['param_grid'] is None:

                lower = 2. ** -30
                upper = 2. ** 2
                n_steps = 7

                alpha_values = np.power(2., np.linspace(np.log2(lower),
                                                        np.log2(upper),
                                                        n_steps))
                param_grid = {'alpha': alpha_values}
                param_info = {'alpha': {'scaling': 'log2'}}

                grid_params['param_grid'] = param_grid
                grid_params['param_info'] = param_info

            grid = ParamSearchCV(self.__model__, scorer=self.metric,
                                 **grid_params)

            t0 = time.time()
            grid.fit(X, Y)

            # Train model using best parameters
            self.__model__.alpha = grid.best_params_['alpha']
            self.__model__.fit(X, Y)
            t_total = time.time() - t0

            self.t_fit = t_total
            self.t_total = t_total

        else:

            t0 = time.time()
            self.__model__.fit(X, Y)
            t_fit = time.time() - t0

            self.t_fit = t_fit
            self.t_total = t_fit
