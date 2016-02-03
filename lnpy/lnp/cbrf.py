#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    A classification-based receptive field estimation algorithm

    For details on the CbRF method see:

    Meyer AF, Diepenbrock J-P, Happel MFK, Ohl FW, Anemueller J (2014)
    Discriminative Learning of Receptive Fields from Responses to Non-Gaussian
    Stimulus Ensembles. PLoS ONE 9(4): e93062.

    doi: 10.1371/journal.pone.0093062
    link: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0093062

    The SGD-based stochastic approximation scheme has been described in:
    Arne AF, Diepenbrock JP, Ohl FW, Anemüller J (2015)
    Fast and robust estimation of spectro-temporal receptive fields using
    stochastic approximations. Journal of Neuroscience Methods, 246, 119-133.

    doi: http://dx.doi.org/10.1016/j.jneumeth.2015.02.009.
    link: http://www.sciencedirect.com/science/article/pii/S0165027015000618
"""

from __future__ import division

from sklearn.linear_model import SGDClassifier

import numpy as np
import time
import warnings

from base import LNPEstimator
from ..learn.svm import SVM
from ..learn.sgd import SGD, ASGD
from ..learn import GaussianPrior, SquaredHingeLoss
from ..learn.grid_search import ParamSearchCV


class CbRF(LNPEstimator):
    """Classification-based RF estimation class

    Computes full solution of the CbRF method by using batch gradient
    descent.

    Parameters
    ----------
    optimize : boolean
        Perform automatic model optimization?

    metric : string
        The metric used for model optimization, e.g., 'AUC' or 'MI'. See
        grid_seach module for further metrics.

    prior : Prior
         Prior of the model, e.g., GaussianPrior or LaplacePrior. See learn
         module for further priors

    tolerance : float
        TRON solver tolerance

    alpha, gamma : double
        Prior hyperparameters

    Optional parameters will be passed to the grid search class
    lnpy.learn.grid_search.ParamSearchCV

    """

    def __init__(self, optimize=True, metric='AUC', prior=GaussianPrior(),
                 alpha=1., gamma=0., **kwargs):

        super(CbRF, self).__init__(**kwargs)

        self.optimize = optimize
        self.metric = metric
        self.prior = prior
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, Y):
        """Ttain model and extract parameters"""

        prior = self.prior
        n_spikefilt = self.n_spikefilt

        X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
        Y = np.require(Y, dtype=np.float64, requirements=['C', 'A'])

        if n_spikefilt > 0:
            nx = X.shape[1]
            nw = nx - n_spikefilt
            prior.set_ndim(nw)

        model = SVM(prior=prior, verbose=0, tolerance=.1,
                    bias_multiplier=1., weighting='inv_prob',
                    alpha=self.alpha, gamma=self.gamma)

        labels = np.unique(Y)
        if np.sum(labels == np.array([-1, 1])) != len(labels):
            warnings.warn('Casting Y to [-1, 1] (and setting Y > 0 to 1)!')
            Y = np.copy(Y)
            Y[Y <= 0] = -1
            Y[Y > 0] = 1

        if self.optimize:

            grid_params = self.grid_params
            if 'param_grid' not in grid_params.keys() or \
                    grid_params['param_grid'] is None:
                param_grid, param_info = prior.get_default_grid()
            else:
                param_grid = grid_params['param_grid']
                param_info = grid_params['param_info']

            # Grid search parameters
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


class StochasticCbRF(LNPEstimator):
    """Stochastic gradient descent version of the CbRF method

    Approximates the solution to the CbRF method by stochastic gradient
    descent (with Gaussian prior)

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

    param_range : {ndarray, list}
        Prior hyperparameters

    weighting : {'permutation', 'inv_prob'}
        Permute samples from spike and no spike classes or use inverse
        class priors?

    Optional parameters will be passed to the grid search class
    lnpy.learn.grid_search.ParamSearchCV

    """
    def __init__(self, optimize=True, metric='AUC', loss=SquaredHingeLoss(),
                 alpha=1, n_epochs=1, algorithm='sgd', suffix='',
                 permutation='auto', weighting=None,
                 param_range=None, avg_decay=2., warm_start=False,
                 eta0=-1, **kwargs):

        super(StochasticCbRF, self).__init__(**kwargs)

        self.optimize = optimize
        self.metric = metric
        self.n_epochs = n_epochs
        self.algorithm = algorithm
        self.suffix = suffix
        self.param_range = param_range
        self.avg_decay = avg_decay
        self.eta0 = eta0

        if algorithm.upper() == 'SKLEARN':
            model = SGDClassifier(loss=loss, penalty='l2',
                                  fit_intercept=True, n_iter=n_epochs,
                                  alpha=alpha, learning_rate='optimal',
                                  class_weight=None, warm_start=True,
                                  shuffle=True, random_state=0,
                                  verbose=False)

        elif algorithm.upper() == 'SGD':
            model = SGD(loss=loss, fit_bias=True,
                        n_epochs=n_epochs, alpha=alpha,
                        permutation=permutation, weighting=weighting,
                        rand_seed=0)

        elif algorithm.upper() == 'ASGD':
            model = ASGD(loss=loss, n_epochs=n_epochs,
                         alpha=alpha, fit_bias=True, permutation=permutation,
                         weighting=weighting, start_avg=0, rand_seed=0,
                         avg_decay=avg_decay)
        else:
            raise ValueError('Invalid fitting algorithm: {}'.format(algorithm))
        self.__model__ = model

    @property
    def name(self):
        if self.algorithm.lower() == 'sgd':
            return 'StochasticCbRF_SGD' + self.suffix

        elif self.algorithm.lower() == 'asgd':
            return 'StochasticCbRF_ASGD' + self.suffix

        elif self.algorithm.lower() == 'adagrad':
            return 'StochasticCbRF_AdaGrad' + self.suffix

    @property
    def alpha(self):
        return self.__model__.alpha

    @alpha.setter
    def alpha(self, value):
        self.__model__.alpha = value

    def reset(self):
        """Resets algorithm parameters to initial values"""
        self.__model__.reset()

        self.t_fit = 0.

    def fit(self, X, Y):
        """Train model and extract parameters"""

        X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
        Y = np.require(Y, dtype=np.float64, requirements=['C', 'A'])

        model = self.__model__

        labels = np.unique(Y)

        n_classes = labels.shape[0]
        if n_classes < 2:
            warnings.warn("Y contains only 1 class")
        elif n_classes > 2:
            raise ValueError("Y contains more than two classes")

        if 0 in labels:
            Y = np.copy(Y)
            Y[Y == 0] = -1
            labels[labels == 0] = -1

        if self.optimize:

            # Grid search parameters
#            lower = 2. ** -30
#            upper = 2. ** 2
#            if self.param_range is not None:
#                lower = self.param_range[0]
#                upper = self.param_range[1]
#
#            alpha_values = np.power(2., np.linspace(np.log2(lower),
#                                                    np.log2(upper), 7))

            grid_params = self.grid_params
            if 'param_grid' not in grid_params.keys() or \
                    grid_params['param_grid'] is None:

                if self.param_range is not None:
                    lower = self.param_range[0]
                    upper = self.param_range[1]
                else:
                    lower = 2. ** -30
                    upper = 2. ** 2

                param_grid = 2. ** np.linspace(np.log2(lower),
                                               np.log2(upper), 7)
            else:
                param_grid = grid_params['param_grid']

            param_info = {'alpha': {'scaling': 'log2'}}
            grid = ParamSearchCV(model, param_grid, param_info,
                                 scorer=self.metric, fit_final=True)

            t0 = time.time()
            grid.fit(X, Y)
            t_fit = time.time() - t0

        else:
            t0 = time.time()
            model.fit(X, Y)
            t_fit = time.time() - t0

        self.t_fit = t_fit

        self.coef_ = np.copy(model.coef_)
        self.intercept_ = np.copy(model.intercept_)
