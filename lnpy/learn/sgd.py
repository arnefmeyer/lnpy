#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Parameter optimization using stochatic gradient descent (SGD)

    Currently, all methods use a zero-mean Gaussian prior.
"""

from __future__ import division

import numpy as np
from base import LinearBaseEstimator
from . import (
    DenseProblem, SparseProblem, GaussianLoss
)
import pyhelper


class BaseSGDEstimator(LinearBaseEstimator):
    """Base class for all SGD classifiers

    Parameters
    ==========

    n_epochs : float
        The number of SGD iterations of the data set

    loss : {'log', 'hinge', 'squared_hinge'}
        Loss function in the large-margin objective

    permutation : {'auto', list, numpy array}
        Permutation of feature vectors

    class_weight : {'auto', None}
        Weighting of misclassification errors

    warm_start : boolean
        Use previous model parameters to fit new model?

    rand_seed : float
        Random seed for drawing SGD examples

    bias_multiplier : float
        Multiplier of bias term, e.g., 1.0 for dense and 0.1 for sparse
        data

    """

    def __init__(self, n_epochs=1, loss=GaussianLoss(), permutation='auto',
                 weighting=None, warm_start=False, rand_seed=0, **kwargs):

        super(BaseSGDEstimator, self).__init__(**kwargs)

        self.n_epochs = n_epochs
        self.loss = loss
        self.permutation = permutation
        self.weighting = weighting
        self.warm_start = warm_start
        self.rand_seed = rand_seed

        self.n_iter = 0
        self.t_fit = 0.

    def reset(self):
        """Reset classifier parameters (weights, bias, and # iterations)"""

        self.n_iter = 0
        self.t_fit = 0

        self.intercept_ = 0
        if self.coef_ is not None:
            self.coef_[:] = 0


class SGD(BaseSGDEstimator):
    """Plain stochastic gradient descent-based binary classifier

        Parameters
        ==========
        alpha : float
            The regularization parameter. Note that we have to define it
            in the constructor to make use of sklearn's hyperparameter
            grid search
    """

    def __init__(self, alpha=1, **kwargs):
        super(SGD, self).__init__(**kwargs)

        self.alpha = alpha

    def fit(self, X, Y):

        if self.optimize:
            self._fit_optimize(X, Y)

        else:
            self._fit(X, Y)

    def _fit(self, X, Y, is_sparse=False):

        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must contain the same number of '
                             'observations!')

        bias = self.bias_multiplier
        n_epochs = self.n_epochs
        loss = self.loss
        alpha = self.alpha
        warm_start = self.warm_start
        permutation = self.permutation
        weighting = self.weighting
        rand_seed = self.rand_seed

        if self.warm_start is False:
            self.reset()

        w = self.coef_
        if w is None:
            N, n_dim = X.shape
            p = n_dim + int(bias > 0)
            w = np.zeros((p,), dtype=np.float64, order='C')

        elif bias > 0:
            w = np.append(w, self.intercept_)

        if is_sparse:
            prob = SparseProblem(X, Y.ravel(), bias,
                                 binary_targets=loss.binary_targets)
        else:
            prob = DenseProblem(X, Y.ravel(), bias,
                                binary_targets=loss.binary_targets)

        perm = permutation.lower()
        if isinstance(perm, str) and perm in ['auto', 'balance_classes']:
            if prob.binary_targets:
                prob.permute('balance_classes', size=n_epochs, seed=rand_seed)
            else:
                prob.permute(size=n_epochs)

        if isinstance(weighting, str) and weighting.lower() == 'inv_prob':
            prob.set_weighting('inv_prob')

        model = pyhelper.PySGD(w, bias, n_epochs=n_epochs, loss=loss,
                               alpha=alpha, warm_start=warm_start)
        n_iter = model.fit(prob, start_iter=self.n_iter)

        self._set_coef_intercept_from_w(w)
        self.n_iter += n_iter


class ASGD(BaseSGDEstimator):
    """polynomial-decay averaging SGD (Shamir and Zhang ICML 2013)

    Parameters
    ----------
    start_avg : int
        Start averaging after start_avg iterations

    avg_decay : float
        Polynomial decay term in the running averaging scheme (eta in Shamir
        & Zhang's paper, eq. 10)

    """

    def __init__(self, alpha=1, start_avg=0, avg_decay=2., **kwargs):
        super(ASGD, self).__init__(**kwargs)

        self.alpha = alpha
        self.start_avg = 0
        self.avg_decay = avg_decay

        self.coef_sgd_ = None

    def reset(self):
        super(ASGD, self).reset()

        if self.coef_sgd_ is not None:
            self.coef_sgd_[:] = 0

    def fit(self, X, Y):
        if self.optimize:
            self._fit_optimize(X, Y)

        else:
            self._fit(X, Y)

    def _fit(self, X, Y, is_sparse=False):
            if X.shape[0] != Y.shape[0]:
                raise ValueError('X and Y must contain the same number of '
                                 'observations!')

            Y = np.require(Y, dtype=np.float64, requirements=['C'])

            bias = self.bias_multiplier
            n_epochs = self.n_epochs
            loss = self.loss
            alpha = self.alpha
            warm_start = self.warm_start
            permutation = self.permutation
            weighting = self.weighting
            rand_seed = self.rand_seed
            avg_decay = self.avg_decay

            if self.warm_start is False:
                self.reset()

            if bias > 0 and self.coef_ is not None:
                w = np.append(self.coef_, self.intercept_)
            else:
                w = self.coef_

            wsgd = self.coef_sgd_
            if w is None or wsgd is None:
                N, n_dim = X.shape
                p = n_dim + int(bias > 0)
                w = np.zeros((p,), dtype=np.float, order='C')
                wsgd = np.zeros((p,), dtype=np.float, order='C')

            if is_sparse:
                prob = SparseProblem(X, Y.ravel(), bias,
                                     binary_targets=loss.binary_targets)
            else:
                prob = DenseProblem(X, Y.ravel(), bias,
                                    binary_targets=loss.binary_targets)

            if isinstance(permutation, str) and permutation.lower() == 'auto':
                if prob.binary_targets:
                    prob.permute('balance_classes', size=n_epochs,
                                 seed=rand_seed)
                else:
                    prob.permute(size=n_epochs)

            if isinstance(weighting, str) and weighting.lower() == 'inv_prob':
                prob.set_weighting('inv_prob')

            model = pyhelper.PyASGD(wsgd, w, bias, n_epochs=n_epochs,
                                    loss=loss, warm_start=warm_start,
                                    alpha=alpha, poly_decay=avg_decay)
            n_iter = model.fit(prob, start_iter=self.n_iter)

            self._set_coef_intercept_from_w(w)
            self.coef_sgd = wsgd
            self.n_iter += n_iter
