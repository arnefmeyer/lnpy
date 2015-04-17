#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Base and helper classes for fitting linear-nonlinear models
"""

import numpy as np
from sklearn.base import BaseEstimator as SKBaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone as _clone
from pyhelper import PyDenseProblem, PySparseProblem
import warnings

from .grid_search import ParamSearchCV


class BaseEstimator(SKBaseEstimator, ClassifierMixin):
    """Base class for all estimators"""

    def __setattr__(self, name, value):
        """Catch prior hyperparameter attributes"""

        if name in ['alpha', 'gamma']:
            if hasattr(self, 'prior') and self.prior is not None:
                if hasattr(self.prior, name) and value is not None:
                    setattr(self.prior, name, value)
            else:
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value

    def clone(self, safe=True):
        return _clone(self, safe=safe)

    @property
    def name(self):
        return self.__class__.__name__


class LinearBaseEstimator(BaseEstimator):
    """Base class for all linear estimators"""

    def __init__(self, bias_multiplier=1., optimize=False, **kwargs):

        self.bias_multiplier = bias_multiplier
        self.optimize = optimize
        self.grid_parameters = kwargs

        self.coef_ = None
        self.intercept_ = 0.

    def get_weights(self):
        return self.coef_

    def get_bias(self):
        return self.intercept_

    def get_weights_and_bias(self):
        return np.append(self.coef_, self.intercept_)

    def score(self, X, y):
        return self.default_scorer(self, X, y)

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

    def _fit_optimize(self, X, Y, stratify_folds=False):

        grid_params = self.grid_parameters
        if 'stratify_folds' not in grid_params.keys():
            grid_params.update({'stratify_folds': stratify_folds})
        else:
            grid_params['stratify_folds'] = stratify_folds

        # Avoid infinite recurrence
        optim_state = self.optimize
        self.optimize = False

        grid = ParamSearchCV(self, **grid_params)
        grid.fit(X, Y)

        self.optimize = optim_state

    def _set_coef_intercept_from_w(self, w):

        if self.bias_multiplier > 0:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]

        else:
            self.coef_ = w
            self.intercept_ = 0.

    def _center_data(self, X, Y):

        X_mean = X.mean(axis=0)
        X = X - X_mean
        Y_mean = Y.mean(axis=0)
        Y = Y - Y_mean

        return X, Y, X_mean, Y_mean

    def _set_intercept(self, X_mean, Y_mean):
        self.intercept_ = Y_mean - np.dot(X_mean, self.coef_.T)


def get_num_trials(Y):
    """Return the number of trials in Y"""

    if Y.ndim == 1:
        return 1
    else:
        return Y.shape[1]


def create_derivative_matrix(nt, nf, order='C'):
    """Create matrix with first-order derivatives

        Parameters
        ----------
        nt : int
            Number of points in temporal dimensions (axis = 0)

        nf : int
            Number of points in frequency dimension (axis = 1)

        order : str
            Numpy memory layout ('C' or 'F')
    """

    N = nt*nf
    Zt = np.zeros((N, N))
    Zf = np.zeros((N, N))
    I = np.eye(N)

    # Create separate grid for time and frequency dimensions
    FF, TT = np.meshgrid(range(nf), range(nt))
    ff = FF.ravel(order=order)
    tt = TT.ravel(order=order)

    for ii in range(N):
        for jj in range(ii, N):

            dt = np.abs(tt[ii] - tt[jj])
            df = np.abs(ff[ii] - ff[jj])

            if dt == 1 and df == 0:
                Zt[ii, jj] = -1

            if dt == 0 and df == 1:
                Zf[ii, jj] = -1

    # First-order derivatives
    Ztf = I + Zt + Zf
    Dtf = np.dot(Ztf.T, Ztf)

    return Dtf


class Problem():

    def __init__(self):
        pass

    def permute(self, arg=None, size=1., seed=0):
        """Permutation of observations and targets"""

        Y = self.get_Y()

        if isinstance(arg, (np.ndarray, list)):
            perm = np.asarray(arg, dtype=np.uint64)
            self.set_permutation(perm.ravel())

        elif isinstance(arg, str) and arg.lower() in ['balance_classes']:
            if self.binary_targets:
                # Manually balance number of negative/positive examples
                N = Y.size
                pos_idx = (Y > 0).nonzero()[0]
                neg_idx = (Y <= 0).nonzero()[0]
                n_pos = len(pos_idx)
                n_neg = len(neg_idx)
                n_iter = int(np.ceil(size * N / 2.) * 2)
                perm = np.zeros((n_iter,), dtype=np.uint64)

                # For reproducible results during parameter optimization
                rng = np.random.RandomState(seed)

                if n_pos > 0 and n_neg > 0:
                    perm[0::2] = pos_idx[rng.randint(0, n_pos, (n_iter/2,))]
                    perm[1::2] = neg_idx[rng.randint(0, n_neg, (n_iter/2,))]
                elif n_pos > 0 and n_neg == 0:
                    perm[:] = pos_idx[rng.randint(0, n_pos, (n_iter,))]
                elif n_neg > 0 and n_pos == 0:
                    perm[:] = neg_idx[rng.randint(0, n_neg, (n_iter,))]

                perm = perm.astype(np.uint64)
                self.set_permutation(perm.ravel())

            else:
                warnings.warn("Permutation 'balance_classes' makes only sense",
                              "for binary targets")

        elif arg is None:

            N = Y.size
            n_iter = int(np.ceil(size * N))
            rng = np.random.RandomState(seed)
            perm = rng.randint(0, N, n_iter).astype(np.uint64)

            self.set_permutation(perm.ravel())

    def set_weights(self, arg):
        """Set weights of observations

        Parameters
        ----------
        arg : np.ndarray, list, str
            If arg is an np.array or a list it is supposed that each
            element contains the weight for the corresponding observation.
            If arg is the string 'inv_prob' each instance is weighted by
            the inverse probability of the corresponding label. Note that
            this makes only sense for binary targets!

        """

        Y = self.get_Y()

        if isinstance(arg, (np.ndarray, list)):
            # Weighting vector given
            if len(arg) != self.get_N() * self.get_ntrials():
                raise ValueError("Number of weights != number of observations")

            C = np.asarray(arg, dtype=np.float64)
            self.set_C(C.ravel())

        elif isinstance(arg, str):
            # Inverse class probability for binary targets
            if arg.lower() == 'inv_prob':
                if self.binary_targets:
                    labels = np.unique(Y)
                    N = Y.size
                    C = np.zeros_like(Y)
                    for label in labels:
                        ind = Y == label
                        C[ind] = np.sum(ind > 0) / float(N)
                    self.set_weights(C.ravel())

                else:
                    warnings.warn("Weighting 'inv_prob' makes only sense",
                                  "for binary targets")


class SparseProblem(PySparseProblem, Problem):
    """Wrapper class for computationally efficient liblinear data format"""

    # Automatically calls the underlying cython constructor
    def __init__(self, X, Y, bias=1., C=None, binary_targets=False):
        self.binary_targets = binary_targets


class DenseProblem(PyDenseProblem, Problem):
    """Wrapper class for dense data format"""

    # Automatically calls the underlying cython constructor
    def __init__(self, X, Y, bias=1., C=None, binary_targets=False):
        self.binary_targets = binary_targets
