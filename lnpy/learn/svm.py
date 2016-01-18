#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Suport vector machine (SVM) classifier (using TRON solver)
"""


import numpy as np
import warnings

from .base import LinearBaseEstimator
from . import GaussianPrior, SparseProblem, DenseProblem
from .pyhelper import PySVM
from .grid_search import scorer_AUC


class SVM(LinearBaseEstimator):
    """Bernoulli GLM with canonical link function

    Parameters
    ----------
    prior : PyPrior
        Prior used for inference of parameters (default: GaussianPrior)
    """

    def __init__(self, prior=GaussianPrior(), tolerance=.1,
                 bias_multiplier=1., weighting='inv_prob',
                 alpha=None, gamma=None, **kwargs):
        super(SVM, self).__init__(**kwargs)

        self.prior = prior
        self.tolerance = tolerance
        self.bias_multiplier = bias_multiplier
        self.weighting = weighting

    def fit(self, X, Y, is_sparse=True):

        X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
        Y = np.require(Y, dtype=np.float64, requirements=['C', 'A'])

        Y = Y.astype(np.float64)
        bias = self.bias_multiplier
        N, p = X.shape
        n_trials = 1
        if Y.ndim > 1:
            n_trials = Y.shape[1]

        labels = np.unique(Y)

        n_classes = labels.shape[0]
        if n_classes < 2:
            warnings.warn("Y contains only 1 class")
        elif n_classes > 2:
            raise ValueError("Y contains more than two classes")

        if np.sum(labels == np.array([-1, 1])) != 2:
            Y = np.copy(Y)
            Y[Y <= 0] = -1
            Y[Y > 0] = 1
            labels = np.array([-1, 1])

        C = None
        if self.weighting.lower() == 'inv_prob':
            C = np.zeros_like(Y)
            for label in labels:
                ii = Y == label
                n = np.sum(ii)
                C[ii] = float(N * n_trials) / n
            C = C.ravel()

        if is_sparse:
            prob = SparseProblem(X, Y.ravel(), bias=1, C=C,
                                 binary_targets=True)
        else:
            prob = DenseProblem(X, Y.ravel(), bias=1, C=C,
                                binary_targets=True)

        w = np.zeros((p+int(bias > 0),))
        model = PySVM(self.prior)
        model.fit(prob, w, verbose=0, tolerance=self.tolerance)

        self._set_coef_intercept_from_w(w)

    @property
    def default_scorer(self):
        return scorer_AUC
