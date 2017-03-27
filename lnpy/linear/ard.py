#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Automatic relevance determination (discussed in Sahani & Linden NIPS 2003a)
"""

import numpy as np
from sklearn.utils import extmath
import time

from .base import LinearModel


def lrard(xx, yy, threshold=1e7, tol=1e-3, maxiter=1500, verbose=True,
          keep_min_dims=0):

    NN, DD = xx.shape
    DDo = DD

    xx2 = np.dot(xx.T, xx)
    xxyy = np.dot(xx.T, yy)
    yy2 = np.sum(yy * yy)

    aa = 2 * np.ones((DD,))
    rv = np.arange(DD)

    nv = yy2 - np.mean(yy)**2 + np.random.rand()

    niter = 0
    t0 = time.time()
    while True:

        niter += 1
        aaold = np.copy(aa)

        if len(aa) == 0 or len(aa) == keep_min_dims:
            break

        # variance of posterior over w
        SS = extmath.pinvh(xx2 / nv + np.diag(aa))

        if np.isnan(1. / np.linalg.cond(SS)):
            raise RuntimeError('Rcond NaN - have to revert to ASD')

        diagSS = np.diag(SS)  # sum ii
        mu = np.dot(SS, xxyy).ravel() / nv  # mean of posterior over w

        # numerator of precision update
        gg = 1 - aa*diagSS

        # precision update
        aa = gg.ravel() / np.abs(mu)**2

        # noise variance update
        nv = (yy2 - 2*np.dot(mu.T, xxyy) +
              np.abs(np.dot(mu.T, xx2).dot(mu)))/(NN - np.sum(gg))

        # remove irrelevant dimensions
        zz = aa > threshold
        if np.sum(zz) > 0:
            zz = ~zz
            rv = rv[zz]
            aa = aa[zz]
            xx = xx[:, zz]
            xx2 = xx2[:, zz]
            xx2 = xx2[zz, :]
            xxyy = xxyy[zz]
            DD = aa.shape[0]

        elif np.all(np.abs(aa - aaold) < tol * aa) or niter > maxiter or \
                np.sum(zz) == 0:
            break

        if verbose:
            t_elapsed = time.time() - t0
            print "iter %d: %d/%d active | nv=%0.4f | %0.2f s" %\
                (niter, DD, DDo, nv, t_elapsed)

    if len(aa) > 0:
        Sw = extmath.pinvh(xx2 / nv + np.diag(aa))

        ww_out = np.zeros((DDo,))
        ww_out[rv] = np.dot(Sw, xxyy) / nv

        Sw_out = np.zeros((DDo, DDo))
        Sw_out[np.ix_(rv, rv)] = Sw

        nv_out = nv

    else:
        ww_out = None
        Sw_out = None
        nv_out = None

    return ww_out, Sw_out, nv_out


class ARD(LinearModel):

    def __init__(self, threshold=1e7, tolerance=1e-3,
                 maxiter=1500, verbose=True, keep_min_dims=0, **kwargs):

        super(ARD, self).__init__(normalize=False, **kwargs)

        self.threshold = threshold
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.verbose = verbose
        self.keep_min_dims = keep_min_dims

        self.cov_posterior = None

    def fit(self, X, y):

        if self.fit_intercept:
            X, y, X_mean, y_mean = self._center_data(
                X, y, copy=True)

        res = lrard(X, y, threshold=self.threshold, tol=self.tolerance,
                    verbose=self.verbose, maxiter=self.maxiter,
                    keep_min_dims=self.keep_min_dims)

        w = res[0]
        if self.fit_intercept:
            if w is not None:
                self.coef_ = w
                self._set_intercept(X_mean, y_mean)
            else:
                self.coef_ = np.zeros((X.shape[1]))
                self.intercept_ = 0

        elif w is not None:
            self.coef_ = np.copy(w)
            self.intercept_ = 0.

        self.cov_posterior = res[1]

    def predict(self, X):

        z = np.dot(X, self.coef_)
        if self.fit_intercept:
            z += self.intercept_

        return z
