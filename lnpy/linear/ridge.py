#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Ridge regression (via an iterative fixed-point algorithm)
"""

from __future__ import print_function

import numpy as np
from scipy import linalg, optimize
from sklearn.utils.extmath import fast_logdet
try:
    from sklearn.utils.extmath import pinvh
except ImportError:
    from scipy.linalg import pinvh

import time

from .base import LinearModel
from .prior import create_derivative_matrix


def ridge_evidence_iter(X, y, penalize_bias=False, maxvalue=1e6, maxiter=1e3,
                        tolerance=1e-3, verbose=1, alpha0=1.):
    """Evidence optimization of ridge regression using fixed-point algorithm.
        See Park and Pillow PLOS Comp Biol 2011 for details.
    """

    N, p = X.shape

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    yTy = np.sum(y * y)

    # Inverse prior variance
    alpha = 10.
    S = np.eye(p)
    I = np.eye(p)

    if not penalize_bias:
        S[0, 0] = 0

    # Initialize mean and noise variance using ridge MAP estimate
    mu = linalg.solve(XTX + alpha * I, XTy, sym_pos=False)
    noisevar = yTy - 2*np.dot(mu.T, XTy) + np.dot(mu.T, XTX).dot(mu)
    alpha = alpha0 / noisevar

    niter = 0
    t0 = time.time()
    while True:

        alpha_old = alpha
        noisevar_old = noisevar

        Cprior_inv = alpha * I

        # Mean and covariance of posterior
        try:
            S = linalg.inv(XTX / noisevar + Cprior_inv)
        except:
            S = pinvh(XTX / noisevar + Cprior_inv)

        mu = np.dot(S, XTy) / noisevar

        # Compute new parameters
        alpha = (p - alpha * np.trace(S)) / np.sum(mu ** 2)
        alpha = float(alpha)
        noisevar = np.sum((y - np.dot(X, mu)) ** 2) \
            / (N - np.sum(1 - alpha*np.diag(S)))

        dd = np.abs(alpha_old - alpha) + np.abs(noisevar_old - noisevar)
        if dd < tolerance or alpha > maxvalue or niter > maxiter:
            break

        niter += 1

        if verbose > 1:
            print("%d | alpha=%0.3f | noisevar=%0.3f | %g | %0.2f s" %\
                (niter, alpha, noisevar, dd, time.time() - t0))

    if verbose > 0:
        t_fit = time.time() - t0
        print("Ridge: finished after %d iterations (%0.2f s)" % (niter, t_fit))

    return mu, S, alpha, noisevar


def _ridge_evidence_fun_grad(theta, X, y, verbose, other):

    nv = theta[0]
    alpha = theta[1]

    XX, Xy, yy = other

    N, p = X.shape
    I = np.eye(p)

    # Prior covariance matrix for current parameter setting
    Cprior = 1./alpha * I
    Cprior[0, 0] = 0

    invCprior = alpha * I
    invCprior[0, 0] = 0

    # Posterior covariance and mean
    SS = linalg.pinv(XX/nv + invCprior)
    mu = np.dot(SS, Xy)/nv

    # (1) Compute log-evidence
    term1 = .5*(fast_logdet(2*np.pi*SS) - p*np.log(2*np.pi/alpha)
                - p*np.log(2*np.pi*nv))
    term2 = -.5*(yy/nv - np.dot(Xy.T, np.dot(SS, Xy))/nv**2)
    logE = term1 + term2

    # Gradient with respect to the ridge parameter
#    dAlpha = .5 * np.trace(1./alpha*I + SS + np.outer(mu, mu))
    dAlpha = p/(2*alpha) - .5*np.sum(mu*mu) - .5*np.trace(SS)

    # Gradient with respect to the noise variance
    SSinvC = np.dot(SS, invCprior)
    rss = yy - 2*np.dot(mu.T, Xy) + np.dot(mu.T, XX).dot(mu)
    dNsevar = -N/nv + np.trace(I - SSinvC)/nv + rss/nv**2

    dEE = np.array([dNsevar.item(), dAlpha])

    if verbose:
        print("-logE: %0.3f | nv: %0.3f | alpha: %0.3f" % (-logE, nv, alpha))

    return -logE, -dEE


def ridge_evidence_optim(X, y, init_params=None, verbose=False):

    if init_params is None:
        init_params = np.asarray([10., 10.])

    N, p = X.shape

    # Precompute covariances and noise variance of OLS solution
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    yy = np.sum(y * y)
    I = np.eye(p)

    w0 = linalg.solve(XX + 10*I, Xy, sym_pos=False)
    nv = np.sum((y - np.dot(X, w0))**2) / N
    init_params[0] = nv

    other = [XX, Xy, yy]

    # Call optimization routine
    fun = _ridge_evidence_fun_grad
    x0 = init_params
    fprime = None
    args = (X, y, verbose, other)
    bounds = [(1e-6, 1e6), (1e-6, 1e6)]

    print("x0:", x0)
    res = optimize.fmin_tnc(fun, x0, fprime=fprime, args=args,
                            approx_grad=0, bounds=bounds, epsilon=1e-08)

    opt_params = res[0]
    nv = opt_params[0]
    alpha = opt_params[1]

    if verbose:
        print("nv: %0.3f | alpha: %0.3f" % (nv, alpha))

    # Compute final quantities
    Cprior = 1./alpha * I
    invC = alpha * I
    SS = linalg.pinv(XX/nv + invC)
    mu = np.dot(SS,  Xy) / nv

    return mu, SS, nv, Cprior, alpha


class Ridge(LinearModel):

    def __init__(self, verbose=1, maxiter=1e3,
                 tolerance=1e-3, solver='iter', alpha0=1., **kwargs):

        super(Ridge, self).__init__(**kwargs)

        self.verbose = verbose
        self.maxiter = maxiter
        self.tolerance = tolerance
        self.solver = solver
        self.alpha0 = alpha0

        self.noisevar = None
        self.cov_prior = None
        self.alpha = None
        self.cov_posterior = None

    def fit(self, X, y):

        if y.ndim > 1 and y.shape[1] > 1:
            y = np.sum(y, axis=1)
            y = np.atleast_2d(y).T

        if y.ndim == 1:
            y = np.atleast_2d(y).T

        if self.fit_intercept:
            N = X.shape[0]
            column = np.ones((N, 1))
            X = np.concatenate((column, X), axis=1)

        if self.solver == 'iter':
            # iterative fixed-point method
            res = ridge_evidence_iter(X, y, maxiter=self.maxiter,
                                      tolerance=self.tolerance,
                                      verbose=self.verbose,
                                      alpha0=self.alpha0)
            mu, SS, alpha, nv = res
            CC = 1./alpha * np.eye(X.shape[1])
            CC[0, 0] = 0

            if self.verbose > 0:
                print("alpha:", alpha)

        elif self.solver == 'optim':
            # use gradient descent and scipy's optimization routines
            res = ridge_evidence_optim(X, y, init_params=[1., self.alpha0],
                                       verbose=self.verbose)
            mu, SS, nv, CC, alpha = res

        elif self.solver == 'map':
            # Maximum a posteriori estimate using alpha0
            XTX = np.dot(X.T, X)
            XTy = np.dot(X.T, y)
            yTy = np.sum(y * y)
#
            alpha = self.alpha0
            I = np.eye(X.shape[1])
            CC = alpha * I
            mu = linalg.solve(XTX + CC, XTy, sym_pos=False)

#            mu = np.dot(np.linalg.pinv(XTX), XTy)
            nv = yTy - 2*np.dot(mu.T, XTy) + np.dot(mu.T, XTX).dot(mu)
            SS = None

        if self.fit_intercept:
            self.coef_ = mu[1:].ravel()
            self.intercept_ = mu[0]

        else:
            self.coef_ = np.copy(mu.ravel())

        self.cov_prior = CC
        self.alpha = float(alpha)
        self.noisevar = nv
        self.cov_posterior = SS

    def predict(self, X):

        z = np.dot(X, self.coef_)
        if self.fit_intercept:
            z += self.intercept_

        return z


def _ridge_smooth_inverse(D, alphas, fit_intercept=True, as_matrix=False):

    wsize = D.shape[:2]
    nD = D.shape[2]
    invC = np.zeros(wsize)
    for j in range(nD):
        invC += alphas[j] * D[:, :, j]

    if fit_intercept:
        Cprior = np.zeros_like(invC)
        try:
            Cprior[1:, 1:] = linalg.inv(invC[1:, 1:])
        except:
            Cprior[1:, 1:] = linalg.pinv(invC[1:, 1:])
    else:
        Cprior = linalg.pinv(invC)

    if as_matrix:
        Cprior = np.matrix(Cprior, copy=False)
        invC = np.matrix(invC, copy=False)

    return Cprior, invC


def _ridge_smooth_fun_grad(theta, X, y, D, verbose, other):

    nv = theta[0]
    alphas = theta[1:]

    XX, Xy, yy, fit_intercept = other

    N, p = X.shape
    nD = D.shape[2]
    I = np.eye(p)

    # Prior covariance matrix for current parameter setting
    Cprior, invC = _ridge_smooth_inverse(D, alphas,
                                         fit_intercept=fit_intercept)

    # Posterior covariance and mean
    SS = linalg.pinv(XX/nv + invC)
    mu = np.dot(SS, Xy)/nv

    # Compute log-evidence
    term1 = .5*(fast_logdet(2*np.pi*SS) - p*np.log(2*np.pi*nv)
                - p*np.log(2*np.pi) - 1./fast_logdet(invC))
    term2 = -.5*(yy/nv - np.dot(Xy.T, np.dot(SS, Xy))/nv**2)
    logE = term1 + term2

    # Derivative with respect to covariance hyperparameters
    dAlphas = np.zeros((nD,))
    for i in range(nD):
        A = Cprior - SS - np.outer(mu, mu)
        dAlphas[i] = .5*np.trace(np.dot(A, D[:, :, i]))

    # Gradient with respect to the noise variance
    SSinvC = np.dot(SS, invC)
    rss = yy - 2*np.dot(mu.T, Xy) + np.dot(mu.T, XX).dot(mu)
    dNsevar = -N/nv + np.trace(I - SSinvC)/nv + rss/nv**2

    dEE = np.append(dNsevar.item(), dAlphas)

    if verbose:
        ss =("-logE: %0.3f | nv: %0.3f | alphas: (" % (-logE, nv))
        for alpha in alphas:
            ss += ("%0.3g, " % alpha)
        print(s[:-2] + ")")

    return -logE, -dEE


def ridge_smooth(X, y, D, init_params=None, verbose=False, fit_intercept=True,
                 optimizer='L-BFGS'):

    n_alphas = D.shape[2]
    if init_params is None:
        init_params = 10. * np.ones((1+n_alphas,))

    N, p = X.shape

    # Precompute covariances and noise variance of OLS solution
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    yy = np.sum(y * y)

    res = ridge_evidence_iter(X, y, penalize_bias=False, maxvalue=1e6,
                              maxiter=1e3, tolerance=1e-3, verbose=False,
                              alpha0=1.)
    mu, S, alpha, noisevar = res
    init_params[0] = noisevar

    other = [XX, Xy, yy, fit_intercept]

    # Call optimization routine
    fun = _ridge_smooth_fun_grad
    x0 = init_params
    fprime = None
    args = (X, y, D, verbose, other)
    bounds = (1+n_alphas) * [(1e-6, 1e6)]

    if optimizer == 'TNC':
        optim_fun = optimize.fmin_tnc
    elif optimizer == 'L-BFGS':
        optim_fun = optimize.fmin_l_bfgs_b

    res = optim_fun(fun, x0, fprime=fprime, args=args,
                    approx_grad=0, bounds=bounds, disp=0)

    opt_params = res[0]
    nv = opt_params[0]
    alphas = opt_params[1:]

    if verbose:
        ss = "  nv: %0.3f | alphas: (" % nv
        for alpha in alphas:
            ss += ("%0.3f, " % alpha)
        print(ss[:-1] + ")")

    # Prior covariance matrix for current parameter setting
    Cprior, invC = _ridge_smooth_inverse(D, alphas, fit_intercept=True)

    # Compute final quantities
    try:
        SS = linalg.inv(XX/nv + invC)
    except:
        SS = linalg.pinv(XX/nv + invC)
    mu = np.dot(SS,  Xy) / nv

    return mu, SS, nv, Cprior, alphas


class SmoothRidge(LinearModel):

    def __init__(self, D, verbose=True, zero_order=True,
                 first_order=True, second_order=False, equal_smoothing=False,
                 init_params=None, optimizer='L-BFGS', **kwargs):

        super(SmoothRidge, self).__init__(**kwargs)

        if isinstance(D, (tuple, list)) and len(D) == 2:
            nx, ny = D
            D = create_derivative_matrix(nx, ny, int(self.fit_intercept),
                                         zero_order=zero_order,
                                         first_order=first_order,
                                         second_order=second_order,
                                         equal_smoothing=equal_smoothing)

        elif isinstance(D, np.ndarray):
            nx, ny = D.shape[:2]

        else:
            raise ValueError('Argument D must be either a tuple of length'
                             ' 2 or a derivative matrix')

        self.verbose = verbose
        self.init_params = init_params
        self.optimizer = optimizer

        self.wsize = (nx, ny)
        self.D = D

        self.cov_posterior = None
        self.cov_prior = None
        self.scale_param = None
        self.smooth_params = None
        self.noisevar = None
        self.alphas = None

    def fit(self, X, y, append_ones=True):

        if y.ndim > 1 and y.shape[1] > 1:
            y = np.sum(y, axis=1)
            y = np.atleast_2d(y).T

        if y.ndim == 1:
            y = np.atleast_2d(y).T

        if self.fit_intercept and append_ones:
            N = X.shape[0]
            column = np.ones((N, 1))
            X = np.concatenate((column, X), axis=1)

        D = self.D

        init_params = self.init_params
        if init_params is None:
            n_alphas = D.shape[2]
            init_params = np.ones((1+n_alphas,))

        res = ridge_smooth(X, y, D, init_params=init_params,
                           verbose=self.verbose, optimizer=self.optimizer)
        mu, SS, nv, Cprior, alphas = res

        if self.fit_intercept:
            self.coef_ = mu[1:].flatten()
            self.intercept_ = mu[0]

        else:
            self.coef_ = np.copy(mu.ravel())

        self.cov_prior = Cprior
        self.cov_posterior = SS
        self.alphas = alphas
        self.noisevar = nv

    def predict(self, X):

        z = np.dot(X, self.coef_)
        if self.fit_intercept:
            z += self.intercept_

        return z

