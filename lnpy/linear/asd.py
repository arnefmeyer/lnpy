#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Automatic smoothness determination (Sahani & Linden NIPS 2003a). Some of
    this code is based on Maneesh's matlab implementation.
"""

import numpy as np
from scipy import linalg, optimize
from sklearn.utils import extmath
import sys

from .base import LinearModel
from .prior import create_distance_matrix
from .ard import lrard


class Matrix(np.matrix):
    """Extension of numpy's matrix class that adds the Schur matrix product"""

    def schur(self, X):
        """Element-wise matrix (Schur) product"""

        # Call numpy's element-wise multiplication function
        return np.multiply(self, X)


def lrasd(X, y, D, init_params=[.1, .1, .1], maxiter=100, step_size=0.01,
          smooth_min=.01, tolerance=.1, XX=None, Xy=None, verbose=True):

    # Setting copy=False creates a new view to the same data in memory
    X = Matrix(X, copy=False)
    if y.ndim == 1:
        y = Matrix(y, copy=False).T
    else:
        y = Matrix(y, copy=False)

    if XX is None:
        XX = X.T * X
    if Xy is None:
        Xy = X.T * y

    N, d = X.shape

    # OLS solution; x ** 2 denotes the matrix exponential. However, we
    # can simply use the array view by calling matrix.A, which supports
    # element-wise exponentiation via x ** n.
    nv = np.sum((y - X * np.linalg.lstsq(X, y)[0]).A ** 2) / N
    if np.isnan(nv) or np.isinf(nv):
        II = 1 * np.diag(np.ones((XX.shape[0],)))
        nv = np.sum((y - X * linalg.solve(XX + II, Xy)).A ** 2) / N

    rr = np.array(init_params[0])
    ss = np.array(init_params[1:])

    nD = D.shape[-1]
    I = np.eye(d)
    dEEm = None
    for i in range(maxiter):

        if nv < 1e-18:
            # for numerical stability
            nv = 1e-12

        if verbose:
            print "%d: r=%.4g | s=(%.4g, %.4g) | nv=%0.4f" %\
                (i+1, rr, ss[0], ss[1], nv)
            sys.stdout.flush()

        # Compute covariance matrix; Numpy's nice broadcasting rules make
        # this quite simple (automatic broadcasting along last dimension)
        Dw = np.sum(D / ss ** 2, axis=2)
        CC = np.exp(-rr - 0.5 * Matrix(Dw))

        AASw = linalg.inv(XX * CC / nv + I)
        muw = CC * AASw * Xy / nv
        QQ = I - AASw - (AASw * Xy) / nv * muw.T
        QQAA = QQ * linalg.pinv(CC)

        dEE = np.zeros((1 + nD,))
        dEE[0] = .5*np.trace(QQ)
        for j in range(nD):
            # schur performs element-wise multiplication (see class Matrix)
            dEE[1+j] = -.5*np.trace(CC.schur(D[:, :, j]) * QQAA) / ss[j]**3

        mom = 0.99
        if dEEm is not None:
            if np.vdot(dEE, dEEm) < 0:
                step_size = 0.8 * step_size
            dEEm = dEE + (dEE*dEEm > 0) * (mom*dEEm)
        else:
            dEEm = dEE

        rr = rr + step_size * dEEm[0]
        ss = ss + step_size * dEEm[1:]

        ss[ss < smooth_min] = smooth_min
        rr = max(rr, -20)
        rr = min(rr, 20)

        nv = np.sum((y - X * muw).A ** 2)/(N - np.trace(I - AASw))
        nv = max(nv, 1e-6)

        if np.all(np.abs(dEE) < tolerance):
            break

    # Compute covariance matrix
    Dw = np.sum(D / ss ** 2, axis=2)
    CC = np.exp(-rr - 0.5 * Matrix(Dw))

    AASw = linalg.pinv(XX * CC / nv + I)
    Sw_out = CC * AASw
    ww_out = Sw_out * Xy / nv
    ss_out = np.append(rr, ss)

    return np.asarray(ww_out), Sw_out, nv, CC, ss_out


def _asd_fun_grad(theta, X, y, D, verbose, other):

    nv = theta[0]
    rr = theta[1]
    ss = theta[2:]

    XX, Xy, yy = other

    N, p = X.shape
    nD = D.shape[2]
    wsize = D.shape[:2]
    I = np.eye(p)

    # Prior covariance matrix for current parameter setting
    DD = np.zeros(wsize)
    for j in range(nD):
        DD += D[:, :, j] / ss[j] ** 2
    Cprior = np.exp(- rr/100. - 0.5 * DD)  # equation 10 in paper

    # Some useful quantities we will need later on
    CXX = np.dot(Cprior, XX)/nv
    CXy = np.dot(Cprior, Xy)
    try:
        SSinvC = linalg.inv(CXX + I)  # AASw
    except:
        SSinvC = linalg.pinv(CXX + I)

    SSinvCXy = np.dot(SSinvC, CXy)

    # (1) Compute log-evidence
    term1 = -.5*(extmath.fast_logdet(CXX+I) + p*np.log(2*np.pi*nv))
    term2 = -.5*(yy/nv - np.dot(Xy.T, SSinvCXy)/nv**2)
    logE = term1 + term2

    # (2) Compute posterior covariance
    SS = np.dot(SSinvC, Cprior)

    # (3) Compute posterior mean
    mu = np.dot(SS, Xy)/nv

    # (4) Compute gradient info
    AA = I - SSinvC - np.dot(SSinvC, Xy/nv).dot(mu.T)
    invCprior = linalg.pinv(Cprior)
    AAinvC = np.dot(AA, invCprior)

    # Gradient with respect to the scale parameter
    dr = .5 * np.trace(AA) / 100.

    # Gradient with respect to the smoothness parameter
    dSmooth = np.zeros((nD,))
    for j in range(nD):
        U = np.dot(Cprior * D[:, :, j], AAinvC)
        dSmooth[j] = -.5*np.trace(U) / ss[j]**3

    # Gradient with respect to the noise variance
    rss = yy - 2*np.dot(mu.T, Xy) + np.dot(mu.T, XX).dot(mu)
    dNsevar = -N/nv + np.trace(I - SSinvC)/nv + rss/nv**2

    dEE = np.array([dNsevar.item(), dr, dSmooth[0], dSmooth[1]])

    if verbose:
        print "-logE: %0.3f | nv: %0.3f | r: %0.3f | s: (%0.3f, %0.3f)" %\
            (-logE, nv, rr, ss[0], ss[1])
        sys.stdout.flush()

    return -logE, -dEE


def calc_ASD_optim(X, y, D, init_params=None, verbose=False,
                   optimizer='L-BFGS', maxiter=None):

    if init_params is None:
        init_params = np.asarray([7., 2., 2., 1.])

    N, p = X.shape

    # Precompute covariances and noise variance of OLS solution
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    yy = np.sum(y * y)
    I = np.eye(p)
    alpha = 0.

#    w_ols = linalg.lstsq(X, y)[0]
    w0 = linalg.solve(XX + alpha*I, Xy, sym_pos=False)
    nv = np.sum((y - np.dot(X, w0))**2) / X.shape[0]
    init_params[0] = nv

    other = [XX, Xy, yy]

    # Call optimization routine
    fun = _asd_fun_grad
    x0 = init_params   # noisevar, rho, delta_s, delta_t
    fprime = None
    args = (X, y, D, verbose, other)
    bounds = [(1e-6, 1e6), (-2000, 2000), (1e-6, 1e6), (1e-6, 1e6)]

    if optimizer == 'L-BFGS':
        optimizer = optimize.fmin_l_bfgs_b

    elif optimizer == 'TNC':
        optimizer = optimize.fmin_tnc

    res = optimizer(fun, x0, fprime=fprime, args=args, approx_grad=0,
                    bounds=bounds, disp=0)

    opt_params = res[0]
    nv = opt_params[0]
    rr = opt_params[1]
    ss = opt_params[2:]

    if verbose:
        print "nv: %0.3f | r: %0.3f | sx: %0.3f | sy: %0.3f" %\
            (nv, rr, ss[0], ss[1])

    # Prior covariance matrix for current parameter setting
    wsize = D.shape[:2]
    DD = np.zeros(wsize)
    nD = D.shape[2]
    for j in range(nD):
        DD += D[:, :, j] / ss[j] ** 2
    CC = np.exp(-rr - 0.5 * DD)  # equation 10 in paper

    # Compute final quantities
    I = np.eye(CC.shape[0])
    SSinvC = linalg.inv(np.dot(CC, XX) / nv + I)
    SS = np.dot(SSinvC, CC)
    mu = np.dot(SS,  Xy) / nv
    ss_out = np.append(rr, ss)

    return mu, SS, nv, CC, ss_out


def calc_ASD_fixed_params(X, y, noisevar, D, rr, ss, XX=None, Xy=None):

    N = X.shape[0]
    I = np.eye(X.shape[1])

    Dw = np.sum(D / ss ** 2, axis=2)
    CC = np.exp(-rr - 0.5 * Dw)

    if XX is None:
        XX = np.dot(X.T, X)

    if Xy is None:
        Xy = np.dot(X.T, y)

    AASw = linalg.inv(np.dot(XX, CC) / noisevar + np.eye(CC.shape[0]))
    Sw_out = np.dot(CC, AASw)
    w = np.dot(Sw_out,  Xy) / noisevar
    noisevar = np.sum((y - np.dot(X, w)) ** 2)/(N - np.trace(I - AASw))

    return w.ravel(), noisevar


class ASD(LinearModel):

    def __init__(self, D=(15, 15), verbose=True, maxiter=100,
                 stepsize=0.01, solver='iter', optimizer='iter',
                 smooth_min=.01, init_params=[6, 2, 2], init_coef=None,
                 init_intercept=None, tolerance=0.1, **kwargs):

        super(ASD, self).__init__(**kwargs)

        if isinstance(D, tuple) and len(D) == 2:
            nx, ny = D
            distance_matrix = create_distance_matrix(nx, ny,
                                                     int(self.fit_intercept))

        elif isinstance(D, np.ndarray):
            distance_matrix = D
            nx, ny = D.shape[:2]

        else:
            raise ValueError('Argument D must be either a tuple of length'
                             ' 2 or a distance matrix')

        D = distance_matrix

        self.verbose = verbose
        self.maxiter = maxiter
        self.stepsize = stepsize
        self.solver = solver
        self.optimizer = optimizer
        self.smooth_min = smooth_min
        self.init_params = init_params
        self.tolerance = tolerance

        self.wsize = (nx, ny)
        self.D = D

        if init_coef is not None:
            self.coef_ = init_coef
        else:
            self.coef_ = np.zeros((nx*ny,))

        if init_intercept is not None:
            self.intercept_ = init_intercept
        else:
            self.intercept_ = 0.

        self.cov_posterior = None
        self.cov_prior = None
        self.scale_param = None
        self.smooth_params = None
        self.noisevar = None

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

        init_params = self.init_params
        if init_params is None:
            init_params = np.asarray([6, 2, 2])

        D = self.D
        if self.solver == 'iter':

            res = lrasd(X, y, D, init_params=init_params,
                        maxiter=self.maxiter, step_size=self.stepsize,
                        verbose=self.verbose, smooth_min=self.smooth_min,
                        tolerance=self.tolerance)
            mu, Sw, nv, Cprior, ss = res
            rr = ss[0]
            ss = ss[1:]
            ww = mu.ravel()

        elif self.solver == 'optim':

            if len(init_params) < 4:
                # prepend noise variance initial parameter
                init_params = np.append(1., init_params)

            res = calc_ASD_optim(X, y, D, init_params=init_params,
                                 verbose=self.verbose, maxiter=self.maxiter,
                                 optimizer=self.optimizer)
            mu, Sw, nv, Cprior, ss = res
            rr = ss[0]
            ss = ss[1:]
            ww = mu.ravel()

        elif self.solver == 'fixed':

            nv = self.noisevar
            D = self.D
            rr = self.scale_param
            ss = self.smooth_params

            if nv is None:
                nv = np.mean((y - X .dot(np.linalg.lstsq(X, y)[0])) ** 2)
            if rr is None:
                rr = init_params[0]
            if ss is None:
                ss = init_params[1:3]

            mu, nv = calc_ASD_fixed_params(X, y, nv, D, rr, ss)
            ww = mu.ravel()

            Cprior = self.cov_prior
            Sw = self.cov_posterior

        if self.fit_intercept:
            self.coef_ = ww[1:]
            self.intercept_ = ww[0]

        else:
            self.coef_ = np.copy(ww)
            self.intercept_ = 0.

        self.cov_prior = Cprior
        self.cov_posterior = Sw
        self.scale_param = rr
        self.smooth_params = ss
        self.noisevar = nv

    def predict(self, X):

        z = np.dot(X, self.coef_)
        if self.fit_intercept:
            z += self.intercept_

        return z

    def show(self, *args, **kwargs):

        fig = super(ASD, self).show(self.wsize, **kwargs)

        return fig


class ASDRD(LinearModel):

    def __init__(self, D=(15, 15), verbose=True,
                 threshold=1e7, tolerance=1e-3, init_params=None,
                 maxiter_ard=1500, **kwargs):

        super(ASDRD, self).__init__(**kwargs)

        if isinstance(D, tuple) and len(D) == 2:
            nx, ny = D
            D = create_distance_matrix(nx, ny, int(self.fit_intercept))

        elif isinstance(D, np.ndarray):
            ny, nx = D.shape

        else:
            raise ValueError('Argument D must be either a tuple of length'
                             ' 2 or a distance matrix')

        self.verbose = verbose
        self.threshold = threshold
        self.tolerance = tolerance
        self.maxiter_ard = maxiter_ard
        self.init_params = init_params

        self.wsize = (nx, ny)
        self.D = D

        self.cov_posterior = None
        self.cov_prior = None
        self.scale_param = None
        self.smooth_params = None
        self.noisevar = None

    def fit(self, X, y):

        if y.ndim == 1:
            y = np.atleast_2d(y).T

        if self.fit_intercept:
            N = X.shape[0]
            column = np.ones((N, 1))
            X = np.concatenate((column, X), axis=1)

        D = self.D
        model = ASD(D, fit_intercept=False, verbose=self.verbose,
                    **self._kwargs)
        model.fit(X, y)
        Cprior = model.cov_prior
        scale_param = model.scale_param
        smooth_params = model.smooth_params
        noisevar = model.noisevar

        R = linalg.sqrtm(Cprior)  # basis
        R = np.real(R)

        XR = np.dot(X, R)

        res_ARD = lrard(XR, y, threshold=self.threshold, tol=self.tolerance,
                        verbose=self.verbose, maxiter=self.maxiter_ard)
        ww = res_ARD[0]

        w = np.dot(R, ww)

        if self.fit_intercept:
            self.coef_ = w[1:]
            self.intercept_ = w[0]

        else:
            self.coef_ = np.copy(w)

        self.cov_prior = Cprior
        self.cov_posterior = model.cov_posterior
        self.scale_param = scale_param
        self.smooth_params = smooth_params
        self.noisevar = noisevar

    def predict(self, X):

        z = np.dot(X, self.coef_)
        if self.fit_intercept:
            z += self.intercept_

        return z
