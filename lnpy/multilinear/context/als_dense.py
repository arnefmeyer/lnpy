#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Context model implementation for dense stimuli. To speed up everything most
    of the computations are done in a separate cython file (~C speed)

    Note that this implementation might differ slightly from Misha's and Ross'
    Matlab implementations.
"""

from __future__ import print_function

import numpy as np
import sys

from ...linear import ASD

from . import context_fast as ctxtools


def fit_context_model(S, Y, J, K, M, N, reg_iter=3, max_iter=100,
                      c2=1., tol=1e-5, wrap_around=True, solver='iter',
                      smooth_min=1e-3, init_params_cgf=[6., 2., 2.],
                      init_params_prf=[7, 4, 4]):

    assert isinstance(S, (np.ndarray, list)), \
        "stimulus S must be numpy array or list or numpy arrays"
    assert isinstance(Y, (np.ndarray, list)), \
        "response y must be numpy array or list or numpy arrays"

    assert type(S) == type(Y), "S and y must be of the same type"

    if isinstance(S, np.ndarray):
        # for now let's put everything in lists to make things easier
        S = [S]
        Y = [Y]

    S_pad = []
    for s in S:
        # Pad zeros around stimulus to simplify subsequent computations
        s_pad = pad_stimulus(s, J, K, M, N, wrap_around=wrap_around)
        S_pad.append(s_pad)

    # Initialize context parameters using STRF estimate
    model_strf = ASD(D=(J, K), fit_intercept=True, verbose=True, maxiter=100,
                     stepsize=0.01, solver=solver, init_params=init_params_prf,
                     smooth_min=smooth_min, tolerance=0.1)

    SS = []
    YY = []
    for s, y in zip(S, Y):
        SS.append(segment_spectrogram(s, J, order='C', prepend_zeros=False))
        YY.append(y[J-1:])

    SS = np.concatenate(SS)
    YY = np.concatenate(YY)
    model_strf.fit(SS, YY)

    init_params_prf = np.append(model_strf.scale_param,
                                model_strf.smooth_params)

    init_params_cgf = np.asarray(init_params_cgf)

    # Create PRF and CGF models
    model_prf = ASD(D=(J, K), fit_intercept=True, verbose=True, maxiter=100,
                    stepsize=0.01, solver=solver, init_params=init_params_prf,
                    init_coef=1e-3 * np.ones((J*K)), init_intercept=0.1,
                    smooth_min=smooth_min, tolerance=0.1)
    model_prf.noisevar = model_strf.noisevar

    model_cgf = ASD(D=(M+1, 2*N+1), fit_intercept=False, verbose=True,
                    maxiter=100,
                    stepsize=0.01, solver=solver, init_params=init_params_cgf,
                    init_coef=-1e-3 * np.ones(((M+1)*(2*N+1),)),
                    init_intercept=1., smooth_min=smooth_min, tolerance=0.1)

    models = [model_prf, model_cgf]

    T = y.shape[0]
    y = y.flatten()

    mse_before = 1e12
    for i in range(max_iter):

        print("iter {}/{}".format(i+1, max_iter))
        sys.stdout.flush()

        for j, model in enumerate(models):

            print("  step %d/%d" % (j+1, len(models)))
            sys.stdout.flush()

            XX = []
            Y_hat = []
            for s, y in zip(S_pad, Y):

                if j == 0:
                    X, y_hat = compute_A_matrix(s, y, models[1], J, K, M, N,
                                                c2)
                else:
                    X, y_hat = compute_B_matrix(s, y, models[0], J, K, M, N)

                XX.append(X)
                Y_hat.append(y_hat)

            XX = np.concatenate(XX)
            Y_hat = np.concatenate(Y_hat)
            run_als_update(XX, Y_hat, model, regularize=i < reg_iter)

        Y_pred = []
        for s in S_pad:
            y_pred = predict_response_context(s, model_prf, model_cgf,
                                              T, J, K, M, N, c2,
                                              pad_zeros=False)
            Y_pred.append(y_pred)

        Y_pred = np.concatenate(Y_pred)

        mse = np.mean((y - y_pred)**2)
        print("  mean squared error: %g" % mse)
        sys.stdout.flush()

        # Check termination conditionl
        rel_err = np.abs(mse_before - mse) / mse
        if i >= reg_iter and rel_err <= tol:
            print("Relative error (%g) smaller than tolerance (%g)." \
                  "Exiting" % (rel_err, tol))
            break

#        elif mse > mse_before:
#            print "Error increased. Exiting."
#            break

        mse_before = mse

    model_cgf.coef_[N] = 0

    all_models = [model_strf]
    all_models.extend(models)

    return all_models


def pad_stimulus(S, J, K, M, N, wrap_around=True):

    T = S.shape[0]
    pad_len = J-1+M
    S_pad = np.zeros((pad_len+T, K+2*N))
    S_pad[pad_len:, N:-N] = S

    if wrap_around:
        S_pad[:pad_len, N:-N] = S[-pad_len:, :]

    return S_pad


def segment_spectrogram(S, lag, order='C', prepend_zeros=False,
                        flip_temp=False, add_ones=False):
    """create lag vectors from spectrogram"""

    nf = S.shape[1]
    nt = lag
    p = nt * nf

    if prepend_zeros:
        S = np.vstack((np.zeros((lag-1, nf)), S))

    if flip_temp:
        f = lambda x: np.flipud(x)
    else:
        f = lambda x: x

    N = S.shape[0] - nt + 1
    X = np.ones((N, p + int(add_ones)))
    for i in range(N):
        xx = S[i:i+nt, :p]
        X[i, :p] = f(xx).ravel(order=order)

    return X


def compute_A_matrix(S, y, model, J, K, M, N, c2=1.):
    """
        Calculate A matrix according to:

            A_ijk = s(i-j+1,k) + s(i-j+1,k)*sum_mn wcgf_mn s(i-j+1-m,k+n)

    """

    w_cgf = model.coef_

    T = y.shape[0]
    A = np.zeros((T, J*K))
    ctxtools.compute_A_matrix(S, w_cgf, A, T, J, K, M, N, c2=c2)

    return A, y


def compute_B_matrix(S, y, model, J, K, M, N):

    w_prf = model.coef_
    c1 = model.intercept_

    T = y.shape[0]

    y_prf = np.zeros_like(y)
    ctxtools.predict_y_prf(S, w_prf, y_prf, T, J, K, M, N)
    y_cgf = y - y_prf - c1

    B = np.zeros((T, (M+1)*(2*N+1)))
    ctxtools.compute_B_matrix(S, w_prf, B, T, J, K, M, N)

    return B, y_cgf


def run_als_update(S, y, model, regularize=False):

    solver = 'iter'
    if not regularize:
        solver = 'fixed'
    model.solver = solver

    model.fit(S, y)

    model.init_params = np.append(model.scale_param, model.smooth_params)


def predict_response_context(S, model_prf, model_cgf, T, J, K, M, N,
                             c2=1., pad_zeros=True, wrap_around=True):

    if pad_zeros:
        S_pad = pad_stimulus(S, J, K, M, N, wrap_around=wrap_around)
    else:
        S_pad = S

    if isinstance(model_prf, ASD):
        w_prf = model_prf.coef_
        c1 = model_prf.intercept_
    else:
        w_prf = model_prf[1:]
        c1 = model_prf[0]

    if isinstance(model_cgf, ASD):
        w_cgf = model_cgf.coef_
        c3 = model_cgf.intercept_
    else:
        w_cgf = model_cgf[1:]
        c3 = model_cgf[0]

    w_cgf[N] = 0

    y_pred = np.zeros((T,))
    ctxtools.predict_y_context(S_pad, w_prf, w_cgf, y_pred, c1, c2, c3, T,
                               J, K, M, N)

    return y_pred
