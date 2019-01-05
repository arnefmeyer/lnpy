#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    NOT WORKING YET!
"""

from __future__ import print_function

import numpy as np


def fit_context_model(S, y, J, K, M, N, reg_iter=3, max_iter=100,
                      c2=1., tol=1e-5, wrap_around=True):

    # Pad zeros around stimulus to simplify subsequent computations
    S_pad = pad_stimulus(S, J, K, M, N, wrap_around=wrap_around)

    mse_before = 1e12
    for i in range(max_iter):

        print("iter {}/{}".format(i+1, max_iter))

        for j, model in enumerate(models):

            print("  step {}/{}".format(j+1, len(models)))
            sys.stdout.flush()

            if j == 0:
                X, y_hat = compute_A_matrix(S_pad, y, models[1], J, K, M, N,
                                            c2)
            else:
                X, y_hat = compute_B_matrix(S_pad, y, models[0], J, K, M, N)

            run_als_update(X, y_hat, model, regularize=i < reg_iter)

        y_pred = predict_response_context(S_pad, model_prf, model_cgf,
                                          T, J, K, M, N, c2, pad_zeros=False)

        mse = np.mean((y - y_pred)**2)
        print("  mean squared error: %g" % mse)

        # Check termination conditionl
        rel_err = np.abs(mse_before - mse) / mse
        if i >= reg_iter and rel_err <= tol:
            print("Relative error (%g) smaller than tolerance (%g)." \
                  "Exiting" % (rel_err, tol))
            break

        mse_before = mse


def pad_stimulus(S, J, K, M, N, wrap_around=True):

    T = S.shape[0]
    pad_len = J-1+M
    S_pad = np.zeros((pad_len+T, K+2*N))
    S_pad[pad_len:, N:-N] = S

    if wrap_around:
        S_pad[:pad_len, N:-N] = S[-pad_len:, :]

    return S_pad
