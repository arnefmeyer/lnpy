#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Fitting of context model using sparse (Matlab) and dense (Python)
    implementations of the alternating least squares (ALS) algorithm
"""

import numpy as np
from os.path import split, join
import matplotlib.pyplot as plt

from lnpy.multilinear.context import (
    ContextModel, plot_context_model, create_toy_data,
    predict_STRF_from_PRF_and_CGF
)


def run_toy_example():

    T = 1000  # number of observations
    J = 9  # time lag STRF/PRF
    K = 9  # number of frequency channels
    M = 5  # time lag CGF
    N = 2  # frequency lag CGF

    max_iter = 100  # max. number of ALS iterations
    reg_iter = 0  # stop hyperparameter optimization after reg_iter iterations

    X, _, y, w_prf, w_cgf = create_toy_data(T=T, J=J, K=K, M=M, N=N,
                                            c1=1., noisevar=.05)

    fig = plot_context_model(np.zeros_like(w_prf), w_prf, w_cgf, J, M, N,
                             dt=0.02, cmap='RdBu_r', windowtitle='True')
    fig.tight_layout()

    model = ContextModel(J=J, M=M, N=N, algorithm='als_dense',
                         max_iter=max_iter, reg_iter=reg_iter)
#    model.fit(X, y)
    model.fit([X, X], [y, y])
    model.show(show_now=False)
    y_pred = model.predict(X)

    print "MSE =", np.mean((y - y_pred)**2)
    print "var(y) =", np.var(y)

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(y_pred, 'k-', label='pred')
    ax.plot(y, 'r-', label='true')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    w_prf = model.w_prf
    w_cgf = model.w_cgf
    c1 = model.b_prf
    b_strf, w_strf = predict_STRF_from_PRF_and_CGF(X, w_prf, w_cgf, c1,
                                                   J, M, N)

    vmax = np.max(np.abs(w_strf))
    ax.imshow(w_strf[::-1, :].T, interpolation='nearest', aspect='auto',
              vmin=-vmax, vmax=vmax, cmap='RdBu_r', origin='lower')
    fig.tight_layout()

    plt.show()


def run_recording_example():

    fit_linear_model = True

    dt = 0.02
    max_iter = 50
    reg_iter = 0

    J = 15
    M = 12
    N = 12

    data_dir = join(split(__file__)[0], 'data')
    mat_file = join(data_dir, 'm598_X_Y.mat')
    data = loadmat(mat_file)
    X = np.ascontiguousarray(data['X'], dtype=np.float64)
    y = np.ascontiguousarray(data['Y'], dtype=np.float64).ravel()

    if np.max(X) > 1:
        # amplitude level with respect to 1mw  (dBm); used for transforming
        # dB-scaled data to linear scale; max(X) <= 1 assumes that the data
        # are already scaled linearly.
        ind = X > 0
        X[ind] = 10. ** ((X[ind] - np.max(X[ind]))/20.)

    if fit_linear_model:
        rfsize = (J, X.shape[1])
        XX = segment_spectrogram(X, J, order='C', prepend_zeros=True)

        model = ASD(D=rfsize, fit_bias=True, verbose=True,
                    maxiter=100, stepsize=0.01, solver='iter',
                    optimizer='L-BFGS', smooth_min=.5, init_params=[7, 4, 4],
                    tolerance=0.1)
        model.fit(XX, y)
        model.show(dt=dt, show_now=False)

    for algo in ['matlab', 'dense']:
        model = ContextModel(J=J, M=M, N=N, algorithm='als_' + algo,
                             max_iter=max_iter, reg_iter=reg_iter,
                             als_solver='iter', tolerance=1e-5)
        model.fit(X, y)
        model.show(dt=dt, show_now=False, windowtitle=algo)

    plt.show()


if __name__ == '__main__':

    run_toy_example()
#    run_recording_example()
