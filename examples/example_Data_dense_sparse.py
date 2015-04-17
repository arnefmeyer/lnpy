#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""

Demonstrate usage of dense and sparse data handling for C-based classes
wrapped using Cython

"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import split, join
import sys
sys.path.append(join(split(__file__)[0], '..'))

from lnpy.lnp.tools import SimpleCell, vdotNormal
from lnpy.learn import pyhelper

from ipdb import set_trace as db


def vecnorm(x):

    return np.sqrt(np.sum(x * x))


def create_toy_data(N=10, ndim=2):

    k_true = np.zeros((ndim, 1))
    k_true[::2] = -1
    k_true[1::2] = 1

    X = np.random.randn(N, ndim)
    cell = SimpleCell(k_true, threshold=.75, stddev=0.1, n_trials=1,
                      rectify=True, dtype=np.float64, seed=None)
    y = cell.simulate(X)

    return X, y, k_true


def run_example():

    X, y, k_true = create_toy_data(N=100, ndim=10)
    print np.mean(y)

    # Create dense and sparse data sets
    bias = 1.
    dense = pyhelper.PyDenseProblem(X, y.ravel(), bias=bias, C=None)
    sparse = pyhelper.PySparseProblem(X, y.ravel(), bias=bias, C=None)

    prior = pyhelper.PyGaussianPrior()
    model = pyhelper.PySVM(prior=prior)

    ws = np.zeros((k_true.shape[0] + 1,))
    model.fit(sparse, ws, tolerance=1e-2, max_iter=100, verbose=0)
    print "sparse: cc = %0.2f" % vdotNormal(k_true, ws[:-1])
    print ws

    wd = np.zeros((k_true.shape[0] + 1,))
    model.fit(dense, wd, tolerance=1e-2, max_iter=100, verbose=0)
    print "dense: cc = %0.2f" % vdotNormal(k_true, wd[:-1])
    print wd

    plt.plot(k_true / vecnorm(k_true), 'k', label='true')
    plt.plot(ws[:-1] / vecnorm(ws[:-1]), 'r', label='sparse')
    plt.plot(wd[:-1] / vecnorm(wd[:-1]), 'b', label='dense')
    plt.xlabel('Time index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_example()
