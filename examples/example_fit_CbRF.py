#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

import numpy as np
import matplotlib.pyplot as plt
from os.path import split, join
import sys
sys.path.append(join(split(__file__)[0], '..'))

from lnpy.lnp.tools import createRF, SimpleCell
from lnpy.learn.base import create_derivative_matrix
from lnpy.learn import SmoothnessPrior
from lnpy.lnp.cbrf import CbRF


def create_toy_data(N=1000, n_trials=1):

    # Create Gabor kernel
    rfsize = (15, 15)
    K_true = createRF(name='gabor', size=rfsize, threshold=0.2,
                      dtype=np.float64, frequency=0.5, sigma=2*[0.35])
    K_true[K_true < 0] *= 1.25

    # GWN Stimulus
    X = np.random.randn(N, rfsize[0] * rfsize[1])

    # Simulate response of LNP model
    cell = SimpleCell(K_true, threshold=1.5, stddev=0.5, n_trials=n_trials,
                      rectify=True)
    Y = cell.simulate(X).astype(np.float64)
    print "%d spikes (p(spike) = %0.3f)" % (np.sum(Y > 0),
                                            np.mean(Y > 0))

    return X, Y, K_true


def run_example():

    X, Y, K_true = create_toy_data(N=2500)

    nt, nf = K_true.shape
    D = create_derivative_matrix(nt, nf, order='C')
    prior = SmoothnessPrior(D=D)
    model = CbRF(optimize=True, metric='AUC', prior=prior,
                 verbose=True, n_griditer=3, n_jobs=1)
    model.fit(X, Y)
    k = model.get_weights()
    K = np.reshape(k, K_true.shape)

    fig, axarr = plt.subplots(nrows=1, ncols=2)

    ax = axarr[0]
    ax.set_title('True')
    vmax = np.max(np.abs(K_true))
    ax.imshow(K_true, interpolation='nearest', vmin=-vmax, vmax=vmax)

    ax = axarr[1]
    ax.set_title('CbRF')
    vmax = np.max(np.abs(K))
    ax.imshow(K, interpolation='nearest', vmin=-vmax, vmax=vmax)

    plt.show()


if __name__ == '__main__':
    run_example()
