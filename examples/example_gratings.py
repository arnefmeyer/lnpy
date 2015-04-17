#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    This example demonstrates robustness of the CbRF method to higher-order
    correlations in the stimulus ensemble (cf. Fig 5 in Meyer et al. PLOS ONE
    2014).
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import split, join
import sys
sys.path.append(join(split(__file__)[0], '..'))

from lnpy.lnp.tools import createGratings, createRF
from lnpy.learn import GaussianPrior
from lnpy.lnp.cbrf import CbRF
from lnpy.lnp.sta import STA


def create_toy_data(N=1000, nonlin_order=3):

    # Create Gabor kernel
    rfsize = (25, 25)
    K_true = createRF(name='gabor', size=rfsize, threshold=0.2,
                      dtype=np.float64, frequency=0.5, sigma=2*[0.35])

    # GWN Stimulus)
    X = createGratings(size=rfsize, N=N, center=True, whiten=True)

    # Poisson spike generation
    z = np.dot(X, K_true.ravel())
    z[z < 0] = 0
    z = z ** nonlin_order
    z /= z.max()
    Y = (z > np.random.rand(N)).astype(np.float64)
    print "%d spikes (p(spike) = %0.3f)" % (np.sum(Y > 0),
                                            np.mean(Y > 0))

    return X, Y, K_true


def run_example():

    X, Y, K_true = create_toy_data(N=25000, nonlin_order=3)

    # Estimate STA
    model = STA()
    print "Estimating STA"
    model.fit(X, Y)
    k_sta = model.get_weights()
    K_sta = np.reshape(k_sta, K_true.shape)

    # Fit CbRF parameters
    prior = GaussianPrior()
    model = CbRF(optimize=True, metric='AUC', prior=prior,
                 verbose=True, n_griditer=3, n_jobs=-1,
                 param_grid={'alpha': 2**np.linspace(-20, 0, 7)})
    print "This may take a couple of minutes. Time to grab a tea or coffee ..."
    model.fit(X, Y)
    k_cbrf = model.get_weights()
    K_cbrf = np.reshape(k_cbrf, K_true.shape)

    fig, axarr = plt.subplots(nrows=1, ncols=3)

    ax = axarr[0]
    ax.set_title('True')
    vmax = np.max(np.abs(K_true))
    ax.imshow(K_true, interpolation='nearest', vmin=-vmax, vmax=vmax)

    ax = axarr[1]
    ax.set_title('STA')
    vmax = np.max(np.abs(K_sta))
    ax.imshow(K_sta, interpolation='nearest', vmin=-vmax, vmax=vmax)

    ax = axarr[2]
    ax.set_title('CbRF')
    vmax = np.max(np.abs(K_cbrf))
    ax.imshow(K_cbrf, interpolation='nearest', vmin=-vmax, vmax=vmax)

    plt.show()


if __name__ == '__main__':
    run_example()
