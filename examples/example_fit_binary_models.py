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

from lnpy.lnp.util import createRF, SimpleCell
from lnpy.learn.base import create_derivative_matrix
from lnpy.learn import SmoothnessPrior
from lnpy.lnp.cbrf import CbRF
from lnpy.lnp.glm import BernoulliGLM


def create_toy_data(N=1000, n_trials=1):

    # Create Gabor kernel
    rfsize = (15, 15)
    K_true = createRF(name='gabor', size=rfsize, threshold=0.2,
                      dtype=np.float64, frequency=0.5, sigma=2*[0.35])

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

    # create data set
    X, Y, K_true = create_toy_data(N=2500)

    # use a smoothness prior
    nt, nf = K_true.shape
    D = create_derivative_matrix(nt, nf)
    prior = SmoothnessPrior(D=D)

    # create CbRF and Bernoulli GLM
    params = dict(optimize=True,  # do hyperparameter optimization
                  verbose=True,  # output some results
                  n_griditer=3,  # do 3 iterated grid searches
                  n_jobs=1  # number of jobs (-1 = all)
                  )
    models = [CbRF(metric='AUC', prior=prior, **params),
              BernoulliGLM(metric='BernoulliLL', prior=prior, **params)]

    # fit models
    fig, axarr = plt.subplots(nrows=len(models), ncols=2)

    for i, model in enumerate(models):

        print "fitting model {}".format(model.name)

        model.fit(X, Y)
        k = model.get_weights()
        K = np.reshape(k, K_true.shape)

        ax = axarr[i, 0]
        ax.set_title('True')
        vmax = np.max(np.abs(K_true))
        ax.imshow(K_true, interpolation='nearest', vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r')

        ax = axarr[i, 1]
        ax.set_title(model.name)
        vmax = np.max(np.abs(K))
        ax.imshow(K, interpolation='nearest', vmin=-vmax, vmax=vmax,
                  cmap='RdBu_r')

    for ax in axarr.flat:
        ax.axis('off')
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    run_example()
