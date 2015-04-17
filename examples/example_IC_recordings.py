#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    This example demonstrates STRF estimation from neural recordings
    using different methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import split, join
import sys
sys.path.append(join(split(__file__)[0], '..'))

from lnpy.io.hdf import NeoHdf5Reader
from lnpy.transform.gammatone import GammatoneFilterbank
from lnpy.lnp.util import DataConverter

from lnpy.lnp.sta import STA
from lnpy.lnp.cbrf import CbRF
from lnpy.lnp.glm import GaussianGLM, PoissonGLM
from lnpy.lnp.base import STRF


def load_data(data_dir, h5_file):

    reader = NeoHdf5Reader(join(data_dir, h5_file))
    block = reader.read()

    return block


def convert_data(block):
    """return stimulus and response matrix"""

    print "---------- Converting data set ----------"

    fs_stim = 16000.
    f_cutoff = (500., 8000.)

    win_len = 0.04
    fs_spec = 400.  # bin with: 2 ms
    filt_per_erb = 1.
    n_samples = np.Inf

    fb = GammatoneFilterbank(samplerate=fs_stim, f_cutoff=f_cutoff,
                             filt_per_erb=filt_per_erb,
                             spectype='magnitude', bw_factor=1.)

    converter = DataConverter(fb, win_len=win_len, samplerate=fs_spec,
                              verbose=True, n_samples=n_samples,
                              scaling='dB', dynamic_range=60.,
                              center=True)
    X, Y, rfsize, axes, stim_pos = converter.process(block)

    print "Stimulus matrix: %d temp. steps x %d features" % X.shape
    print "Spike    matrix: %d temp. steps x %d trials" % Y.shape
    print "%d spikes (%0.3f spikes per sample)" % (np.sum(Y), np.mean(Y))

    return X, Y, rfsize, axes, fs_spec


def run_example():

    data_dir = join(split(__file__)[0], 'data')
    data_file = 'STRFs_ChirpsBlocks_IC_2012-01-31_50dB_chan01_unit01.h5'

    block = load_data(data_dir, data_file)
    X, Y, rfsize, axes, fs = convert_data(block)

    # Add some models
    n_jobs = 3  # number of processes used for hyperparameter optimization
    n_griditer = 3
    models = []

    model = STA()
    models.append(model)

    model = GaussianGLM(n_griditer=n_griditer, n_jobs=n_jobs,
                        optimizer='ridge')
    models.append(model)

    model = CbRF(n_griditer=n_griditer, n_jobs=n_jobs)
    models.append(model)

    model = PoissonGLM(n_griditer=n_griditer, n_jobs=n_jobs)
    models.append(model)

    # Fit models
    for model in models:
        print 50 * '-'
        print "Fitting model:", model.name
        print 50 * '-'
        model.fit(X, Y)

    # Plot STRFs
    fig, axarr = plt.subplots(nrows=1, ncols=len(models), sharex=True,
                              sharey=True)
    for i, model in enumerate(models):

        ax = axarr[i]
        ax.set_title(model.name)

        W = np.reshape(model.coef_, rfsize)
        rf = STRF(W, fs, time=axes[0].values.base,
                  frequency=axes[1].values.base)
        rf.show(ax=ax, show_now=False, colorbar=False)

        if i > 0:
            ax.set_ylabel('')

    fig.set_size_inches(8, 2.5)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_example()
