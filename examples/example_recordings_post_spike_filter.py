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
from lnpy.lnp.util import DataConverter, create_postfilt_features

from lnpy.lnp.cbrf import CbRF
from lnpy.lnp.base import STRF


#from ipdb import set_trace as db


def load_data(data_dir, h5_file):

    reader = NeoHdf5Reader(join(data_dir, h5_file))
    block = reader.read()

    return block


def convert_data(block, t_spikefilt=0.015):
    """return stimulus and response matrix"""

    print "---------- Converting data set ----------"

    fs_stim = 16000.
    f_cutoff = (500., 8000.)

    win_len = 0.04
    fs_spec = 500.  # bin with: 2 ms
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

    # Add post-spike filter features
    n_spikefilt = int(np.ceil(t_spikefilt * fs_spec))

    Y = Y[:, 0]  # works only for single trials
    Z = create_postfilt_features(Y, n_spikefilt)
    X = np.hstack((X, Z))  # append to feature matrix

    print "Stimulus matrix: %d temp. steps x %d features" % X.shape
    print "%d spikes (%0.3f spikes per sample)" % (np.sum(Y), np.mean(Y))
    print "Post-spike filter length: %d samples" % n_spikefilt

    return X, Y, rfsize, axes, fs_spec, n_spikefilt


def run_example():

    t_spikefilt = 0.05

    data_dir = join(split(__file__)[0], 'data')
    data_file = 'STRFs_ChirpsBlocks_IC_2012-01-31_50dB_chan01_unit01.h5'
#    data_file = 'STRFs_ChirpsBlocks_IC_2012-04-26_Pos_02_30dB_chan01_unit01.h5'

    block = load_data(data_dir, data_file)
    X, Y, rfsize, axes, fs, n_spikefilt = convert_data(block, t_spikefilt)

    # Add some models
    n_jobs = 3  # number of workers used for hyperparameter optimization
    n_griditer = 3
    models = []

#    model = STA(n_postfilt=n_postfilt)
#    models.append(model)

#    model = GaussianGLM(n_griditer=n_griditer, n_jobs=n_jobs,
#                        optimizer='ridge')
#    models.append(model)

    model = CbRF(n_griditer=n_griditer, n_jobs=n_jobs, n_spikefilt=n_spikefilt)
    models.append(model)
#
#    model = PoissonGLM(n_griditer=n_griditer, n_jobs=n_jobs)
#    models.append(model)

    # Fit models
    for model in models:
        print 50 * '-'
        print "Fitting model:", model.name
        print 50 * '-'
        model.fit(X, Y)

    # Plot STRFs
    fig, axarr = plt.subplots(nrows=len(models), ncols=1+int(n_spikefilt > 0))
    axarr = np.atleast_2d(axarr)
    for i, model in enumerate(models):

        ax = axarr[i, 0]
        ax.set_title(model.name)

        W = np.reshape(model.get_weights(), rfsize)
        rf = STRF(W, fs, time=axes[0].values.base,
                  frequency=axes[1].values.base)
        rf.show(ax=ax, show_now=False, colorbar=False)

        if n_spikefilt > 0:
            # Post-spike filter
            ax = axarr[i, 1]
            xx = np.linspace(-t_spikefilt, 0, n_spikefilt) * 1000
            h = model.get_spikefilt()
            ax.plot(xx, h)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Gain (a.u.)')

    fig.set_size_inches(8, 4)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_example()
