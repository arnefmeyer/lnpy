#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3
"""
    Wrapper for automatic locality determination matlab code by Mijung Park and
    Jonathan Pillow
"""

import numpy as np
from os.path import split, join

from ..base import LinearModel


class ALD(LinearModel):

    def __init__(self, wsize, fit_bias=True, verbose=True, maxiter=100,
                 **kwargs):

        self.wsize = wsize

        self.fit_bias = fit_bias
        self.verbose = verbose
        self.maxiter = maxiter
        self.optargs = kwargs

        self.coef_ = np.zeros(tuple(wsize))
        self.intercept_ = 0.

        self.cov_posterior = None
        self.cov_prior = None
        self.scale_param = None
        self.smooth_params = None
        self.noisevar = None

    def fit(self, X, y):

        self._call_matlab_code(X, y)

    def predict(self, X):

        z = np.dot(X, self.coef_)
        if self.fit_bias:
            z += self.intercept_

        return z

    def show(self, *args, **kwargs):

        fig = super(ALD, self).show(self.wsize, **kwargs)

        return fig

    def _call_matlab_code(self, X, y):

        from matcall import MatlabCaller

        thisdir = split(__file__)[0]
        paths = [join(thisdir, 'matlab_code'),
                 join(thisdir, 'matlab_code', 'tools')]

        caller = MatlabCaller(addpath=paths, tempdir=None,
                              verbose=self.verbose, **self.optargs)

        y = np.reshape(y, (y.shape[0], 1)).astype(np.float64)
        wsize = np.asarray(self.wsize, dtype=np.float64)
        if len(wsize) > 1:
            wsize = np.reshape(wsize, (wsize.shape[0], 1))

        input_dict = dict(x=X, y=y, spatialdims=wsize, nkt=1)
        input_order = ['x', 'y', 'spatialdims', 'nkt']
        result = caller.call('runALD', input_dict, input_order=input_order,
                             kwarg_names=None, output_names=['khatALD',
                                                             'kRidge'])

        khatALD = result['khatALD'].item()
        K = np.reshape(khatALD.khatSF, self.wsize, order='F')
        self.coef_ = K.flatten(order='C')
        self.intercept_ = np.mean(y)
