#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Spike-triggered average (STA) stimulus
"""

from base import LNPEstimator
import numpy as np
import time


class STA(LNPEstimator):
    """Spike-triggered average (STA) reptive field estimator"""

    def __init__(self, **kwargs):
        super(STA, self).__init__(**kwargs)

    def fit(self, X, Y):

        if Y.ndim > 1:
            spike_cnt = np.sum(Y, axis=1)
        else:
            spike_cnt = Y

        t0 = time.time()
        N = np.sum(spike_cnt > 0)
        w = np.dot((X - np.mean(X, axis=0)).T, spike_cnt) / N
        self.t_fit = time.time() - t0

        self._split_coef_spikefilt(w)
        self.intercept_ = np.mean(spike_cnt)
