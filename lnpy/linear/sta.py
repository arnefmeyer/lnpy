#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Spike-triggered average
"""

import numpy as np
from scipy import linalg

from .base import LinearModel


class STA(LinearModel):

    def __init__(self, whiten=False, **kwargs):

        super(STA, self).__init__(**kwargs)

        self.whiten = whiten

    def fit(self, X, y):

        if y.ndim > 1:
            y = np.sum(y, axis=1)

        X_centered = X - np.mean(X, axis=0)
        if self.whiten:
            y_mean = np.mean(y)
            self.coef_ = linalg.lstsq(X_centered, y - y_mean)[0]
            self.intercept_ = y_mean

        else:
            self.coef_ = np.dot(X_centered.T, y)
            self.intercept_ = np.mean(y)
