#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3
"""
    Prior-related classes for linear-Gaussian models
"""

import numpy as np
from scipy import linalg
from abc import ABCMeta, abstractmethod


def create_distance_matrix(nt, nf, offset=1, order='C', penalize_bias=False):

    N = offset + nt*nf
    Dt = 1e12 * np.ones((N, N))
    Df = 1e12 * np.ones((N, N))

    # Create separate grid for time and frequency dimensions
    FF, TT = np.meshgrid(range(nf), range(nt))
    ff = FF.ravel(order=order)
    tt = TT.ravel(order=order)

    for ii in range(nt*nf):
        for jj in range(nt*nf):
            Dt[offset+ii, offset+jj] = (tt[ii] - tt[jj]) ** 2
            Df[offset+ii, offset+jj] = (ff[ii] - ff[jj]) ** 2

    # Don't penalize bias term
    if not penalize_bias:
        for D in [Dt, Df]:
            for i in range(offset):
                D[i, i] = 0

    D = np.dstack((Dt, Df))
    return D


def create_derivative_matrix(nt, nf, offset=1, zero_order=True,
                             first_order=True, second_order=False, order='C',
                             equal_smoothing=False, penalize_bias=False,
                             bias_penalty=1.):

    N = nt*nf
    Zt = np.zeros((N, N))
    Zf = np.zeros((N, N))
    I = np.eye(N)

    # Create separate grid for time and frequency dimensions
    FF, TT = np.meshgrid(range(nf), range(nt))
    ff = FF.ravel(order=order)
    tt = TT.ravel(order=order)

    for ii in range(N):
        for jj in range(ii, N):

            dt = np.abs(tt[ii] - tt[jj])
            df = np.abs(ff[ii] - ff[jj])

            if dt == 1 and df == 0:
                Zt[ii, jj] = -1

            if dt == 0 and df == 1:
                Zf[ii, jj] = -1

    # Diagonal (ridge-like) covariance matrix
    LL = []
    if zero_order:
        D0 = np.eye(N)
        LL.append(D0)

    Dt1 = None
    Df1 = None

    # First-order derivatives
    if equal_smoothing:
        Ztf = I + Zt + Zf
        Dtf = np.dot(Ztf.T, Ztf)
        if first_order:
            LL.append(Dtf)
    else:
        Zt += I
        Zf += I
        Dt1 = np.dot(Zt.T, Zt)
        Df1 = np.dot(Zf.T, Zf)
        if first_order:
            LL.extend([Dt1, Df1])

    if second_order:
        # Second-order derivatives
        if equal_smoothing:
            Dtf2 = np.dot(Dtf.T, Dtf)
            LL.append(Dtf2)
        else:
            Dt2 = np.dot(Dt1.T, Dt1)
            Df2 = np.dot(Df1.T, Df1)
            LL.extend([Dt2, Df2])

    # Create
    nD = len(LL)
    DD = np.zeros((offset+N, offset+N, nD))
    for i, dd in enumerate(LL):
        DD[offset:, offset:, i] = dd

    if penalize_bias:
        # Put penalty on bias term?
        for i in range(offset):
            DD[i, i, :] = bias_penalty

    return DD


class Prior(object):

    __meta__ = ABCMeta

    @abstractmethod
    def get_cov(self):
        """Return prior covariance matrix"""

    @abstractmethod
    def get_inv_cov(self):
        """Return inverse of prior covariance matrix"""

    @abstractmethod
    def get_derivative_cov(self):
        """First derivative with respect to hyperparameter(s)"""


class RidgePrior(Prior):

    def __init__(self, ndim, alpha=1., offset_bias=1, penalize_bias=False,
                 bias_penalty=0.):

        self.ndim = ndim
        self.alpha = alpha
        self.offset_bias = offset_bias
        self.penalize_bias = penalize_bias
        self.bias_penality = bias_penalty

    def get_cov(self):

        d = self.ndim + self.offset_bias
        C = self.alpha * np.eye(d)

        if self.penalize_bias is False:
            for i in range(self.offset_bias):
                C[i, i] = self.bias_penality

        return C

    def get_inv_cov(self):

        C = self.get_cov()
        if self.offset_bias == 0 or self.bias_penalty > 0:
            return linalg.inv(C)
        else:
            return linalg.pinv(C)

    def get_derivative_cov(self):

        C = self.get_cov()
        return C / self.alpha


class DistancePrior(Prior):

    def __init__(self, ndims, scale=1., alphas=[1., 1.], offset_bias=1,
                 penalize_bias=False, bias_penalty=0., equal_smoothing=False,
                 order='C'):

        if len(ndims) > 2:
            raise ValueError("Distance prior only defined for <= 2 dimensions")

        self.ndims = tuple(ndims)
        self.scale = scale
        self.alphas = alphas
        self.offset_bias = offset_bias
        self.penalize_bias = penalize_bias
        self.bias_penality = bias_penalty

        self.equal_smoothing = equal_smoothing
        self.order = order

        self.DD = None
        self.recompute_precision_matrix()

    def recompute_precision_matrix(self):

        if len(self.ndims) > 1:
            nt, nf = self.dims
        else:
            nt = self.ndims[0]
            nf = 1

        self.DD = create_distance_matrix(nt, nf, offset=self.offset_bias,
                                         order=self.order,
                                         penalize_bias=self.penalize_bias)

    def get_cov(self):

        D = self.get_inv_cov()

        offset = self.offset_bias
        if offset == 0 or (self.penalize_bias and self.bias_penality > 0):
            return linalg.inv(D)
        else:
            return linalg.pinv(D)

    def get_inv_cov(self):

        D = np.sum(self.DD, axis=2)

        return D

    def get_derivative_cov(self):

        pass
