#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    blubb
"""

import numpy as np
from scipy import signal

from .util import whiten_matrix


class SimpleCell():
    """Simple cell simulation based on linear-nonlinear model

        Create one or more spike trains using a given receptive field

        Parameters
        ==========
        pattern : ndarray
            The true RF of the model cell

        threshold : float
            Spiking threshold in terms of standard deviations of filtered
            stimulus

        stddev : float
            Standard deviation of the Gaussian noise around the threshold

        n_trials : int
            Number of trials

        rectify : boolean
            Set negative filtered values to zero?

        dtype : numpy.dtype
            The data type of the response values

        seed : int or None
            The seed for the random number generator

    """

    def __init__(self, pattern, threshold=2.0, stddev=0.5, n_trials=1,
                 rectify=True, dtype=np.float64, seed=None):

        self.k = pattern
        self.theta = threshold
        self.sigma = stddev
        self.ntrials = n_trials
        self.rectify = rectify
        self.dtype = dtype
        self.seed = seed

    def simulate(self, S):
        """Generates spike for given input stimulus

            Input:
                S - stimulus array with dimensions (samples, features)

            Output:
                0-1-valued spike array with dimensions (samples, n_trials)
        """

        # Projections onto linear part
        proj = S.dot(self.k.flatten())
        if np.isscalar(proj):
            proj = np.atleast_1d(proj)
        proj /= np.std(proj)

        if self.rectify:
            proj[proj < 0] = 0.

        # Poisson-like spike generation
        rng = np.random.RandomState(self.seed)
        T = np.tile(proj, (self.ntrials, 1))

        if self.ntrials > 1:
            R = self.sigma * rng.randn(len(proj), self.ntrials)
            spikes = (T.T + R - self.theta) >= 0

        else:
            R = self.sigma * rng.randn(len(proj))
            spikes = (T + R - self.theta > 0).T

        return np.ascontiguousarray(spikes, dtype=self.dtype)


def create_rf(name='gabor', size=(25, 25), angle=45., phase=0.0, frequency=0.5,
              sigma=(0.35, 0.35), threshold=0.01, dtype=np.float, xoffset=0,
              yoffset=0, extent=(-8, 4, -7, 5)):
    """create typical 2D receptive field pattern

        Currently available: 'gabor' and 'onset'

    """

    if name.lower() == 'gabor':
        # Create grid
        nx = size[1]
        ny = size[0]
        x = np.linspace(-1., 1., nx)
        y = np.linspace(-1., 1., ny)
        xx, yy = np.meshgrid(x, y)

        # Create 2D gaussian and remove values below threshold
        sigmax = sigma[1]
        sigmay = sigma[0]
        G = np.exp(-(xx - 0.)**2 / (2. * sigmax**2) -
                    (yy - 0.)**2 / (2. * sigmay ** 2))
        G[np.abs(G) * np.amax(np.abs(G)) < threshold] = 0.

        # Create 2D sinusoidal grating
        rad = (angle/360.0) * 2 * np.pi
        xxr = xx * np.cos(rad)
        yyr = yy * np.sin(rad)
        xyr = (xxr + yyr) * 2 * np.pi * 2 * frequency
        S = np.cos(xyr + phase)

        # The gabor patch is simply the product of the two patches
        K = G * S
        K = K / np.amax(np.abs(K))

    if name.lower() == 'onset':
        # Simple onset RF
        nx = size[1]
        ny = size[0]
        x_lim = [extent[0] - yoffset, extent[1] - yoffset]
        y_lim = [extent[2] - xoffset, extent[3] - xoffset]
        x = np.linspace(x_lim[0], x_lim[1], nx)
        y = np.linspace(y_lim[0], y_lim[1], ny)
        xx, yy = np.meshgrid(x, y)
        K = yy * np.exp(-xx**2 - yy**2)

    K[np.abs(K) / np.amax(np.abs(K)) < threshold] = 0.

    return K.astype(dtype)


def create_onset_rf(size_x, size_y, mu_x, mu_y, sigma_x, sigma_y):
    """return narrow-band onset RF"""

    x = 1. + np.arange(size_x)
    y = 1. + np.arange(size_y)
    xx, yy = np.meshgrid(x, y)
    G = np.exp(- np.power(xx - mu_x, 2) / (2. * sigma_x) -
               np.power(yy - mu_y, 2) / (2. * sigma_y))
    G *= (yy - mu_y)
    G /= np.amax(np.abs(G))

    return G


def create_gabor_rf(size_x, size_y, mu_x, mu_y, sigma_x, sigma_y,
                    angle=45, frequency=0.5, phase=0.):

    x = 1. + np.arange(size_x)
    y = 1. + np.arange(size_y)
    xx, yy = np.meshgrid(x, y)

    G = np.exp(- np.power(xx - mu_x, 2) / (2. * sigma_x) -
               np.power(yy - mu_y, 2) / (2. * sigma_y))

    phi = 2. * np.pi * (angle/360.)
    xxr = xx * np.cos(phi)
    yyr = yy * np.sin(phi)
    xyr = (xxr + yyr) * 2. * np.pi * 2. * frequency
    S = np.cos(xyr + phase)

    K = G * S
    K /= np.amax(np.abs(K))

    return K


def smooth_rf(rf, filtsize=(3, 3), scale=1):
    """smooth 2D RF using 2D Gaussian filter"""

    # Create 2D Gaussian filter
    sizex = int(np.floor(filtsize[0]/2.))
    sizey = int(np.floor(filtsize[1]/2.))
    x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
    G = np.exp(-scale*(x**2/float(sizex)+y**2/float(sizey)))
    G /= G.sum()

    if isinstance(rf, np.ndarray):
        KG = signal.convolve2d(rf, G, mode='same', boundary='symm')
        return KG

    else:
        K = np.reshape(rf.coef, rf.shape)
        KG = signal.convolve2d(K, G, mode='same', boundary='symm')
        rf.coef = KG.flatten()


#################################################################
#                   Stimulus generation                         #
#################################################################

def create_grating(size=(25, 25), angle=45., phase=0.0, f=0.5):
    """Creates single sinusoidal grating of given size

    Parameters
    ==========
    size : tuple, list, ndarray
        The size of the grating

    angle : float
        The angle in defees

    phase : float
        The phase in radians

    f : float
     Normalized frequency

    """

    # Create x-y grid
    nx = size[1]
    ny = size[0]
    x = np.pi * np.linspace(-1., 1., nx)
    y = np.pi * np.linspace(-1., 1., ny)
    xx, yy = np.meshgrid(x, y)

    rad = (angle/360.0) * 2 * np.pi
    xxr = xx * np.cos(rad)
    yyr = yy * np.sin(rad)
    xyr = (xxr + yyr) * 2 * np.pi * f

    return np.cos(xyr + phase)


def create_gratings(size=(25, 25), N=10**4, center=True, dtype=np.float64,
                    whiten=False, random_seed=None, f_min=0., f_max=1.,
                    order='C'):
    """Create ensemble of sinusoidal gratings with random parameters

    Parameters
    ==========
    size : tuple, list, ndarray
        The size of the grating

    N : int
        The number of gratings

    center : boolean
        Element-wise centering of grating ensemble

    dtype : numpy.dtype
        Data type of gratings

    whiten : boolean
        PCA-based whitening of grating matrix?

    random_seed : int or None
        Random seed for numpy.random.RandomState

    f_min : float
        Lower normalized spatial frequency

    f_max : float
        Upper normalized spatial frequency

    order : str
        Flatten gratings in 'C' or 'F' order. The output will always be in
        'C' order.
    """

    # Create x-y grid
    nx = size[1]
    ny = size[0]
    x = np.linspace(-1., 1., nx)
    y = np.linspace(-1., 1., ny)
    xx, yy = np.meshgrid(x, y)

    # for reproducible results
    rng = np.random.RandomState(random_seed)

    # Create gratings using randomly drawn parameters
    G = np.zeros((N, size[0]*size[1]), dtype=dtype, order='C')
    for i in range(int(N)):

        angle = rng.rand(1) * 180
        phase = rng.rand(1) * 2 * np.pi
        f = f_min + rng.rand(1) * (f_max - f_min)
        G[i, :] = create_grating(size, angle, phase, f).ravel(order=order)

    if whiten:
        G, _ = whiten_matrix(G, eps=1e-8)

    if center:
        G -= np.mean(G, axis=0)

    return G
