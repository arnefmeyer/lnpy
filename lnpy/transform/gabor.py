#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""

Python implementation of the gabor filterbank from

[1] M.R. Schädler, B.T. Meyer, B. Kollmeier
"Spectro-temporal modulation subspace-spanning filter bank features
for robust automatic speech recognition ", J. Acoust. Soc. Am. Volume 131,
Issue 5, pp. 4134-4151 (2012)

"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axes_grid import ImageGrid

from base import BaseTransform
from ..base import Spectrogram, ModulationSpectrogram


def _hann_win(width):
    """ A hanning window function that accepts non-integer width and always
        returns a symmetric window with an odd number of samples.
    """

    x_center = 0.5

    xx = np.arange((x_center - 1./(width+1.)), 0, -1./(width+1.))
    yy = np.arange(x_center, 1, 1./(width+1.))

    x_values = np.concatenate((xx[::-1], yy))
    valid_values_mask = np.logical_and(x_values > 0, x_values < 1)
    h = 0.5 * (1. - (np.cos(2 * np.pi * x_values[valid_values_mask])))

    return h


def _fftconv2(in1, in2, shape):
    """2D convolution in terms of the 2D FFT that substitutes conv2"""

    size_y = in1.shape[0] + in2.shape[0] - 1
    size_x = in1.shape[1] + in2.shape[1] - 1
    fft_size_x = int(2 ** np.ceil(np.log2(size_x)))
    fft_size_y = int(2 ** np.ceil(np.log2(size_y)))

    in1_fft = np.fft.fft2(in1, (fft_size_y, fft_size_x))
    in2_fft = np.fft.fft2(in2, (fft_size_y, fft_size_x))

    out_fft = in1_fft * in2_fft
    out_padd = np.fft.ifft2(out_fft)
    out_padd = out_padd[:size_y, :size_x]

    if shape == 'same':
        y_offset = int(np.floor(in2.shape[0] / 2.))
        x_offset = int(np.floor(in2.shape[1] / 2.))
        out = out_padd[y_offset:in1.shape[0] + y_offset,
                       x_offset:in1.shape[1] + x_offset]

    elif shape == 'full':
        out = out_padd

    return out


class GaborFilter(object):
    """
    Gabor filter class
    """

    def __init__(self, size_n, size_k, omega_n=np.pi/4, omega_k=np.pi/4,
                 nu_n=3.5, nu_k=3.5, index=None):

        self.size_n = size_n
        self.size_k = size_k
        self.omega_n = omega_n
        self.omega_k = omega_k
        self.nu_n = nu_n
        self.nu_k = nu_k
        self.index = index

        self._coef = None
        self._create_filter()

    def get_coef(self):

        return self._coef

    def get_omega(self):

        return self.omega_n, self.omega_k

    def get_size(self):

        return self.size_n, self.size_k

    def process(self, X, output_type='real', mode='same', boundary='fill',
                fillvalue=0):

        Z = self._filter(X)

        if output_type.lower() == 'real':
            Z = np.real(Z)

        return Z

    def _filter(self, X):

        gfilter = self._coef

        if np.any(gfilter < 0):
            # Compare this code to the compensation for the DC part in the
            # 'gfilter_gen' function. This is an online version of it removing
            # the DC part of the filters by subtracting an appropriate part
            # of the filters' envelope

            gfilter_abs_norm = np.abs(gfilter) / np.sum(np.abs(gfilter))
            gfilter_dc_map = _fftconv2(np.ones_like(X), gfilter, 'same')
            env_dc_map = _fftconv2(np.ones_like(X), gfilter_abs_norm, 'same')
            dc_map = _fftconv2(X, gfilter_abs_norm, 'same') \
                / env_dc_map * gfilter_dc_map

        else:
            # Dont' remove the DC part if it is the DC filter
            dc_map = 0

        # Filter log_mel_spec with the 2d Gabor filter and remove the DC parts
        Z = _fftconv2(X, gfilter, 'same') - dc_map

        return Z

    def _create_filter(self):
        # Generates a gabor filter function with:
        #  omega_k       spectral mod. freq. in rad
        #  omega_n       temporal mod. freq. in rad
        #  nu_k        number of half waves unter the envelope in spectral dim.
        #  nu_n        number of half waves unter the envelope in temporal dim.
        #  size_max_k    max. allowed extension in spectral dimension
        #  size_max_n    max. allowed extension in temporal dimension

        omega_n = self.omega_n
        omega_k = self.omega_k
        size_max_n = self.size_n
        size_max_k = self.size_k
        nu_n = self.nu_n
        nu_k = self.nu_k

        # Calculate windows width.
        w_n = 2 * np.pi / np.abs(omega_n) * nu_n / 2
        w_k = 2 * np.pi / np.abs(omega_k) * nu_k / 2

        # If the size exceeds the max. allowed extension in a dimension set the
        # corresponding mod. freq. to zero.
        if w_n > size_max_n:
            w_n = size_max_n
            omega_n = 0

        if w_k > size_max_k:
            w_k = size_max_k
            omega_k = 0

        # Separable hanning envelope, cf. Eq. (1c).
        env_n = _hann_win(w_n - 1.)
        env_k = _hann_win(w_k - 1.)
        envelope = np.outer(env_k, env_n)
        win_size_k, win_size_n = envelope.shape

        # Sinusoid carrier, cf. Eq. (1c).
        n_0 = (win_size_n + 1.) / 2.
        k_0 = (win_size_k + 1.) / 2.
        n, k = np.meshgrid(np.r_[1:win_size_n+1.], np.r_[1:win_size_k+1.])
        sinusoid = np.exp(1j*omega_n*(n - n_0) + 1j*omega_k*(k - k_0))

        # Eq. 1c
        gfilter = envelope * sinusoid

        # Compensate the DC part by subtracting an appropiate part
        # of the envelope if filter is not the DC filter.
        envelope_mean = np.mean(envelope)
        gfilter_mean = np.mean(gfilter)
        if (omega_n != 0) or (omega_k != 0):
            gfilter -= (envelope/envelope_mean * gfilter_mean)

        else:
            # Add an imaginary part to DC filter for a fair real/imag
            # comparison.
            gfilter += 1j * gfilter

        # Normalize filter to have gains <= 1.
        gfilter /= np.max(np.abs(np.fft.fft2(gfilter)))

        self._coef = gfilter.T
        self.omega_n = omega_n
        self.omega_k = omega_k
        self.size_n = w_n
        self.size_k = w_k


class GaborFilterbank(BaseTransform):
    """
    Gabor filterbank
    """

    def __init__(self, samplerate=400., t_maxlen=0.04, f_maxlen=22,
                 design='ASR'):

        self.samplerate = samplerate
        self.f_maxlen = f_maxlen
        self.t_maxlen = t_maxlen
        self.design = design

        nt = int(np.ceil(self.t_maxlen * self.samplerate))
        nf = self.f_maxlen
        size_max = np.asarray([nf, nt])
        self._create_filters(size_max)

    def get_filters(self):

        return self._filters

    def process(self, spec, verbose=False):

        fs = self.samplerate

        if isinstance(spec, Spectrogram):
            X = spec.get_data()
            fc = spec.get_frequency()
            t = spec.get_time()

        else:
            X = spec
            fc = np.linspace(0, fs/2., X.shape[1])
            dt = 1. / fs
            t = np.linspace(dt/2., X.shape[0] * dt - dt/2., X.shape[0])

        # Filter spectrogram
        filters = self._filters
        Z = []
        cnt = 0
        for filt in filters:

            if verbose:
                print("{}/{}".format(cnt+1, len(filters)))

            z = filt.process(X, output_type='real')
            Z.append(z)

        Y = np.asarray(Z)
        Y = np.rollaxis(Y, 0, 3)
        fm = np.arange(Y.shape[2])
        modspec = ModulationSpectrogram(Y, fs, time=t, f_center=fc, f_mod=fm)

        return modspec

    def plot_filters(self, fig=None, show_now=False):

        if fig is None:
            fig = plt.figure()

        filters = self._filters
        indices = np.asarray([f.index for f in filters])

        ind_k = indices[:, 0]
        ind_n = indices[:, 1]
        n_omega_k = np.unique(ind_k).shape[0]
        n_omega_n = np.unique(ind_n).shape[0]

        grid = ImageGrid(fig, 111, nrows_ncols=(n_omega_k, n_omega_n),
                         share_all=False)

        for filt in filters:

            i_k, i_n = filt.index
            idx = i_k * n_omega_n + i_n

            G = np.real(filt.get_coef())
            vmax = np.amax(np.abs(G))

            ax = grid[idx]
            ax.imshow(G.T, interpolation='nearest', origin='lower',
                      vmin=-vmax, vmax=vmax)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()

        if show_now:
            plt.show()

        return fig

    def _create_filters(self, size_max):

        # Standard settings
        omega_max = np.pi * np.array([.5, .5])
        nu = np.array([3.5, 3.5])
        distance = np.array([0.3, 0.2])
#        distance = np.array([0.2, 0.2])

        # Filter parameters
        if self.design.lower() == 'asr':
            omega_n, omega_k = self._calcaxis_ASR(omega_max, size_max,
                                                  nu, distance)
            scale_n = np.ones_like(omega_n)
            scale_k = np.ones_like(omega_k)

        elif self.design.lower() == 'temporal':
            omega_n, omega_k = self._calcaxis_temp(omega_max, size_max,
                                                   nu)
            scale_n = np.ones_like(omega_n)
            scale_k = np.power(2., np.arange(0, -omega_k.shape[0], -1))

        else:
            raise ValueError('Unknown design parameter: %s' % self.design)

        # Create filters
        omega_n_num = omega_n.shape[0]
        omega_k_num = omega_k.shape[0]
        filters = []
        for i in range(omega_k_num):

            for j in range(omega_n_num):

                if not (omega_k[i] < 0 and omega_n[j] == 0):
                    size_n = size_max[0] * scale_n[j]
                    size_k = size_max[1] * scale_k[i]
                    gf = GaborFilter(size_n, size_k,
                                     omega_n=omega_n[j], omega_k=omega_k[i],
                                     nu_n=nu[0], nu_k=nu[1], index=(i, j))
                    filters.append(gf)

        self._filters = filters

    def _calcaxis_ASR(self, omega_max, size_max, nu, distance):
        """ Calculates the modulation center frequencies iteratively.
            Termination condition for iteration is reaching omega_min, which
            is derived from size_max.
        """

        omega_min = (np.pi * nu) / size_max

        # Eq. (2b)
        c = distance * 8. / nu

        # Second factor of Eq. (2a)
        space_n = (1 + c[1]/2.) / (1 - c[1]/2.)

        # Iterate starting with omega_max in spectral dimension
        count_n = 0
        omega_n = [omega_max[1]]
        while omega_n[-1]/space_n > omega_min[1]:
            tmp = omega_max[1] / np.power(space_n, count_n)
            if count_n > 0:
                omega_n.append(tmp)
            else:
                omega_n[0] = tmp
            count_n += 1

        omega_n = np.asarray(omega_n)
        omega_n = omega_n[::-1]

        # Add DC
        omega_n = np.append(0, omega_n)

        # Second factor of Eq. (2a)
        space_k = (1 + c[0]/2.) / (1 - c[0]/2.)

        # Iterate starting with omega_max in temporal dimension
        count_k = 0
        omega_k = [omega_max[0]]
        while omega_k[-1]/space_k > omega_min[0]:
            tmp = omega_max[0] / np.power(space_k, count_k)
            if count_k > 0:
                omega_k.append(tmp)
            else:
                omega_k[0] = tmp
            count_k += 1

        omega_k = np.asarray(omega_k)
        # Add DC and negative MFs for spectro-temporal opposite
        # filters (upward/downward)
        omega_k = np.concatenate((-omega_k, np.array([0]), omega_k[::-1]))

        return omega_n, omega_k

    def _calcaxis_temp(self, omega_max, size_max, nu):
        """purely temporal filters at different resolutions"""

        omega_min = (np.pi * nu) / size_max

        omega_n = [omega_max[1]]
        omega_k = [0]
        while omega_n[-1] / 1.5 > omega_min[1]:
            omega_n.append(omega_n[-1] / 1.5)
            omega_k.append(0)

        omega_k = [omega_max[0]]
        omega_n.append(0)

        while omega_k[-1] / 1.5 > omega_min[0]:
            omega_k.append(omega_k[-1] / 1.5)
            omega_n.append(0)

#        omega_n.append(0)
#        omega_k = np.zeros((1,))

        return np.asarray(omega_n), np.asarray(omega_k)
