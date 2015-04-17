#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from __future__ import division

import numpy as np
from base import BaseTransform
from gammatone import GammatoneFilterbank
from spectrogram import STFT
from ..base import ModulationSpectrogram


class AMS(BaseTransform):
    """Amplitude modulation spectrogram class

    """

    def __init__(self, first_transform=GammatoneFilterbank(),
                 second_transform=STFT(), dynamic_range=60.):

        self.first_transform = first_transform
        self.second_transform = second_transform
        self.dynamic_range = dynamic_range

    def get_center_frequencies(self):
        return self.first_transform.get_center_frequencies()

    def get_modulation_frequencies(self):
        return self.second_transform.get_center_frequencies()

    def get_samplerate(self):
        return self.first_transform.get_samplerate()

    def to_string(self):
        """Returns unique parameter string for filterbank object

        """
        return self.first_transform.to_string() + '_' + \
            self.second_transform.to_string()

    def process(self, signal):
        """
        Transforms a spectrogram into a modulation spectrogram
        """

        fb1 = self.first_transform
        fb2 = self.second_transform

        spec = fb1.process(signal)

        # Logarithmic compression
        spec.scale(scaling='dB', dynamic_range=self.dynamic_range)

        # Resample frequency channels?
        if fb1.samplerate != fb2.samplerate:
            spec.resample(fb2.samplerate)

        # Compute modulation frequencies for each frequency band
        mod_spec = fb2.process(spec)

        return mod_spec

    def analyze(self, signal):
        return self.process(signal)


class ModSpec2D(BaseTransform):
    """Spectro-temporal modulation spectrogram class"""

    def __init__(self, first_transform=GammatoneFilterbank(),
                 dynamic_range=60., samplerate=2., win_len=0.03,
                 nfft_t=16, nfft_f=16, f_cutoff_t=(0., 100.),
                 f_cutoff_f=(0, 20.), mode='same'):

        self.first_transform = first_transform
        self.dynamic_range = dynamic_range
        self.samplerate = samplerate
        self.win_len = win_len
        self.nfft_t = nfft_t
        self.nfft_f = nfft_f
        self.f_cutoff_t = f_cutoff_t
        self.f_cutoff_f = f_cutoff_f
        self.mode = mode

    def get_center_frequencies(self):

        return self.first_transform.get_center_frequencies()

    def get_modulation_frequencies(self):

        N = int(np.round(self.win_len * self.samplerate))
        return np.linspace(0., self.samplerate/2., int(N/2. + 1))

    def get_samplerate(self):

        return self.first_transform.get_samplerate()

    def to_string(self):
        """Returns unique parameter string for filterbank object"""

#        return self.first_transform.to_string() + '_FFT2'

    def process(self, signal):
        """Transform spectrogram into a modulation spectrogram"""

        fb1 = self.first_transform

        spec = fb1.process(signal)

        # Logarithmic compression
        spec.scale(scaling='dB', dynamic_range=60.)

        # Resample frequency channels?
        fs_spec = self.samplerate
        if fb1.samplerate != fs_spec:
            spec.resample(fs_spec)

        nwin = int(np.round(fs_spec * self.win_len))
        win_half = int(np.floor(nwin/2.) + 1)

        S = spec.get_data()
#        spec_time = spec.get_time()

        # Handling of boundaries
        nt, nfc = S.shape
#        t_offset = spec_time[int(win/2.)]
#        if self.mode == 'same':
#            t_offset = spec_time[0]
        S_begin = np.zeros((win_half - 1, nfc))
        S_end = np.zeros((win_half - 1, nfc))
        S = np.concatenate((S_begin, S, S_end), axis=0)

        # Create energy-normalized window
        win = np.hanning(nwin)
        win /= np.sum(win * win)

        # Compute modulation frequencies for each frequency band
        nfft_t = self.nfft_t
        nfft_f = self.nfft_f
        if nfft_f is None:
            nfft_f = S.shape[1]

        M = None
        for i in range(0, S.shape[0] - nwin):

            ind = np.arange(0, nwin, dtype=np.int)
            X = (S[ind, :].T * win).T

            Z = np.fft.rfft2(X, s=(nfft_t, nfft_f), axes=(0, 1))
            Z = np.abs(Z)

            if M is None:
                ntt = S.shape[0] - nwin
                nfm, nfw = Z.shape
                M = np.zeros((ntt, nfw, nfm))

            M[i, :, :] = Z.T

#            if i == 20:
#                import matplotlib.pyplot as plt
#                plt.imshow(Z, aspect='auto', interpolation='nearest')
#                plt.show()

        # Create time and frequency vectors
        if self.mode == 'same':
            t = spec.get_time()
        else:
            t = spec.get_time()[win_half:-win_half+1]

        fmt = np.fft.rfftfreq(nfft_t) * fs_spec
        fmw = np.fft.rfftfreq(nfft_f)

        # Check cutoff frequencies
        fcutt = self.f_cutoff_t
        valid = np.logical_and(fmt >= fcutt[0], fmt <= fcutt[1])
        M = M[:, :, valid]
        fmt = fmt[valid]

        modspec = ModulationSpectrogram(M, fs_spec, time=t, f_center=fmw,
                                        f_mod=fmt)

        return modspec

#        N = int(np.round(self.win_len * self.samplerate))
#        n_steps = spec.shape()[0] - N
#        M = None
#        for i in range(n_steps):
#
#            s = out1[i:i+N, :]
#            x = np.abs(np.fft.rfft2(s, s=(nfft_t, nfft_f), axes=(0, 1)))
#
#            if M is None:
#                nt, nwt, nwf = n_steps, x.shape[0], x.shape[1]
#                M = np.zeros((nt, nwt, nwf), dtype=np.float64)
#
#            M[i, :, :] = x
#
#        t2 = 1. / self.samplerate * np.arange(N/2., (n_steps - N), 1.)
#        fmt = np.linspace(0., self.samplerate/2., nwt)
#        fmf = np.linspace(0., nwf, nwf)
#
#        M = np.log(M)
#
#        return M, t2 + t1[0], (fmt, fmf)

#     def __calc_stft(self, x):
#
#        nwin, nshift, nfft, fs, noverlap = self.__get_fft_params()
#
#        # Create energy-normalized window
#        win = np.hanning(nwin)
#        win /= np.sum(win * win)
#
#        if self.mode == 'same':
#            z_begin = np.zeros((int(nwin/2 - 1,)))
#            z_end = np.zeros((int(nwin/2 - 1,)))
#            x = np.concatenate((z_begin, x, z_end))
#
#        # The fast way ...
#        X = np.multiply(win, segment(x, nwin, nshift))
#        if self.spectype.lower() == 'dct':
#            if nfft > nwin:
#                nn = nfft - nwin
#                X = np.concatenate((X, np.zeros((X.shape[0], nn),
#                                                dtype=X.dtype)), axis=1)
#
#            if self.dct_type is None or self.dct_type == 2:
#                S = dct(X, type=2, axis=1)
#            else:
#                S = dct(X, type=self.dct_type, axis=1)
#
#        else:
#            S = np.fft.rfft(X, n=nfft, axis=1)
#            S = self._convert_spectrum_type(S, self.spectype)
#
#            # Compenstate for power of omitted negative frequencies except for
#            # DC and Nyquist frequencies
#            S[1:-1, :] *= 2.
#
#        # Create time and frequency vectors
#        if self.mode == 'same':
#            t = 1./fs * np.arange(0, S.shape[0]*nshift, nshift)
#        else:
#            t = 1./fs * np.arange(nwin/2., nwin/2. + S.shape[0]*nshift, nshift)
#
#        f = np.linspace(0., fs/2., S.shape[1])
#
#        return S, f, t
