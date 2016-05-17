#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

from base import BaseTransform
from ..lnp.util import segment
from ..base import Stimulus, Spectrogram, ModulationSpectrogram
import quantities as pq
import numpy as np
from matplotlib.mlab import specgram
from scipy.fftpack import dct


def hannwin(n):
    """Create a von Hann window of given length (in samples)"""
    x = np.arange(0, n, dtype=np.float)
    return 0.5 * (1. - np.cos(2. * np.pi * x / (n-1)))


def tukeywin(n, alpha=0.5):
    """Create a Tukey window of given length (in samples)"""

    win = None
    if alpha <= 0:
        win = np.ones((n,))

    elif alpha >= 1:
        win = np.hanning(n)

    else:
        x = np.linspace(0, 1, n)
        win = np.ones(x.shape)

        # 0 <= x < alpha/2
        vv = x < .5 * alpha
        win[vv] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[vv] - alpha/2)))

        # 1 - alpha / 2 <= x <= 1
        vv = x >= (1 - .5 * alpha)
        win[vv] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[vv] - 1 + alpha/2)))

    return win


class STFT(BaseTransform):
    """Short-time Fourier transform class"""

    def __init__(self, win_len=256, win_shift=128, nfft=256,
                 samplerate=2., use_quantities=False, f_cutoff=None,
                 spectype='magnitude', mode='valid', dct_type=None,
                 normalize=False):

        self.win_len = win_len
        self.win_shift = win_shift
        self.nfft = nfft
        self.samplerate = samplerate
        self.use_quantities = use_quantities
        self.f_cutoff = f_cutoff
        self.spectype = spectype
        self.mode = mode
        self.dct_type = dct_type
        self.normalize = normalize

    def to_string(self):
        """Return string with transform parameters"""

        param_str = 'winlen_%d_shift_%d_nfft_%d_fs_%d' % \
            (self.win_len, self.win_shift, self.nfft, int(self.samplerate))
        if self.f_cutoff is not None:
            param_str + '_cutoff_%d_%d' % (self.f_cutoff[0], self.f_cutoff[1])
        return param_str

    def get_center_frequencies(self):
        """
        Returns center frequencies of filter bank channels

        """
        nwin, nshift, nfft, fs, noverlap = self.__get_fft_params()
        _, fc, _ = specgram(np.zeros((2*nwin,)), NFFT=nfft, Fs=fs,
                            noverlap=noverlap)
        if self.f_cutoff is not None:
            valid = np.logical_and(fc >= self.f_cutoff[0],
                                   fc <= self.f_cutoff[1])
            fc = fc[valid]
        return fc

    def get_samplerate(self):
        """
        Returns sample rate of filter bank output
        """

        return self.samplerate

    def process(self, signal):
        """Short-time Fourier transform of time signal"""

        x, _, t0 = self._parse_arguments(signal)

        nwin, nshift, nfft, fs, noverlap = self.__get_fft_params()

        if x.ndim == 1:
            # x is a time signal
            Pxx, f, t = self.__calc_stft(x)
            if self.f_cutoff is not None:
                valid = np.logical_and(f >= self.f_cutoff[0],
                                       f <= self.f_cutoff[1])
                f = f[valid]
                Pxx = Pxx[:, valid]

        else:
            # x contains several channels (one per column) or is
            # a spectrogram and we transform along each frequency channel
            nt, nc = x.shape
            Pxx = None
            for i in range(nc):
                tmp, f, t = self.__calc_stft(x[:, i])

                if self.f_cutoff is not None:
                    valid = np.logical_and(f >= self.f_cutoff[0],
                                           f <= self.f_cutoff[1])
                    f = f[valid]
                    tmp = tmp[:, valid]

                if Pxx is None:
                    nt = len(t)
                    nf = len(f)
                    Pxx = np.zeros((nt, nc, nf), dtype=tmp.dtype)

                Pxx[:, i, :] = tmp

        # Create output signal
        if isinstance(signal, (np.ndarray, Stimulus)):
            fs_spec = fs / nshift
            spec = Spectrogram(Pxx, fs_spec, time=t0 + t, frequency=f)

        elif isinstance(signal, Spectrogram):
            fs_mod = fs
            spec = ModulationSpectrogram(Pxx, fs_mod, time=t0 + t,
                                         f_center=signal.get_frequency(),
                                         f_mod=f)

        if self.normalize:
            spec.normalize()

        return spec

    def __get_fft_params(self):

        # Check attributes for units
        if self.use_quantities:
            if isinstance(self.samplerate, pq.Quantity):
                fs = self.samplerate.rescale(pq.Hz).magnitude

            else:
                raise ValueError('samplerate must be a quantity if '
                                 'use_quantities is set to True')

            if isinstance(self.win_len, pq.Quantity):
                win_len_sec = self.win_len.rescale(pq.s).magnitude
                nwin = int(np.round(win_len_sec * fs))

            else:
                raise ValueError('win_len must be a quantity if '
                                 'use_quantities is set to True')

            if isinstance(self.win_shift, pq.Quantity):
                x = self.win_shift.rescale(pq.s).magnitude
                nshift = int(np.round(x * fs))

            else:
                raise ValueError('win_shift must be a quantity if '
                                 'use_quantities is set to True')

        else:
            fs = self.samplerate
            nwin = self.win_len
            nshift = self.win_shift

        noverlap = nwin - nshift
        nfft = self.nfft
        if nfft is None:
            nfft = nwin

        return int(nwin), int(nshift), int(nfft), fs, int(noverlap)

    def __calc_stft(self, x):

        nwin, nshift, nfft, fs, noverlap = self.__get_fft_params()
        spectype = self.spectype.lower()

        # Create energy-normalized window
        win = np.hanning(nwin)
        win /= np.sum(win*win)

        nwin_half = int(np.floor(nwin/2.) + 1)

        if self.mode == 'same':

            n_start = nwin_half - 1
            n_end = n_start

            z_begin = np.zeros((n_start,))
            z_end = np.zeros((n_end,))
            x = np.concatenate((z_begin, x, z_end))

        # The fast way ...
        X = np.multiply(win, segment(x, nwin, nshift))

        if spectype == 'dct':
            if nfft > nwin:
                nn = nfft - nwin
                X = np.concatenate((X, np.zeros((X.shape[0], nn),
                                                dtype=X.dtype)), axis=1)

            if self.dct_type is None or self.dct_type == 2:
                S = dct(X, type=2, axis=1)
            else:
                S = dct(X, type=self.dct_type, axis=1)

        elif 'rand' in spectype:
            if nfft > nwin:
                nn = nfft - nwin
                X = np.concatenate((X, np.zeros((X.shape[0], nn),
                                                dtype=X.dtype)), axis=1)

            # For reproducible results
            rng = np.random.RandomState(0)

            # Create transformation matrix
            nf = int(np.floor(nfft/2.)) + 1
            ntw = X.shape[1]

            if spectype == 'rand':
                T = rng.rand(nf, ntw) + 1j * rng.rand(nf, ntw)
            elif spectype == 'randn':
                T = rng.randn(nf, ntw) + 1j * rng.randn(nf, ntw)
            else:
                raise ValueError('Unknown random spectrum type:', spectype)

            # DC part
            T[0, :] = 1 + 1j * 0

            # Actual transformation
            S = X.dot(T.T)
            S = self._convert_spectrum_type(S, self.spectype)

        else:
            S = np.fft.rfft(X, n=nfft, axis=1)
            S = self._convert_spectrum_type(S, self.spectype)

            # Compenstate for power of omitted negative frequencies except for
            # DC and Nyquist frequencies
            S[1:-1, :] *= 2.

        # Create time and frequency vectors
        if self.mode == 'same':
            t = 1./fs * np.arange(0, S.shape[0]*nshift, nshift)
        else:
            t = 1./fs * np.arange(nwin/2., nwin/2. + S.shape[0]*nshift, nshift)

        f = np.linspace(0., fs/2., S.shape[1])

        return S, f, t

