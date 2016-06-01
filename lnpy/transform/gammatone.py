#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Wrapper code around Volker Hohmann's gammatone filterbank implementation:

    Volker Hohmann, Frequency analysis and synthesis using a Gammatone
    filterbank, Acta acustica/Acustica, vol. 88, no. 3, pp. 433-442, 2002.

    Tobias Herzke and Volker Hohmann, Improved numerical methods for gammatone
    filterbank analysis and synthesis,  Acta Acustica united with Acustica,
    vol. 93, pp. 498-500, 2007.

    Remark
    ------
    Due to copyright restrictions, the C-code must be downloaded from

    http://www.uni-oldenburg.de/mediphysik-akustik/mediphysik/downloads/

    and placed into the src/gammatone folder.

"""

import numpy as np
import math

import wrap_gtfb as wrapper

from base import BaseTransform
from ..base import Stimulus, Spectrogram, ModulationSpectrogram


GFB_L = 24.7   # see equation (17) in [Hohmann 2002]
GFB_Q = 9.265  # see equation (17) in [Hohmann 2002]


def _hz2erbscale(f):
    """Hertz to ERB frequency conversion"""

    return GFB_Q * np.log(1. + np.asarray(f) / (GFB_L * GFB_Q))


def _erbscale2hz(f):
    """ERB to Hertz frequency conversion"""

    return (np.exp(f / GFB_Q) - 1.) * (GFB_L * GFB_Q)


class GammatoneFilter:
    """Gammatone filter class

        Parameters
        ----------
        f_center : float
            Filter center frequency (in Hz)

        samplerate : float
            Sampling frequency (in Hz)

        order : int
            Filter order (default: 4)

        bw_factor : float
            Bandwidth factor (default: 1.)

    """

    def __init__(self, f_center, samplerate, order=4, bw_factor=1.0):

        self.f_center = f_center
        self.samplerate = samplerate
        self.order = order
        self.bw_factor = bw_factor

        fc_erb = (GFB_L + self.f_center / GFB_Q) * self.bw_factor

        # equation (14), line 3 [Hohmann 2002]:
        a = np.pi * math.factorial(2.*order - 2.) * 2.**-(2.*order - 2.)
        b = math.factorial(order - 1.)**2
        a_gamma = a / b

        # equation (14), line 2 [Hohmann 2002]:
        b = fc_erb / a_gamma

        # equation (14), line 1 [Hohmann 2002]:
        lamda = np.exp(-2*np.pi*b / samplerate)

        # equation (10) [Hohmann 2002]:
        beta = 2 * np.pi * f_center / samplerate

        # equation (1), line 2 [Hohmann 2002]:
        self.coef = lamda * np.exp(0 + 1j*beta)
        if not np.isscalar(self.coef):
            # make sure that coef is a scalar
            self.coef = self.coef.item()

        # normalization factor from section 2.2 (text) [Hohmann 2002]:
        self.normalization_factor = 2. * (1 - np.abs(self.coef)) ** order
        self.state = np.zeros(order, dtype=np.complex, order='c')

    def get_center_frequency(self):
        return self.f_center

    def get_order(self):
        return self.order

    def get_coefficients(self):
        return self.coef

    def get_norm_factor(self):
        return self.normalization_factor

    def get_state(self):
        return self.state

    def reset_state(self):
        self.state[:] = 0

    def process(self, x, spectype='complex'):

        fb = GammatoneFilterbank(samplerate=self.samplerate, f_cutoff=(0, 0),
                                 order=self.order, spectype=spectype)

        fb.filters = [self]
        return fb.process(x)


class GammatoneFilterbank(BaseTransform):
    """Gammatone filterbank class

        Parameters
        ----------
        samplerate : float
            Sampling frequency of input signal (in Hz; defaul: 16000)

        f_cutoff : list or tuple
            Lower and upper cutoff frequencies (in Hz; default: (250, 6700))

        f_carrier : float
            Carrier frequency of filters (in Hz; default: 1000)

        filt_per_erb : float
            Filter density per ERB band (default: 1)

        order : int
            Filter order (default: 4)

        bw_factor : float
            Bandwidth of filters (default: 1)

        spectype : {'complex', 'magnitude', 'real', 'imag'}
            Type of spectrum returned by the process function
            (default: 'complex')

        warm_start : boolean
            Use filter states from previous run? (default: false)

        chunk_size : int
            Process signal in multiple chunks or a single chunk if
            chunk_size is None (default: None). Note: currently only
            works for signals with a single channel.
    """

    def __init__(self, samplerate=16000., f_cutoff=(250., 6700.),
                 f_carrier=1000., filt_per_erb=1.0, order=4,
                 bw_factor=1., spectype='complex', warm_start=False,
                 chunk_size=None, verbose=False, dtype=np.float64):

        order = np.uint8(order)

        self.samplerate = samplerate
        self.f_cutoff = f_cutoff
        self.filt_per_erb = filt_per_erb
        self.order = order
        self.bw_factor = bw_factor
        self.spectype = spectype.lower()
        self.f_carrier = f_carrier
        self.warm_start = warm_start
        self.chunk_size = chunk_size
        self.verbose = verbose

        if isinstance(f_cutoff, (int, float)):
            f_center = [f_cutoff]

        elif isinstance(f_cutoff, np.ndarray):
            f_center = f_cutoff.ravel()

        else:
            # Compute filter spacing on ERB scale
            f_cutoff_erb = _hz2erbscale(f_cutoff)
            f_carrier_erb = _hz2erbscale(f_carrier)

            erbs_below = f_carrier_erb - f_cutoff_erb[0]
            n_filt_below = np.floor(erbs_below * filt_per_erb)

            f_start_erb = f_carrier_erb - n_filt_below / filt_per_erb
            fc_erb = np.arange(f_start_erb, f_cutoff_erb[1], 1. / filt_per_erb)

            f_center = _erbscale2hz(fc_erb)

        # Create filters
        self.filters = []
        for cf in f_center:
            self.filters.append(GammatoneFilter(cf, self.samplerate,
                                                self.order, self.bw_factor))

    def get_center_frequencies(self):
        """Return center frequencies of filterbank"""
        fc = [f.get_center_frequency() for f in self.filters]
        return np.asarray(fc)

    def get_samplerate(self):
        """Return input sampling frequency in Hz"""
        return self.samplerate

    def to_string(self):
        """Return unique parameter string for filterbank object"""

        param_str = 'GtFb_fs_%d_Hz_fl_%d_Hz_fu_%d_Hz_fc_%d_Hz_' \
                    'filterb_%0.1f_order_%d_bw_%0.2f_%s' % \
                    (self.samplerate, self.f_cutoff[0], self.f_cutoff[1],
                     self.f_carrier, self.filt_per_erb, self.order,
                     self.bw_factor, self.spectype)
        return param_str

    def analyze(self, signal):
        """Decompose time signal into frequency bands

            Parameters
            ----------
            signal : numpy array or Signal object

            Returns
            -------
            spec : lnpy.base.Spectrogram
                If signal is a numpy array the filtered signal (t x f),
                the time points and the frequency vector are returned as
                tuple. If signal is a list, a list of tuples will be
                returned.
        """

        x, _, t0 = self._parse_arguments(signal)

        if x.ndim == 1:

            if not x.flags['C_CONTIGUOUS']:
                x = np.ascontiguousarray(x)

            pars = self._unpack_params()
            spectype = self.spectype
            chunk_size = self.chunk_size
            fs = self.samplerate

            N = x.shape[0]
            if chunk_size is None:
                # single chunk
                chunk_ind = [(0, N - 1)]

            else:
                # multiple (equal-sized) chunks
                samples_per_chunk = int(np.ceil(chunk_size * fs))
                n_chunks = int(np.ceil(float(N) / samples_per_chunk))

                chunk_ind = []
                for i in range(n_chunks):
                    chunk_ind.append((i*samples_per_chunk,
                                      min((i+1) * samples_per_chunk-1, N-1)))

            n_channels = len(self.filters)
            tmp = np.zeros((N, n_channels), dtype=np.float)
            for i, ii in enumerate(chunk_ind):

                if self.verbose:
                    print "processing chunk {}/{}".format(i+1, n_chunks)

                xx = wrapper.process(n_channels, self.order,
                                     pars[0], pars[1], pars[2], pars[3],
                                     pars[4], x[ii[0]:ii[1]])

                tmp[ii[0]:ii[1], :] = self._convert_spectrum_type(xx,
                                                                  spectype)

            if self.warm_start:
                self._update_filter_states(pars[3], pars[4])

        else:
            # Process each channel separately
            n_channels = x.shape[1]
            tmp = None
            pp = self._unpack_params()
            for i in range(n_channels):

                xx = x[:, i]

                if not xx.flags['C_CONTIGUOUS']:
                    xx = np.ascontiguousarray(xx)

                # warm_start option does not make sense; pass copy of states
                y = wrapper.process(len(self.f_center), self.order,
                                    pp[0], pp[1], pp[2], pp[3].copy(),
                                    pp[4].copy(), xx)
                y = self._convert_spectrum_type(y, self.spectype)

                if tmp is None:
                    tmp = np.zeros((y.shape[0], n_channels, y.shape[1]),
                                   dtype=y.dtype)
                tmp[:, i, :] = y

        t = np.arange(0, tmp.shape[0], dtype=x.dtype) / self.samplerate
        f = self.get_center_frequencies()

        if isinstance(signal, (np.ndarray, Stimulus)):
            spec = Spectrogram(tmp, self.samplerate, time=t, frequency=f)

        else:
            fc = signal.frequency
            spec = ModulationSpectrogram(tmp, self.samplerate, time=t,
                                         f_center=fc, f_mod=f)

        return spec

    def process(self, signal):
        """For backward compatibility"""
        return self.analyze(signal)

    def reset_states(self):
        """Resets states of all filters"""
        for f in self.filters:
            f.reset_state()

    def _unpack_params(self):
        """Get filter parameters as vectors"""
        cr = np.asarray([f.get_coefficients().real for f in self.filters])
        ci = np.asarray([f.get_coefficients().imag for f in self.filters])
        nf = np.asarray([f.get_norm_factor() for f in self.filters])
        sr = np.asarray([f.get_state().real for f in self.filters]).ravel()
        si = np.asarray([f.get_state().imag for f in self.filters]).ravel()

        return cr, ci, nf, sr, si

    def _update_filter_states(self, sr, si):
        """Update states of filter objects"""
        for i, f in enumerate(self.filters):
            order = f.get_order()
            ind = np.arange(order)
            state = sr[i*order + ind] + 1j*si[i*order + ind]
            f.set_state(state)
