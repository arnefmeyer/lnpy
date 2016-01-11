#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Data base classes
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io.wavfile import read as _waveread

from .util import resample as _resample_fun


class Axis(object):
    """General axis class with values, label, and unit.

    """
    def __init__(self, values=None, label=None, unit=None):
        self.values = values
        self.label = label
        self.unit = unit


class Signal(object):
    """General signal class

    Parameters
    ----------
    data : array-like
        Signal data

    samplerate : float
        Samplerate of the signal (in Hz)

    time : array-like
        Vector with time stamps

    All other keyword arguments will be added as annotations.

    """

    def __init__(self, data=None, samplerate=2., time=None, **kwargs):
        self.data = data
        self.samplerate = samplerate
        self.time = time
        self.annotations = kwargs

    def get_data(self):
        """return data array"""

        return self.data

    def get_samplerate(self):
        """return samplerate of signal"""

        return self.samplerate

    def get_time(self):
        """sample time stamps"""

        return self.time

    def get_axis(self, label):
        """return Axis for a given dimensions"""

        if label.lower() == 'time':
            ax = Axis(label='Time', unit='s', values=self.time)

        return ax

    def clip(self, lower=None, upper=None):
        """Limit range of values to lower and upper, respectively"""

        if lower is not None:
            self.data[self.data < lower] = lower

        if upper is not None:
            self.data[self.data > upper] = upper

    def max(self, ignore_inf=False):
        """Return maximum value"""

        if ignore_inf:
            return np.amax(self.data[~np.isinf(self.data)])
        else:
            return np.amax(self.data)

    def min(self, ignore_inf=False):
        """Return minimum value"""

        if ignore_inf:
            return np.amin(self.data[~np.isinf(self.data)])
        else:
            return np.amin(self.data)

    def shape(self):
        """Return shape of data"""

        return self.data.shape

    def __mul__(self, x):
        """Overload left-sided scalar multiplication"""
        self.data *= x

    def __rmul__(self, x):
        """Overload right-sided scalar multiplication"""
        self.data *= x

    def __itruediv__(self, x):
        """Overload division operator"""
        self.data /= x
        return self


class Stimulus(Signal):
    """Stimulus class"""

    def __init__(self, data, samplerate, t0=0., **kwargs):
        super(Stimulus, self).__init__(data=data, samplerate=samplerate,
                                       **kwargs)
        self.t0 = t0

    def get_nchannels(self):

        if self.data.ndim > 1:
            return self.data.shape[1]
        else:
            return 1

    def merge_channels(self, factor=.5):

        self.data = factor * self.data.sum(axis=1)

    def get_axis(self, label):
        """return Axis for a given dimensions"""

        if label.lower() == 'time':
            t = self.t0 + np.arange(self.data.shape[0]) / self.samplerate
            ax = Axis(label='Time', unit='s', values=t)

        return ax

    def get_axes(self):
        """return all axes of this object"""

        ax = self.get_axis('Time')
        return [ax]

    def append(self, x):

        if self.samplerate != x.samplerate:
            raise ValueError('Samplerates do not match!')

        self.data = np.concatenate((self.data, x.get_data()))
        t_new = self.time.max() + x.get_time()
        if x.get_time()[0] == 0:
            t_new += 1. / self.samplerate
        self.time = np.concatenate((self.time, t_new))

    def length(self):

        return self.shape()[0] / self.samplerate

    @staticmethod
    def from_file(wavfile, fs_resample=None, dtype=np.float,
                  resample_algo='scikits', normalize=False):
        """Get stimulus objec from wave file"""

        samplerate, data = _waveread(wavfile)
        if data.dtype != dtype:
            data = data.astype(dtype)

        if normalize:
            data /= np.amax(np.abs(data))

        stim = Stimulus(data, samplerate)
        if fs_resample is not None and samplerate != fs_resample:
            stim.resample(fs_resample, resample_algo)

        len_sec = stim.shape()[0] / stim.samplerate
        stim.time = np.linspace(0, len_sec, stim.shape()[0])

        return stim

    def resample(self, samplerate, algorithm='sinc_fastest'):
        """Resample simulus"""

        if samplerate != self.samplerate:
            self.data = _resample_fun(self.data, self.samplerate,
                                      samplerate, axis=0,
                                      algorithm=algorithm)
            self.samplerate = samplerate

    def normalize(self):
        """Normalize stimulus to standard deviation one"""

        self.data /= self.data.std()

    def time_to_index(self, t0):
        """Return index corresponding to a time instant"""

        if t0 < self.t0 or t0 > self.t0 + self.data.shape[0] / self.samplerate:
            return None

        else:
            t = self.t0 + np.arange(self.data.shape[0]) / self.samplerate
            idx = np.argmin(np.abs(t - t0))
            return idx

    def show(self, ax=None, show_now=True, subsample_factor=None,
             vmin=None, vmax=None):

        if ax is None:
            ax = plt.gca()

        t = self.t0 + np.arange(self.data.shape[0]) / self.samplerate

        if subsample_factor is not None and subsample_factor > 1:
            subsample_factor = int(np.ceil(subsample_factor))
            ax.plot(t[::subsample_factor],
                    self.data[::subsample_factor], 'k')
        else:
            ax.plot(t, self.data, 'k')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        ax.set_xlim(np.amin(t), np.amax(t))
        if vmin is None:
            vmin = -1.05 * np.amax(np.abs(self.data))
        if vmax is None:
            vmax = 1.05 * np.amax(np.abs(self.data))
        ax.set_ylim(vmin, vmax)

        ax.tick_params(axis='both', labelsize=10)

        if show_now:
            plt.show()

    def __str__(self):

        s = "Stimulus object\n"
        s += "  fs: %0.2f Hz\n" % self.samplerate
        s += "  samples: %d\n" % self.data.shape[0]
        s += "  t0: " + str(self.t0) + " s"

        return s


class Spectrogram(Signal):
    """Spectrogram class

        Parameters
        ----------

        data : numpy.ndarray
            2D numpy array with dimensionality time x frequency

        fs : float
            Sampling frequency in Hz

        time : numpy.ndarray, optional
            Time steps (in seconds). If not given, the time steps
            will be created using the sampling frequency.

        frequency : numpy.ndarray, optional
            Vector with center frequencies (in Hz).

    """

    def __init__(self, data, samplerate, time=None, frequency=None,
                 **kwargs):
        super(Spectrogram, self).__init__(**kwargs)

        self.data = data
        self.samplerate = samplerate
        self.time = time
        self.frequency = frequency

    def get_frequency(self):
        """return frequency vector"""

        return self.frequency

    def get_time(self):
        """return frequency vector"""

        return self.time

    def get_axis(self, label):
        """return Axis for a given dimensions"""

        if label.lower() == 'time':
            ax = Axis(label='Time', unit='s', values=self.time)

        elif label.lower() == 'frequency':
            ax = Axis(label='Frequency', unit='Hz', values=self.frequency)

        return ax

    def get_axes(self):
        """return all axes of this object"""

        axx = [self.get_axis('Time'), self.get_axis('Frequency')]
        return axx

    def scale(self, scaling='dB', dynamic_range=60., fill_inf=None,
              fill_nan=None):
        """Log-scaling of coefficient"""

        x = self.data
        x = 20 * np.log10(x)
        x[np.isnan(x)] = -np.inf
        if dynamic_range is not None:
            xmax = np.amax(x)
            x[x < xmax - dynamic_range] = xmax - dynamic_range
        self.data = x

        if fill_inf is not None:
            self.data[np.isinf(self.data)] = fill_inf

        if fill_nan is not None:
            self.data[np.isnan(self.data)] = fill_nan

    def normalize(self):
        """Normalize each frequency channel to standard deviation one"""

        self.data /= self.data.std(axis=0)

    def resample(self, fs, algorithm='scikits'):
        """Resample spectrogram in temporal direction"""

        if fs != self.samplerate:

            t = self.time
            if t is not None:
                t0 = self.time[0]
                t1 = self.time[-1]

            else:
                t0 = 0.
                t1 = (self.data.shape[0] - 1) / self.samplerate

            self.data = _resample_fun(self.data, self.samplerate,
                                      fs, axis=0, algorithm=algorithm)

            self.time = np.linspace(t0, t1, self.data.shape[0])
            self.samplerate = fs

    def time_to_index(self, t0):
        """Return index corresponding to a time instant"""

        t = self.time
        if t is None:
            t = np.arange(self.data.shape[0]) / self.samplerate

        if t0 < np.amin(t) or t0 > np.amax(t):
            return None

        else:
            idx = np.argmin(np.abs(t - t0))
            return idx

    def append(self, spec):

        if self.samplerate != spec.samplerate:
            raise ValueError('Cannot append spectra with different '
                             'sample rates!')

        self.data = np.vstack((self.data, spec.get_data()))
        t_new = self.time.max() + 1./self.samplerate + spec.get_time()
        self.time = np.concatenate((self.time, t_new))

    def segment(self, seg_len, shift, order='C'):
        """segment spectrogram into overlapping parts

            Parameters
            ----------
            seg_len : int
                segment length in samples

            shift : int
                segment shift in samples

            Returns
            -------
            X : ndarray
                the segmented spectrogram

        """

        X = _segment(self.data.ravel(order=order), seg_len, shift)

        return X

    def show(self, ax=None, f_ticks=None, show_now=True, colorbar=True,
             interpolation='nearest', fscale='kHz', logfscale=True,
             t_lim=None, vmin=None, vmax=None, scaling=None,
             f_ticks_base=250., cmap=None, t_ticks=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.gca()

        else:
            fig = ax.figure

        tt = self.time
        if isinstance(tt, Axis):
            tt = tt.values.base
        elif hasattr(tt, 'units'):
            tt = tt.base

        t0 = 0.
        t1 = (self.data.shape[0] - 1) / self.samplerate
        if tt is not None:
            t0 = np.amin(tt)
            t1 = np.amax(tt)

        fc = self.frequency
        if fc is None:
            fc = np.linspace(0., self.samplerate/2., self.data.shape[1])
        elif isinstance(fc, Axis):
            fc = fc.values.base
        elif hasattr(fc, 'units'):
            fc = fc.base

        if f_ticks is None:
            if logfscale is True:
                tmp = np.arange(0, np.floor(np.log2(fc[-1])))
                f_ticks = f_ticks_base * np.power(2., tmp)
            else:
                f_ticks = self.get_axis('frequency').values

        if fscale == 'kHz':
            fc = fc / 1000.
            f_ticks /= 1000.

        X = self.data
        if scaling is not None:
            if scaling.lower() in ['db', 'decibel', 'decible']:
                X = 20 * np.log10(X)
                vmax = X.max()
                vmin = vmax - 60.
            elif scaling.lower() in ['sqrt3']:
                X = np.power(X, 1./3.)

        else:
            if vmax is None:
                vmax = X.max()
            if vmin is None:
                vmin = X.min()

        if cmap is None:
            cmap = plt.cm.RdBu_r

        if logfscale:
            extent = [t0, t1, 1, len(fc)]
        else:
            extent = [t0, t1, fc.min(), fc.max()]
        im = ax.imshow(X.T, aspect='auto', origin='lower',
                       extent=extent, interpolation=interpolation,
                       vmin=vmin, vmax=vmax, cmap=cmap)

        if t_ticks is not None:
            ax.set_xticks(t_ticks)

        if logfscale:
            invfunc = interp1d(fc, np.arange(0, len(fc)), kind='cubic')
            f_ticks = f_ticks[np.logical_and(f_ticks >= np.amin(fc),
                                             f_ticks <= np.amax(fc))]
            fc_loc = invfunc(f_ticks)
            ax.yaxis.set_ticks(fc_loc)

        else:
            ax.yaxis.set_ticks(f_ticks)

        ax.yaxis.set_ticklabels(f_ticks)

        ax.set_xlabel('Time (s)')
        if fscale is 'kHz':
            ax.set_ylabel('Frequency (kHz)')
        else:
            ax.set_ylabel('Frequency (Hz)')
        ax.tick_params(axis='both', labelsize=10)

        if t_lim is not None:
            ax.set_xlim(t_lim[0], t_lim[1])

        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.locator = MaxNLocator(5)
            cbar.update_ticks()
            cbar.ax.tick_params(axis='both', labelsize=10)

        if show_now:
            plt.show()

        elif colorbar:
            return (fig, im, cbar)

        else:
            return fig, im

    def smooth(self, n=4):

        win = signal.hanning(n, sym=True)
        win /= np.sum(win)

        K = self.data

        M, N = K.shape
        for i in range(M):
            K[i, :] = np.convolve(K[i, :], win, mode='same')
        for j in range(N):
            K[:, j] = np.convolve(K[:, j], win, mode='same')

    def __str__(self):
        s = ""
        s += "Spectrogram object:\n"
        s += "  fs: %0.2f Hz\n" % self.samplerate
        s += "  samples: %d\n" % self.data.shape[0]
        if self.time is not None:
            s += "  time: [" + str(self.time[0]) + " s, " + \
                str(self.time[-1]) + " s]\n"
        if self.frequency is not None:
            s += "  frequency:" + str(self.frequency) + " Hz"
        return s


class ModulationSpectrogram(Signal):
    """Modulation spectrogram class

        Parameters
        ----------

        data : numpy.ndarray
            3D numpy array with dimensionality time x freq. x mod. freq.

        fs : float
            Sampling frequency in Hz

        t0 : float
            Temporal offset in s

    """

    def __init__(self, data, samplerate, time=None, f_center=None,
                 f_mod=None, **kwargs):
        super(ModulationSpectrogram, self).__init__(**kwargs)

        self.data = data
        self.samplerate = samplerate
        self.time = time
        self.f_center = f_center
        self.f_mod = f_mod

    def normalize(self):
        """Normalize modulation channels to standard deviation one"""

        nfm = self.data.shape[2]
        for i in range(nfm):
            self.data[:, :, i] /= np.std(self.data[:, :, i])

    def time_to_index(self, t0):
        """Return index corresponding to a time instant"""

        t = self.time
        if t is None:
            t = np.arange(self.data.shape[0]) / self.samplerate

        if t0 < np.amin(t) or t0 > np.amax(t):
            return None

        else:
            idx = np.argmin(np.abs(t - t0))
            return idx

    def get_frequency(self, which):
        """return frequency vector"""

        if which.lower() in ['center', 'f_center', 'f_c', 'fc']:
            f = self.f_center

        elif which.lower() in ['mod', 'modulation', 'f_mod', 'fm', 'f_m']:
            f = self.f_mod

        return f

    def get_axis(self, label):
        """return Axis for a given dimensions"""

        if label.lower() in ['t', 'time', 'temporal']:
            ax = Axis(label='Time', unit='s', values=self.time)

        elif label.lower() in ['center', 'f_center', 'f_c', 'fc']:
            ax = Axis(label='Center frequency', unit='Hz',
                      values=self.f_center)

        elif label.lower() in ['mod', 'modulation', 'f_mod', 'fm', 'f_m']:
            ax = Axis(label='Modulation frequency', unit='Hz',
                      values=self.f_mod)

        return ax

    def get_axes(self):
        """return all axes of this object"""

        axx = [self.get_axis('Time'), self.get_axis('f_center'),
               self.get_axis('f_mod')]
        return axx

    def append(self, modspec):

        if self.samplerate != modspec.samplerate:
            raise ValueError('Cannot append modulation spectra with '
                             'different sample rates!')

        self.data = np.concatenate((self.data, modspec.get_data()))
        t_new = self.time.max() + 1./self.samplerate + modspec.get_time()
        self.time = np.concatenate((self.time, t_new))

    def segment(self, timesteps=None, win_len=None):
        """segment spectrogram into overlapping parts

            Parameters
            ----------
            seg_len : int
                segment length in samples

            shift : int
                segment shift in samples

            timesteps : list or ndarray
                the relative indices to be considered for each segment

            Returns
            -------
            X : ndarray
                the segmented modulation spectrogram

        """

        if timesteps is None:
            timesteps = np.array([0], dtype=np.int)

        else:
            timesteps = np.array(timesteps)

        if win_len is None:
            win_len = np.abs(timesteps)
            win_len = np.amax([win_len, 1])

        data = self.data
        nt, nfc, nfm = data.shape
        nf = nfc * nfm

        nsteps = timesteps.shape[0]
        ntt = nt - win_len + 1
        X = np.zeros((ntt, nf * nsteps))

        for j in range(win_len-1, nt):
            tmp = data[j + timesteps, :, :]
            X[j - win_len + 1, :] = tmp.flatten()

        return X

    def get_slice(self, t):

        idx = np.argmin(np.abs(self.time - t))

        return self.data[idx, :, :]

    def show_slice(self, t0, ax=None, show_now=True, colorbar=True,
                   fc_ticks=None, fm_ticks=None, fc_scale='kHz',
                   interpolation='nearest', vmin=None, vmax=None):

        if ax is None:
            ax = plt.gca()

        fc = self.get_frequency('f_c').copy()
        if fc is None:
            fc = np.linspace(0., self.samplerate/2., self.data.shape[1])

        if fc_ticks is None:
            values = np.arange(0, np.floor(np.log2(fc[-1])))
            fc_ticks = 250. * np.power(2., values)

        if fc_scale == 'kHz':
            fc /= 1000.
            fc_ticks /= 1000

        fm = self.get_frequency('f_m').copy()

        extent = [fm.min(), fm.max(), 1, len(fc)]
        S = self.get_slice(t0)

        maxabs = np.amax(np.abs(S))
        if vmax is None:
            vmax = maxabs
        if vmin is None:
            vmin = -vmax

        im = ax.imshow(S, aspect='auto', origin='lower',
                       extent=extent, interpolation=interpolation,
                       vmin=vmin, vmax=vmax)

        invfunc = interp1d(fc, np.arange(0, len(fc)), kind='cubic')
        fc_ticks = fc_ticks[np.logical_and(fc_ticks >= np.amin(fc),
                                           fc_ticks <= np.amax(fc))]
        fc_loc = invfunc(fc_ticks)
        ax.yaxis.set_ticks(fc_loc)
        ax.yaxis.set_ticklabels(fc_ticks)

        ax.xaxis.set_major_locator(MaxNLocator(3))

        ax.set_xlabel('Modulaton frequency (Hz)')
        if fc_scale is 'kHz':
            ax.set_ylabel('Center frequency (kHz)')
        else:
            ax.set_ylabel('Center frequency (Hz)')
        ax.tick_params(axis='both', labelsize=10)

        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.locator = MaxNLocator(5)
            cbar.update_ticks()

        if show_now:
            plt.show()

        elif colorbar:
            return (im, cbar)

        else:
            return im

    def show_all_slices(self, max_slices_per_row=6, **kwargs):

        time = self.get_axis('time').values
        n_slices = time.shape[0]

        n_rows = 1
        n_cols = n_slices
        if n_slices > max_slices_per_row:
            n_cols = max_slices_per_row
            n_rows = int(np.ceil(float(n_slices) / n_cols))

        fig, axarr = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True,
                                  sharey=True)

        maxval = np.max(np.abs(self.data))
        if 'vmin' not in kwargs.keys():
            kwargs.update({'vmin': -maxval})

        if 'vmax' not in kwargs.keys():
            kwargs.update({'vmax': maxval})

        for i, t0 in enumerate(time.tolist()):

            if n_slices > 1:
                ax = axarr.ravel()[i]
            else:
                ax = axarr

            self.show_slice(t0, ax=ax, colorbar=False, **kwargs)

            if i > 0:
                ax.set_xlabel('')
                ax.set_ylabel('')

        if isinstance(axarr, np.ndarray) and n_rows * n_cols > time.shape[0]:
            for ax in axarr.flatten()[time.shape[0]]:
                ax.axis('off')

        fig.tight_layout()

        return fig

    def smooth(self, n=4):

        win = signal.hanning(n, sym=True)
        win /= np.sum(win)

        K = self.data
        nt, M, N = K.shape
        for t in range(nt):
            for i in range(M):
                K[t, i, :] = np.convolve(K[t, i, :], win, mode='same')
            for j in range(N):
                K[t, :, j] = np.convolve(K[t, :, j], win, mode='same')

    def __str__(self):
        s = ""
        s += "Spectrogram object:\n"
        s += "  fs: %0.2f Hz\n" % self.samplerate
        s += "  samples: %d\n" % self.data.shape[0]
        if self.time is not None:
            s += "  time: [" + str(self.time[0]) + " s, " + \
                str(self.time[-1]) + " s]\n"
        if self.frequency is not None:
            s += "  frequency:" + str(self.frequency) + " Hz"
        return s


class GaborSpectrogram(ModulationSpectrogram):
    """Modulation spectrogram class for Gabor filterbank

        Parameters
        ----------

        data : numpy.ndarray
            3D numpy array with dimensionality time x freq. x gabor filter

        fs : float
            Sampling frequency in Hz

        t0 : float
            Temporal offset in s

    """

    def __init__(self, data, samplerate, time=None, f_center=None,
                 filters=None):
        super(GaborSpectrogram, self).__init__(data, samplerate, time=time,
                                               f_center=f_center)

        self.f_mod = np.arange(self.data.shape[2])
        self.filters = filters

    def segment(self, win_len=None):
        """segment spectrogram into overlapping parts

            Parameters
            ----------
            seg_len : int
                segment length in samples

            shift : int
                segment shift in samples

            timesteps : list or ndarray
                the relative indices to be considered for each segment

            Returns
            -------
            X : ndarray
                the segmented modulation spectrogram

        """

        data = self.data
        nt, nfc, n_filt = data.shape

        filters = self.filters
        filt_len = np.asarray([f.get_coef().shape[0] for f in filters])

        # Cover the whole time window preceding the response
        # with a temporal filter overlap of 0.5
        t_indices = []
        n_ind_tot = 0
        n_filters = len(filters)
        for i in range(n_filters):

            f_len = filt_len[i]
            ts = np.arange(-f_len/2., -win_len + f_len/2., -f_len/2.)

            t_ind = np.round(ts).astype(np.int)
            t_indices.append(t_ind)

            n_ind_tot += t_ind.shape[0]

        # Segment features
        ntt = nt - win_len + 1
        X = np.zeros((ntt, nfc * n_ind_tot))
        for j in range(win_len-1, nt):

            xx = []
            for i in range(n_filters):
                tmp = data[j + t_indices[i], :, i]
                xx.append(tmp)

            X[j - win_len + 1, :] = np.concatenate(tuple(xx)).flatten()

        return X, t_indices


def _segment(data, seg_len, shift):
    """Rearrange vector by buffering overlapping segments as rows"""

    total_len = data.shape[0]
    num_seg = np.int(np.ceil((total_len - seg_len + shift) / shift))
    out = np.zeros([num_seg, seg_len])
    for seg in range(0, num_seg):
        tmp = data[seg*shift:seg*shift+seg_len]
        N = tmp.shape[0]
        out[seg, :N] = tmp
    return out
