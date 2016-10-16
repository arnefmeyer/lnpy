#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Some helper for LNP model estimation
"""

from __future__ import division

import warnings
from os import listdir
from os.path import isfile, join
from copy import deepcopy

import numpy as np
import array
import quantities as pq
from scipy.signal import convolve2d
from pylab import mlab
from PIL import Image

from ..base import Axis
from ..base import Stimulus

import fast_tools


class ModelBootstrapper(object):
    """Bootstrapping of RF model

    Parameters
    ==========

    model : neuropy.receptivefields.BaseEstimator
        a fitted model

    size : float, optional
        fraction of the data used for each bootstrap run. Defaults to 0.1

    runs : int, optional
        number of bootstrap runs. Defaults to 100.

    verbose : bool, optional
        tell something. Defaults to False.

    """

    def __init__(self, model, size=0.1, runs=100, verbose=False,
                 random_seed=None):

        self.size = size
        self.runs = runs
        self.verbose = verbose
        self.random_seed = random_seed

        if hasattr(model, 'optimize'):
            self.model = deepcopy(model)
            self.model.optimize = False
            self.model.verbose = False
        else:
            self.model = model

    def process(self, X, Y, return_models=False):
        """run bootstrapping on data with trained model

        Parameters
        ==========

        X : array-like
            stimulus data with dimensions: observations x features

        Y : array-like
            spike data with dimensions: observations x trials

        return_models : bool, optional
            return models for each run if True. Otherwise the standard
            deviation across all runs. Defaults to False

        Returns
        =======

        W : array-like
            array with bootstrapped models (one per row)

        """

        n_samples, n_features = X.shape

        model = self.model
        n_runs = self.runs
        bs_size = self.size
        bs_samples = int(np.round(bs_size * n_samples))

        rng = np.random.RandomState(self.random_seed)

        if self.verbose:
            print "----------------- Bootstrapping model -----------------",

        W = np.zeros((n_runs, n_features))
        for i in range(n_runs):

            if self.verbose and (i+1) % 10 == 0:
                update_progress((i+1) / n_runs)

            perm_stim = rng.permutation(n_samples)
            stim_idx = perm_stim[:bs_samples]

            perm_resp = rng.permutation(stim_idx.shape[0])
            resp_idx = stim_idx[perm_resp]

            if Y.ndim == 1:
                y = Y[resp_idx]
            else:
                y = Y[resp_idx, :]

            model.fit(X[stim_idx, :], y)

            w = model.coef_
            W[i, :] = w.flatten()

        if return_models:
            return W
        else:
            return np.std(W, axis=0)


class DataConverter(object):
    """Converts neo block to stimulus and response matrices

    Parameters
    ----------
    transform : An instance derived from class BaseTransform
        Object used for time-frequency or modulation analysis

    win_len : float
        the temporal window length preceding the response

    sample_rate : float
        sampling frequency of the features in the stimulus matrix

    freq_lim : list
        upper and lower frequency limits [obsolete; use samplerate of
        transform]

    filters_per_erb : float
        number of filters per ERB of gammatone filterbank [obsolete]

    verbose : boolean
        output some information

    n_samples : int
        the maximum number of samples

    scaling : {'dB', 'decible', 'none'}
        scaling of spectro-temporal amplitudes

    dynamic_range : float
        limit amplitude range (after scaling) to dynamic_range below max.
        value

    center : boolean
        normalize each feature dimension to zero mean

    timesteps : list
        time steps preceding the response for modulation features

    multi_spike_warning : bool, optional
        print warning if more than one spike is located in a time bin

    history_len : float, optional
        length of the post-spike history term in seconds. If history_len > 0,
        the post-spike history will be appended to the end of the feature
        vectors. Defaults to None (no history term).

    """

    def __init__(self, transform, win_len=0.1 * pq.s,
                 samplerate=500 * pq.Hz, verbose=False, n_samples=np.Inf,
                 scaling='dB', dynamic_range=60., center=True,
                 normalize=True, timesteps=None,
                 multi_spike_warning=True):

        self.transform = transform
        self.win_len = win_len
        self.samplerate = samplerate
        self.verbose = verbose
        self.n_samples = n_samples
        self.scaling = scaling
        self.dynamic_range = dynamic_range
        self.center = center
        self.normalize = normalize
        self.multi_spike_warning = multi_spike_warning

    def __get_rescaled_params(self):
        """return parameters rescaled to seconds and Hz"""

        fs_spec = self.samplerate
        if hasattr(fs_spec, 'magnitude'):
            fs_spec = fs_spec.rescale(pq.Hz).magnitude

        win_len = self.win_len
        if hasattr(win_len, 'magnitude'):
            win_len = win_len.rescale(pq.s).magnitude

        return fs_spec, win_len

    def process(self, block):
        """Convert stimulus/response data set to matrices

        Parameters
        ----------
        block : neo block
            Neo block containing stimulus and response data

        Returns
        -------
        XX : numpy array
            n x d array (n samples x d features)

        YY : numpy array
            n x m array (n samples x m trials)

        patch_size : list
            dimensionality of the RF patch

        axes : list
            list of Axis instances

        stim_offset : numpy array
            offset positions of single stimuli
        """

        fs_spec, win_len = self.__get_rescaled_params()
        n_temp = int(np.ceil(fs_spec * win_len))
        transform = self.transform

        max_samples = self.n_samples

        XX = []
        YY = []
        sample_cnt = 0
        for i, seg in enumerate(block.segments):

            # Get stimulus signal
            stim = seg.annotations['wav_signal']

            data = stim.base
            fs_stim = stim.sampling_rate.rescale(pq.Hz).item()
            stim = Stimulus(data, fs_stim)

            if stim.get_samplerate() != transform.get_samplerate():
                stim.resample(transform.get_samplerate())

            # Transform stimulus
            spec = transform.process(stim)

            if spec.get_samplerate() != fs_spec:
                ratio = spec.get_samplerate() / fs_spec
                if ratio > 256:
                    warnings.warn("Sampling rate ratio too high. "
                                  "Resampling two times!")
                    spec.resample(spec.get_samplerate() / 128.)

                spec.resample(fs_spec)

            # Create time-lagged spectrogram
            f_center = spec.get_frequency()
            n_fc = f_center.shape[0]
            n_features = n_temp * n_fc
            X = spec.segment(n_features, n_fc)

            # Create binary (0/1) spike array
            n_samples = X.shape[0]
            n_trials = len(seg.spiketrains)
            Y = np.zeros((n_samples, n_trials), dtype=np.float64)
            t0 = spec.get_time()[0]
            for j, train in enumerate(seg.spiketrains):

                train = train.rescale(pq.s)

                spike_ind = np.array(np.round((train - t0 * pq.s) *
                                     self.samplerate), dtype=int)
                spike_ind -= n_temp - 1
                spike_ind = spike_ind[spike_ind < n_samples]

                for ii in spike_ind:
                    Y[ii, j] += 1

                if self.multi_spike_warning:
                    cnt = np.sum(Y[:, j] > 1)
                    if cnt > 0:
                        warnings.warn("Detected multiple spikes in "
                                      "at least %d time bins" % cnt)

            # Append data to lists
            XX.append(X)
            YY.append(Y)

            sample_cnt += X.shape[0]

            if self.verbose:
                print "  Stimulus matrix: %d temp. steps x %d features" % \
                    (sample_cnt, X.shape[1])
                print "  Spike    matrix: %d temp. steps x %d trials" % \
                    (sample_cnt, Y.shape[1])

            if max_samples >= 1 and sample_cnt >= max_samples:
                break

        stim_len = [x.shape[0] for x in XX]
        stim_offset = np.append(np.array([0], dtype=np.int),
                                np.cumsum(stim_len[:-1]))

        XX = np.concatenate(tuple(XX))
        YY = np.concatenate(tuple(YY))

        if max_samples < 1:
            max_samples = int(np.round(XX.shape[0] * max_samples))

        if XX.shape[0] > max_samples:
            XX = XX[:max_samples, :]
            YY = YY[:max_samples, :]

        # Create RF axes
        len_ms = win_len * 1000
        axes = []

        # Time and center frequency axes
        t = np.linspace(-len_ms, 0, n_temp)
        axes.append(Axis(values=t * pq.ms, label='Time', unit='ms'))
        axes.append(Axis(values=f_center * pq.Hz, label='Frequency',
                         unit='Hz'))

        patch_size = [n_temp, n_fc]

        scaling = self.scaling.lower()
        if scaling in ["db", "decible"]:
            valid = XX > 0
            XX[valid] = 20 * np.log10(XX[valid])
            max_val = np.max(XX[valid])
            XX[valid == 0] = max_val - self.dynamic_range
            XX[XX < max_val - self.dynamic_range] = \
                max_val - self.dynamic_range

        elif 'sqrt' in scaling:
            n = 2
            if len(scaling) > 4:
                n = float(scaling[-1])
            XX = np.power(XX, 1./n)

        if self.center:
            # Center stimulus matrix along each dimension
            XX -= np.mean(XX, axis=0)

        if self.normalize:
            # Normalize data to make regularization easier
            max_norm = np.max(np.sqrt(np.sum(XX * XX, axis=1)))
            XX /= max_norm

        return XX, YY, patch_size, axes, stim_offset

    @staticmethod
    def unwrap_data(n_samples, n_trials, stim_offset, stim_perm):
        """get order of (trial-)unwrapped stimulus and response examples

        Parameters
        ----------
        n_samples : int
            The number of samples

        n_trials : int
            The number of trials

        stim_offset : array-like
            Offset of each stimulus segment in X/Y matrices

        stim_perm : array-like
            Permutation of stimulus segments during experiment
        """

        time_ind = []
        trial_ind = []
        stim_ind = []

        n_stim = stim_offset.shape[0]
        trial_offset = np.zeros((n_stim,), dtype=np.int)
        for i, stim_idx in enumerate(stim_perm):

            i0 = stim_offset[stim_idx - 1]
            if stim_idx == n_stim:
                i1 = n_samples
            else:
                i1 = stim_offset[stim_idx]

            ind = np.arange(i0, i1)
            time_ind.append(ind)

            N = ind.shape[0]

            trial_offset[stim_idx - 1] += 1
            ind = trial_offset[stim_idx - 1] * np.ones((N, ))
            trial_ind.append(ind)

            ind = stim_idx * np.ones((N,))
            stim_ind.append(ind)

        time_ind = np.concatenate(tuple(time_ind)).astype(np.int)
        trial_ind = np.concatenate(tuple(trial_ind)).astype(np.int)
        stim_ind = np.concatenate(tuple(stim_ind)).astype(np.int)

        return time_ind, trial_ind, stim_ind


def create_postfilt_features(y, n_postfilt):
    """creates post-spike filter features from spike response

    Parameters
    ----------
    y : ndarray
        Vector holding response values

    n_postfilt : int
        Length of the post-spike filter (in samples)

    Returns
    -------
    Y : ndarray
        Matrix with post-spike data for every time step in y
    """

    # The first steps don't have a complete spike history
    zeros_pre = np.zeros((n_postfilt,), dtype=y.dtype)
    yy = np.append(zeros_pre, y)

    # Create features for every time step (and don't include current time!)
    Y = segment(yy[:-1], n_postfilt, 1)

    return Y


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


#################################################################
#                   Some test functions                         #
#################################################################
def calcAUC(Y, score, num=250):
    """Calculate area under ROC curve"""

    if Y.ndim == 1:
        return fast_tools.calc_auc(Y, score, num)
    else:
        return fast_tools.calc_auc_trials(Y, score, num)


def calcROC(proj, spikes, n_steps=250):
    """slow way of computing the ROC curve"""

    proj_min = np.amin(proj)
    proj_max = np.amax(proj)
    thresholds = np.linspace(proj_min - 10.**-6, proj_max + 10.**-6, n_steps)
    tpr = np.zeros((n_steps,))
    fpr = np.zeros((n_steps,))
    balance = np.mean(spikes > 0)

    n_trials = 1
    if spikes.ndim > 1:
        n_trials = spikes.shape[1]

    auc = 0.0
    for i, b in enumerate(thresholds):
        pred = np.sign(proj + b)
        tprt = 0.
        fprt = 0.
        for t in range(n_trials):
            if n_trials == 1:
                tprt += np.mean(np.logical_and(spikes > 0, pred > 0))
                fprt += np.mean(np.logical_and(spikes <= 0, pred > 0))

            else:
                tprt += np.mean(np.logical_and(spikes[:, t] > 0, pred > 0))
                fprt += np.mean(np.logical_and(spikes[:, t] <= 0, pred > 0))
        tpr[i] = tprt / n_trials / balance
        fpr[i] = fprt / n_trials / (1. - balance)
        auc = sum((tpr[:-1] + tpr[1:]) / 2. * (fpr[1:] - fpr[:-1]))
    return fpr, tpr, thresholds, auc


def calcCoherence(pred, Y, nfft=256, noverlap=128):
    """Coherence between true and predicted response"""

    if Y.ndim > 1:
        PSTH = np.mean(Y, axis=1)
    else:
        PSTH = Y

    if pred.shape[0] < 2 * nfft:
        nfft = int(np.floor(pred.shape[0]/2))
        noverlap = int(nfft/2)

    cxy, f = mlab.cohere(PSTH, pred, NFFT=nfft, noverlap=noverlap)

    return cxy, f


def calcLogLikelihood(z, Y, dt=1., family='poissonexp'):
    """compute negative log likelihood for a given GLM family"""

    if family.lower() == 'poissonexp':

        z[z < -120] = -120
        z[z > 50] = 50
        rate = np.exp(z)

        if Y.ndim == 2:
            y = np.sum(Y, axis=1)
        else:
            y = Y
        ll = np.sum(y * z - dt * rate)

    if family.lower() == 'poissonexpquad':

        if Y.ndim == 2:
            y = np.sum(Y, axis=1)
        else:
            y = Y

        ll = 0

        ind = z <= 0
        r = np.exp(z[ind])
        ll += np.sum(y[ind] * z[ind] - dt * r)

        ind = z > 0
        u = 1. + z[ind] + np.power(z[ind], 2)
        ll += np.sum(y[ind] * np.log(u) - dt * u)

    elif family.lower() == 'binomlogit':

        rate = 1. / (1 + np.exp(-z))
        eps = np.finfo(z.dtype).eps

        if Y.ndim == 1:
            ll = np.vdot(Y, np.log(rate + eps)) + \
                np.vdot((1. - Y), np.log((1. - rate) + eps))

        else:
            n_trials = Y.shape[1]
            ll = 0.
            for i in range(n_trials):
                y = Y[:, i]
                ll += np.vdot(y, np.log(rate + eps)) + \
                    np.vdot((1. - y), np.log((1. - rate) + eps))

    return ll


def calcDPrime(stim, spikes, k, equal_variance=False):
    """
    Calculate d-prime between spike-conditional and no spike-conditional
    distributions
    """

    proj = np.dot(k, stim.T)
    n_trials = 1
    if spikes.ndim > 1:
        n_trials = spikes.shape[1]
        X = np.tile(proj, (n_trials, 1)).T
        row_idx, col_idx = np.where(spikes > 0)
        proj_spike = X[row_idx, col_idx]
        row_idx, col_idx = np.where(spikes <= 0)
        proj_non = X[row_idx, col_idx]
    else:
        proj_spike = proj[np.where(spikes > 0)[0]]
        proj_non = proj[np.where(spikes <= 0)[0]]

    mean_spike = np.mean(proj_spike)
    mean_non = np.mean(proj_non)

    # Divide by standard deviation
    std_spike = np.std(proj_spike)
    std_non = np.std(proj_non)
    if equal_variance:
        dprime = (mean_spike - mean_non) / std_non
    else:
        dprime = (mean_spike - mean_non) / \
            np.sqrt((std_non * std_non + std_spike * std_spike) / 2.)
    return dprime


def calcMutualInformation(Y, z, n_bins=50, distributions=False,
                          correct_bias=False):
    """
    Calculate mutual information transmitted by the RF k
    """

    if np.sum(Y > 0) == 0:
        if distributions:
            return np.nan, np.nan, np.nan
        else:
            return np.nan

    # Histogram of stimulus projections (all stimulus examples)
    hist_raw, edges_raw = np.histogram(z, n_bins, density=True)
    hist_raw /= np.sum(hist_raw)

    # Histogram of spike-triggered stimulus projections
    N = z.shape[0]
    n_trials = 1
    if Y.ndim > 1:
        n_trials = Y.shape[1]

    hist_spk = np.zeros((n_bins,))
    for trial in range(0, n_trials):
        if n_trials > 1:
            spikes = Y[:, trial] > 0
            proj_spk = z[spikes]
        else:
            spikes = (Y > 0).nonzero()
            proj_spk = z[spikes[0]]

        hist_spk += np.histogram(proj_spk, edges_raw)[0]

    hist_spk /= np.sum(hist_spk)

    # Kullback-Leibler divergence between the two distributions
    if not distributions:
        valid = hist_spk > 0
        mi = np.dot(hist_spk[valid].T, np.log2(np.divide(hist_spk[valid],
                    hist_raw[valid])))
        if correct_bias:
            mi -= n_bins / (2. * np.log(2) * N * n_trials)
        return mi

    else:
        return hist_raw, hist_spk, edges_raw


def calcMI(*args, **kwargs):
    """Short-named version of MI calculation routine"""
    return calcMutualInformation(*args, **kwargs)


def calcKLD(stim_mat, spike_mat, k, num_bins=50, distributions=False,
            correct_bias=False):
    """Calculate KLD between conditional distributions of projections"""

    # Histogram of stimulus projections (all stimulus examples)
    proj_all = np.dot(stim_mat, k)
    proj_min = np.amin(proj_all) - 10. ** -6
    proj_max = np.amax(proj_all) + 10. ** -6

    if np.sum(spike_mat > 0) == 0:
        if distributions:
            return np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan
    # Histogram of spike-triggered stimulus projections
    num_trials = 1
    if spike_mat.ndim > 1:
        num_trials = spike_mat.shape[1]
    hist_spk = np.zeros((num_bins,))
    hist_non = np.zeros((num_bins,))
    for trial in range(0, num_trials):
        if num_trials > 1:
            spike_idx = spike_mat[:, trial] > 0
            non_idx = spike_mat[:, trial] <= 0
            proj_spk = np.dot(stim_mat[spike_idx, :], k)
            proj_non = np.dot(stim_mat[non_idx, :], k)
        else:
            spike_idx = spike_mat > 0
            non_idx = spike_mat <= 0
            proj_spk = np.dot(stim_mat[spike_idx], k)
            proj_non = np.dot(stim_mat[non_idx], k)

        hist_tmp, _ = np.histogram(proj_spk, bins=num_bins,
                                   range=(proj_min, proj_max))
        hist_spk += hist_tmp
        hist_tmp, _ = np.histogram(proj_non, bins=num_bins,
                                   range=(proj_min, proj_max))
        hist_non += hist_tmp

    hist_spk /= np.sum(hist_spk)
    hist_non /= np.sum(hist_non)

    # Kullback-Leibler divergence between the two distributions

    if distributions:
        hist_all, edges_all = np.histogram(proj_all, num_bins, density=True,
                                           range=(proj_min, proj_max))
        return hist_spk, hist_non, edges_all, np.std(proj_all)
    else:
        valid = np.logical_and(hist_spk > 0, hist_non > 0)
        kld = np.dot(hist_spk[valid].T, np.log2(np.divide(hist_spk[valid],
                     hist_non[valid])))
        if correct_bias:
            kld -= num_bins / (2. * np.log(2) * stim_mat.shape[0] * num_trials)
        return kld


def vdotNormal(x, y):
    """Normalized projection (aka correlation) between to arrays

        This functions calculates the normalized projection between two
        vectors x and y. If x and/or are matrices the normalized projection
        between the flattened arrays is calculated.

        Inputs:
            x,y (numpy arrays)

        Outputs:
            z - A scalar between -1. and 1.
    """

    norm_x = np.sqrt(np.sum(x * x))
    norm_y = np.sqrt(np.sum(y * y))

    if norm_x > 0 and norm_y > 0:
        return np.vdot(x.flatten(), y.flatten()) / norm_x / norm_y

    else:
        return 0.


#################################################################
#                   Little helper functions                     #
#################################################################

def segment(data, seg_len, shift, zero_padding=True):
    """Rearranges a vector by buffering overlapping segments as rows

    Parameters
    ==========
        data : array-like
            a vector (or flattened view) of the array to be segmented

        seg_len : int
            length of each segment in samples

        shift : int
            the segment shift in samples

        zero_padding : bool, optional
            append zeros to data if data does not contain enough
            samples to fill all segments. If set to False, the
            last segment will be omitted. Defaults to True.

    """
    total_len = data.shape[0]

    if zero_padding is True:
        num_seg = np.int(np.ceil((total_len - seg_len + shift) / shift))
    else:
        num_seg = np.int(np.floor((total_len - seg_len + shift) / shift))

    out = np.zeros([num_seg, seg_len])
    for i in range(num_seg):
        i0 = i * shift
        i1 = i0 + seg_len
        if i1 <= data.shape[0]:
            out[i, :] = data[i0:i1]

        else:
            j1 = data.shape[0] - i0
            out[i, :j1] = data[i0:]

    return out


def calcSWDF(wf, bins=25):
    """
    Calculates spike waveform density function of spike waveform matrix
        every row in the matrix is assumed to represent a spike waveform
    """

    # Calculate histograms for all time steps
    num_samples = wf.shape[1]
    prob = np.zeros((bins, num_samples))
    edges = np.linspace(np.min(wf) - 10.0**-6, np.max(wf) + 10.0**-6, bins+1)
    for i in range(num_samples):
        prob[:, i], _ = np.histogram(wf[:, i], edges)

    # Normalize histograms to form ditributions
    prob = np.divide(prob, np.mean(prob, axis=0))

    return prob, edges


def update_progress(progress):
    """A simple console-based progress bar"""
    print "\r[{0:50s}] {1:.1f}%".format('#' * int(progress * 50),
                                        progress * 100),


#################################################################
#                          RF creation                          #
#################################################################
def createRF(name='gabor', size=(25, 25), angle=45., phase=0.0, frequency=0.5,
             sigma=(0.35, 0.35), threshold=0.01, dtype=np.float, xoffset=0,
             yoffset=0, extent=(-8, 4, -7, 5)):
    """Create artificial receptive field pattern

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


def smoothRF(rf, filtsize=(3, 3), scale=1):
    """smooth 2D RF using 2D Gaussian filter"""

    # Create 2D Gaussian filter
    sizex = int(np.floor(filtsize[0]/2.))
    sizey = int(np.floor(filtsize[1]/2.))
    x, y = np.mgrid[-sizex:sizex+1, -sizey:sizey+1]
    G = np.exp(-scale*(x**2/float(sizex)+y**2/float(sizey)))
    G /= G.sum()

    if isinstance(rf, np.ndarray):
        KG = convolve2d(rf, G, mode='same', boundary='symm')
        return KG

    else:
        K = np.reshape(rf.coef, rf.shape)
        KG = convolve2d(K, G, mode='same', boundary='symm')
        rf.coef = KG.flatten()


#################################################################
#                   Stimulus generation                         #
#################################################################

def createGrating(size=(25, 25), angle=45., phase=0.0, f=0.5):
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


def createGratings(size=(25, 25), N=10**4, center=True, dtype=np.float64,
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
        G[i, :] = createGrating(size, angle, phase, f).ravel(order=order)

    if whiten:
        G, _ = whiten_matrix(G, eps=1e-8)

    if center:
        G -= np.mean(G, axis=0)

    return G


def loadImages(location, samples=10**2, size=(25, 25), center=True,
               randomize=False, random_seed=None, scale=0.5, images=10,
               normalize=True, dtype=np.float64, remove_dc=True):
    """Load van Hateren images and recast them as vectors in a stimulus matrix

    """

    rng = np.random.RandomState(random_seed)

    # Get all image files
    img_dir = location
    img_files = [f for f in listdir(img_dir)
                 if isfile(join(img_dir, f)) and f.endswith('.iml')]
    img_files.sort()

    if randomize:
        idx = rng.permutation(len(img_files))
        img_files = img_files[idx]

    if images < len(img_files):
        img_files = img_files[:images]

    ndims = np.prod(size)
    X = np.zeros((samples, ndims), dtype=dtype)
    offset = 0
    samples_per_image = np.int(np.ceil(float(samples) / images))
    sample = 0
    for i in range(images):
        idx = i - offset
        img_file = join(img_dir, img_files[idx])

        # Read image data
        fin = open(img_file, 'rb')
        tmp = fin.read()
        fin.close()
        arr = array.array('H', tmp)
        arr.byteswap()
        data = np.array(arr, dtype='uint16').reshape(1024, 1536)

        # Resize image?
        img = Image.fromarray(data, 'I;16')
        if scale < 1.0:
            nx = np.int(np.round(img.size[0] * scale))
            ny = np.int(np.round(img.size[1] * scale))
            img = img.resize((nx, ny))

        width, height = img.size
        D = np.reshape(np.array(img.getdata(), dtype=dtype), (height, width))
        for p in range(samples_per_image):

            # Randomly select patch
            x0 = rng.randint(0, width - size[1] + 1)
            y0 = rng.randint(0, height - size[0] + 1)
            region = D[y0:y0+size[0], x0:x0+size[1]]

            # Store patch in data matrix
            X[sample, :] = region.flatten()
            if remove_dc:
                X[sample, :] -= np.mean(X[sample, :])
            sample += 1

            if sample >= samples:
                break

        if sample >= samples:
            break

        if (i+1) % len(img_files) == 0:
            offset += len(img_files)

    if center:
        X -= np.mean(X, axis=0)

    if normalize:
        max_norm = np.max(np.sqrt(np.sum(X * X, axis=1)))
        X /= max_norm

    return X


def whiten_matrix(X, eps=1e-12):
    """whitening of data matrix using PCA"""

    C = np.dot(X.T, X)
    S, V = np.linalg.eigh(C)
    D = np.diag(1. / np.sqrt(S + eps))

    W = np.dot(np.dot(V, D), V.T)

    Xw = np.dot(X, W)

    return Xw, W
