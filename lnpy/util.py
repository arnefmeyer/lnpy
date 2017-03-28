#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Some (hopefully) useful utilities
"""

from __future__ import division

import os
import os.path as op
import numpy as np
from scipy import signal
import array
import warnings
import quantities as pq
from copy import deepcopy

import base

# Try to use scikits' samplerate method
use_scikits_resample = False
try:
    import scikits.samplerate as sks
    use_scikits_resample = True
    max_ratio = 256.
except:
    pass


def makedirs_save(f):
    """simple and save directory generation"""

    if not os.path.exists(f):
        try:
            os.makedirs(f)
        except:
            pass


def update_progress(progress):
    """A simple console-based progress bar"""
    print "\r[{0:50s}] {1:.1f}%".format('#' * int(progress * 50),
                                        progress * 100),


def resample(x, fs_old, fs_new, axis=0, algorithm='scipy'):
    """Resample signal

    If available resampling is done using scikit samplerate. Otherwise,
    scipy's FFT-based resample function will be used.

    Converters available in scikits.samplerate:
    - sinc_medium
    - linear
    - sinc_fastest
    - zero_order_hold
    - sinc_best

    """

    if fs_old == fs_new:
        return x

    else:

        ratio = float(fs_new) / fs_old
        if use_scikits_resample and algorithm != 'scipy':

            if algorithm in ['scikits', 'scikit']:
                algo = 'sinc_medium'
            else:
                algo = algorithm

            if axis == 0:
                tmp = sks.resample(x, ratio, algo)
            else:
                tmp = sks.resample(x.T, ratio, algo)

            if tmp.dtype != x.dtype:
                tmp = tmp.astype(x.dtype, casting='safe')

            return tmp

        else:
            if axis == 0:
                n_samples_new = int(np.round(x.shape[0] * ratio))
                return signal.resample(x, n_samples_new)
            else:
                n_samples_new = int(np.round(x.shape[1] * ratio))
                return signal.resample(x, n_samples_new, axis=axis)


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
            stim = base.Stimulus(data, fs_stim)

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
        axes.append(base.Axis(values=t * pq.ms, label='Time', unit='ms'))
        axes.append(base.Axis(values=f_center * pq.Hz, label='Frequency',
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


def load_images_VanHateren(location, samples=10**2, size=(25, 25), center=True,
                           randomize=False, random_seed=None, scale=0.5,
                           images=10, normalize=True, dtype=np.float64,
                           remove_dc=True):
    """Load van Hateren database images as matrix"""

    from PIL import Image

    rng = np.random.RandomState(random_seed)

    # Get all image files
    img_dir = location
    img_files = [f for f in os.listdir(img_dir)
                 if op.isfile(op.join(img_dir, f)) and f.endswith('.iml')]
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
        img_file = op.join(img_dir, img_files[idx])

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


def calc_SWDF(wf, bins=25):
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
