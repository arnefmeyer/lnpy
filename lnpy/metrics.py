#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Some helper for LNP model estimation
"""

from __future__ import print_function, division

import numpy as np
from sklearn.metrics import r2_score
from pylab import mlab
import matplotlib.pyplot as plt
from scipy import stats

from . import fast_tools
from .plotting import set_font_axes


def AUC(Y, score, num=250):
    """Calculate area under ROC curve"""

    if Y.ndim == 1:
        return fast_tools.calc_auc(Y, score, num)
    else:
        return fast_tools.calc_auc_trials(Y, score, num)


def ROC(proj, spikes, n_steps=250):
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


def coherence(pred, Y, nfft=256, noverlap=128):
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


def logLikelihood(z, Y, dt=1., family='poissonexp'):
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


def dPrime(stim, spikes, k, equal_variance=False):
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


def mutualInformation(Y, z, n_bins=50, distributions=False,
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


def MI(*args, **kwargs):
    """Short-named version of MI calculation routine"""
    return mutualInformation(*args, **kwargs)


def KLD(stim_mat, spike_mat, k, num_bins=50, distributions=False,
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


def vdot_normal(x, y):
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


def srfpower(yy, indep_noise=False, verbose=False):
    """Estimate signal and noise power as described in Sahani & Linden 2003b"""

    TT, NN = yy.shape

    ybar = np.mean(yy, axis=1)
    yvar = np.var(yy, axis=1, ddof=1)
    ybarm = np.mean(ybar)
    yvarm = np.mean(yvar)

    Pyy = np.mean(np.var(yy, axis=0, ddof=1))  # = Psignal + Pnoise
    Pybar = np.var(ybar, ddof=1)  # = Psignal + Pnoise/N

    Psignal = (NN*Pybar - Pyy) / (NN - 1.)
    Pnoise = Pyy - Psignal

    if indep_noise:
        Esignal = 1./TT * np.sqrt(
            4./NN * (np.dot(yvar.T, ybar**2) -
            2 * ybarm * (np.sum(yvar*ybar)) + TT * yvarm*ybarm**2) +
            2./NN/(NN-1.)*((1. - 2./TT) * np.sum(yvar*yvar) + yvarm ** 2)
            )
    else:
        ydelt = (yy.T - ybar).T  # repmat(ybar,1,size(yy,2));
        ycovm = np.dot(ydelt, np.mean(ydelt, axis=0).T) / (NN - 1.)
        ycovmm = np.mean(ycovm)
        e1 = 4. / NN * (np.var(np.dot(ybar.T, ydelt), ddof=1) / TT **2 -
                2 * ybarm * (np.dot(ycovm.T, ybar)) / TT + ybarm ** 2 * ycovmm)
        e2 = 2./NN/(NN-1.)*(np.sum(np.dot(ydelt.T, ydelt)**2)/(NN - 1.)**2/TT**2 -
                2 * np.dot(ycovm.T, ycovm) / TT + ycovmm**2)
        Esignal = np.sqrt(e1 + e2)

    if verbose:
        print("signal power = %g +/- %g" % (Psignal, Esignal))
        print("noise power = %g" % Pnoise)
        print("s/n power ratio = %g +/- %g" % (Psignal/Pnoise, Esignal/Pnoise))

    return Psignal, Pnoise, Esignal


def plot_srfpower(Psignal, Pnoise, Esignal=None, ax=None, ls_line=True,
                  color=3*[.5], **kwargs):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.errorbar(Pnoise, Psignal, yerr=Esignal, fmt='o', color=color,
                **kwargs)
    ax.set_xlabel(r'Noise power (spikes$^2$/bin)')
    ax.set_ylabel(r'Signal power (spikes$^2$/bin)')

    # Diagonal line
    vmin = min(np.min(Pnoise), np.min(Psignal))
    vmax = min(np.max(Pnoise), np.max(Psignal))
    xx = np.linspace(vmin, vmax, 100)
    ax.plot(xx, xx, '--', linewidth=1, color=3*[.25], zorder=0)

    if ls_line:
        slope, intercept, r_value, _, _ = stats.linregress(Pnoise, Psignal)
        vmin = min(np.min(Pnoise), np.min(Psignal))
        vmax = 1.1 * max(np.max(Pnoise), np.max(Psignal))
        xx = np.linspace(vmin, vmax, 100)
        ax.plot(xx, slope * xx + intercept, '-', linewidth=1, color=3*[0])

    set_font_axes(ax)

    return ax.get_figure()


def plot_normalized_pred_power(Psignal, Pnoise, pp_cv, pp_insample=None,
                               ax=None, lsline=True, legend=True,
                               label_cv='cv', label_insample='insample',
                               **kwargs):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    xx = Pnoise / Psignal
    if pp_insample is not None:
        ax.errorbar(xx, pp_insample, fmt='o', color=3*[.95],
                    label=label_insample, **kwargs)
    ax.errorbar(xx, pp_cv, fmt='o', color=3*[.4], label=label_cv, **kwargs)
    ax.set_xlabel(r'Normalized noise power')
    ax.set_ylabel(r'Normalized linearly predictable power')

    if lsline:
        vmin = min(np.min(xx), 0)
        vmax = 1.05 * np.max(xx)
        vv = np.linspace(vmin, vmax, 100)

        if pp_insample is not None:
            slope, intercept, r_value, _, _ = stats.linregress(xx, pp_insample)
            ax.plot(vv, slope * vv + intercept, '-', linewidth=1.5,
                    color=3*[.8])

        slope, intercept, r_value, _, _ = stats.linregress(xx, pp_cv)
        ax.plot(vv, slope * vv + intercept, '-', linewidth=1.5, color=3*[0])

    if legend:
        ax.legend(loc='best', numpoints=1, fontsize=8)

    set_font_axes(ax, add_size=0, size_ticks=6, size_labels=8,
                  size_text=8, size_title=8, family='Arial')

    return ax.get_figure()
