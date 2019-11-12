#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Context model. For details see Ahrens et al. J Neurosci 2008 and
    Williamson et al. Neuron 2016.
"""

import numpy as np
from os.path import join, split
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.sparse import rand as sprand
from scipy.ndimage.filters import gaussian_filter
from sklearn.base import BaseEstimator, RegressorMixin

from .als_dense import fit_context_model as _fit_context_als
from .als_dense import predict_response_context as _predict_response_context
from .vb_dense import fit_context_model as _fit_context_vb
from . import context_fast


def create_toy_data(T=1000, J=9, K=9, M=5, N=2, dt=0.02, c1=0., c2=1., c3=0.,
                    noisevar=0.):

    # Create PRF and CGF
    prf_size = (J, K)
    cgf_size = (M+1, 2*N+1)

    w_prf = np.zeros(prf_size)
    nt = int(np.round(J/3.))
    nf = int(np.round(K/3.))
    w_prf[nt:2*nt, nf:2*nf] = np.tile(np.linspace(-1, 1, nt), (nf, 1)).T

    w_cgf = np.zeros(cgf_size)
    nt = int(np.round(M/2.))
    nf = int(np.round(N/2.))
    w_cgf[-nt:-2*nt:-1, N-nf:N+nf+1] = -1
    w_cgf[-1, 0] = -1
    w_cgf[-1, -1] = 1

    w_prf = gaussian_filter(w_prf, .75)
    w_cgf = gaussian_filter(w_cgf, .75)
    w_cgf[0, N] = 0

    # DRC stimulus
    S = np.asarray(sprand(T, K, density=.165).todense())

    # Pad zeros around stimulus to simplify subsequent computations
    S_pad = np.zeros((J-1+M+T, K+2*N))
    S_pad[J-1+M:, N:-N] = S

    # Predict response of full context model
    y = np.zeros((T,))

    context_fast.predict_y_context(S_pad, w_prf.ravel(), w_cgf.ravel(), y,
                                   c1, c2, c3, T, J, K, M, N)

    # Add some noise
    y += np.sqrt(noisevar) * np.random.randn(y.shape[0])

    return S, S_pad, y, w_prf, w_cgf


def plot_context_model(w_strf, w_prf, w_cgf, J, M, N,
                       dt=0.02,
                       cmap='RdBu_r',
                       timeticks_prf=None,
                       timeticks_cgf=None,
                       colorbar=True,
                       colorbar_num_ticks=3,
                       frequencies=None,
                       freqticks=[4000, 8000, 16000, 32000],
                       logfscale=True,
                       windowtitle=None,
                       axes=None,
                       plot_STRF=False):

    w_strf = w_strf[::-1, :]
    w_prf = w_prf[::-1, :]
    w_cgf = w_cgf[::-1, :]

    if frequencies is None:
        frequencies = 2.**np.arange(w_prf.shape[1])

    # frequency axis
    fc = frequencies
    df_log = np.diff(np.log2(fc))
    n_octaves = round(N * np.mean(df_log) * 10) / 10

#    w_cgf[-1, N] = 0
    n_plots = 2 + plot_STRF

    # Plot RFs
    new_figure = False
    if axes is None:
        new_figure = True
        fig, axes = plt.subplots(nrows=1, ncols=n_plots)
    else:
        axes = np.asarray(axes)
        fig = axes[0].get_figure()

    plt_args = dict(interpolation='nearest',
                    aspect='auto',
                    origin='lower')
    plot_index = 0
    images = []

    if fc is not None:

        fc = np.asarray(fc) / 1000.
        f_unit = 'kHz'

        if freqticks is not None:
            freqticks = np.asarray(freqticks) / 1000.

        # set ticks
        if logfscale:
            f_extent = [.5, fc.shape[0] + .5]
        else:
            f_extent = [fc.min(), fc.max()]

    else:
        f_extent = [.5, w_prf.shape[1] + .5]
        f_unit = 'channels'

    if plot_STRF:
        ax1 = axes[plot_index]
        plot_index += 1

        ax1.set_title('STRF')
        v_max = np.max(np.abs(w_strf))
        v_eps = max(1e-3 * v_max, 1e-12)
        extent = (-J*dt*1000, 0, f_extent[0], f_extent[1])
        im = ax1.imshow(w_strf.T,
                        vmin=-v_max - v_eps,
                        vmax=v_max + v_eps,
                        cmap=cmap,
                        extent=extent,
                        **plt_args)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Frequency (%s)' % f_unit)
        images.append(im)

    ax = axes[plot_index]
    plot_index += 1
    ax.set_title('PRF')
    v_max = np.max(np.abs(w_prf))
    v_eps = max(1e-3 * v_max, 1e-12)
    extent = (-J*dt*1000, 0, f_extent[0], f_extent[1])
    im = ax.imshow(w_prf.T,
                   vmin=-v_max - v_eps,
                   vmax=v_max + v_eps,
                   cmap=cmap,
                   extent=extent,
                   **plt_args)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (%s)' % f_unit)
    images.append(im)

    # Set ticks for STRF and PRF
    for ax in axes[:1+plot_STRF]:
        if timeticks_prf is not None:
            ax.set_xticks(timeticks_prf)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(4))

    ax = axes[plot_index]
    ax.set_title('CGF')
    v_max = np.max(np.abs(w_cgf))
    v_eps = max(1e-3 * v_max, 1e-12)
    extent = (-M*dt*1000, 0, -1 - .5/N, 1 + .5/N)
    im = ax.imshow(w_cgf.T,
                   vmin=-v_max - v_eps,
                   vmax=v_max + v_eps,
                   cmap=cmap,
                   extent=extent,
                   **plt_args)
    ax.set_xlabel('Time shift (ms)')
    ax.set_ylabel(r'$\Phi$ (oct)')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-n_octaves, 0, n_octaves])
    images.append(im)

    if timeticks_cgf is not None:
        ax.set_xticks(timeticks_cgf)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(4))

    if windowtitle is not None:
        fig.canvas.set_window_title(windowtitle)

    if colorbar:
        for im, ax in zip(images, axes.tolist()):
            cbar = plt.colorbar(mappable=im, ax=ax)
            cbar.locator = MaxNLocator(colorbar_num_ticks)
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=5,
                                pad=2)

    if new_figure:
        fig.set_size_inches(6 + plot_STRF*2, 2.5)
        fig.tight_layout()

    return fig


def predict_STRF_from_PRF_and_CGF(S, w_prf, w_cgf, c1, J, M, N):
    """predict STRF from PRF and CGF as given in the appendix of ???"""

    K = S.shape[1]

    s_bar = np.mean(S)  # eq S4
    W_cgf = np.sum(w_cgf)  # eq S9

    beta = 1 + s_bar * W_cgf

    # Carry out the sum in eq S14
    w_conv = np.zeros_like(w_prf)
    for p in range(J):
        for q in range(K):
            a = 0
            for j in range(p):
                for k in range(max(0, q-N), min(K, q+N)):
                    if p - j < M+1:
                        a += w_prf[j, k] * w_cgf[p - j, q - k + N]

            w_conv[p, q] = a

    W_conv = np.sum(w_conv)  # eq S17

    b_strf = c1 - s_bar ** 2 * W_conv  # eq S17
    w_strf = beta * w_prf + s_bar * w_conv  # eq S17

    return b_strf, w_strf


class ContextModel(BaseEstimator, RegressorMixin):
    """Context model as described in Ahrens et al. J Neurosci 2008

        Parameters
        ----------
        J : int
            Time lag (history length) of the PRF

        M : int
            Time lag of the CGF

        N : int
            Frequency shift of the CGF

        algorithm : str
            The algorithm used to fit the model's parameters, which can be
            either 'als_matlab' (calls Misha's Matlab code), 'als_dense'
            (a dense Python implementation), or 'vb' (variational Bayesian
            fitting; not implemented yet)

        max_iter : int
            The maximum number of ALS iterations

        reg_iter : int
            The number of ALS iterations in which the prior hyperparameters
            will be optimized (and kept fixed afterwards)

        smooth_min : float
            Minimum samount of moothing of the ASD algorithm

        tolerance : float
            ALS procedure terminates if error difference is smaller
            than tolerance

    """

    def __init__(self,
                 J=15,
                 M=12,
                 N=10,
                 algorithm='als_dense',
                 max_iter=100,
                 reg_iter=3,
                 als_solver='iter',
                 init_params_cgf=[6., 2., 2.],
                 smooth_min=0.5,
                 tolerance=1e-5,
                 validate=False,
                 init_params_prf=[7, 4, 4]):

        self.J = J
        self.M = M
        self.N = N

        self.algorithm = algorithm
        self.max_iter = max_iter
        self.reg_iter = reg_iter
        self.als_solver = als_solver
        self.init_params_cgf = init_params_cgf
        self.smooth_min = smooth_min
        self.tolerance = tolerance
        self.validate = validate
        self.init_params_prf = init_params_prf

        self.w_strf = None
        self.b_strf = 0.
        self.w_prf = None
        self.b_prf = 0.
        self.w_cgf = None
        self.b_cgf = 0.

        self._validation = {}
        self.stats = None

    def fit(self, X, y):

        algo = self.algorithm.lower()
        if algo == 'als_matlab':
            self._fit_matlab(X, y)

        elif algo in ['als_dense', 'vb_dense']:
            self._fit_dense(X, y)

    def _fit_matlab(self, X, y):

        from matcall import MatlabCaller

        code_dir = join(split(__file__)[0], 'context_code')
        mat_func = 'fit_fullrank_context_inputs'

        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0], 1))

        input_dict = dict(X=X,
                          y=y,
                          J=float(self.J),
                          M=float(self.M),
                          N=float(self.N),
                          context_smoothing_choice=1.,
                          distance_choice=1.,
                          maxiter=self.max_iter,
                          regiter=self.reg_iter,
                          validate=self.validate)
        input_order = ['X', 'y', 'J', 'M', 'N', 'context_smoothing_choice',
                       'distance_choice', 'maxiter', 'regiter', 'validate']
        kwarg_names = ['maxiter', 'regiter', 'validate']

        output_names = ['results']

        mc = MatlabCaller(addpath=code_dir)
        res = mc.call(mat_func,
                      input_dict=input_dict,
                      input_order=input_order,
                      kwarg_names=kwarg_names,
                      output_names=output_names)

        results = res['results']

        w_strf = results.strf.ww
        w_cgf = results.full_rank_sparse_rep.wtauphi
        b_cgf = 0.
        w_prf = results.full_rank_sparse_rep.wtf
        b_prf = results.full_rank_sparse_rep.c
        pred_resp = results.full_rank_sparse_rep.predictedResp

        if self.validate:
            self._validation = {'pred_power_train_context': results.full_rank_sparse_rep.tpp,
                                'pred_power_cv_context': results.full_rank_sparse_rep.pp,
                                'pred_power_train_strf': results.strf.tpp,
                                'pred_power_cv_strf': results.strf.pp}
        else:
            self._validation = {}

        self.stats['signal_power'] = results.stats.signalpower
        self.stats['noise_power'] = results.stats.noisepower
        self.stats['error_signal'] = results.stats.error

        b_strf = w_strf[0]
        w_strf = np.reshape(w_strf[1:], w_prf.shape, order='F')
        w_strf = w_strf[::-1, :]

        self.w_strf = np.ascontiguousarray(w_strf)
        self.b_strf = b_strf
        self.w_prf = np.ascontiguousarray(w_prf)
        self.w_cgf = np.ascontiguousarray(w_cgf)
        self.b_cgf = b_cgf
        self.b_prf = b_prf
        self._pred_resp = pred_resp

    def _fit_dense(self, X, y):

        J = self.J
        M = self.M
        N = self.N

        if isinstance(X, np.ndarray):
            K = X.shape[1]
        else:
            K = X[0].shape[1]

        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = np.atleast_2d(y).T
        elif isinstance(y, list):
            for i, yi in enumerate(y):
                if yi.ndim == 1:
                    y[i] = np.atleast_2d(yi).T

        max_iter = self.max_iter
        reg_iter = self.reg_iter

        if self.algorithm.lower() == 'als_dense':
            models = _fit_context_als(X, y, J, K, M, N,
                                      reg_iter=reg_iter,
                                      max_iter=max_iter,
                                      c2=1.,
                                      tol=self.tolerance,
                                      wrap_around=True,
                                      solver=self.als_solver,
                                      init_params_cgf=self.init_params_cgf,
                                      smooth_min=self.smooth_min,
                                      init_params_prf=self.init_params_prf)

        elif self.algorithm.lower() == 'vb_dense':

            raise NotImplementedError()

            models = _fit_context_vb(X, y, J, K, M, N,
                                     reg_iter=reg_iter,
                                     max_iter=max_iter,
                                     c2=1.,
                                     tol=self.tolerance,
                                     wrap_around=True,
                                     solver=self.als_solver,
                                     init_params_cgf=self.init_params_cgf,
                                     smooth_min=self.smooth_min)

        else:
            raise ValueError("Invalid algorithm:", self.algorithm)

        model_strf, model_prf, model_cgf = models
        w_strf = np.reshape(model_strf.coef_, (J, K))
        w_prf = np.reshape(model_prf.coef_, (J, K))
        w_cgf = np.reshape(model_cgf.coef_, (M+1, 2*N+1))

        self.w_strf = w_strf[::-1, :]
        self.b_strf = model_strf.intercept_
        self.w_prf = w_prf
        self.b_prf = model_prf.intercept_
        self.w_cgf = w_cgf
        self.b_cgf = model_cgf.intercept_

        if isinstance(y, np.ndarray):
            p_signal, p_noise, e_signal = srfpower(y)
            self.stats = {'signal_power': p_signal,
                          'noise_power': p_noise,
                          'error_signal': e_signal}

        elif isinstance(y, list):
            p_signal = []
            p_noise = []
            e_signal = []
            for yi in y:
                ps, pn, es = srfpower(yi)
                p_signal.append(ps)
                p_noise.append(pn)
                e_signal.append(es)

            self.stats = {'signal_power': np.mean(p_signal),
                          'noise_power': np.mean(p_noise),
                          'error_signal': np.mean(e_signal)}

    def predict(self, X):

        model_prf = np.append(self.b_prf, self.w_prf.ravel())
        model_cgf = np.append(self.b_cgf, self.w_cgf.ravel())
        J = self.J
        T, K = X.shape
        M = self.M
        N = self.N
        y_pred = _predict_response_context(X, model_prf, model_cgf, T, J,
                                           K, M, N,
                                           c2=1.,
                                           pad_zeros=True,
                                           wrap_around=True)

        return y_pred

    def show(self, dt=0.02, cmap='RdBu_r', show_now=True, **kwargs):

        if hasattr(self, 'dt'):
            dt = self.dt

        fig = plot_context_model(self.w_strf, self.w_prf, self.w_cgf,
                                 self.J, self.M, self.N,
                                 dt=dt,
                                 cmap=cmap,
                                 **kwargs)

        if show_now:
            plt.show()

        return fig
