#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Adapted sklearn cv-based grid search to LNP estimation setting
"""

import numpy as np
import time
import sys

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.base import clone as _clone_estimator

from ..lnp.util import calcMI, calcCoherence, calcLogLikelihood


def _calc_AUC(Y, z):

    if Y.ndim > 1:
        n_trials = Y.shape[1]
        AUC = np.zeros((n_trials,))
        for i in range(n_trials):
            y = np.ascontiguousarray(Y[:, i])
            AUC[i] = roc_auc_score(y, z)
    else:
        AUC = roc_auc_score(Y, z)

    return np.mean(AUC)


def _calc_MI(Y, z):

    mi = calcMI(Y, z, n_bins=50, distributions=False, correct_bias=True)
    return np.mean(mi)


def _calc_Coherence(Y, z):

    z[z < 0] = 0.

    cxy, f = calcCoherence(z, Y)

    return np.mean(cxy)


def _calc_PoissonLL(Y, z):

    ll = calcLogLikelihood(z, Y, dt=1., family='poissonexp')

    return ll


def _calc_ModPoissonLL(Y, z):

    ll = calcLogLikelihood(z, Y, dt=1., family='poissonexpquad')

    return ll


def _calc_BernoulliLL(Y, z):

    valid = np.where(np.logical_and(~np.isnan(Y), Y >= -1000))
    pred = z[valid[0]]

    if len(valid) == 2:
        ll = calcLogLikelihood(pred, Y[valid[0], valid[1]], dt=1.,
                               family='binomlogit')
    else:
        ll = calcLogLikelihood(pred, Y[valid[0]], dt=1.,
                               family='binomlogit')

    return ll


def _calc_MSE(Y, z):

    if Y.ndim > 1:
        y = Y.mean(axis=1)
    else:
        y = Y

    mse = np.mean((z - y) * (z - y))
    return mse


def _calc_PredPower(Y, z):

    if Y.ndim > 1:
        y = Y.mean(axis=1)
    else:
        y = Y

    pred_err = np.mean((z - y) ** 2)
    pred_pow = np.var(Y, ddof=1)

    pp = pred_pow - pred_err

    return pp


def _calc_r_squared(Y, z):

    if Y.ndim > 1:
        y = Y.mean(axis=1)
    else:
        y = Y

    r2 = r2_score(y, z)
    return r2


scorer_BernoulliLL = make_scorer(_calc_BernoulliLL, greater_is_better=True,
                                 needs_threshold=False)

scorer_AUC = make_scorer(_calc_AUC, greater_is_better=True,
                         needs_threshold=False)

scorer_MI = make_scorer(_calc_MI, greater_is_better=True,
                        needs_threshold=False)

scorer_Coherence = make_scorer(_calc_Coherence, greater_is_better=True,
                               needs_threshold=False)

scorer_PoissonLL = make_scorer(_calc_PoissonLL,
                               greater_is_better=True,
                               needs_threshold=False)

scorer_ModPoissonLL = make_scorer(_calc_ModPoissonLL,
                                  greater_is_better=True,
                                  needs_threshold=False)

scorer_MSE = make_scorer(_calc_MSE, greater_is_better=False,
                         needs_threshold=False)

scorer_PredPower = make_scorer(_calc_PredPower,
                               greater_is_better=True,
                               needs_threshold=False)

scorer_r_squared = make_scorer(_calc_r_squared, greater_is_better=True,
                               needs_threshold=False)


def get_scorer_from_name(name):

    fn = None
    exec('fn = scorer_%s' % name)
    return fn


def get_scorer_name(scorer):

    name = scorer._score_func.func_name
    if name.startswith('_calc_'):
        name = name[6:]

    return name


def get_scorer_names(scorers):

    names = []
    for s in scorers:
        if isinstance(s, str):
            names.append(s)
        else:
            names.append(get_scorer_name(s))

    return names


class ParamSearchCV(object):

    def __init__(self, model, param_grid, param_info=None, n_griditer=5,
                 n_jobs=-1, scorer=scorer_AUC, verbose=1, n_folds=5,
                 stratify_folds=True, fit_final=True, random_state=0,
                 param_scaling='log2'):

        if isinstance(scorer, str):
            scorer = get_scorer_from_name(scorer)

        self.model = model
        self.param_grid = param_grid
        self.param_info = param_info
        self.n_griditer = n_griditer
        self.n_jobs = n_jobs
        self.scorer = scorer
        self.verbose = verbose
        self.n_folds = n_folds
        self.stratify_folds = stratify_folds
        self.fit_final = fit_final
        self.random_state = random_state
        self.param_scaling = param_scaling

        self.best_params_ = None
        self.best_score_ = 0.
        self.t_fit = 0.

    def fit(self, X, Y):

        n_folds = self.n_folds
        if self.stratify_folds:
            if Y.ndim == 1:
                y = Y
            else:
                y = np.sum(Y, axis=1)
            cv = StratifiedKFold(y, n_folds=n_folds)

        else:
            cv = KFold(Y.shape[0], n_folds=n_folds, shuffle=False,
                       random_state=self.random_state)

        model = self.model
        scorer = self.scorer
        scorer_name = get_scorer_name(scorer)

        n_griditer = self.n_griditer
        param_grid = self.param_grid.copy()
        param_info = self.param_info

        if param_info is None:
            param_info = {}
            for k in param_grid.keys():
                param_info.update({k: {'scaling': self.param_scaling}})

        param_hist = []
        self.t_fit = 0.

        for i in range(n_griditer):

            grid = GridSearchCV(model, param_grid, scoring=scorer,
                                n_jobs=self.n_jobs, iid=True, cv=cv,
                                refit=False, verbose=self.verbose,
                                pre_dispatch='2*n_jobs')

            t0 = time.time()
            grid.fit(X, Y)
            t1 = time.time() - t0
            self.t_fit += t1

            print_s = "iter %d/%d (%0.2f s) | %s = %0.3f " % \
                (i + 1, n_griditer, t1, scorer_name, grid.best_score_)

            for key in grid.best_params_.keys():

                param_hist.append(grid.grid_scores_)

                # "zoom in"
                best_param = grid.best_params_[key]
                idx = np.where(best_param == param_grid[key])[0][0]
                old_values = param_grid[key]
                n_params = old_values.shape[0]
                i1 = np.amax([idx - 1, 0])
                i2 = np.amin([idx + 1, n_params - 1])
                p1 = old_values[i1]
                p2 = old_values[i2]

                new_values = None
                s = ''
                if param_info[key]['scaling'] in [None, '', 'linear']:
                    new_values = np.linspace(p1, p2, n_params)
                    s = ("%s = " % key) + str(best_param)

                elif param_info[key]['scaling'] is 'log2':
                    new_values = np.power(2., np.linspace(np.log2(p1),
                                                          np.log2(p2),
                                                          n_params))
                    s = ("log2(%s) = " % key) + str(np.log2(best_param))

                elif param_info[key]['scaling'] is 'log10':
                    new_values = np.power(10., np.linspace(np.log10(p1),
                                                           np.log10(p2),
                                                           n_params))
                    s = ("log10(%s) = " % key) + str(np.log10(best_param))

                param_grid[key] = new_values
                print_s += " | " + s
            print print_s

            sys.stdout.flush()

        self.best_params_ = grid.best_params_
        self.best_score_ = grid.best_score_

        if self.fit_final:
            for name in self.best_params_:
                setattr(model, name, self.best_params_[name])
            model.fit(X, Y)


class CVEvaluator():
    """Model evaluation using cross-validation

    Parameters
    ----------
    model : BaseEstimator
        A model class derived from BaseEstimator

    n_folds : int
        number of CV folds

    metrics : list
        one or more metrics used for the evaluation, e.g., 'AUC', 'MI', ...

    eval_folds: array-like
        evaluate only the given folds

    """

    def __init__(self, model, scorers, n_folds=5, verbose=True,
                 eval_folds='all', stratify_folds=False, random_state=0,
                 save_coef=True):

        self.model = model
        self.n_folds = n_folds
        self.scorers = scorers
        self.verbose = verbose
        self.eval_folds = eval_folds
        self.stratify_folds = stratify_folds
        self.random_state = random_state
        self.save_coef = save_coef

        self.results = None
        self.K = None

    def process(self, X, Y):
        """Runs cross-validation-based evaluation of models"""

        verbose = self.verbose
        n_folds = self.n_folds
        scorers = self.scorers
        stratify_folds = self.stratify_folds
        random_state = self.random_state
        model = self.model

        eval_folds = self._get_folds()
        scorer_names = []
        scorer_funcs = []
        for s in scorers:
            if isinstance(s, str):
                scorer_names.append(s)
                scorer_funcs.append(get_scorer_from_name(s))
            else:
                scorer_names.append(get_scorer_name(s))
                scorer_funcs.append(s)

        if stratify_folds:
            if Y.ndim == 1:
                y = Y
            else:
                y = np.sum(Y, axis=1)
            cv = StratifiedKFold(y, n_folds=n_folds)  # , indices=True)

        else:
            cv = KFold(X.shape[0], n_folds,  # indices=True,
                       random_state=random_state)

        results = {'n_folds': n_folds, 'scorers': scorer_names,
                   'eval_folds': eval_folds}
        for name in scorer_names:
            results.update({name: []})

        K = []
        for i, (train_index, test_index) in enumerate(cv):

            if i+1 in eval_folds:
                if verbose:
                    print 25 * '-', 'Fold %d/%d' % (i+1, n_folds), 25 * '-'

                # Learn model
                if Y.ndim > 1:
                    y_train = Y[train_index, :]
                    y_test = Y[test_index, :]

                else:
                    y_train = Y[train_index]
                    y_test = Y[test_index]

                model_copy = _clone_estimator(model)
                model_copy.fit(X[train_index, :], y_train)

                if self.save_coef and hasattr(model_copy, 'coef_'):
                    K.append(model_copy.coef_)

                # Evaluate model
                for j, scorer in enumerate(scorer_funcs):
                    name = scorer_names[j]
                    z = model_copy.predict(X[test_index, :])
                    value = scorer._score_func(y_test, z)
                    results[name].append(value)

                print ""

            else:
                for name in scorer_names:
                    results[name].append(None)

        self.results = results
        self.K = K

    def print_summary(self):
        """output formatted summary of results"""

        print "------------------------------------------------------"
        print "CVEvaluator results"
        print "------------------------------------------------------"

        results = self.results
        scorer_names = get_scorer_names(self.scorers)
        eval_folds = self._get_folds()

        for j, name in enumerate(scorer_names):
            x = np.asarray(results[name])[eval_folds - 1]
            print "%s:" % name, np.mean(x), "+-", np.std(x)

        print "------------------------------------------------------"

    def _get_folds(self):

        n_folds = self.n_folds
        fold_arg = self.eval_folds
        if fold_arg is None or isinstance(fold_arg, str) and fold_arg == 'all':
            eval_folds = np.arange(1, n_folds+1)
        else:
            eval_folds = np.asarray(fold_arg)

        return eval_folds
