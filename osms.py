"""
Code inspired from https://github.com/rasbt/mlxtend/blob/3e501e645ee7d54726344d9b0a25f2840c7ad6f8/mlxtend/feature_selection/sequential_feature_selector.py
Thanks Dr. Raschka Sebastian
"""
import datetime
import sys
from collections import defaultdict
from copy import deepcopy
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn.base import MetaEstimatorMixin, clone
from sklearn.metrics import *
from sklearn.model_selection import cross_validate
from sklearn.utils.metaestimators import _BaseComposition

from aqosd_experiments.config import DATE_FORMAT
from aqosd_experiments.scorers import process_odms_score


class BaseOSDMComposition(_BaseComposition):
    def _set_params(self, attr, named_attr, **params):
        if attr in params:
            setattr(self, attr, params.pop(attr))
        items = getattr(self, named_attr)
        names = []
        if items:
            names, estimators = zip(*items)
            estimators = list(estimators)
        for name in list(iter(params.keys)):
            if '__' not in name and name in names:
                for i, est_name in enumerate(names):
                    if est_name == name:
                        new_val = params.pop(name)
                        if new_val is None:
                            del estimators[i]
                        else:
                            estimators[i] = new_val
                        break
                setattr(self, attr, estimators)
        super(BaseOSDMComposition, self).set_params(**params)
        return self


def _calc_cost(selection, all_costs):
    vp = [all_costs[i] for i in selection]
    return np.sum(vp)


def _calc_score(selector, X, y, selected_metrics):
    cost = _calc_cost(selected_metrics, selector.overheads)
    if cost <= selector.overhead_budget:
        scores = cross_validate(selector.mbi, X[:, selected_metrics], y, cv=selector.cv, scoring=selector.scorer,
                                n_jobs=1, pre_dispatch=selector.pre_dispatch)
    else:
        scores = -1  # penalty
    scores = process_odms_score(scores, selector.scorer, selector.penalty)
    return selected_metrics, scores, cost


def _get_metricnames(subsets_dict, metric_idx, custom_metric_names):
    metric_names = None
    if metric_idx is not None:
        if custom_metric_names is not None:
            metric_names = tuple((custom_metric_names[i] for i in metric_idx))
        else:
            metric_names = tuple(str(i) for i in metric_idx)
    subsets_dict_ = deepcopy(subsets_dict)
    for key in subsets_dict_:
        if custom_metric_names is not None:
            new_tuple = tuple((custom_metric_names[i] for i in subsets_dict[key]['metric_idx']))
        else:
            new_tuple = tuple(str(i) for i in subsets_dict[key]['metric_idx'])
        subsets_dict_[key]['metric_names'] = new_tuple
    return subsets_dict_, metric_names


def _process_work_results(selector, all_avg_scores, all_cv_scores, all_subsets, all_costs, work):
    for new_subset, cv_scores, cost in work:
        for score in cv_scores:
            if score in all_cv_scores:
                all_cv_scores[score].append(cv_scores[score])
            else:
                all_cv_scores[score] = list()
                all_cv_scores[score].append(cv_scores[score])
        all_avg_scores.append(np.nanmean(cv_scores[selector.optim_scoring]))
        all_subsets.append(new_subset)
        all_costs.append(cost)
    best = np.argmax(all_avg_scores)  # np.argmin(all_avg_scores)  #
    for score in cv_scores:
        all_cv_scores[score] = all_cv_scores[score][best]
    return all_subsets[best], all_avg_scores[best], all_cv_scores, all_costs[best]


def _name_estimators(estimators):
    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for _, name in zip(estimators, names):
        namecount[name] += 1
    for k, v in list(iter(namecount.items)):
        if v == 1:
            del namecount[k]
    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1
    return list(zip(names, estimators))


class OverheadSensitiveMetricSelection(BaseOSDMComposition, MetaEstimatorMixin):
    def __init__(self, classifier, overheads, overhead_budget, k_metrics="all", verbose=0, scoring=None, cv=None,
                 test_indexes=None, n_jobs=1, optim_scoring='test_precision'):
        self.clf = classifier
        self.k_metrics = k_metrics
        self.pre_dispatch = '2*n_jobs'
        self.cv = cv if cv else PredefinedHoldoutSplit(test_indexes)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.mbi = clone(self.clf)
        self.overhead_budget = overhead_budget
        self.overheads = overheads
        self.optim_scoring = optim_scoring
        self.scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
        self.subsets_ = {}
        self.interrupted_ = False
        self._USER_STOP = False
        self.penalty = -3

    @property
    def named_estimators(self):
        return _name_estimators([self.clf])

    def get_params(self, deep=True):
        return self._get_params('named_estimators', deep=deep)

    def set_params(self, **params):
        self._set_params('estimator', 'named_estimators', **params)
        return self

    def fit(self, X, y, user_metric_names=None):
        # cleaning
        self.subsets_ = {}
        self.interrupted_ = False
        self.k_metric_idx_ = None
        self.k_metric_names_ = None
        self.k_score_ = None
        X_ = X
        min_k, max_k = 1, X_.shape[1]
        orig_set = set(range(X_.shape[1]))
        k_to_select = max_k
        k_idx, k = (), 0
        best_subset = None
        try:
            while k != k_to_select:
                prev_subset = set(k_idx)
                k_idx, k_score, cv_scores, cost = self.include(orig_set=orig_set, subset=prev_subset, X=X_, y=y)
                continuation_cond_1, continuation_cond_2 = len(k_idx), True
                ran_step_1, new_metric = True, None
                while continuation_cond_1 >= 2 and continuation_cond_2:
                    if ran_step_1:
                        (new_metric,) = set(k_idx).symmetric_difference(prev_subset)
                    k_idx_c, k_score_c, cv_scores_c, cost_c = self.exclude(metric_set=k_idx, X=X_, y=y,
                                                                           fixed_metric=({new_metric}))
                    if k_score_c is not None and k_score_c > k_score:  # <
                        cached_score = self.subsets_[len(k_idx_c)]['avg_score'] if len(
                            k_idx_c) in self.subsets_ else None
                        if cached_score is None or k_score_c > cached_score:  # <
                            prev_subset = set(k_idx)
                            k_idx, k_score, cv_scores, cost = k_idx_c, k_score_c, cv_scores_c, cost_c
                            continuation_cond_1 = len(k_idx)
                            ran_step_1 = False
                        else:
                            continuation_cond_2 = False
                    else:
                        continuation_cond_2 = False
                k = len(k_idx)
                if k not in self.subsets_ or (k_score > self.subsets_[k]['avg_score']):  # >
                    k_idx = tuple(sorted(k_idx))
                    self.subsets_[k] = {'metric_idx': k_idx, 'cv_scores': cv_scores, 'avg_score': k_score, 'cost': cost}
                if self.verbose > 0:
                    if k_score != self.penalty:  # penalty
                        self.pprint(cost, k_idx, k_score, k_to_select)
                if self._USER_STOP:
                    self.subsets_, self.k_metric_names_ = _get_metricnames(self.subsets_, self.k_metric_idx_,
                                                                           user_metric_names)
                    raise KeyboardInterrupt
                if k_score == self.penalty:  # penalty
                    break
        except KeyboardInterrupt:
            self.interrupted_ = True
            sys.stderr.write('\nSTOPPING EARLY DUE TO KEYBOARD INTERRUPT...')
        max_score = float('-inf')
        for k in self.subsets_:
            if k < min_k or k > max_k:
                continue
            if self.subsets_[k]['avg_score'] > max_score:  # >
                max_score = self.subsets_[k]['avg_score']
                best_subset = k
        k_score = max_score
        k_idx = self.subsets_[best_subset]['metric_idx']
        self.k_metric_idx_, self.k_score_ = k_idx, k_score
        self.subsets_, self.k_metric_names_ = _get_metricnames(self.subsets_, self.k_metric_idx_, user_metric_names)
        return self

    def pprint(self, cost, k_idx, k_score, k_to_select):
        sys.stdout.write('\n[%s] Metrics: %d/%s -- %s score: %s -- cost : %s/%s \n' % (
            datetime.datetime.now().strftime(DATE_FORMAT), len(k_idx), k_to_select,
            self.optim_scoring.replace('test_', '').capitalize(), k_score, cost, self.overhead_budget))

    def include(self, orig_set, subset, X, y):
        remaining = orig_set - subset
        if remaining:
            metrics = len(remaining)
            n_jobs = min(self.n_jobs, metrics)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)
            work = parallel(
                delayed(_calc_score)(self, X, y, tuple(subset.union({metric}))) for metric
                in remaining)
        return _process_work_results(self, [], {}, [], [], work)

    def exclude(self, metric_set, X, y, fixed_metric=None):
        n = len(metric_set)
        if n > 1:
            metrics = n
            n_jobs = min(self.n_jobs, metrics)
            parallel = Parallel(n_jobs=n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)
            work = parallel(delayed(_calc_score)(self, X, y, p) for p in
                            combinations(metric_set, r=n - 1) if not fixed_metric or fixed_metric.issubset(set(p)))
        return _process_work_results(self, [], {}, [], [], work)

    def transform(self, X):
        X_ = X
        return X_[:, self.k_metric_idx_]

    def fit_transform(self, X, y, groups=None, **fit_params):
        self.fit(X, y, groups=groups, **fit_params)
        return self.transform(X)

    def get_results(self):
        fdict = deepcopy(self.subsets_)
        del fdict[list(fdict.keys())[-1]]
        return fdict
