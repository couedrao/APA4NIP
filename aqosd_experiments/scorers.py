"""
TODO docstring
"""
import math
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import *

from aqosd_experiments.config import K_FOLD, AVERAGE

e = sys.float_info.epsilon  # noqa : handle zero div and nan


def _coverage_error_wrapper(y_true, y_score):
    if hasattr(y_score, 'todense'):
        y_score = y_score.todense()
    return coverage_error(y_true, y_score)


def _label_ranking_loss_wrapper(y_true, y_score):
    if hasattr(y_score, 'todense'):
        y_score = y_score.todense()
    return label_ranking_loss(y_true, y_score)


def _hamming_loss_wrapper(y_true, y_score):
    if hasattr(y_score, 'todense'):
        y_score = y_score.todense()
    return hamming_loss(y_true, y_score)


def _label_ranking_average_precision_score_wrapper(y_true, y_score):
    if hasattr(y_score, 'todense'):
        y_score = y_score.todense()
    return label_ranking_average_precision_score(y_true, y_score)


def user_defined_matthews_corrcoef(y_true, y_pred, *, average='yes'):
    out = dict()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        out[i] = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if average != 'yes':
        return out
    else:
        mcc = np.mean(list(out.values()))
        if np.isnan(mcc):
            return 0.
        else:
            return mcc


def user_defined_specificity(y_true, y_pred, *, averaging='yes'):
    out = dict()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        out[i] = tn / (tn + fp + e)
    if averaging != 'yes':
        return out
    else:
        spe = np.nanmean(list(out.values()))
        if np.isnan(spe):
            return 0.
        else:
            return spe


def user_defined_dor(y_true, y_pred, *, minimun='yes'):
    out = dict()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        out[i] = math.log10((tp / fn + e) / (fp / tn + e))
    if minimun != 'yes':
        return out
    else:
        dor = min(list(out.values()))
        if np.isnan(dor):
            return 0.
        else:
            return dor


SCORING = OrderedDict({
    'sensitivity': make_scorer(recall_score, average=AVERAGE),
    'specificity': make_scorer(user_defined_specificity),
    'coverage error': make_scorer(_coverage_error_wrapper),
    'subset accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average=AVERAGE),
    # 'hamming loss': make_scorer(_hamming_loss_wrapper),
    # 'coverage error': make_scorer(_coverage_error_wrapper),
    # 'ranking loss': make_scorer(_label_ranking_loss_wrapper),
    # 'zero one loss': make_scorer(zero_one_loss),
})


def process_score(classifier_name, scores, scoring):
    classifier_output = {classifier_name: {}}
    scoring_keys = scoring.keys()
    mean_scores = {k: v.mean() for k, v in scores.items()}
    test_keys = map(lambda x: 'test_{}'.format(x), scoring_keys)
    for k in test_keys:
        classifier_output[classifier_name][k] = mean_scores[k]
        classifier_output[classifier_name]['{}_std'.format(k)] = scores[k].std()
    time_keys = ['fit_time', 'score_time']
    for k in time_keys:
        classifier_output[classifier_name][k] = mean_scores[k]
    return pd.DataFrame.from_dict(classifier_output).T


def process_odms_score(scores, scoring, p):
    output = {}
    scoring_keys = scoring.keys()
    if scores == -1:  # penalty
        penalty = p * np.array(np.ones(K_FOLD))
        scores = {'fit_time': penalty, 'score_time': penalty,
                  'test_sensitivity': penalty, 'train_sensitivity': penalty,
                  'test_specificity': penalty, 'train_specificity': penalty,
                  # 'test_dor': penalty, 'train_dor': penalty,
                  # 'test_mcc': penalty, 'train_mcc': penalty,
                  'test_subset accuracy': penalty, 'train_subset accuracy': penalty,
                  'test_precision': penalty, 'train_precision': penalty,
                  'test_coverage error': penalty, 'train_coverage error': penalty,
                  # 'test_zero one loss': penalty, 'train_zero one loss': penalty,
                  # 'test_ranking loss': penalty, 'train_ranking loss': penalty,
                  # 'test_hamming loss': penalty, 'train_hamming loss': penalty,
                  }
    mean_scores = {k: v.mean() for k, v in scores.items()}
    test_keys = map(lambda x: 'test_{}'.format(x), scoring_keys)
    for k in test_keys:
        output[k] = mean_scores[k]
        output['{}_std'.format(k)] = scores[k].std()
    time_keys = ['fit_time', 'score_time']
    for k in time_keys:
        output[k] = mean_scores[k]
    return output
