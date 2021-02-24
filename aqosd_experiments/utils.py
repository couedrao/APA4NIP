"""
helpers to update raw output and save analysis files
"""
import os
import os.path
import sys

import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score

from aqosd_experiments.config import ROUND, AVERAGE
from aqosd_experiments.scorers import user_defined_matthews_corrcoef, user_defined_specificity, user_defined_dor

e = sys.float_info.epsilon  # noqa : handle zero div and nan


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        sys.stderr.write(path, "already exist!")
    else:
        sys.stderr.write(path, "successfully created!")
    return path + '/'


def compute_precision_sensitivity_specificity(y_true, y_pred, bottleneck_names):
    columns = ['LR+', 'LR-', 'MCC', 'DOR']
    out = dict()
    for b in bottleneck_names:
        out[b] = dict()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        tpr = tp / (tp + fn + e)
        fnr = fn / (tp + fn + e)
        fpr = fp / (fp + tn + e)
        tnr = tn / (fp + tn + e)
        out[bottleneck_names[i]][columns[0]] = round(tpr / (fpr + e), ROUND)
        out[bottleneck_names[i]][columns[1]] = round(fnr / (tnr + e), ROUND)
    mcc = user_defined_matthews_corrcoef(y_true, y_pred, minimun='no')
    dor = user_defined_dor(y_true, y_pred, minimun='no')
    for i in mcc.keys():
        out[bottleneck_names[i]][columns[2]] = mcc[i]
        out[bottleneck_names[i]][columns[3]] = dor[i]
    df = pd.DataFrame.from_dict(out, orient='index', columns=columns)
    print('Min ' + columns[3], '=', df[columns[3]].min(), '(', df[columns[3]].idxmax(), ')')
    print('Min ' + columns[2], '=', df[columns[2]].min(), '(', df[columns[2]].idxmax(), ')')
    print('Min ' + columns[0], '=', df[columns[0]].min(), '(', df[columns[0]].idxmax(), ')')
    print('Max ' + columns[1], '=', df[columns[1]].max(), '(', df[columns[1]].idxmin(), ')')
    return df


def print_metrics(y_test, y_pred):
    a = accuracy_score(y_test, y_pred)
    m = user_defined_matthews_corrcoef(y_test, y_pred)
    s = user_defined_specificity(y_test, y_pred)
    r = recall_score(y_test, y_pred, pos_label=1, average=AVERAGE)
    print('Accuracy:', round(a, ROUND), '|', 'Specificity:', round(s, ROUND), '|', 'Sensitivity:', round(r, ROUND), '|',
          'MCC', round(m, ROUND))
