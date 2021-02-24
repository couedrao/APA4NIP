"""
helpers to plot results
"""
import datetime
import itertools
import time
import warnings
from math import pi

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from scipy import sparse
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.metrics import *

from aqosd_experiments.config import DATE_FORMAT, HOST_LIST, ROUND, AVERAGE
from aqosd_experiments.scorers import SCORING, user_defined_matthews_corrcoef, user_defined_specificity, \
    user_defined_dor

warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'grid', 'no-latex'])


def plt_long_stats(raw_dataset_path, h_list):
    fig, axes = plt.subplots(nrows=1, ncols=len(h_list), sharey=True, figsize=(3 * len(h_list), 1.5))
    log = pd.read_csv(raw_dataset_path + 'bottlenecks_in_time.csv', index_col=0, parse_dates=True)
    log['bottleneck'] = log['bottleneck'].str.replace('_', ' ')
    bottlenecks_types = sorted(list(log['bottleneck'].unique()), reverse=True)
    for host, i in zip(h_list, range(len(h_list))):
        df = log.loc[log['node'] == host]
        min_date, max_date = [datetime.datetime.strptime(item, DATE_FORMAT) for item in
                              (min(df['start_at']), max(df['end_at']))]
        reset_time = pd.Timedelta(time.mktime(min_date.timetuple()), unit='s') + pd.Timedelta(1, unit='h')
        df['start_at'] = pd.to_datetime(df['start_at'], format=DATE_FORMAT) - reset_time
        df['end_at'] = pd.to_datetime(df['end_at'], format=DATE_FORMAT) - reset_time
        min_date, max_date = min(df['start_at']), max(df['end_at'])
        ax = axes[i]
        for bottleneck in bottlenecks_types:
            for row in df[df['bottleneck'] == bottleneck].itertuples():
                left, right = [mdates.date2num(item) for item in (row.start_at, row.end_at)]
                ax.barh(bottleneck, left=left, width=right - left, height=0.5, color='k', alpha=1)
        ax.set_xlabel("time (hours)")
        ax.set_title(host, fontweight='bold')
        max_date = min_date + datetime.timedelta(minutes=30)
        ax.set_xlim(mdates.date2num(min_date), mdates.date2num(max_date))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  #
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if i == 0:
            ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='y', which='both', length=0)
        ax.spines['bottom'].set_bounds(mdates.date2num(min_date), mdates.date2num(max_date))
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    return fig


def plt_corr_metrics(data, c3='w', method='spearman'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    ax.set_title('Metrics correlations')
    ax_bounds = .5, len(data.columns) - .5
    corr = data.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    colors = [c3, '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    cmap = mpl.colors.ListedColormap(colors[::-1] + [c3, '#eff3ff', '#bdd7e7', '#6baed6', '#2171b5'])
    sns.heatmap(corr, ax=ax, cmap=cmap, mask=mask, annot=False, vmin=-1, vmax=1, square=True, center=0,
                cbar_kws={"shrink": .5, 'spacing': 'proportional'})
    ax.collections[0].colorbar.ax.tick_params(size=0, labelsize=11)
    mpl.rcParams['text.usetex'] = True
    ax.collections[0].colorbar.ax.set_title(r'$\rho$')
    mpl.rcParams['text.usetex'] = False
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_bounds(ax_bounds)
    ax.spines['bottom'].set_bounds(ax_bounds)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    [item.set_fontsize(11) for item in
     ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())]
    return fig


def plot_multicollinear_metrics(X):
    fig, ax = plt.subplots(1, 1, figsize=(3, 8))
    corr = stats.spearmanr(X.values).correlation
    corr_linkage = hierarchy.ward(corr)
    hierarchy.dendrogram(corr_linkage, labels=list(X.columns), ax=ax, orientation="right")  # , leaf_rotation=90)
    return fig


def plot_number_of_instance(data):
    fig, ax = plt.subplots(figsize=(3, .6))
    df = data.melt(value_vars=data.columns)
    df = df[df["value"] != 0]
    df = 100 * df.variable.value_counts(normalize=True, ascending=False)
    df.plot(ax=ax, kind='bar')
    print(sum(df))
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.tick_params(axis='x', which='both', size=0)
    ax2 = ax.twinx()
    ax2.tick_params(axis='y', which='both', size=0)
    ax2.set_ylabel('')
    ax2.set_yticks([])
    xticks = [i for i in range(len(data.columns))]
    ax.set_xticks(xticks)
    ax.set_xticklabels([x for x in xticks], rotation=45)
    [item.set_fontsize(5) for item in
     ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.set_xlim([None, max(xticks) + 0.3])
    ax.set_ylim([None, 6.01])
    return fig


def plt_all_data(data):
    n = len(list(data.columns)) // len(HOST_LIST)
    if n % 2 != 0:
        n += 1
    fig, axes = plt.subplots(nrows=1 + n, ncols=len(HOST_LIST), sharex=True, figsize=(3 * len(HOST_LIST), 1 + n),
                             gridspec_kw={"height_ratios": [0.02] + n * [1], 'hspace': 0.5})
    for i, host in enumerate(HOST_LIST):
        df = data[[i for i in list(data.columns) if host + '.' in i]]
        for j, c in enumerate(df.columns):
            ax = axes[j + 1][i]
            df[c].plot(ax=ax, color='k', title=c)  # marker='.', ms=.1,
            [item.set_fontsize(6) for item in
             ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels())]
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    for i, ax in enumerate(axes.flatten()[:len(HOST_LIST)]):
        ax.axis("off")
        ax.set_title(HOST_LIST[i], fontweight='bold')
    return fig


def plt_corr_bottlenecks(data, c3='w', method='spearman'):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('Bottleneck correlations')
    corr = data.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    colors = [c3, '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
    cmap = mpl.colors.ListedColormap(colors[::-1] + [c3, '#eff3ff', '#bdd7e7', '#6baed6', '#2171b5'])
    sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, center=0, annot=False, square=True,
                cbar_kws={"shrink": .5, 'spacing': 'proportional'})
    ax.collections[0].colorbar.ax.tick_params(size=0, labelsize=11)
    mpl.rcParams['text.usetex'] = True
    ax.collections[0].colorbar.ax.set_title(r'$\rho$')
    mpl.rcParams['text.usetex'] = False
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return fig


def plot_multi_confusion_matrix(y_true, y_pred, label_to_class, algo):
    def plot_confusion_matrix(cm, classes, title, ax):
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greys)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j], horizontalalignment="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks), ax.xaxis.set_ticklabels(classes)
        ax.set_yticks(tick_marks), ax.yaxis.set_ticklabels(classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Truth')
        ax.set_title(title)
        ax.grid(False)

    fig, axes = plt.subplots(int(len(label_to_class) / 4), 4, figsize=(8, 15))
    fig.suptitle("Confusion Matrix -- " + algo, y=-.01, fontsize=16)
    axes = axes.flatten()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        plot_confusion_matrix(np.array([[tp, fn], [fp, tn]]), classes=['+', '-'], title=f'{label_to_class[i]}',
                              ax=axes[i])
        plt.tight_layout()
    return fig


def plot_osdm(osdms):
    colors = iter(['k', 'r', 'b', 'darkorange', 'g'])
    markers = iter([',', ',', ',', '+', 'x'])
    results = osdms.get_results()
    best = results[len(osdms.k_metric_idx_)]['cost']
    k1 = map(lambda x: 'test_{}'.format(x), [s for s in list(SCORING.keys())])
    fig, ax = plt.subplots(figsize=(3, 2))
    ax2 = ax.twinx()
    k_metric = sorted(results.keys())
    costs = [results[k]['cost'] for k in k_metric]
    lns = []
    ax2.set_ylabel('Coverage Error', color='b')
    ax2.tick_params(axis='y', colors='b')
    ax.set_ylabel('Precision, Sens., Spec., Acc.')
    plt.setp(ax.get_yticklabels(), color="b")
    metric_min, metric_max = len(results[k_metric[0]]['metric_idx']), len(results[k_metric[-1]]['metric_idx'])
    ax.set_yticks(range(metric_min, metric_max + 1), range(metric_min, metric_max + 1))
    ax.set_xlabel("Metrics total cost")
    for score in k1:
        avg = [results[k]['cv_scores'][score] for k in k_metric]
        upper = [results[k]['cv_scores'][score] + results[k]['cv_scores'][score + '_std'] for k in k_metric]
        lower = [results[k]['cv_scores'][score] - results[k]['cv_scores'][score + '_std'] for k in k_metric]
        l = score.replace('test_', '').capitalize()
        if 'cov' not in score:
            c = next(colors)
            lns += ax.plot(costs, avg, marker=next(markers), markersize=3, color=c, label=l)
            ax.fill_between(costs, upper, lower, alpha=0.2, color=c, lw=1)
        else:
            c = next(colors)
            lns += ax2.plot(costs, avg, marker=next(markers), markersize=3, color=c, label=l)
            ax2.fill_between(costs, upper, lower, color=c, alpha=0.2, lw=1)
    ax.axvline(best, color='#f0027f')
    for a in (ax, ax2):
        [item.set_fontsize(8) for item in
         ([a.title, a.xaxis.label, a.yaxis.label] + a.get_xticklabels() + a.get_yticklabels())]
    labs = [lab.get_label() for lab in lns]
    ax.legend(lns, labs, loc=8, ncol=1)
    ax.set_xlim(min(costs) - 1, max(costs) + 1)
    plt.grid(False)
    plt.tight_layout()
    return fig, pd.DataFrame.from_dict(results).T

def compute_classification_score(results, y_test, many=True):
    dic = {}
    if not many:
        y_pred = results
        compute_evaluate_score(dic, 'Baseline', y_test, y_pred)
    else:
        for algorithm in results:
            y_pred = results[algorithm]
            compute_evaluate_score(dic, algorithm, y_test, y_pred)
    return pd.DataFrame.from_dict(dic, orient='index')


def compute_evaluate_score(dic, algorithm, y_test, y_pred):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "Hamming loss", "Jaccard index"]
    dic[algorithm] = {}
    dic[algorithm][metrics[0]] = round(accuracy_score(y_test, y_pred), ROUND)
    dic[algorithm][metrics[1]] = round(precision_score(y_test, y_pred, average=AVERAGE), ROUND)
    dic[algorithm][metrics[2]] = round(recall_score(y_test, y_pred, average=AVERAGE), ROUND)
    dic[algorithm][metrics[3]] = round(f1_score(y_test, y_pred, average=AVERAGE), ROUND)
    dic[algorithm][metrics[4]] = round(hamming_loss(y_test, y_pred), ROUND)
    dic[algorithm][metrics[5]] = round(jaccard_score(y_test, y_pred, average=AVERAGE), ROUND)


def perf_viz(results, y_test):
    c = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'][::-1]
    metrics = ['MCC', 'Specificity', 'Sensitivity', 'log(DOR)']
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey")
    for i, algorithm in enumerate(results):
        values = list()
        y_pred = results[algorithm]
        values.append(user_defined_matthews_corrcoef(y_test, y_pred))
        values.append(user_defined_specificity(y_test, y_pred))
        values.append(recall_score(y_test, y_pred, average=AVERAGE))
        values.append(user_defined_dor(y_test, y_pred))
        values += values[:1]
        ax.plot(angles, values, color=c[i], linestyle='solid', label=algorithm)
        ax.fill(angles, values, color=c[i], alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=2.5)
    return fig


def compute_multilabel_ranking_metrics(results, y_test):
    metrics = ['Coverage error', 'Label ranking average precision', 'Label ranking loss']
    dic = {}
    for algorithm in results:
        dic[algorithm] = {}
        y_pred = results[algorithm]
        if hasattr(y_pred, 'toarray'):
            y_pred = y_pred.toarray()
        dic[algorithm][metrics[0]] = round(coverage_error(y_test, y_pred), ROUND)
        dic[algorithm][metrics[1]] = round(label_ranking_average_precision_score(y_test, y_pred), ROUND)
        dic[algorithm][metrics[2]] = round(label_ranking_loss(y_test, y_pred), ROUND)
    return pd.DataFrame.from_dict(dic, orient='index')


metrics_per_label = ['Accuracy', 'Precision', 'Recall', 'F1', 'Fbeta', 'Roc auc']


def compute_measure_per_label(results, y_test, labels):
    def mesure_per_lab(measure, y_true, y_predicted, beta, average):
        dic = {}
        for i in range(y_true.shape[1]):
            if beta != 0:  # need beta
                dic[labels[i]] = round(measure(y_true[:, i].toarray(), y_predicted[:, i].toarray(), beta), ROUND)
            elif average != 'no':
                dic[labels[i]] = round(measure(y_true[:, i].toarray(), y_predicted[:, i].toarray(), average), ROUND)
            else:
                dic[labels[i]] = round(measure(y_true[:, i].toarray(), y_predicted[:, i].toarray()), ROUND)
        return dic

    y_true = sparse.csr_matrix(y_test)
    dic = {}
    for a in results:
        dic[a] = {}
        y_pred = results[a]
        if hasattr(y_pred, 'toarray'):
            y_pred = y_pred.toarray()
        dic[a][metrics_per_label[0]] = mesure_per_lab(accuracy_score, y_true, results[a], 0, average='no')
        dic[a][metrics_per_label[1]] = mesure_per_lab(precision_score, y_true, results[a], 0, average='no')
        dic[a][metrics_per_label[2]] = mesure_per_lab(recall_score, y_true, results[a], 0, average='no')
        dic[a][metrics_per_label[3]] = mesure_per_lab(f1_score, y_true, results[a], 0, average='no')
        dic[a][metrics_per_label[4]] = mesure_per_lab(fbeta_score, y_true, results[a], beta=0.5, average='no')
        dic[a][metrics_per_label[5]] = mesure_per_lab(roc_auc_score, y_true, results[a], 0, average=AVERAGE)
    return pd.DataFrame.from_dict({(i, j): dic[i][j] for i in dic.keys() for j in dic[i].keys()}, orient='index')


def plt_roc_auc(ax, clf_name, y_score, y_test, labels):
    if not hasattr(y_score, 'toarray'):
        if isinstance(y_score, list):
            y_score = np.array(y_score).T[1]
        else:
            y_score = np.array(y_score)
    if hasattr(y_score, 'A'):
        y_score = y_score.A
    n_classes, lw = len(labels), 1
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr[AVERAGE], tpr[AVERAGE], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc[AVERAGE] = auc(fpr[AVERAGE], tpr[AVERAGE])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = sum([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)]) / n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    print("Average AUC: ", round(sum(roc_auc.values()) / len(roc_auc), ROUND))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color='darkorange', label='Bottlenecks ROC curve', alpha=0.2,
                lw=lw) if i == 0 else ax.plot(fpr[i], tpr[i], color='darkorange', alpha=0.2, lw=lw)
    ax.plot(fpr[AVERAGE], tpr[AVERAGE], label='Micro average ROC curve (area = {0:0.2f})' ''.format(roc_auc[AVERAGE]),
            color='r', linewidth=lw)
    ax.plot(fpr["macro"], tpr["macro"], label='Macro average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]),
            color='b', linewidth=lw)
    ax.plot([0, 1], [0, 1], color='k', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(clf_name + ' ROC curve')
    ax.legend(loc='lower right', fontsize=7)


def plot_mcc(results, y_test, bottleneck_names):
    def confusion(ax, y_pred, y_test, len_b, algorithm):
        angles = [n / float(len_b) * 2 * pi for n in range(1, len_b)]
        angles += angles[:1]
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        bbot, values = [str(i) for i in range(1, len_b)], list()
        ax.set_xticklabels(bbot)
        ax.set_rlabel_position(0)
        values = list(user_defined_matthews_corrcoef(y_test, y_pred, averaging='no').values())
        values += values[:1]
        ax.plot(angles, values, label=algorithm)
        ax.fill(angles, values, alpha=0.1)
        ax.tick_params(axis="x", labelsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(algorithm)

    fig, axes = plt.subplots(2, 4, subplot_kw=dict(projection='polar'), figsize=(15, 8))
    axes = axes.flat
    len_b = len(bottleneck_names) + 1
    for i, algorithm in enumerate(results):
        y_pred = results[algorithm]
        confusion(axes[i], y_pred, y_test, len_b, algorithm)
    return fig
