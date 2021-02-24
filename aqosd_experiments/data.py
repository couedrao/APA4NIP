"""
helpers to load datasets
"""
import datetime

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def get_flatten_labels(raw_dataset_path):
    df0 = pd.read_csv(raw_dataset_path + 'bottlenecks_in_time.csv', index_col=0, parse_dates=True)
    df0 = df0.loc[df0['status'] == 'COMPLETED']
    df0['bottleneck'] = df0['bottleneck'].str.replace('_', ' ')
    df0[['start_at', 'end_at']] = df0.iloc[:, 0:2].apply(pd.to_datetime)
    df0 = df0.reset_index(drop=True)
    return df0


def import_and_prepare_data(raw_dataset_path, host_list, ct='10S', select='24'):
    df = load_host_df(host_list[0], raw_dataset_path)
    df.columns = [host_list[0] + '.' + str(col) for col in df.columns]
    for i in range(1, len(host_list)):
        temp = load_host_df(host_list[i], raw_dataset_path)
        temp.columns = [host_list[i] + '.' + str(col) for col in temp.columns]
        df = pd.merge(df, temp, left_index=True, right_index=True, sort=True)
    df = df[df.apply(pd.Series.value_counts).dropna(thresh=2, axis=1).columns]
    df.interpolate(method='time', inplace=True)
    df = df.resample(ct).mean()
    df.dropna(inplace=True)
    if select != 'all':
        end = max(df.index)
        start = end - datetime.timedelta(hours=int(select))
        df = df[(df.index > start) & (df.index <= end)]
    print(max(df.index) - min(df.index), 'of Data')
    l_df = get_flatten_labels(raw_dataset_path)
    packed = list(zip(l_df.start_at, l_df.end_at, l_df.node, l_df.bottleneck))
    df['bottleneck_node'] = [[n + '.' + b for start, end, n, b in packed if start <= el <= end] for el in df.index]
    df.reset_index(drop=True, inplace=True)
    X = df[[i for i in list(df.columns) if i not in ['bottleneck_node']]]
    y = binarizer_bottlenecks(df[['bottleneck_node']].copy())
    return X, y


def load_host_df(host, raw_dataset_path):
    df1 = pd.read_csv(raw_dataset_path + host + '_raw.csv', index_col=0, parse_dates=True)
    df1 = df1.pivot_table('val', ['cycle'], 'name')
    if 'System local time' in list(df1.columns):
        df1.drop(['System local time', 'System uptime', 'CPU nice time', 'Zabbix agent availability'], axis=1, inplace=True)
    df1.index = pd.to_datetime(df1.index)
    df1 = df1.groupby(pd.Grouper(freq='1S', label='right')).mean()
    return df1


def binarizer_bottlenecks(y, col='bottleneck_node'):
    y = y[col].astype(str).str.strip('[]').str.split(', ')
    mlb = MultiLabelBinarizer()
    y = pd.DataFrame(mlb.fit_transform(y), columns=mlb.classes_)
    y.columns.values[0] = "NaN"
    del y["NaN"]
    y.columns = [header.strip("'") for header in y.columns]
    return y


def scale_metrics(X, scaler):
    df = X.copy()
    scaled_features = scaler.fit_transform(df.values)
    X = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    return X
