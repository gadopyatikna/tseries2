import pandas as pd
import numpy as np
from functools import wraps

import matplotlib.pyplot as plt
import matplotlib as mpl

from plotly import __version__
from plotly.offline import download_plotlyjs, plot, iplot
from plotly import graph_objs as go

def plotly_df(df, title=''):
    data = []

    for column in df.columns:
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column
        )
        data.append(trace)

    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False)

def plot_rolling_mean(series, n, confidence=1.96):
    roll_mean = series.rolling(window=n).mean()

    rolling_std =  series.rolling(window=n).std()
    upper_bond = roll_mean+confidence*rolling_std
    lower_bond = roll_mean-confidence*rolling_std

    roll_mean['upper_bnd'] = upper_bond
    roll_mean['lower_bnd'] = lower_bond
    roll_mean['original_ts'] = series.values

    # print(roll_mean[n:].head())
    plotly_df(roll_mean[n:])
    return roll_mean[n: ]

def double_exp_smoth(series, alpha, beta):
    result = [series[0]]
    level = series[0]
    trend = series[1] - series[0]

    for n in range(2, len(series)):
        value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)

    return result

def plot_double_exp_smoth(series):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(20, 8))
        for alpha in [0.9, 0.02]:
            for beta in [0.9, 0.02]:
                plt.plot(double_exp_smoth(series.values, alpha, beta),
                         label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label="Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        plt.show()


def load_data(path):
    dataset = pd.read_csv(path, index_col=['Time'], parse_dates=['Time'])
    # print(dataset.head())
    return dataset

def plotly_anomalies(predicted, observed, lower, upper, index, layout):

    Anomalies = np.array([np.NaN] * len(observed))

    Anomalies[observed < lower] = observed[observed < lower]
    Anomalies[observed > upper] = observed[observed > upper]

    plotting_data = []

    anom_trace = go.Scatter(x=index, y=Anomalies, mode='markers', marker=dict(size=16, color='red'), name='anomalies')
    plotting_data.append(anom_trace)

    orig_data = go.Scatter(x=index, y=observed, mode='lines', line=dict(color='blue'), name='observed')
    plotting_data.append(orig_data)

    pred_data = go.Scatter(x=index, y=predicted, mode='lines', line=dict(color='black'), name='predicted')
    plotting_data.append(pred_data)

    for ts, fill in zip([lower, upper], (None, 'tonexty')):
        trace = go.Scatter(
            x=index,
            y=ts,
            fill=fill,
            mode='lines',
            line=dict(dash='dash', color='grey'),
            name='confidence interval'
        )
        plotting_data.append(trace)


    fig = dict(data=plotting_data, layout=layout)
    plot(fig, show_link=False)