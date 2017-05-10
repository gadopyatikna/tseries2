import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import download_plotlyjs, plot, iplot
from plotly import graph_objs as go
import helper

def train_hw(dataset):

    def timeseriesCVscore(x):
        # вектор ошибок
        errors = []

        values = data.values
        alpha, beta, gamma = x

        # задаём число фолдов для кросс-валидации
        tscv = TimeSeriesSplit(n_splits=3)

        # идем по фолдам, на каждом обучаем модель, строим прогноз на отложенной выборке и считаем ошибку
        for train, test in tscv.split(values):
            model = HoltWinters(series=values[train], slen=24 * 7, alpha=alpha, beta=beta, gamma=gamma,
                                n_preds=len(test))
            model.fit_triple_exp_smoth()

            predictions = model.result[-len(test):]
            actual = values[test]
            error = mean_squared_error(predictions, actual)
            errors.append(error)

        # Возвращаем средний квадрат ошибки по вектору ошибок
        return np.mean(np.array(errors))

    # training process
    # for testing
    data = dataset.Users[:-500]
    # init params
    x = [0, 0, 0]

    start = time.time()
    opt = minimize(timeseriesCVscore, x0=x, method="TNC", bounds = ((0, 1), (0, 1), (0, 1)))
    end = time.time()

    alpha_final, beta_final, gamma_final = opt.x

    print('H-W parameters estimation: {} {} {}'.format(alpha_final,
                                                       beta_final, gamma_final))
    print('Elapsed time: {}'.format(end-start))
    return alpha_final, beta_final, gamma_final

class HoltWinters:
    """
    https://fedcsis.org/proceedings/2012/pliks/118.pdf

    # slen - season length
    # alpha, beta, gamma - H-W params
    # series, n_preds, confidence - speaks for itself

    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, confidence=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = confidence

    # stands for b
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # вычисляем сезонные средние
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))

        # вычисляем начальные значения
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]

            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def fit_triple_exp_smoth(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()
        smooth = self.series[0]
        trend = self.initial_trend()

        self.result.append(self.series[0])
        self.Smooth.append(smooth)
        self.Trend.append(trend)
        self.Season.append(seasonals[0 % self.slen])
        self.PredictedDeviation.append(0)

        self.UpperBond.append(self.result[0] +
                              self.scaling_factor *
                              self.PredictedDeviation[0])

        self.LowerBond.append(self.result[0] -
                              self.scaling_factor *
                              self.PredictedDeviation[0])

        for i in range(1, len(self.series) + self.n_preds):

            if i >= len(self.series):  # прогнозируем
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])
                # во время прогноза с каждым шагом увеличиваем неопределенность
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (
                smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Отклонение рассчитывается в соответствии с алгоритмом Брутлага
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i])
                                               + (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] + self.scaling_factor * self.PredictedDeviation[-1])
            self.LowerBond.append(self.result[-1] - self.scaling_factor * self.PredictedDeviation[-1])
            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])


def plotHW(model, data, title=''):
    index = data.index
    data = data.Users
    model.result = np.array(model.result).reshape(1, -1)[0]
    model.LowerBond = np.array(model.LowerBond).reshape(1, -1)[0]
    model.UpperBond = np.array(model.UpperBond).reshape(1, -1)[0]

    Anomalies = np.array([np.NaN] * len(data))
    Anomalies[data.values < model.LowerBond] = data.values[data.values < model.LowerBond]
    # print(data.values)

    # return
    layout = {
        # to highlight the timestamp we use shapes and create a rectangular
        'shapes': [
            # 1st highlight during Feb 4 - Feb 6
            {
                'type': 'rect',
                # x-reference is assigned to the x-values
                'xref': 'x',
                # y-reference is assigned to the plot paper [0,1]
                'yref': 'paper',
                'x0': index[-model.n_preds],
                'y0': 0,
                'x1': index[-1],
                'y1': 1,
                'fillcolor': 'yellow',
                'opacity': 0.3,
                'line': {
                    'width': 0,
                }
            }]
    }

    helper.plotly_anomalies(model.result, data.values, model.LowerBond, model.UpperBond, index, layout)

    # plotting_data = []
    #
    # anom_trace = go.Scatter(x=index, y=Anomalies, mode='markers', marker=dict(size=16, color='red'), name='anomalies')
    # plotting_data.append(anom_trace)
    #
    # orig_data = go.Scatter(x=index, y=data, mode='lines', line=dict(color='blue'), name='original')
    # plotting_data.append(orig_data)
    #
    # for ts, fill in zip([model.LowerBond, model.UpperBond], (None, 'tonexty')):
    #     trace = go.Scatter(
    #         x=index,
    #         y=ts,
    #         fill=fill,
    #         mode='lines',
    #         line=dict(dash='dash', color='grey'),
    #         name='confidence interval'
    #     )
    #     plotting_data.append(trace)
    #
    # layout = {
    #     # to highlight the timestamp we use shapes and create a rectangular
    #     'shapes': [
    #         # 1st highlight during Feb 4 - Feb 6
    #         {
    #             'type': 'rect',
    #             # x-reference is assigned to the x-values
    #             'xref': 'x',
    #             # y-reference is assigned to the plot paper [0,1]
    #             'yref': 'paper',
    #             'x0': index[-model.n_preds],
    #             'y0': 0,
    #             'x1': index[-1],
    #             'y1': 1,
    #             'fillcolor': 'yellow',
    #             'opacity': 0.3,
    #             'line': {
    #                 'width': 0,
    #             }
    #         }]
    # }
    #
    # fig = dict(data=plotting_data, layout=layout)
    # plot(fig, show_link=False)


# def plotHW(model, data):
#     data = data.Users
#
#     model.LowerBond = np.array(model.LowerBond).reshape(1,-1)[0]
#     model.UpperBond = np.array(model.UpperBond).reshape(1,-1)[0]
#
#     Anomalies = np.array([np.NaN] * len(data))
#     Anomalies[data.values < model.LowerBond] = data.values[data.values < model.LowerBond]
#
#     plt.figure(figsize=(25, 10))
#     plt.plot(model.result, label = "Modelled")
#     plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Upper/lower bnd")
#     plt.plot(model.LowerBond, "r--", alpha=0.5)
#     plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, y2=model.LowerBond, alpha=0.5, color = "grey")
#     plt.plot(data.values, label = "Observed")
#     plt.plot(Anomalies, "o", markersize=10, label = "Anomalies")
#
#     plt.axvspan(len(data)-128, len(data), alpha=0.5, color='lightgrey')
#
#     plt.grid(True)
#     plt.axis('tight')
#     plt.legend(loc="best", fontsize=13)
#     plt.show()