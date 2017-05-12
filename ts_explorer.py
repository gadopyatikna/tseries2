import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

def load_csv(path='/media/sf_Shared/db_export/dlk_load_instances_sample_5k.csv'):
    df = pd.read_csv(path, '|', index_col=['DATETIME'],
                     parse_dates=['DATETIME'])
    df = df.sort()
    return df

def plot_samples():
    df = load_csv()
    df=df.sort()
    reccnt = df.groupby(['SRCID', 'KEY'])['RECCNT'].apply(list)
    sysids, _ = reccnt.index.levels
    i=0
    for sysid in sysids:
        keys = reccnt[sysid].index
        for key in keys:
            i+=1
            ts = reccnt[sysid, key]
            plt.clf()
            plt.plot(range(len(ts)), ts, 'bo--')
            plt.title(str(key))
            print(i, sysid, key)
            plt.savefig('/media/sf_Shared/plots/{}_{}_{}.png'.format(sysid, key, i))

def df_test(ts):
    test = sm.tsa.adfuller(ts)
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print('non-stationary')
    else:
        print('stationary')

def jb_test(ts):
    row = [u'JB', u'p-value', u'skew', u'kurtosis']
    jb_test = sm.stats.stattools.jarque_bera(ts)
    a = np.vstack([jb_test])
    result = pd.DataFrame(data=a, columns=row)
    print(result)

plot_samples()

'''

df = load_csv()
ts = df[df.SRCID=='ATM']
ts = ts[ts.KEY==17075]

ts_index = ts.index
# print(ts.head().index)
# df_test(ts)
# jb_test(ts)


#linear regression


from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import linear_reg as lreg
import helper

ts = pd.DataFrame(data=ts.RECCNT.values, columns=['reccnt'])
x_trn, x_tst, y_trn, y_tst = lreg.add_features_mk_split(ts, lag_start=1, lag_end=7)
n_pr = len(x_tst)

lr = sm.OLS(y_trn, x_trn).fit()
pred_tst = lr.predict(x_tst)

print('simple linreg mean abs err: ', mean_absolute_error(y_tst, pred_tst))

from statsmodels.sandbox.regression.predstd import wls_prediction_std

alpha=0.05
_, fitted_lower, fitted_upper = wls_prediction_std(res=lr, alpha=alpha)
_, pred_lower, pred_upper = wls_prediction_std(res=lr, exog=x_tst, alpha=alpha)

lower = np.hstack((fitted_lower, pred_lower))
upper = np.hstack((fitted_upper, pred_upper))

y_full = ts[5:].reccnt.values
index = ts_index[5:]

Anomalies = np.array([np.NaN] * len(y_full))
Anomalies[y_full < lower] = y_full[y_full < lower]

# helper.plotly_df(pd.DataFrame(data=np.array([y_full, lower, upper]).T, columns=['y','low','up']))

# layout = {
#         # to highlight the timestamp we use shapes and create a rectangular
#         'shapes': [
#             # 1st highlight during Feb 4 - Feb 6
#             {
#                 'type': 'rect',
#                 # x-reference is assigned to the x-values
#                 'xref': 'x',
#                 # y-reference is assigned to the plot paper [0,1]
#                 'yref': 'paper',
#                 'x0': index[-n_pr],
#                 'y0': 0,
#                 'x1': index[-1],
#                 'y1': 1,
#                 'fillcolor': 'yellow',
#                 'opacity': 0.3,
#                 'line': {
#                     'width': 0,
#                 }
#             }]
#     }

pred = np.hstack((lr.fittedvalues, pred_tst))
# print(pred.shape, y_full.shape, lower.shape)
# print(index)
rng = range(len(pred))
# plt.plot(rng,pred)
# plt.plot(rng,y_full)
# plt.show()
print(pred.shape)
layout={'title':'test'}
print(index)
# index = [x for x in rng]
helper.plotly_anomalies(pred, y_full, lower, upper, index, layout)
'''