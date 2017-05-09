import numpy as np
import helper
import holt_winters as hw
import linear_reg as lreg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

df = helper.load_data('/home/dunno/Projects/ods/data/hour_online.csv')
# helper.plotly_df(df, title="Online users")
# helper.plot_rolling_mean(df,24*7)
# helper.plot_double_exp_smoth(df)
a, b, g = 0.00663426706434, 0.0, 0.0467652042897#hw.train_hw(df) #

'''
hand-tuned parameters are
    n_pr - number of predictions to make
    slen - length of the SEASON
'''
# n_pr = 128
# model = hw.HoltWinters(df[:-n_pr].values, slen=24*7, alpha=a, beta=b, gamma=g, n_preds=n_pr, confidence=2.56)
# model.fit_triple_exp_smoth()
# hw.plotHW(model, df)
# print('HW mean abs error: ',mean_absolute_error(df.Users.values, model.result))

'''
linear regression
'''

''' '''
x_trn, x_tst, y_trn, y_tst = lreg.add_features_mk_split(df, lag_start=12, lag_end=48)

model, pred_tst = lreg.fit_lr(x_trn, x_tst, y_trn, y_tst)
# err = lreg.performTimeSeriesCV(x_trn, y_trn, 5, model, 'ABS')
print('simple linreg mean abs err: ', mean_absolute_error(y_tst, pred_tst))

import xgboost as xgb

dtrain = xgb.DMatrix(x_trn, label=y_trn)
dtest = xgb.DMatrix(x_tst)

def gridCV():
    errs = np.zeros((5*10,4))
    i=0

    for lrate in np.linspace(0.1, 0.2, 5):
        for estimators in range(100, 1001, 100):
            params = {
                'objective': 'reg:linear',
                'booster': 'gblinear',
                'learning_rate':lrate,
                'n_estimators': estimators,
                'seed':0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight':1,
                'max_depth':1
            }
            crossval = xgb.cv(params, dtrain, metrics=('rmse'), verbose_eval=False, nfold=10, show_stdv=False)
            bst = xgb.train(params, dtrain)
            pred_tst = bst.predict(dtest)

            errs[i] = [mean_absolute_error(pred_tst, y_tst), min(crossval['test-rmse-mean']), lrate, estimators]

            print(i)
            # print(errs[i])
            minind = np.argmin(errs[:,0][:i+1])
            print(errs[minind])

            i+=1
    print(errs[np.argmin(errs[:,0][:i+1])])

def boost_lr():
    params = {
                'objective': 'reg:linear',
                'booster': 'gblinear',
                'learning_rate':0.2,
                'n_estimators': 900,
                'seed':0,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight':1,
                'max_depth':1
            }


    cv = xgb.cv(params, dtrain, metrics=('rmse'), verbose_eval=False, nfold=10, show_stdv=False)
    bst = xgb.train(params, dtrain)

    pred_trn = bst.predict(dtrain)
    pred_tst = bst.predict(dtest)
    pred_full = np.concatenate((pred_trn, pred_tst))

    dev = min(cv['test-rmse-mean'])
    confidence = 1.96
    lower = pred_full - confidence*dev
    upper = pred_full + confidence*dev

    title = 'boosted linear regression'
    layout = {'title':title}

    helper.plotly_anomalies(pred_full, np.concatenate((y_trn, y_tst)), lower, upper, df.index, layout)

    print('xgboost linreg mean absolute error: ',mean_absolute_error(y_tst, pred_tst))


# gridCV()
boost_lr()
