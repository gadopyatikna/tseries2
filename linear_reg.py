import numpy as np
import pandas as pd
import helper
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression

# def simple_lr(x_trn, x_tst, y_trn, y_tst):
#     # print(x_trn.head())
#
#     lr = LinearRegression()
#     lr.fit(x_trn, y_trn)
#     pred = lr.predict(x_tst)
#     return lr, pred

def plot_simple_lr(x, y, model):
    pred = model.predict(x)

    lr_df = pd.DataFrame(data=np.array([pred, y]).T, columns=['prediction', 'observed'])
    helper.plotly_df(lr_df, 'linear regression')

def get_temporal_features(df):
    if type(df.head().index) != pd.tseries.index.DatetimeIndex:
        df.index = df.index.to_datetime()

    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df.weekday.isin([5,6])*1

def get_means(df, cat_feat, real_feat):
    return df.groupby(cat_feat)[real_feat].mean().to_dict()

# TAKES PANDAS.DATAFRAME
def add_features_mk_split(data, lag_start=5, lag_end=50, test_size=0.15):
    y = data.columns[0]
    # print(y)
    test_idx = int(len(data) * (1 - test_size))

    # ??
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data[y].shift(i)

    get_temporal_features(data)
    mapping = get_means(data, 'weekday', y)
    data['weekday_avg'] = data.weekday.map(mapping).values

    mapping = get_means(data, 'hour', y)
    data['hour_avg'] = data.hour.map(mapping).values

    data.drop(["hour", "weekday"], axis=1, inplace=True)

    data = data.dropna()
    data = data.reset_index(drop=True)
    # print(data.head())
    x_trn = data.loc[:test_idx].drop([y], axis=1)
    y_trn = data.loc[:test_idx][y]
    x_tst = data.loc[test_idx:].drop([y], axis=1)
    y_tst = data.loc[test_idx:][y]

    return x_trn, x_tst, y_trn, y_tst

def performTimeSeriesCV(X_train, y_train, number_folds, model, metrics='ABS'):
    if metrics == 'MSE':
        metrics = mean_squared_error
    elif metrics == 'ABS':
        metrics = mean_absolute_error

    print('Size train set: {}'.format(X_train.shape))

    k = int(np.floor(float(X_train.shape[0]) / number_folds))
    print('Size of each fold: {}'.format(k))

    errors = np.zeros(number_folds - 1)

    # loop from the first 2 folds to the total number of folds
    for i in range(2, number_folds + 1):
        print
        ''
        split = float(i - 1) / i
        print('Splitting the first ' + str(i) + ' chunks at ' + str(i - 1) + '/' + str(i))

        X = X_train[:(k * i)]
        y = y_train[:(k * i)]
        print('Size of train + test: {}'.format(X.shape))  # the size of the dataframe is going to be k*i

        index = int(np.floor(X.shape[0] * split))

        # folds used to train the model
        X_trainFolds = X[:index]
        y_trainFolds = y[:index]

        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]

        model.fit(X_trainFolds, y_trainFolds)
        errors[i - 2] = metrics(model.predict(X_testFold), y_testFold)

    # the function returns the mean of the errors on the n-1 folds
    return errors.mean()