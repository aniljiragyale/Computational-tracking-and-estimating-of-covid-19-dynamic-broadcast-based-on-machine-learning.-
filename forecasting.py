import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from datetime import datetime
import itertools
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm


data = pd.read_csv('train.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date',inplace=True)
print(data.head())

cases_us = data[(data['Target'] == 'ConfirmedCases') & (data['Country_Region'] == 'US')]
deaths_us = data[(data['Target'] == 'Fatalities') & (data['Country_Region'] == 'US')]
print(cases_us.head())

# Inspect the TaretValues in the data
print('There are {} records that contains negative value of cases.'
      .format(cases_us[cases_us['TargetValue']<0].count()['TargetValue']))
# Replace them by 0, which means that they wont be counted into the final aggregation
cases_us = cases_us[~cases_us['TargetValue']<0]


# Inspect the TaretValues in the data
print('There are {} records that contains negative value of deaths.'
      .format(deaths_us[deaths_us['TargetValue']<0].count()['TargetValue']))
# Replace them by 0, which means that they wont be counted into the final aggregation
deaths_us = deaths_us[~deaths_us['TargetValue']<0]

# two time series for the cases and deaths in the US
cases_us = cases_us.groupby(level = 'Date')['TargetValue'].sum()
deaths_us= deaths_us.groupby(level = 'Date')['TargetValue'].sum()
df = cases_us.to_frame()
df = df.rename(columns = {'TargetValue': 'Cases'})
df['Deaths'] = deaths_us

# We will get rid of the days when the US have less than 10 cases and deaths each day
# So the forecasting of the later days would be less affected by values on those days
df =df[(df['Cases']>10)&(df['Deaths']>10)]

print(df.shape)

df['Cases'].rolling(7).mean().plot(label='Weekly Rolling Mean')
df['Cases'].rolling(7).std().plot(label='Weekly Rolling Std')
df['Cases'].plot()
plt.legend()
df['cases_4_rolling'] = df['Cases'].rolling(4).mean().dropna()


df['Deaths'].rolling(7).mean().plot(label='Weekly Rolling Mean')
df['Deaths'].rolling(7).std().plot(label='Weekly Rolling Std')
df['Deaths'].plot()
plt.legend()
df['deaths_4_rolling'] = df['Deaths'].rolling(4).mean().dropna()



# Store in a function for later use!
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        

adf_check(df['cases_4_rolling'].dropna())


adf_check(df['deaths_4_rolling'].dropna())


# First difference
df['Death first difference'] = df['deaths_4_rolling'] - df['deaths_4_rolling'].shift(1)
adf_check(df['Death first difference'].dropna())


# Second difference
df['Death second difference'] = df['Death first difference'] - df['Death first difference'].shift(1)
adf_check(df['Death second difference'].dropna())

df['Death second difference'].plot()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

acf_cases = plot_acf(df['cases_4_rolling'].dropna())


pacf_cases = plot_pacf(df['cases_4_rolling'].dropna())


acf_cases = plot_acf(df['Death second difference'].dropna())


# evaluate an ARIMA model for a given order (p,d,q)

## Using root mean quared error metric
def rmse(test, pred):
    return sqrt(mean_squared_error(test, pred))
## Using mean absolute percentage error metric:
def mape(test, pred):
    test, pred = np.array(test), np.array(pred)
    return np.mean(np.abs((test - pred)/test))*100

def arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    return test, predictions



# evaluate combinations of p, d and q values for an ARIMA model

def GridSearch_arima(dataset, p_values, d_values, q_values):
    '''
    We will choose the arima model that has the lowest RMSE (You can switch to another one if you'd like)
    '''
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    #rmse = rmse(arima_model(dataset, order)[0], arima_model(dataset, order)[1])
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    
    
# evaluate parameters

p_values = range(0, 7)
d_values = range(0, 3)
q_values = range(0, 7)


import warnings
warnings.filterwarnings("ignore")

GridSearch_arima(df['Cases'].values, p_values, d_values, q_values)

# run the selected model
model = sm.tsa.arima.ARIMA(df['Cases'], order=(6,1,2)) 
results = model.fit()
predictions = results.predict(len(df['Cases'])-20, len(df['Cases'])-1, typ = 'levels').rename("Predictions") 

plt.figure(figsize = [10,5])
df['Cases'].plot(legend = True)
predictions.plot(legend = True) 
plt.title('Forecasting Result using ARIMA on 93 training time steps')

# Evaluation
print('RMSE: {}'.format(rmse(df.iloc[-20:]['Cases'],predictions.values)))
print('MAPE: {}'.format(mape(df.iloc[-20:]['Cases'],predictions.values)))


# Get the daily cases of US from sources 
real_data = pd.read_csv('time_series_covid19_confirmed_global.csv')

real_usdata = real_data[(real_data['Country/Region']  == 'US')]
real_usdata = real_usdata.stack().reset_index().drop('level_0',axis = 1).iloc[3:,:].rename(columns = {'level_1': 'Date', 0: 'real_cases'})
real_usdata['Date'] = pd.to_datetime(real_usdata['Date'])
real_usdata = real_usdata.set_index('Date')
real_usdata = real_usdata[real_usdata.index >= '2020-06-10']
real_usdaily = real_usdata - real_usdata.shift(1)

# Create a dataframe that has extended datetime index to store the predicted and real # of cases
from pandas.tseries.offsets import DateOffset
future_dates = [df.index[-1] + DateOffset(days=x) for x in range(0,27) ]
future_dates_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)
future_df = pd.concat([df,future_dates_df])

future_df['pred'] = predictions
future_df['real'] = real_usdaily
future_df.tail(10)


print(predictions)
