from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
import tensorflow as tf
#from keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
plt.rcParams['figure.figsize'] = [20, 5]
from tqdm import tqdm
import pandas as pd

df = pd.read_csv('train.csv')

COUNTRIES = ['India']

data = df.loc[(df['Country_Region'] == COUNTRIES[0]) & (df['Target'] == 'ConfirmedCases')]
df_date = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in data['Date']]
df_conf = data['TargetValue']
df_fata = df.loc[(df['Country_Region'] == COUNTRIES[0]) & (df['Target'] == 'Fatalities')]['TargetValue']


plt.plot( df_date, df_conf, 'b' ), plt.title('Confirmed Cases'), plt.show();
plt.plot( df_date, df_fata, 'r' ), plt.title('Fatalities'), plt.show();



df_maxs = [max(df_conf), max(df_fata)]
df_mins = [min(df_conf), min(df_fata)]
def normalize(x, d):
    return (x-df_mins[d])/(df_maxs[d]-df_mins[d])
def denormalize(x, d):
    x = np.array(x)
    return x*(df_maxs[d]-df_mins[d])+df_mins[d]



data = []
for f1, f2 in zip(df_conf, df_fata):
    data.append([normalize(f1, 0), normalize(f2, 1)])
data = np.array(data)


x, y, seq = [], [], 16
for i in range(0, len(df_conf.values)-seq):
    x.append( data[i:i+seq] )
    y.append( data[i+seq] )
x = np.array(x)
y = np.array(y)

print('x shape:', x.shape)
print('y shape:', y.shape)
in_dim = (seq, 2)
out_dim = 2

model = Sequential()

model.add(LSTM(units=512, input_shape=in_dim, activation="relu")) 
model.add(Dense(256, activation="relu")) 
# model.add(Dense(16, activation="relu")) 
model.add(Dense(out_dim))

# model.compile(loss='mse', optimizer='adam') 
model.compile(loss='mae', optimizer='adam') 
 
model.summary()
hist = model.fit(x, y, epochs=2, verbose=0)


plt.plot([i for i in range(len(hist.history['loss']))], hist.history['loss']) 
plt.title('Training Loss over Epoch'), plt.ylabel('Loss'), plt.xlabel('Epoch'), plt.show()
print('Final loss value:', hist.history['loss'][-1])



yp = model.predict(x)


yp1, yp2 = zip(*yp) # Prediction
y1, y2 = zip(*y)    # Actual Data

print("Forecasted Output Values : ")

print("conformed Cases", yp1)
print("conformed Deaths", yp2)

plt.title('Confirmed Cases'), plt.plot(denormalize(yp1, 0), 'b'), plt.plot(denormalize(y1, 0), 'r'), plt.show();
plt.title('Fatalities'), plt.plot(denormalize(yp2, 1), 'b'), plt.plot(denormalize(y2, 1), 'r'), plt.show();
print('Blue = Prediction, Red = Real Data')



mae = tf.keras.losses.MeanAbsoluteError()

countries = df['Country_Region'].unique()
seq = 16
in_dim = (seq, 2)
out_dim = 2

results = []
for country in tqdm(countries):
    df_conf = df.loc[(df['Country_Region'] == country) & (df['Target'] == 'ConfirmedCases')]['TargetValue']
    df_conf = ( df_conf - min(df_conf) )/( max(df_conf) - min(df_conf) )
    df_fata = df.loc[(df['Country_Region'] == country) & (df['Target'] == 'Fatalities')]['TargetValue']
    df_fata = ( df_fata - min(df_fata) )/( max(df_fata) - min(df_fata) )
    data = np.array([ [f1, f2] for f1, f2 in zip(df_conf, df_fata) ])
    x, y = [], []
    for i in range(0, len(df_conf.values)-seq):
        x.append( data[i:i+seq] ), y.append( data[i+seq] )
    x = np.array(x)
    y = np.array(y)
    yp = model.predict(x)
    if not np.isnan(mae(y, yp).numpy()):
        results.append( [mae(y, yp).numpy(), country] )
print('Done')


results.sort()
sorted_errors, sorted_countries = zip(*results)
sorted_errors = (np.array(sorted_errors) - min(sorted_errors))/(max(sorted_errors) - min(sorted_errors))*100
y_pos = np.arange(len(sorted_countries))
TOP_K = 10

## First Plot: Error Chart For Every Country
plt.title('Covid19 Jan-Jun 2021 Global Forecast')
plt.ylabel('Covid Rate')
plt.bar(y_pos, sorted_errors, align='center', alpha=0.5)
plt.show()
print('Error mean:', np.mean(sorted_errors))
print('Error std:', np.std(sorted_errors))
print('Total No. Countires Which Cannot predict (NaN): ', len(countries) - len(y_pos))

## Second Plot: Top 10 correct prediction
plt.title('Top 10 least error prediction')
plt.ylabel('Errors')
plt.bar(y_pos[:TOP_K], sorted_errors[:TOP_K], align='center', alpha=0.5)
plt.xticks(y_pos[:TOP_K], sorted_countries[:TOP_K])
plt.show()

## Third Plot: Top 10 inaccurate prediction
plt.title('Top 10 most error prediction')
plt.ylabel('Errors')
plt.bar(y_pos[-TOP_K:], sorted_errors[-TOP_K:], align='center', alpha=0.5)
plt.xticks(y_pos[-TOP_K:], sorted_countries[-TOP_K:])
plt.show()



