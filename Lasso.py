
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
plt.rcParams['figure.figsize'] = [20, 5]
import pandas as pd
from sklearn.linear_model import Lasso


df = pd.read_csv('train.csv')
dt = pd.read_csv('test.csv')
df['Target'].replace(['ConfirmedCases','Fatalities'],[0,1],inplace=True)
dt['Target'].replace(['ConfirmedCases','Fatalities'],[0,1],inplace=True)

df['date_dup'] = pd.to_datetime(df['Date'])
df['month'] = 0
list1=[]
for i in df['date_dup']:
    list1.append(i.month)
df['month'] = list1
df['date'] = 0
list1=[]
for i in df['date_dup']:
    list1.append(i.day)
df['date'] = list1

dt['date_dup'] = pd.to_datetime(dt['Date'])
dt['month'] = 0
list1=[]
for i in dt['date_dup']:
    list1.append(i.month)
dt['month'] = list1
dt['date'] = 0
list1=[]
for i in dt['date_dup']:
    list1.append(i.day)
dt['date'] = list1

#dt['ForecastId'] =dt['Id']
dt.rename(columns={"ForecastId": "Id"},inplace = True) 
df.drop(['County','Province_State','Country_Region','Date','date_dup'],axis = 1,inplace = True)
dt.drop(['County','Province_State','Country_Region','Date','date_dup'],axis = 1,inplace = True)


X=df.drop(['TargetValue'],axis=1)
y=df['TargetValue']

x_test = dt


from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)  # You can tune the alpha parameter
lasso.fit(X, y)


y_pred = lasso.predict(x_test)
print(y_pred)

pred=pd.DataFrame(y_pred)
#print(pred)
sub_df=pd.read_csv('submission.csv')
#print(sub_df.shape)
datasets=pd.concat([sub_df['ForecastId_Quantile'],pred],axis=1)
datasets.columns=['ForecastId_Quantile','TargetValue']
datasets.to_csv('samplesubmission.csv',index=False)
