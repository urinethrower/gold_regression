#%%
from cmath import sqrt
from sklearn import linear_model
from sklearn.impute import KNNImputer
import yfinance as yf
from email.utils import parsedate
import json
from pickle import TRUE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing.sharedctypes import Value
from pdb import runcall
from urllib.request import urlopen

#%% Download Gold Price from yfinance
xau = yf.download('GC=F', period='max')[['Adj Close']]

#%% Download data from FRED
def json_to_pandas(url):
    with urlopen(url) as response:
        source = response.read()

    data = json.loads(source)
    data = data['observations']
    data = pd.DataFrame(data)
    data = data[['date','value']]
    data['date'] = pd.to_datetime(data['date']) #Optional: , format='%Y-%m-%d'
    data.set_index('date', inplace=True)
    
    return data
#%%
url = "https://api.stlouisfed.org/fred"
url += "/series/observations?series_id=DFII10"
url += "&api_key=*****"
url += "&file_type=json"
ryld = json_to_pandas(url)
ryld['value'].replace('.', np.nan, inplace = True)
ryld['value']=ryld['value'].astype(float)

#%% Merging
xau = xau.reset_index()
ryld = ryld.reset_index()
xau.columns = ['date','close']
ryld.columns = ['date', 'real_yield']
df = pd.merge(xau, ryld, how='left')
df.set_index('date', inplace=True)

#%% Imputation
# impute = KNNImputer(n_neighbors=2)
# imputed = impute.fit_transform(df)
# imp_df = df.drop(columns=['close','real_yield'])
# imp_df.insert(0,'close',imputed[:,0],True)
# imp_df.insert(1,'real_yield',imputed[:,1],True)

#%% Cleaned df (without imputation)
# df.isnull().sum() / len(df) #impact check
cln_df = df
cln_df.dropna(axis=0, how='any', inplace=True)

#%% Linear Regression Model for 15 years
reg = linear_model.LinearRegression()
reg.fit(cln_df['2006':].drop('close',axis=1,inplace=False), cln_df.close['2006':])
r2 = reg.score(cln_df['2006':].drop('close',axis=1,inplace=False), 
cln_df.close['2006':])
feed = cln_df['real_yield'].to_numpy()
feed = feed.reshape(-1,1)
model = pd.DataFrame(reg.predict(feed), columns=['predicted_close'])
model = model.join(cln_df.reset_index())
model.set_index('date', inplace=True)

#%% Plotting Actual vs Model
plot = sns.lineplot(x='date', y='close', data=model['2006':])
plot = sns.lineplot(x='date', y='predicted_close', data=model['2006':])
plot.set(xlabel='Year', ylabel='Gold Price')
plot.set_title('Valuing Gold Price with 10Y Real Yields')
plt.legend(labels=["Actual","Prediction"])

#%% Model Performance
def RMSE(df, actual, predict):
    actual = df[actual]
    predict = df[predict]
    summation = 0
    n = len(actual)
    for i in range (0,n):
        difference = actual[i] - predict[i]
        squared_difference = difference**2
        summation += squared_difference
    RMSE = sqrt(summation/n)
    print("The Root Mean Squared Error is: " , round(RMSE.real, 2))

RMSE(model, 'close', 'predicted_close')
print("The R2 Value is ", round(r2, 2))
#%%