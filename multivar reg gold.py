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
def fred_to_pd(url):
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
url_fred = "https://api.stlouisfed.org/fred"
url_fred += "/series/observations?series_id=DFII10"
url_fred += "&api_key=*****"
url_fred += "&file_type=json"
ryld = fred_to_pd(url_fred)
ryld['value'].replace('.', np.nan, inplace = True)
ryld['value']=ryld['value'].astype(float)

#%% Download data from EIA
def eia_to_pd(url):
    with urlopen(url) as response:
        source = response.read()

    data = json.loads(source)
    data = pd.DataFrame(data['series'][0]['data'])
    data.columns = ['date','stock(kbbl)']
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data.set_index('date', inplace=True)
    
    return data
#%%
url_eia = "https://api.eia.gov/series"
url_eia += "/?api_key=*****"
url_eia += "&series_id=PET.WTESTUS1.W"
stk = eia_to_pd(url_eia)

#%% Merging
xau = xau.reset_index()
ryld = ryld.reset_index()
xau.columns = ['date','close']
ryld.columns = ['date', 'real_yield']
df = pd.merge(xau, ryld, how='left')
df = pd.merge(df, stk.reset_index(), how='left')
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
X = cln_df['2006':][['real_yield','stock(kbbl)']]
y = cln_df['2006':]['close']
reg.fit(X, y)
reg.score(X, y)
feed = cln_df[['real_yield','stock(kbbl)']].to_numpy()
feed = feed.reshape(-1,2)
model = pd.DataFrame(reg.predict(feed), columns=['predicted_close'])
model = model.join(cln_df.reset_index())
model.set_index('date', inplace=True)

#%% Plotting Actual vs Model
plot = sns.lineplot(x='date', y='close', data=model['2006':])
plot = sns.lineplot(x='date', y='predicted_close', data=model['2006':])
plot.set(xlabel='Year', ylabel='Gold Price')
plot.set_title('Valuing Gold Price with 10Y Real Yields and Cushing Stocks')
plt.legend(labels=["Actual","Prediction"])

#%% Model performance
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
print("The R2 Value is ", round(reg.score(X, y),2))
#%%
