import math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib as mpl
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Lasso,BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 2, 9)

df = web.DataReader("AAPL", 'yahoo', start, end)
print(df.tail())


close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()
dfreg = df.loc[:,['Adj Close','Volume']]

dfreg['HL_PCT'] = (df['High'] -df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0



# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.1, random_state=42)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

clflasso=Lasso()
clflasso.fit(X_train,y_train)

clfbayes=BayesianRidge()
clfbayes.fit(X_train,y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)
confidencebayes=clfbayes.score(X_test,y_test)
confidencelasso=clflasso.score(X_test,y_test)

print(confidencebayes,confidencelasso)

forecast_set = clfpoly3.predict(X_lately)
dfreg['Poly3'] = None
dfreg['Poly3'][-1*len(forecast_set):]=forecast_set

forecast_set = clfknn.predict(X_lately)
dfreg['knn'] = None
dfreg['knn'][-1*len(forecast_set):]=forecast_set

forecast_set = clfbayes.predict(X_lately)
dfreg['Bayes'] = None
dfreg['Bayes'][-1*len(forecast_set):]=forecast_set


forecast_set = clflasso.predict(X_lately)
dfreg['Lasso'] = None
dfreg['Lasso'][-1*len(forecast_set):]=forecast_set


dfreg['Adj Close'].tail(30).plot()
dfreg['Poly3'].tail(30).plot()
dfreg['Bayes'].tail(30).plot()
dfreg['Lasso'].tail(30).plot()
dfreg['knn'].tail(30).plot()

plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Price')
print(dfreg)
plt.show()