import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle



style.use('ggplot')

df = quandl.get('WIKI/GOOGL')


df = df[['Open', 'High', 'Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Open']) / df['Open'] * 100.0

#             price        x           x              x
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna('-99999', inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label', 'Adj. Close'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)    #clf --- classifier
# clf = svm.SVR(kernel='poly')
clf.fit(X_train, Y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


