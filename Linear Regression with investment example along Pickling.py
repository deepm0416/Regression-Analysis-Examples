#!/usr/bin/env python
# coding: utf-8

# In[19]:


pip install Quandl 


# In[43]:


import pandas as pd
import quandl, datetime, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot  as plt
from matplotlib import style
import pickle

style.use('ggplot')


# In[46]:


quandl.ApiConfig.api_key = 'HMPuzvc248VuCMdP6x3z'
df = quandl.get('WIKI/GOOGL')
df.head()


# In[47]:


df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']]


# In[48]:


df.shape


# In[49]:


df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


# In[50]:


df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]


# In[51]:


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)


# In[53]:


forecast_out = int(math.ceil(0.01*len(df)))
df['label']=df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)

X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

#Pickling
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)


# In[35]:


df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


# In[42]:


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:




