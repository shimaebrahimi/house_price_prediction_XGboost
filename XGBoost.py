#!/usr/bin/env python
# coding: utf-8

# In[17]:


pip install xgboost


# In[18]:


import xgboost as xgb
import numpy as np
import pandas as pd


# In[19]:


data=pd.read_csv('kc_house_data.csv')


# In[20]:


data.head()


# In[21]:


x=data.iloc[:,3:]


# In[22]:


y=data.loc[:,'price']


# In[23]:


from sklearn.model_selection import train_test_split 


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[25]:


datamatrix=xgb.DMatrix(data=x,label=y)


# In[27]:


e_xgb=xgb.XGBRegressor(n_estimators=10000,max_depth=7,learning_rate=0.1,verbosity=3,subsample=0.5,colsample_bytree=1,random_state=13)


# In[28]:


e_xgb.fit(x_train,y_train)


# In[29]:


from sklearn import metrics


# In[30]:


y_pred=e_xgb.predict(x_test)


# In[31]:


print(metrics.r2_score(y_test,y_pred),metrics.mean_absolute_error(y_test,y_pred))


# In[32]:


xgb.plot_importance(e_xgb)


# In[ ]:




