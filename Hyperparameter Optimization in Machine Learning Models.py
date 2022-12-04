#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import the dependencies 
import pandas as pd 


# In[3]:


import numpy as np 


# In[5]:


from sklearn.linear_model import LogisticRegression


# In[8]:


#import data 
data = pd.read_csv("C://Users//nourd//Downloads//archive//diabetes.csv") 
data.head()


# In[9]:


data.describe()


# In[11]:


data.head(20)


# In[12]:


data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",header=None)
print((data[[1,2,3,4,5]] == 0).sum())


# In[13]:


#mark zero values as missing or nan 
data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0,np.nan) 
print(data.isnull().sum())


# In[14]:


data.head()


# In[17]:


#fill missing values with mean column values 
data.fillna(data.mean(),inplace = True) 
#count the number of missing values 
print(data.isnull().sum())


# In[19]:


#split the output and inputs 
values = data.values
X = values[:,0:8] 
y = values[:,8] 


# In[24]:


#initiate the LR model with hyperparameters 
lr = LogisticRegression(penalty = 'l2',dual=False,max_iter=110)


# In[25]:


# Pass data to the LR model
lr.fit(X,y)


# In[26]:


lr.score(X,y)


# In[28]:


#applying cross validation 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[30]:


#build the kflod cross validator 
kfold = KFold(n_splits=3,random_state=7,shuffle=True)


# In[31]:


result = cross_val_score(lr, X, y, cv=kfold, scoring='accuracy')
print(result.mean())


# In[32]:


from sklearn.model_selection import GridSearchCV 


# In[35]:


dual = [True,False] 
max_iter = [100,110,120,130,140] 
params_grid = dict(dual=dual,max_iter=max_iter) 


# In[39]:


import time 
lr = LogisticRegression(penalty = 'l2') 
grid = GridSearchCV(estimator = lr,param_grid = params_grid,cv = 3,n_jobs = -1)


# In[41]:


start_time = time.time() 
grid_result = grid.fit(X,y) 
#summarize results 
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[42]:


from sklearn.model_selection import RandomizedSearchCV


# In[44]:


random = RandomizedSearchCV(estimator=lr, param_distributions=params_grid, cv = 3, n_jobs=-1)

start_time = time.time()
random_result = random.fit(X, y)
# Summarize results
print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')


# In[ ]:




