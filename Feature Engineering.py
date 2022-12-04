#!/usr/bin/env python
# coding: utf-8

# In[7]:


data = [
 {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
 {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
 {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
 {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
 ]


# In[8]:


from sklearn.feature_extraction import DictVectorizer


# In[9]:


vec = DictVectorizer(sparse=False, dtype=int) 
vec.fit_transform(data)


# In[10]:


vec.get_feature_names()


# In[11]:


vec = DictVectorizer(sparse=True, dtype=int)
vec.fit_transform(data)


# In[12]:


sample = ['problem of evil',
 'evil queen',
 'horizon problem']


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer 
vec = CountVectorizer() 
X = vec.fit_transform(sample)
X


# In[14]:


import pandas as pd
pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer 
vec = TfidfVectorizer() 
X = vec.fit_transform(sample)
pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) 


# In[16]:


from sklearn.preprocessing import PolynomialFeatures


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y);


# In[18]:


from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit) 


# In[19]:


poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)


# In[20]:


model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)


# Handling Missing Data 

# In[21]:


import numpy as np


# In[22]:


from numpy import nan


# In[23]:


X = np.array([[ nan, 0, 3 ],
 [ 3, 7, 9 ],
 [ 3, 5, 2 ],
 [ 4, nan, 6 ],
 [ 8, 8, 1 ]])
y = np.array([14, 16, -1, 8, -5])


# In[25]:


from sklearn.impute import SimpleImputer


# In[30]:


imp = SimpleImputer(strategy='mean')


# In[31]:


X2 = imp.fit_transform(X)
X2


# In[32]:


from sklearn.pipeline import make_pipeline


# In[33]:


model = make_pipeline(SimpleImputer(strategy='mean'),PolynomialFeatures(degree=2),LinearRegression())


# In[ ]:




