#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[5]:


from sklearn.datasets import make_blobs


# In[7]:


X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)


# In[9]:


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')


# In[10]:


from sklearn.naive_bayes import GaussianNB


# In[11]:


model = GaussianNB()


# In[12]:


model.fit(X, y) 


# In[13]:


rng = np.random.RandomState(0)


# In[16]:


Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)


# In[19]:


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)


# In[21]:


yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)


# Multinomial Naive Bayes 

# In[24]:


from sklearn.datasets import fetch_20newsgroups 
data = fetch_20newsgroups()
data.target_names 


# In[25]:


categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space',
 'comp.graphics'] 
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)


# In[26]:


print(train.data[5])


# In order to use this data for machine learning, we need to be able to convert the content of each string into a vector of numbers. For this we will use the TF–IDF vectorizer, and create a pipeline that
# attaches it to a multinomial naive Bayes classifier

# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# In[28]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[29]:


model.fit(train.data,train.target) 
labels = model.predict(test.data)


# In[31]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
mat


# In[37]:


sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar =False,xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


# Evidently, even this very simple classifier can successfully separate space talk from computer talk, but it gets confused between talk about religion and talk about Christianity. This is perhaps an expected area of confusion!

# In[42]:


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# In[43]:


predict_category('sending a payload to the ISS')


# In[44]:


predict_category('discussing islam vs atheism')


# In[45]:


predict_category('determining the screen resolution')


# In[ ]:




