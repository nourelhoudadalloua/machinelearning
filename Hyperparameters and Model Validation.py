#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_iris 


# In[5]:


iris = load_iris()
iris


# In[6]:


X = iris.data
y = iris.target


# In[7]:


y


# In[8]:


from sklearn.neighbors import KNeighborsClassifier 
model = KNeighborsClassifier(n_neighbors = 1)


# In[9]:


#train the model and predict 
model.fit(X, y)
y_model = model.predict(X)


# In[10]:


from sklearn.metrics import accuracy_score 
accuracy_score(y,y_model)


# In[11]:


#model validation 
from sklearn.model_selection import train_test_split 
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,train_size=0.5)


# In[12]:


#train the data 
model.fit(X1,y1) 
y2_model = model.predict(X2)


# In[13]:


accuracy_score(y2,y2_model)


# In[14]:


from sklearn.model_selection import cross_val_score 
cross_val_score(model, X, y, cv=5) 


# In[15]:


from sklearn.model_selection import LeaveOneOut


# In[16]:


scores = cross_val_score(model, X, y, cv=LeaveOneOut())
scores


# In[17]:


scores.mean()


# In[18]:


from sklearn.preprocessing import PolynomialFeatures


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


from sklearn.pipeline import make_pipeline


# In[21]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),LinearRegression(**kwargs))


# In[22]:


import numpy as np 


# In[23]:


def make_data(N, err=1.0, rseed=1):
 # randomly sample the data
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y 


# In[24]:


X,y = make_data(40)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[26]:


X_test = np.linspace(-0.1, 1.1, 500)[:, None]


# In[27]:


plt.scatter(X.ravel(), y, color='black') 


# In[28]:


for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
    
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 12)
plt.legend(loc='best');


# In[29]:


from sklearn.model_selection import validation_curve


# In[30]:


degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y, cv=7,param_name = 'polynomialfeatures__degree',param_range = degree)


# In[31]:


train_score
val_score


# In[32]:


plt.plot(degree, np.median(train_score, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')


# In[33]:


print(X) 


# In[36]:


print(X.ravel())


# In[37]:


plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test) 
plt.axis(lim)


# In[38]:


# demonstrate that the model complexity depends on the amount of training data 
X2,y2 = make_data(200)
plt.scatter(X2.ravel(),y2)


# In[39]:


#plot the validation curve of the larger dataset and overplot the previous one 

train_score2, val_score2 = validation_curve(PolynomialRegression(), X2, y2, cv=7,param_name = 'polynomialfeatures__degree',param_range = degree)

plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')

plt.plot(degree, np.median(train_score, 1), color='blue',linestyle = 'dashed',alpha = 0.3)
plt.plot(degree, np.median(val_score, 1), color='red',linestyle = 'dashed',alpha = 0.3)
plt.legend(loc='lower center')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')


# In[40]:


#learning curve in sckit-learn 
from sklearn.model_selection import learning_curve


# In[49]:


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1) 
for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),X, y, cv=7,train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray',
    linestyle='dashed') 
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')


# In[50]:


for i, degree in enumerate([2, 9]): 
    print(i)
    print(degree)


# In[51]:


train_lc[-1]


# In[53]:


from sklearn.model_selection import GridSearchCV 


# In[54]:


param_grid = {'polynomialfeatures__degree': np.arange(21),
 'linearregression__fit_intercept': [True, False],
 'linearregression__normalize': [True, False]} 


# In[55]:


grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)


# In[56]:


grid.fit(X,y)


# In[57]:


grid.best_params_


# In[ ]:




