#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import  classification_report,confusion_matrix
statlogdata = pd.read_csv(r'C:\Users\Tusha\Downloads\statlog.csv')


# In[71]:


features = statlogdata.columns
names = []
for x in range(36):
 names.append(features[x])

print(features.size)
print(names)


# In[72]:


X = statlogdata.drop('Class',axis=1)
y = statlogdata['Class']
X.shape


# In[69]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# In[74]:


#Support Vector Machine
svm = SVC()


# In[73]:


svm.fit(X_train,y_train)


# In[23]:


y_pred = svm.predict(X_test)
print(classification_report(y_test,y_pred))


# In[76]:


#Kfold Validation
scores=[]
acc = []
# print(X.shape)
# print(Y.shape)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
kf = KFold(n_splits=10,random_state=None,shuffle=False)
for train_index,test_index in kf.split(X):
    print(test_index.shape)
    print(train_index.shape)
    X_train,X_test,y_train,y_test=X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]
    svm.fit(X_train,y_train)
    y_predict = svm.predict(X_test)
    acc.append(accuracy_score(y_test,y_predict))
    scores.append(svm.score(X_test,y_test))

print(np.mean(acc))
cross_val_score(svm,X,y,cv=10)
#decision Tree 
from sklearn import tree


# In[25]:


clf = tree.DecisionTreeClassifier()


# In[26]:


clf = clf.fit(X_train, y_train)


# In[27]:


y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)


# In[31]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[39]:


clf = RandomForestClassifier(max_depth=2, random_state=1)
clf=clf.fit(X_train, y_train)


# In[46]:


y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))


# In[35]:


#KNeighbors
from sklearn.neighbors import KNeighborsClassifier


# In[36]:


neigh = KNeighborsClassifier(n_neighbors=3)


# In[41]:


neigh.fit(X_train, y_train)


# In[45]:


y_pred = neigh.predict(X_test)
print(classification_report(y_test,y_pred))


# In[78]:



#neural Network
from sklearn.neural_network import MLPClassifier
mlp_classifier  = MLPClassifier(random_state=123)


# In[47]:


mlp=mlp_classifier.fit(X_train, y_train)


# In[52]:


y_pred = mlp.predict(X_test)
print(classification_report(y_test,y_pred))


# In[50]:


#Gaussian Naive Bayes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# In[51]:


gnb = GaussianNB()
gnb= gnb.fit(X_train, y_train)


# In[53]:


y_pred = gnb.predict(X_test)
print(classification_report(y_test,y_pred))


# In[ ]:




