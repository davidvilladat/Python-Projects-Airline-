#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[2]:


print(iris_dataset.keys())


# In[3]:


print(iris_dataset['DESCR'])


# In[5]:


print("Labels: {}".format(iris_dataset['target_names']))


# In[6]:


print("Features: {}".format(iris_dataset['feature_names']))


# In[10]:


iris_dataset['data'][:6]


# In[11]:


iris_dataset['target']


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state = 0)

#by setting the random_select , we are making a deterministic approach, easy to reproduce

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

#the X output always will be (data collected, features)

#while the y output will be (data collected, labels): labels = 1 column 


# In[14]:


print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[17]:


import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_dataframe.head()


# In[23]:


#pair plot
#!pip install mglearn
import mglearn

grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8,cmap=mglearn.cm3)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=1)


# In[25]:


knn.fit(X_train,y_train)


# In[27]:


import numpy as np

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))


# In[31]:


X_train.shape


# In[32]:


#Prediction 

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))


# In[33]:


#Evaluating the model 

y_pred = knn.predict(X_test)
print(y_pred)


# In[34]:


len(y_pred) #=number for test 


# In[35]:


print("Test set score: {:.2f}".format(np.mean(y_pred==y_test)))


# In[36]:


knn.score(X_test,y_test)


# In[37]:


### Supervised machine learning


# In[39]:


###Classification 

from matplotlib import pyplot as plt
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))


# In[40]:


###Regression algorithms 

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")


# In[41]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))


# In[43]:


cancer.data.shape


# In[44]:


print("Sample counts per class:\n{}".format(
 {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))


# In[46]:


"""
n: v: This defines the structure of the dictionary being created. Each entry will have a key (n) and a value (v).
for n, v in zip(...): This iterates over two separate iterables simultaneously using the zip function.
cancer.target_names: This assumes it's a list containing the class names for each sample in the cancer dataset.
np.bincount(cancer.target): This calculates the number of times each class label appears in the cancer.target attribute (likely another list/array containing class labels).
zip(...): This pairs corresponding elements from the two iterables, creating tuples like (class_name, count).
n = class_name, v = count: Inside the loop, each tuple is unpacked, assigning the class name to n and the count to v.
Dictionary creation: This creates a dictionary entry for each (n, v) pair. The class name (n) becomes the key, and the count (v) becomes the value.

"""


# In[47]:


print("Feature names:\n{}".format(cancer.feature_names))


# In[48]:


from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))


# In[49]:


X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))


# In[50]:


mglearn.plots.plot_knn_classification(n_neighbors=1)


# In[51]:


from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[52]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


# In[53]:


clf.fit(X_train, y_train)


# In[54]:


print("Test set predictions: {}".format(clf.predict(X_test)))


# In[55]:


print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# In[56]:


fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
 # the fit method returns the object self, so we can instantiate
 # and fit in one line
 clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
 mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
 mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
 ax.set_title("{} neighbor(s)".format(n_neighbors))
 ax.set_xlabel("feature 0")
 ax.set_ylabel("feature 1")
axes[0].legend(loc=3)


# In[57]:


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer() #load dataset 


#classic form to split train and test data in ML
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=66)

#list created to fill with the values varying the k-neighbors 

training_accuracy = []
test_accuracy = []


# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
 # build the model
 clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    
 clf.fit(X_train, y_train)

 # record training set accuracy
 training_accuracy.append(clf.score(X_train, y_train))

 # record generalization accuracy
 test_accuracy.append(clf.score(X_test, y_test))


#plot results 

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()


# In[58]:


#optimal neighbors = 6 


# In[60]:


mglearn.plots.plot_knn_regression(n_neighbors=8)


# In[61]:


from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)

# fit the model using the training data and training targets
reg.fit(X_train, y_train)


# In[62]:


print("Test set predictions:\n{}".format(reg.predict(X_test)))


# In[63]:


print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


# In[64]:


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 9], axes):
    
 # make predictions using 1, 3, or 9 neighbors
 reg = KNeighborsRegressor(n_neighbors=n_neighbors)
 reg.fit(X_train, y_train)
 ax.plot(line, reg.predict(line))
 ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
 ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
 ax.set_title(
 "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
 n_neighbors, reg.score(X_train, y_train),
 reg.score(X_test, y_test)))


 ax.set_xlabel("Feature")
 ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
 "Test data/target"], loc="best")


# In[ ]:




