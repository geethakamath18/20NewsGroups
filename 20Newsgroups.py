#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import string
import numpy as np
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


X=[]
Y=[]
docs_data=[]
for category in os.listdir('E:/Grad School/Semester 2/ML/Homeworks/20_newsgroups/'):
    for document in os.listdir('E:/Grad School/Semester 2/ML/Homeworks/20_newsgroups/'+category):
        with open('E:/Grad School/Semester 2/ML/Homeworks/20_newsgroups/'+category+'/'+document, "r") as f:
            dat = f.read()
            X.append((document, dat))
            docs_data.append(dat)
            Y.append(category)


# Splitting the data set into Test data represented by Train_Data, Test Data represented by  Test_Data, labels represented by Train_Labels and Test_Labels respectively

# In[3]:


Train_Data, Test_Data, Train_Labels, Test_Labels = model_selection.train_test_split(X, Y, test_size=0.15, random_state=0)


# In[4]:


vocabulary={}
for i in range(len(Train_Data)):
    word=[]
    for word in Train_Data[i][1].split():
        word_new=word.strip(string.punctuation).lower()
        if (len(word_new)>2):  
            if word_new in vocabulary:
                vocabulary[word_new]+=1
            else:
                vocabulary[word_new]=1


# In[5]:


feature_vector=[]
for key in vocabulary:
    if vocabulary[key] >=20:
        feature_vector.append(key)


# In[8]:


Training_Dataset=np.zeros((len(Train_Data),len(feature_vector)))
for i in range(len(Train_Data)):
    words=[ word.strip(string.punctuation).lower() for word in Train_Data[i][1].split()]
    for word in words:
        if word in feature_vector:
            Training_Dataset[i][feature_vector.index(word)] += 1
           


# In[10]:


Test_Dataset=np.zeros((len(Test_Data),len(feature_vector)))
for i in range(len(Test_Data)):
    words=[ word.strip(string.punctuation).lower() for word in Test_Data[i][1].split()]
    for word in words:
        if word in feature_vector:
            Test_Dataset[i][feature_vector.index(word)] += 1


# Question 5a. : Using Multinomial Naive Bayes

# In[11]:


m=MultinomialNB()
m.fit(Training_Dataset,Train_Labels)
MultinomialNB(alpha=1.0,fit_prior=True)
prediction1=m.predict(Test_Dataset)

print(classification_report(Test_Labels,prediction1))


# Question 5b. : Using Bernoulli Naive Bayes

# In[13]:


b=BernoulliNB()
b.fit(Training_Dataset,Train_Labels)
BernoulliNB()
prediction2=b.predict(Test_Dataset)

print(classification_report(Test_Labels, prediction2))


# In[ ]:




