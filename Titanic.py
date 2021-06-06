#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[3]:


train_df = pd.read_csv('Titanic Data/train.csv')


# In[4]:


test_df = pd.read_csv('Titanic Data/test.csv')


# In[5]:


train_df.drop("Name", axis =1, inplace=True)
train_df.drop("Cabin", axis =1, inplace=True)
train_df.drop("Fare", axis =1, inplace=True)
train_df.drop("PassengerId", axis =1, inplace=True)
train_df.drop("Ticket", axis =1, inplace=True)


test_df.drop("Ticket", axis =1, inplace=True)
test_df.drop("PassengerId", axis =1, inplace=True)
test_df.drop("Name", axis =1, inplace=True)
test_df.drop("Cabin", axis =1, inplace=True)
test_df.drop("Fare", axis =1, inplace=True)


# In[6]:


train_df.head()


# In[7]:


test_df.head()


# In[8]:


source = train_df.drop('Survived', axis=1)
target = train_df['Survived']

source.shape, target.shape


# In[9]:


source.head()


# In[10]:


target.head()


# In[11]:


df_cat = source.drop(['Age', 'Parch', 'SibSp', 'Pclass'], axis =1)

cat_attribs = list(df_cat)
cat_attribs


# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer



cat_pipeline = Pipeline([
    ('imputer_cat', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

full_pipeline = ColumnTransformer([
    ("cat", cat_pipeline, cat_attribs),
])

source_final = full_pipeline.fit_transform(source)


# In[13]:


source_final.shape


# In[14]:


target1 = target.to_numpy()


# In[15]:


target.shape


# In[16]:


target = target1.reshape(1,-1)


# In[17]:


target = target.transpose()
target.shape


# In[18]:


#test_final = full_pipeline.transform(test_df)


# In[19]:


from sklearn.model_selection import train_test_split
source_train, source_test, target_train, target_test = train_test_split(source_final, target, test_size = 0.25, random_state=0)


# In[20]:


target_test.shape


# Linear Models

# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score


# In[25]:


LR = LogisticRegression()


# In[26]:


LR.fit(source_train, target_train)


# In[29]:


LR.score(source_test, LRpred)


# In[28]:


LRpred = LR.predict(source_test)


# In[30]:


LRpred


# In[ ]:




