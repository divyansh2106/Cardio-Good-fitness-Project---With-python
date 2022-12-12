#!/usr/bin/env python
# coding: utf-8

# # Cardio Good Fitness Project

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df=pd.read_csv('CardioGoodFitness.csv')


# In[9]:


df.info()


# In[4]:


df.head()


# # Data Preprocessing

# In[5]:


# Changing the datatype from object to category
df.Product=df["Product"].astype("category")
df.Gender=df["Gender"].astype("category")
df.MaritalStatus=df["MaritalStatus"].astype("category")


# In[6]:


df.info()


# In[7]:


# Checking null values
df.isnull().sum()


# In[36]:


# Checking duplicated values
print(df.duplicated())


# # Analysis

# In[9]:


df.describe(include='all')


# In[10]:


plt.figure(figsize=(14,7))
df['Product'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(8,8))
plt.title("Pie chart of Product Sales")
plt.show()


# In[11]:


sns.countplot(x="Product", hue="Gender", data=df)
plt.show()


# In[13]:


# Calculating average number of miles for each product
sns.set(font_scale = 2)
sns.displot(data=df, x="Miles", hue="Product",kind='kde', height=8, aspect=2.5)


# In[14]:


def dist_box_violin(data):
    Name=data.name.upper()
    fig=plt.subplots(1, figsize=(17, 7))
    sns.boxplot(x=data,showmeans=True, orient='h',color="purple")


# In[15]:


dist_box_violin(df.Income)


# In[13]:


sns.relplot(x="Age", y="Income", hue="Product", size="Usage",
            sizes=(40, 400), alpha=.5, palette="plasma",
            height=6, data=df).set(title='INCOME BY AGE ,PRODUCT AND USAGE');


# # Correlation Analysis

# In[17]:


corr = df.corr(method='pearson')
corr


# In[18]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f', center = 1


# # Linear Regression, Logestic Regression and Decision Tree Classifier

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[20]:


X=df[['Usage','Fitness']]
y=df[['Miles']]


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


# In[22]:


# Linear Regression
lr = LinearRegression()
lr.fit(X_test,y_test)
lr.score(X_test,y_test)


# In[23]:


# Logistic Regression
model=LogisticRegression()

model.fit(X_train,y_train)


# In[24]:


y_pred=model.predict(X_test)
print(y_pred)


# In[25]:


from sklearn.metrics import accuracy_score
k=accuracy_score(y_test,y_pred)
print('The accuracy is ',k)


# In[26]:


# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[27]:


dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print("The accuracy of Decision Tree is:", metrics.accuracy_score(predictions, y_test))

