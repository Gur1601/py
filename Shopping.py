#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:32:10 2023

@author: Gurpreet
"""

#Import Libraries

#Reading data
import pandas as pd
#For mathematical operations
import numpy as np
#Visualisation
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt
#Data preprocessing
from sklearn.preprocessing import LabelEncoder 
#Data spliting
from sklearn.model_selection import train_test_split


#Import dataset
df=pd.read_csv(r'/Users/gurpreet/Desktop/Data Science/supermarket_sales - Sheet1.csv')
#head() for display the first 5 rows  
df.head(5).style.set_properties(**{'background-color': '#873600','color': '#E2EEF3'}) #for colored output
df

df['Date'] = pd.to_datetime(df['Date']) 
df['Time'] = pd.to_datetime(df['Time']) 
df.set_index('Date', inplace=True)
df.value_counts()

df.shape
df.info
df.describe()
df.isnull().sum()
#Exploratory Data Analysis
df.hist(figsize=(20,14))
plt.show()
df = df.drop(columns=['gross margin percentage'])
df.shape
df = df.drop(columns=['Invoice ID'], axis=1)
# checking the categorical variables
data_categorical = (df.dtypes == 'object')
data_categorical_objects = list(data_categorical[data_categorical].index)
print(f'The categorical variables  {data_categorical_objects}')
  
df.corr()
plt.figure(figsize = (12,10))

sns.heatmap(df.corr(), annot =True)

plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Unit price", data=df[170:180])
plt.title("Rating vs Unit Price",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Unit Price")
plt.show()
plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Gender", data=df[170:180])
plt.title("Rating vs Gender",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Gender")
plt.show()

plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Rating", y="Quantity", data=df[170:180])
plt.title("Rating vs Quantity",fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Quantity")
plt.show()

# apply Ordinal Encoder 
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
df[data_categorical_objects] = ordinal_encoder.fit_transform(df[data_categorical_objects])
df

X= df.drop('Gender',axis=1)
y= df['Gender']
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",knn.score(x_train,y_train)*100)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)