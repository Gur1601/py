#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:32:10 2023

@author: Gurpreet GURPREET
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

df['Date'] = pd.to_datetime(df['Date']) 
df['Time'] = pd.to_datetime(df['Time']) 
df.set_index('Branch', inplace=True)
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
df = df.drop(columns=['Invoice ID','Time','Date'], axis=1)
df

df[['Product line','Quantity']].groupby(['Product line']).mean().sort_values(by='Quantity',ascending=False).style.background_gradient(cmap='Oranges')


fig=px.histogram(df,x='Product line',y='Quantity',
                color_discrete_sequence=['#6E2C00'],
                text_auto=True)


fig.update_layout(title='<b>The best selling product </b>..',
                  title_font={'size':35,'family': 'Serif'},
                  paper_bgcolor='#F6DDCC',
                  plot_bgcolor='#F6DDCC')



fig.update_yaxes(showgrid=False)

fig.show()

#Obviously, the highest percentage of sales is Electronic accessories.

fig = px.pie(df,values='Quantity',names='Gender',
             hover_data=['Quantity','Gender'],
             labels={'Gender':'Gender'},
             color_discrete_sequence=px.colors.sequential.OrRd_r)


fig.update_traces(textposition='inside',
                  textinfo='percent+label')


fig.update_layout(title='<b> Who buys more : Men or Women?<b>',
                  titlefont={'size': 35,'family': 'Serif'},
                  showlegend=True,
                  paper_bgcolor='#F6DDCC',
                  plot_bgcolor='#F6DDCC')
fig.show()
#It is clear that women buy more than men

df[['Product line','Gender']][(df['Gender']=='Male')].value_counts().plot(kind='bar',title='Interests of men')
plt.show()


#It is clear that the most important interests of men is health and beauty.

df[['Product line']][(df['Gender']=='Female')].value_counts().plot(kind='bar',color='pink',title='Interests of women')
plt.show()

#It is clear that the most important interests of men is Fashion accessories.



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

#2. SVC

#In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis.


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)

y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",svc.score(x_train,y_train)*100)


#NB

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)


y_pred=gnb.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",gnb.score(x_train,y_train)*100)

#DTC
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(x_train,y_train)


y_pred=dtree.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(x_train,y_train)*100)


#RFC

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(x_train,y_train)*100)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

#CONCLUSION :

#ACCURACIES OF DIFFERENT MODELS ARE:

#KNeighbors Classifier=  64.625 %

#SVC=  54.875%

#Naiye Bayes=  54.0 %

#Decision Tree Classifier= 65.625 %

#Random Forest Classifier= 100 %


#I got a good accuracy of about 100 % using Random Forest Classifier.