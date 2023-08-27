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


