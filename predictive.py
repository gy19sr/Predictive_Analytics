# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#import data
data = pd.read_csv("C:/Users/stuar/OneDrive/Documents/onlinecourses/Linkedin/predictive_analytics_through_python/Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Exercise Files/Datasets/insurance.csv")

print(data.head(15))

######## dealing with NA's #################

#print if a value is NA (or in pythonNAN) null
#data.isnull()
#print(data.isnull())

data.isnull().sum().sum()
print(data.isnull().sum())
print("total nulls =",data.isnull().sum().sum()) #all rows and columns combine

#fill missing values with .... (mean or median or most_frequent)

data = pd.read_csv("C:/Users/stuar/OneDrive/Documents/onlinecourses/Linkedin/predictive_analytics_through_python/Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Exercise Files/Datasets/insurance.csv")
imputer = SimpleImputer(strategy='mean')#develope a way to calc mean
imputer.fit(data['bmi'].values.reshape(-1, 1)) #prepare bmi column
data['bmi'] = imputer.transform(data['bmi'].values.reshape(-1, 1)) #mean of bmi and replace
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])
print(data.head(15))
print(data.isnull().sum())


#if want to remove NAs (rows)
"""
data = pd.read_csv("C:/Users/stuar/OneDrive/Documents/onlinecourses/Linkedin/predictive_analytics_through_python/Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Exercise Files/Datasets/insurance.csv")
 # reloading fresh dataset for option 0
#option1 for dropping NAN
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
#check how many values are missing (NaN) - after we filled in the NaN
count_nan = data.isnull().sum() # the number of missing values for every column
print(count_nan[count_nan > 0])
print(data.head(15))
print(data.isnull().sum())
"""

#if want to remove NA columns
"""
data = pd.read_csv("C:/Users/stuar/OneDrive/Documents/onlinecourses/Linkedin/predictive_analytics_through_python/Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Exercise Files/Datasets/insurance.csv")
 # reloading fresh dataset for option 0
data.drop('bmi', axis = 1, inplace = True)
print(data.isnull().sum())
"""

################## Converting Categorical variables to numbers ############################

####label encoding - two distinct values (so binomial)

#id the variables
sex = data.iloc[:,1:2].values
smoker = data.iloc[:,4:5].values

#le (label encoder) for smoker
le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("sklearn label encoder results for smoker ")
print(le_smoker_mapping)
print(smoker[:10])

## le (label encoder) for sex
le = LabelEncoder()
sex[:,0] = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex)
sex.columns = ['sex']
le_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for sex:") 
print(le_sex_mapping)
print(sex[:10])


####One hot encoding - three or more distinct values 

#new columns for each option 1 is present 0 is not

region = data.iloc[:,5:6].values





