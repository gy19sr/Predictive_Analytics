# -*- coding: utf-8 -*-
"""
Spyder Editor

author Stuart Ross
"""
import os
import keras
import pandas as pd
import numpy as np
import seaborn as sns
#Seaborn library for data visulaisation (look in folders)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler




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

####label encoding - two distinct values (so binomial) Yes and No

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
#one hot encoder
ohe = OneHotEncoder()
#create ndarray for one hot encoding (sklearn)
region = ohe.fit_transform(region).toarray()

region = pd.DataFrame(region)

region.columns = ['northeast', 'northwest', 'southeast', 'southwest' ]
print("sklearn one hot encoder results for region")
print(region[:10])

############### divide test and train ####################

#take numerical data from the original data
x_num = data[['age', 'bmi', 'children']]
#take encoded and add numerical
x_final = pd.concat([x_num, sex, smoker, region], axis = 1)

#define y as being the "charges column" from the original dataset
y_final = data[['charges']].copy()
#test and train split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.33, random_state = 0)



#####feature scaling#########
##these are data normalization and data standardization

##normalization (min max scalling)---   (x - min(x))/[max(x)-min(x)]   ---- better with no outliers
#will range from 0 to 1 
#n_scaler = MinMaxScaler()
#x_train = n_scaler.fit_transform(x_train.astype(np.float))
#x_test = n_scaler.transform(x_test.astype(np.float))


#standard scaler (fit transform on train, fit only on test)
#z = (x - mean)/ SD
s_scaler = StandardScaler()
x_train = s_scaler.fit_transform(x_train.astype(np.float))
x_test= s_scaler.transform(x_test.astype(np.float)) #don't need to fit test


############## End of data Preperation ###############


############## Model #############################
"""
##### Types of Machine Learning ##########
Supervised (with labeled Data)
 --- regression (predicts a numerical variable)
 --- Classification (Predicts a categorical variable)
 
 Unsupervised (Without Labeled) 
 --- Clustering (Discover the inherent grouping in the data)
 --- Association (discover rules that describe portions of the data)
 
 Reinforcement Learning
(personal fav)

 
 #####Commonly used regression models#####
 -linear regression
 -polynomial regression
 -support vector regression (SVM)
 -Dcesion tree
 -Random forest regression
 
 """
  
