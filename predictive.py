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


#import data
data = pd.read_csv("C:/Users/stuar/OneDrive/Documents/onlinecourses/Linkedin/predictive_analytics_through_python/Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Ex_Files_Python_Predictive_Analytics/Exercise Files/Datasets/insurance.csv")

print(data.head(15))


#print if a value is NA (or in pythonNAN) null
data.isnull()
print(data.isnull())

data.isnull().sum().sum()
print(data.isnull().sum())
print("total nulls =",data.isnull().sum().sum()) #all rows and columns combine


#fill missing values with .... (mean or median or mode)

#check if still NA's

#remove NA rows

#remove NA columns