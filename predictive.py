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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import r2_score

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
  
##### linear regression ###########
 
lr = LinearRegression().fit(x_train, y_train)
y_train_pred = lr.predict(x_train)
y_test_predict = lr.predict(x_test)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print('lr train score %.3f, lr test score: Z%.3f' % (
lr.score(x_train, y_train),
lr.score(x_test, y_test)))

#test score about 78%



######## Polynomical Regression #########
#linear model to address non linear data

poly = PolynomialFeatures (degree = 2)
x_poly = poly.fit_transform(x_final)

x_train,x_test,y_train,y_test = train_test_split(x_poly,y_final, test_size = 0.33, random_state = 0)

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
x_train = sc.fit_transform(x_train.astype(np.float))
x_test= sc.transform(x_test.astype(np.float))

#fit model
poly_lr = LinearRegression().fit(x_train,y_train)

y_train_pred = poly_lr.predict(x_train)
y_test_pred = poly_lr.predict(x_test)

#print score
print('poly train score %.3f, poly test score: %.3f' % (
poly_lr.score(x_train,y_train),
poly_lr.score(x_test, y_test)))
#about 87% poly test score
#degree is the key affects under or over fitting




######## Support Vector Classification #########
#SVR similar to SVC
#output is a continuous number rather than a category
#Goal is to minimize error and obtain a minimum margin interval
#which contains the maximum number of data points

#commonly used kernel functions include: Linear, RBF (radial basis function), Polynomial, Exponential

svr = SVR(kernel='linear', C = 300)

#test train split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test) #essential to scale with this method as sensitive to outliers
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test= sc.transform(X_test.astype(np.float))

#fit model
svr = svr.fit(X_train,y_train.values.ravel())
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

#print score
print('svr train score %.3f, svr test score: %.3f' % (
svr.score(X_train,y_train),
svr.score(X_test, y_test)))
#test score 63%


############# Decision Tree ######################

dt = DecisionTreeRegressor(random_state=0)

#test train split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test= sc.transform(X_test.astype(np.float))


#fit model
dt = dt.fit(X_train,y_train.values.ravel())
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

#print score
print('dt train score %.3f, dt test score: %.3f' % (
dt.score(X_train,y_train),
dt.score(X_test, y_test)))
#71%
#99 % for train so overfit for trained not generalized for prediction then too specific



############# Random Forest ######################
# --Regression tree = Output: number -- predicted calculated from mean
# --Regression tree = Output: category -- predicted calculated from mode

#does not require scaling

forest = RandomForestRegressor(n_estimators = 100,
                              criterion = 'mse',
                              random_state = 1,
                              n_jobs = -1)
#test train split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test= sc.transform(X_test.astype(np.float))

#fit model
forest.fit(X_train,y_train.values.ravel())
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

#print score
print('forest train score %.3f, forest test score: %.3f' % (
forest.score(X_train,y_train),
forest.score(X_test, y_test)))
#much higher test score of 86%
#with reduced overfitting

################################# Tuning and parameters ##########################################



#Function to print best hyperparamaters: 
def print_best_params(gd_model):
    param_dict = gd_model.best_estimator_.get_params()
    model_str = str(gd_model.estimator).split('(')[0]
    print("\n*** {} Best Parameters ***".format(model_str))
    for k in param_dict:
        print("{}: {}".format(k, param_dict[k]))
    print()

#test train split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size = 0.33, random_state = 0 )

#standard scaler (fit transform on train, fit only on test)
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float))
X_test= sc.transform(X_test.astype(np.float))

###Challenge 1: SVR parameter grid###
param_grid_svr = dict(kernel=[ 'linear', 'poly'],
                     degree=[2],
                     C=[600, 700, 800, 900],
                     epsilon=[0.0001, 0.00001, 0.000001])
svr = GridSearchCV(SVR(), param_grid=param_grid_svr, cv=5, verbose=3)


#fit model
svr = svr.fit(X_train,y_train.values.ravel())

#print score
print('\n\nsvr train score %.3f, svr test score: %.3f' % (
svr.score(X_train,y_train),
svr.score(X_test, y_test)))
#print(svr.best_estimator_.get_params())

print_best_params(svr)


###Challenge 2:Decision Tree parameter grid###
param_grid_dt = dict(min_samples_leaf=np.arange(9, 13, 1, int), #start, stop, step, type
                  max_depth = np.arange(4,7,1, int),
                  min_impurity_decrease = [0, 1, 2],
                 )

dt = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid=param_grid_dt, cv=5,  verbose=3)



#fit model
dt = dt.fit(X_train,y_train.values.ravel())


#print score
print('\n\ndt train score %.3f, dt test score: %.3f' % (
dt.score(X_train,y_train),
dt.score(X_test, y_test)))
print_best_params(dt)



###Challenge 3:Random Forest parameter grid###
param_grid_rf = dict(n_estimators=[20],
                     max_depth=np.arange(1, 13, 2),
                     min_samples_split=[2],
                     min_samples_leaf= np.arange(1, 15, 2, int),
                     bootstrap=[True, False],
                     oob_score=[False, ])


forest = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param_grid_rf, cv=5, verbose=3)

#fit model
forest.fit(X_train,y_train.values.ravel())


#print score
print('\n\nforest train score %.3f, forest test score: %.3f' % (
forest.score(X_train,y_train),
forest.score(X_test, y_test)))

print_best_params(forest)


