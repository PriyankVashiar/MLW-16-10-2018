#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 09:00:18 2018

@author: priyank
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#import dataset
d1 = pd.read_csv('/home/priyank/projects/ML Workshop/DATASET/Salary_Data.csv')
#import the necessary columns
X = d1.iloc[:,0:-1].values
Y = d1.iloc[:,1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
#making the classifier
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#predicting with respect to test data
Y_pred = regressor.predict(X_test)
#graph for prediction line with train data
plt.scatter(X_train, Y_train, color = 'red');
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#graph for prediction line with test data
plt.scatter(X_test, Y_test, color = 'red');
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()