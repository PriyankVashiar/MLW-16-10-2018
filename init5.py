#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:26:57 2018

@author: priyank
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataset = pd.read_csv('/home/priyank/projects/ML Workshop/DATASET/50_Startups (5).xls')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

le_x = LabelEncoder()
X[:,3] = le_x.fit_transform(X[:,3])

one = OneHotEncoder(categorical_features = [3])
X = one.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

X = np.append(np.ones((50, 1), dtype = np.float64), X, axis = 1)
X_opt = X
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_Added = np.append(X, np.random.random((50,1)), axis = 1)
X_opt = X
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

#removed 2nd dummy variable
X_opt = X_opt[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:, [0,2,3,4]]
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:, [0,1,3]]
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X_opt[:, [0,1]]
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()