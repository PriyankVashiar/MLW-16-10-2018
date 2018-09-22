#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:45:29 2018

@author: priyank
"""

#Data Preprocessing

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

#import dataset
d1 = pd.read_csv('/home/priyank/projects/ML Workshop/DATASET/Data.csv')

X = d1.iloc[:,0:3].values
Y = d1.iloc[:,-1].values

#Fill missing data
imputer = Imputer()
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#eNcode categorial data
le_x = LabelEncoder()
X[:,0] = le_x.fit_transform(X[:,0])

one = OneHotEncoder(categorical_features = [0])
X = one.fit_transform(X).toarray()

le_y = LabelEncoder()
Y = le_y.fit_transform(Y)

#Splitting data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc_x = StandardScaler().fit(X_train)
X_train = sc_x.transform(X_train)
X_test = sc_x.transform(X_test)