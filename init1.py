#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 11:40:03 2018

@author: priyank
"""

from sklearn import tree

features = [[100,0], [80, 0], [120,0], [125,1], [140,1], [160,1]]
labels = [0,0,0,1,1,1]

classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(features, labels)
print(classifier.predict([[155, 0]]))

import pandas as pd
data1 = pd.read_csv('/home/priyank/projects/ML Workshop/DATASET/tsv_data.txt', sep = '\t')
data2 = pd.read_table('/home/priyank/projects/ML Workshop/DATASET/tsv_data.txt')

#Reading from database
import pyodbc

sql_conn = pyodbc.connect('DRIVER ={SQL Server=};'
                          'SERVER =servername;'
                          'DATABASE = test;'
                          'Trusted_Connection = yes')
query = "SELECT * FROM tablename"
