#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:08:16 2018

@author: priyank
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

d1 = pd.read_excel('/home/priyank/projects/ML Workshop/New Dataset/salary.xlsx')
X = d1.iloc[:,[0,1]].values

plt.plot(X[:,0],X[:,1])
plt.title('Graph of Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

#Bar Chart
plt.bar(X[:,0],X[:,1],align = 'center', alpha = 1.0)
plt.title('Graph of Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

#Comparison Bar Chart
salary_1 = X[:,1]
salary_2 = X[:,1] * 2
plot_1 = plt.bar(X[:,0], salary_1, align = 'center', alpha = 1, color = 'g', label = 'Salary of group 1')
plot_2 = plt.bar(X[:,0] + 0.8, salary_2, align = 'center', alpha = 0.3, color = 'b', label = 'Salary of group 2')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()