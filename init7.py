#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 09:29:12 2018

@author: priyank
"""

from sklearn.datasets import load_digits

digits = load_digits()

data = digits.data

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))

for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image,(8, 8)), cmap = plt.cm.gray)
    plt.title('Training %i\n' % label, fontsize = 10)
    
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.3, random_state = 10)

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic = logistic.fit(x_train, y_train)

predictions = logistic.predict(x_test)

logistic.score(x_test, y_test)

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)