# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 13:13:20 2017

@author: ANEESH
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
import math

from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import Perceptron

(X,y) =  make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.0,random_state=20)
#we need to add 1 to X values (we can say its bias)
X1 = np.c_[np.ones((X.shape[0])),X]

plt.scatter(X1[:,1],X1[:,2],marker='o',c=y)


clf = Perceptron()
clf.fit(X1,y)
w0,w1,w2 = clf.coef_[0]

x_min = min(X[:,0])
x_max = max(X[:,0])

y_min = min(X[:,1])
y_max = max(X[:,1])

x_axis = np.linspace(x_min,x_max)
y_axis = np.linspace(y_min,y_max)

Y = w0+w1*x_axis+w2*y_axis
plt.plot(x_axis,Y)
plt.show()