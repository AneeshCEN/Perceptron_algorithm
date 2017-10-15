# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 12:31:13 2017

@author: ANEESH
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.linear_model import Perceptron
np.random.seed(6)

def hypothesis_prediction(W,X):
    prediction = np.dot(W,X)
    if prediction>=0:
        return 1
    else:
        return -1




if __name__ == "__main__":
    
    #number of iteration 
    num_iter = 500
    
    eta = 0.01
    
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    plt.scatter(X[:50,0],X[:50,1], c='r',label='class 1')
    plt.scatter(X[50:100,0],X[50:100,1], c='b',label='class 2')
    plt.title('Linearly separable')
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.legend(loc='upper right')
#    plt.show()
    
    training_data = X[:100,:]
    
    # Appending feature vector of ones
    X = np.c_[np.ones(training_data.shape[0]), training_data]
    y = y[:100,]
    
    # Just changing all the 0 lables into -1
    y[np.where(y==0)] = -1
    
    # initialize the weight vector
    weights = np.zeros(X.shape[1])
    
    def pass_to(weights, X, y):
        for x,y in zip(X,y):
            predicted_label = hypothesis_prediction(weights, x)
            error = eta*(y-predicted_label)
            delta_w =  error *x
            weights = weights+delta_w
        return weights
            
        
    for i in range(num_iter):
        weights = pass_to(weights, X, y)
        weights = weights
        
    clf = Perceptron()
    clf.fit(X,y)
    w0,w1,w2 = clf.coef_[0]
    
    x_min = min(X[:100,1])
    x_max = max(X[:100,1])
    
    y_min = min(X[:100,2])
    y_max = max(X[:100,2])
    
    x_axis = np.linspace(x_min,x_max)
    y_axis = np.linspace(y_min,y_max)
        
    Y = -(weights[0]+weights[1]*x_axis)/weights[2]
    Y2 = -(w0+w1*x_axis)/w2
    plt.plot(x_axis,Y)
    plt.plot(x_axis,Y2, c='r')
    plt.ylim(y_min-1,y_max+1)
    plt.legend()
    plt.show()

    
    