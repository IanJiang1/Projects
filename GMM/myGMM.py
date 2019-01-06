#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:11:07 2018

@author: yujia
"""
import numpy as np
import random
from scipy.stats import multivariate_normal

def myGMM(X, K, maxIter):

    '''
    This is a function that performs GMM clustering
    The input is X: N*d, the input data
                 K: integer, number of clusters
                 maxIter: integer, number of iterations
             
    The output is C: K*d the center of clusters
                  I: N*1 the label of data
                  Loss: [maxIter] likelihood in each
                  step  
    '''
    
    # number of vectors in X
    N, d = X.shape

    # construct indicator matrix (each entry corresponds to the cluster of each point in X)
    I = np.zeros((N, 1))
    
    # construct centers matrix
    C = np.zeros((K, d))
    
    # the list to record error
    Loss = []

    #####################################################################
    # TODO: Implement the EM method for Gaussian mixture model          #
    #####################################################################
    
    #Initialize the centers at randomly selected data points and uniform, independent covariance:
    C = X[np.random.choice(N, K, replace=False)]
    Sigmas = [np.eye(d)]*K
    weights = (1/K) * np.ones(K)

    #ALTERNATIVE: initialize gaussian centroids at the periphery of the data
    #x_centre = X.mean(axis=0)
    #distances = ((X - x_centre)**2).sum(axis = 1)
    #max_rad = np.sqrt(distances.max())
    
    #for iter in range(K):
    #    x1 = x_centre[0] + max_rad*(2*(np.random.rand(1)[0] - 0.5))
    #    x2 = x_centre[1] + np.random.choice([1, -1], 1)*np.sqrt(max_rad**2 - (x1 - x_centre[0])**2)
    #    #plt.scatter(x1, x2, c="red"i)
    #    C[iter] = [x1, x2]

    count = 0
    tol = False
    while not tol:
        #initializing multivariate normal objects:
        rv = []

        for k in range(K):
            rv.append(multivariate_normal(mean=C[k],  cov=Sigmas[k]))

        #----------E-step---------

        #calculating responsiblities
        lk = np.vstack([weights[k]*rv[k].pdf(X) for k in range(K)])
        res = lk/lk.sum(axis=0)

        #calculating log likelihood:
        Loss.append((np.log(lk)*res).sum())

        #obtaining predictions
        I = res.argmax(axis=0)

        #recording old values of parameters in order to calculate differences for tolerances
        C_old = C.copy()
        Sigmas_old = Sigmas

        #----------M-step----------

        #updating weights
        weights = res.mean(axis=1)

        #updating means
        C = np.vstack([(r*X.T).sum(axis = 1)/r.sum() for r in res])

        #updating covariance
        for k in range(K):
            outers = np.dstack([np.outer(x - C[k], x - C[k]) for x in X])
            Sigmas[k] = ((res[0].T * outers)/res[0].sum()).sum(axis = 2)
        
        outers = np.dstack([np.outer(x, x) for x in X])
        
        diff = abs(C - C_old).sum() + np.sum(abs(Sigmas[i] - Sigmas_old[i]).sum() for i in range(K))

        count += 1
        tol = (count > maxIter) | (diff < 0.0001)
      
    #####################################################################
    #                      END OF YOUR CODE                             #
    #####################################################################
    return C, I, Loss
