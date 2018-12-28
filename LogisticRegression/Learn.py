# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import sys

def g(z):
    """Computes the sigmoid function at the value specified by z; z can be of any dimensions"""    
    return np.divide(1, 1 + np.exp(-z))

class Learn(object):
    """A class that comprises aspects of the general supervised learning problem"""
    """x: An mxn array of n training examples  of an m-dimensional sample space"""
    """y: An array of labels corresponding to each of the n training examples"""
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y).flatten()
        self.numobs = self.x.shape[0]
        self.numvars = self.x.shape[1]
        self.ostype = sys.platform

class LogReg(Learn):
    """A class that comprises aspects of logistic regression problem"""
    """x: An array of training feature values"""
    """y: An array of labels corresponding to each of the training examples....values are binary with either 0 or 1""" 
    """learning_rate: learning rate of the gradient descent algorithm"""
    """Lambda: regularization parameter"""
    def __init__(self, x, y, learning_rate, Lambda):        
        Learn.__init__(self, x, y)
        self.learning_rate = learning_rate
        self.categories = set(list(self.y))
        self.num_cat = len(self.categories)
        self.Lambda = Lambda
    
    def normalize(self, factor):
        """Normalize features by de-meaning and scaling each feature by its standard deviation calculated from the data"""
        std = self.x.std(axis=0)
        std[std==0] = 1
        self.x = factor*(self.x - self.x.mean(axis=0))/std

    def cost(self, h, param):        
        if np.any(h < 0) | np.any(h > 1):
            raise ValueError("elements in x should be greater than 0 and less than 1")
        return (1/self.numobs)*np.sum(-self.y*np.log(h) - (1 - self.y)*np.log(1 - h)) + self.Lambda*(0.5/self.numobs)*np.sum(param[1:]**2)

    def model(self, param):
        param = np.array(param)
        if self.numvars != param.size:
            print("self.numvars: ", self.numvars)
            print("param size: ", param.size)
            raise ValueError("parameter space should be the same dimension as the domain space")
        return g(np.sum(self.x*param, axis=1))

    def LogGrad(self, inittheta):
        """Calculates theta resulting from one iteration of logistic gradient descent, utilizing the ubiquitous logistic cost function"""        	
        theta = inittheta - self.learning_rate*(1/self.numobs)*(np.sum((g(np.sum(inittheta*self.x, axis = 1)) - self.y)[:, np.newaxis]*self.x, axis = 0) + (self.Lambda/self.numobs)*np.concatenate([[0],inittheta[1:]]))
        return theta

    def LogGrad2(self, inittheta, tolerance):
        """Calculates the value of the coefficients in standard binary logistic regression, using gradient descent, after num_iter iterations in order to test convergence of cost function"""
        theta = inittheta
        cost = [self.cost(self.model(inittheta), theta)]
        diff = 1
        while diff > tolerance:
            theta = self.LogGrad(theta)
            cost = np.concatenate([cost, [self.cost(self.model(theta), theta)]])
            diff = abs(cost[len(cost) - 1] - cost[len(cost) - 2])
        
        plt.scatter(np.arange(cost.size), cost, s = 100)
        plt.show()
        return (theta, cost)

    def LogGradK(self, inittheta, tolerance):
        """Calculates the value of coefficients in standard logistic regression, for all distinct values of the output vector y."""
        if self.num_cat == 2:
            raise ValueError("For multi-class classification, number of classes must be > 2. Please use LogGrad2() method")
        theta_k = np.zeros(inittheta.size)
        y_orig = self.y
        cost = []
        for i in self.categories:
            self.y = np.array(self.y == i, dtype = float)
            print(i)
            tht, cst = self.LogGrad2(inittheta, tolerance)
            theta_k = np.vstack([theta_k, tht])
            cost.append(cst)
            self.y = y_orig
        self.theta = theta_k[1:]
        return theta_k[1:], cost
    
    def PredictK(self, x, theta, normfactor):        
        x_orig = self.x
        self.x = x

        if isinstance(normfactor, float):
            self.normalize(normfactor)

        if self.x.ndim == 1:
            self.x=self.x[:, np.newaxis].T
        result = self.model(theta)
        self.x = x_orig
        return result

    def InitTheta(self, mod, pos, norm):
        """Randomly initializes a theta for the algorithm"""
        if norm:
            if pos:
                return np.abs(mod*np.random.randn(self.numvars))
            else:
                return mod*np.random.randn(self.numvars)
        else:
            if pos:
                return mod*np.random.rand(self.numvars)
            else:
                return 2*mod*(np.random.rand(self.numvars) - 1)

    def CrossVal(self, train_prop, tolerance, normfactor):
        """Creates a training set and a test set, based upon the input proportion of training samples"""
        self.train_size = int(train_prop*self.numobs)
        index = np.array(np.zeros(self.numobs), dtype=bool)
        index[np.random.choice(np.arange(self.numobs), self.train_size, replace=False)] = True
        self.tr_x = self.x[index]
        self.tr_y = self.y[index]
        self.tst_x = self.x[~index] 
        self.tst_y = self.y[~index]
        if self.num_cat > 2:
            costs = self.LogGradK(self.InitTheta(mod=1, pos=False, norm=False), tolerance)[1]
            predict = np.array(list(self.categories))[np.array([self.PredictK(self.tst_x, i, normfactor) for i in self.theta]).T.argmax(axis=1)]
            return np.abs(predict - self.tst_y), (predict==self.tst_y).sum()/self.tst_y.size, costs
        else:
            theta, costs = self.LogGrad2(self.InitTheta(mod=1, pos=False, norm=False), tolerance)
            setattr(self, "theta", theta)            #edict = np.array(list(self.categories))[self.PredictK(self.tst_x, i) for i in self.theta]).T.argmax(axis=1)]
            predict = (self.PredictK(self.tst_x, self.theta, normfactor) > 1).astype("float")

            return np.abs(predict - self.tst_y), (predict==self.tst_y).sum()/self.tst_y.size, costs


    def save(self, **kwargs):
        """Saves learned parameters to a csv file for later use"""
        """Use the following keyword arguments:
            fname: filename as a string which contains the parameters
            curr: True or False: save parameters in most recent 'Paramx.csv' file (instead of creating a new 'Param(x + 1).csv' file"""

        if re.match("win*", self.ostype) is not None:
            findstr = "dir Params*.csv /b"
        else:
            findstr =  "ls Params*.csv" 
       
        if "fname" in kwargs.keys():
            filename = kwargs["fname"]
        else: 
            #Determining first which versions are already within working directory:
            param_files = os.popen(findstr).read().split("\n")[:-1]

            #Determining latest version number:
            if param_files == []:
                last_version = 0
            else:
                last_version = max([int("".join(re.findall("[0123456789]*", x))) for x in param_files])
                if "curr" in kwargs.keys():
                    if "curr" == False:
                        last_version += 1
                    elif "curr" == True:
                        pass
                else:
                    last_version += 1

            #Creating filename:
            filename = "Params" + str(last_version) + ".csv"

        #Saving file
        pd.DataFrame(self.theta).to_csv(filename)

        return True 

    def load(self, filename):
        """Loads learned parameters contained in filename given by 'filename'"""

        #Loading and stripping column and row labels
        self.theta = pd.read_csv(filename).values[:, 1:]


   
