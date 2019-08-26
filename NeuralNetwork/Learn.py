# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import sys

def bound(arr, lower, upper):
    arr[arr < lower] = lower
    arr[arr > upper] = upper
    return arr

def lower_bound(arr, lower):
    arr[arr < lower] = lower
    return arr

def upper_bound(arr, upper):
    arr[arr > upper] = upper
    return arr

def g(z):
    """Computes the sigmoid function at the value specified by z; z can be of any dimensions"""    
    return np.divide(1, 1 + np.exp(-z))

def relu(z, a):
    return a * z * (z > 0)

class Activation(object):
     def __init__(self, fnum, *argv):
         if fnum == 1:
             # Logistic function: f(x) = 1 / ( 1 + exp(-x) )
             self.name = "logistic"
             self.f = g
             self.deriv = lambda x: x * (1 - x)
         if fnum == 2:
             # Relu function: f(x) = max(0, x)
             self.name = "relu"
             self.f = lambda x: relu(x, argv[0])
             self.deriv = lambda x: np.ones(x.shape) - (x < 0)
         if fnum == 3:
             # Hyperbolic Tangent: f(x) = ( 1 - exp(-x) ) / ( 1 + exp(-x) )
             self.name = "tanh"
             self.f = lambda x: argv[0]*np.tanh(argv[1] * x)
             self.deriv = lambda x: argv[0] * (np.cosh(argv[1] * x)**2 - np.sinh(argv[1] * x)**2)/(np.cosh(argv[1] * x)**2)
         if fnum == 4:
             # Softmax: f(x) = exp(x_i) / ( exp(x_1) + exp(x_2) + ... + exp(x_n) )
             self.name = "softmax"
             self.f = lambda x: np.exp(x)/(np.sum(np.exp(x)))
             self.deriv = lambda x: x*(1 - x)

class Cost(object):
    def __init__(self, fnum):
        if fnum == 1:
            # Mean Squared Error
            self.f = lambda actual, predicted: (actual - predicted)**2
            self.deriv = lambda actual, predicted: (actual - predicted)

        if fnum == 2:
            # Cross entropy
            self.f = lambda actual, predicted: -actual * np.log(lower_bound(predicted, 1e-12))
            self.deriv = lambda actual, predicted: predicted - actual

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

    def normalize(self, factor):
        """Normalize features by de-meaning and scaling each feature by its standard deviation calculated from the data"""
        std = self.x.std(axis=0)
        std[std==0] = 1
        self.x = factor*(self.x - self.x.mean(axis=0))/std

    def add_one(self):
        self.x = np.hstack([np.ones(self.numobs)[:, np.newaxis], self.x])

    def shift_register_down(self, x, y):
        self.x_orig = self.x
        self.y_orig = self.y
        setattr(self, "x", x)
        setattr(self, "y", y)
        self.numobs = self.x.shape[0]
        self.numvars = self.x.shape[1]

    def shift_register_up(self):
        setattr(self, "x", self.x_orig)
        setattr(self, "y", self.y_orig)
        self.numobs = self.x.shape[0]
        self.numvars = self.x.shape[1]

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

    def cost(self, h, param):        
        if np.any(h < 0) | np.any(h > 1):
            raise ValueError("elements in x should be greater than 0 and less than 1")
        return (1/self.numobs)*np.sum(-self.y*np.log(h) - (1 - self.y)*np.log(1 - h))
               #+ self.Lambda*(0.5/self.numobs)*np.sum(param[1:]**2)

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

class NN(Learn):
    """A class for neural nets"""
    """x: An array of training feature values"""
    """y: An array of labels corresponding to each of the training examples....values are binary with either 0 or 1""" 
    """learning_rate: learning rate of the gradient descent algorithm"""
    """Lambda: regularization parameter"""
    def __init__(self, x, y, cost, learning_rate, Lambda, activation_func):
        Learn.__init__(self, x, y)
        self.cost = cost
        self.learning_rate = learning_rate
        self.categories = list(set(list(self.y)))
        self.num_cat = len(self.categories)
        self.Lambda = Lambda
        self.activation_func = activation_func
        self.num_layers = len(activation_func)
        self.y_index = np.array([self.categories.index(el) for el in self.y])
        self.y_vec = np.zeros([len(self.y_index), self.num_cat])

        for i in range(len(self.y_index)):
            self.y_vec[i, self.y_index[i]] = 1

    def vectorize(self):
        self.orig_y = self.y
        self.y = self.y_vec

    def one_layer1(self, theta, fun, bias=False):
        if bias:
            return lambda x: np.concatenate([[1], fun(theta.dot(x)).flatten()])
        else:
            return lambda x: fun(theta.dot(x)).flatten()

    def forward1(self, x, thetas, bias=False):

        #Initializing
        layer = x
        layers = []

        if len(x.shape) < 2:
            layer = layer[np.newaxis, :]

        #Calculating layers one-by-one
        count = 0
        for theta in thetas:
            layer = np.apply_along_axis(self.one_layer1(theta, self.activation_func[count].f, bias=bias), 1, layer)
            layers.append(layer)
            count += 1

        #Checking if bias unit is included
        if bias:
            return layers, layer[:, 1]
        else:
            return layers, layer

    def backprop_onesamp1(self, sample, thetas, bias=False):
        """sample consisting of an ordered pair (x, y) where x are the features and y is the label"""

        #Initializing algorithm and getting number of layers
        a, y_pred = self.forward1(sample[0], thetas, bias=bias)
        delta = self.cost.deriv(sample[1], y_pred).flatten()

        #print("delta: " + str(delta))
        layernum = len(thetas)

        #Checking if bias units are used in theta definition
        if bias:
            thetas = [theta[1:, :] for theta in thetas]
            a = [x[1:] for x in a]

        #Initializing gradient increment for sample
        Delta = [np.zeros(theta.shape) for theta in thetas]

        #Backpropagation
        for l in range(layernum - 1, 1, -1):
            Delta[l] += np.outer(a[l - 1], delta).T
            delta = (thetas[l].T.dot(delta)) * self.activation_func[l].deriv(a[l - 1]).flatten()

        return [Delta[l] for l in range(layernum)]

    def backprop1(self, thetas, bias=False):
        """Running backpropagation algorithm for all samples"""
        
        #Initializing gradient increment for sample
        Delta = [np.zeros(theta.shape) for theta in thetas]
        numlayers = len(thetas)
        for i in range(self.numobs):
            D = self.backprop_onesamp1((self.x[i], self.y[i]), thetas, bias=bias)
            Delta = [Delta[l] + D[l] for l in range(numlayers)]

        return [(1 / self.numobs) * (Delta[l] + self.Lambda[l] * thetas[l]) for l in range(numlayers)]

    # def cost(self, h, thetas):
    #     if np.any(h < 0) | np.any(h > 1):
    #         raise ValueError("elements in x should be greater than 0 and less than 1")
    #
    #     return -(1/self.numobs)*np.sum(self.y*np.log(h) + (1 - self.y)*np.log(1 - h))
    #            #+ self.Lambda*(0.5/self.numobs)*sum([(theta ** 2).sum() for theta in thetas])
	
    def rand_thetas(self, output_sizes, bias=False):
        """output_sizes: array of size layernum + 1"""
        
        #Creating an initial theta from size of data and initial theta size value
        input_size = self.x.shape[1]
        layernum = len(output_sizes) - 1
        thetas = [np.random.rand(output_sizes[0], input_size)]

        #Appending random thetas one at a time
        if bias:
            for layer in range(layernum):
                thetas.append(np.random.rand(output_sizes[layer + 1], output_sizes[layer] + 1))
        else:
            for layer in range(layernum):
                thetas.append(np.random.rand(output_sizes[layer + 1], output_sizes[layer]))
        return thetas

    def numgrad(self, layernum, thetas, stepsize=1e-4, bias=False):
        """Numerically computing gradients -- A LOT SLOWER"""

        #Selecting the relevant theta
        theta = thetas[layernum]

        #Initializing the array containing gradients at the end
        grad = np.zeros(theta.shape)

        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                print(j)
                elem = np.zeros(thetas[layernum].shape)      
                elem[i, j] = stepsize
                print("elem: " + str(elem))
                thetas_plus = [t.copy() for t in thetas]
                thetas_minus = [t.copy() for t in thetas]
                print("Length of thetas_plus: " + str(len(thetas_plus)))
                thetas_plus[layernum] = thetas[layernum] + elem
                thetas_minus[layernum] = thetas[layernum] - elem
                print("thetas_plus: " + str(thetas_plus[layernum]))
                print("thetas_minus: " + str(thetas_minus[layernum]))
                _, h_plus = self.forward1(self.x, thetas_plus, bias=bias)
                _, h_minus = self.forward1(self.x, thetas_minus, bias=bias)
                print("Absolute difference in model values: " + str(sum(abs(h_plus - h_minus))))
                print("h_plus size for gradient at " + str(i) + ", " + str(j) + " : " + str(h_plus.shape))
                grad[i, j] = (self.cost(h_plus, thetas_plus) - self.cost(h_minus, thetas_minus))/(2*stepsize)

        return grad

    def gradDesc_iter(self, initthetas, bias=False):
        
        Deltas = self.backprop1(initthetas, bias=bias)
        numlayers = len(Deltas)

        return [initthetas[i] - self.learning_rate[i] * Deltas[i] for i in range(numlayers)]

    def gradDesc(self, initthetas, max_iter, tol, bias=False, verbose=False):

        numlayers = len(initthetas)

        # Converting learning rate to vector
        if isinstance(self.learning_rate, float) | isinstance(self.learning_rate, int):
            self.learning_rate = ([self.learning_rate] * numlayers)

        # Converting regularizations to vector
        if isinstance(self.Lambda, float) | isinstance(self.Lambda, int):
            self.Lambda = ([self.Lambda] * numlayers)

        thetas = initthetas
        cost = 1
        costs =[]
        count = 0
        while ((cost > tol) | np.isnan(cost)) and (count <= max_iter):
             if verbose:
                 print("Running backpropagation and obtaining costs for iteration " + str(count))
             thetas = self.gradDesc_iter(thetas)
             _, h = self.forward1(self.x, thetas, bias=bias)
             cost = (1/self.numobs)*np.sum(self.cost.f(self.y, h))
             costs.append(cost)
             count += 1

        return thetas, costs

    def random_sample(self, frac):
        random_ind = np.random.choice(self.numobs, int(frac*self.numobs), replace=False)
        return self.x[random_ind, :], self.y[random_ind, :]

    def stochasticGradDesc(self, frac, initthetas, max_iter, tol, bias=False):
        # Random Sampling
        x, y = self.random_sample(frac)
        self.shift_register_down(x, y)
        print("x shape: " + str(self.x.shape))
        print("y shape: " + str(self.y.shape))
        thetas, cost = self.gradDesc(initthetas, max_iter, tol, bias=bias)
        self.shift_register_up()
        return thetas, cost

    def sgd(self, initthetas, num_epochs):
        # Running num_epochs number of epochs

