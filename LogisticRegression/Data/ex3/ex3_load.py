# coding: utf-8
from skimage import io
from skimage import color
import numpy as np

def ex3_load(filename):
    data = open(filename, "r+").readlines()
    
    for i in range(len(data)):
        data[i] = str.rsplit(data[i], ",")
     
    data2 = [[float(j) for j in i] for i in data]
    return np.array(data2)

def ex3_compat(filename, factor):
    img = color.rgb2gray(io.imread(filename))
    img = img[np.linspace(0, img.shape[0] - 1, 20, dtype=int), :][:, np.linspace(0, img.shape[1] - 1, 20, dtype=int)]
    return np.hstack([[1], factor*img.reshape(400,)])

def ex3_merge(data, labels, filename, numrpt, newlabel, factor):
    to_merge = ex3_compat(filename, factor)
    return np.vstack([data, np.repeat([to_merge], repeats=numrpt, axis=0)]), np.vstack([labels, np.repeat([[newlabel]], repeats=numrpt, axis=0)])
    
