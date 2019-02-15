
import pandas as pd
import numpy as np

def hm_load(filename,  imp_meth):

    data = pd.read_csv(filename)
    
    level_types = {col:set([type(x).__name__ for x in data[col]]) for col in data.columns}
    mapping = dict()
    
    for col in level_types.keys():
        if "str" in level_types[col]:
            data[col], mapping[col] =  to_num(data[col])
    
    data = impute(data, meth=imp_meth)

    return data, mapping

def to_num(arr):
    levels = set(arr)
    count = 0
    mapping = dict()
    for level in levels:
        if not pd.isnull(level):
            arr[arr == level] = count
            mapping[level] = count
            count += 1
    return arr.astype("float"), mapping

def impute(data, meth):

    if meth=="mean":
        for col in data.columns:
            data[col].fillna(data[col].mean(), inplace=True)
    elif meth=="dropall":
        data.dropna(inplace=True)
    elif meth=="drop50":
        data.dropna(axis=1, thresh=int(0.50*data.shape[0]), inplace=True)
    elif meth=="drop25":
        data.dropna(axis=1, thresh=int(0.25*data.shape[0]), inplace=True)
    elif meth=="drop75":
        data.dropna(axis=1, thresh=int(0.75*data.shape[0]), inplace=True)
    elif meth=="forw":
        data.fillna(method="ffill")
    return data
