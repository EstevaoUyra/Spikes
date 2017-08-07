import pandas as pd
import numpy as np
from itertools import product

def singleRatResults(trials, nSplits = 5, conditions = ['Train late','Train early']):
    split = np.arange(nSplits)
    ratResults = pd.DataFrame(list(product(conditions,split,trials)  ),columns = ['condition','split','trial'] )
    ratResults['correlation']=np.nan
    ratResults['kappa']=np.nan
    return ratResults

def trialToXyT(dataset):
    X = np.transpose(dataset).reshape(-1,dataset.shape[0])
    y = np.arange(X.shape[0])%dataset.shape[1]
    trial = np.arange(X.shape[0])//dataset.shape[1]
    data = pd.DataFrame(X, columns = ['unit'+str(i) for i in np.arange(dataset.shape[0])+1])
    data['y'] = y
    data['trial'] = trial
    data['end'] = data['trial'] > data['trial'].max() - 100
    data['beg'] = data['trial'] < 100
    return data

def getX(data):
    return data[data.columns[['unit' in coli for coli in data.columns]]].as_matrix()
