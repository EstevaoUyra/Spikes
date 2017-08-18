import pandas as pd
import numpy as np
from itertools import product
from spikeHelper.filters import convHist

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

def trialNumber(trialStrings):
    return np.array(list(map(lambda x: int(x[5:]),trialStrings)))

def XyTfromEpoch(epochs, binDuration=50, window=[0,1000]):
    assert epochs.applymap(len).min().min() >  window[1]
    cutEpochs = epochs.applymap(lambda x: x[window[0]:window[1]] )
    bins = (window[1]-window[0])//binDuration
    binnedData = cutEpochs.applymap(lambda x: convHist(x, bins=bins) )
    return np.swapaxes(np.array([np.vstack(binnedData.iloc[i]) for i in range(binnedData.shape[0])]),1,2)

def normRows(k):
    return np.array([k[i,:]/(k.max(axis=1)[i]) for i in range(k.shape[0])])
