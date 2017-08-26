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

def XyTfromEpoch(epochs, getBins=False, minBins=False, maxBins=False):
    trialBins = epochs.applymap(len).iloc[0,:].values

    if getBins == False:
        nBins = trialBins.min()
        getBins = [0,nBins]
        print('Number of bins not defined, getting first',nBins)

    if minBins == False:
        minBins = trialBins.min()
        print('Minimum size not restricted. Using all up from ',minBins)
    else:
        print('Minimum size restricted. Using all up from ',minBins)

    if maxBins == False:
        maxBins = trialBins.max()
        print('Maximum size not restricted. Using all up to ',maxBins)
    else:
        print('Maximum size restricted. Using all up to ',maxBins)

    possibleEpochs = np.logical_and(np.logical_and(trialBins <= maxBins, trialBins >= minBins),trialBins >=getBins[1])
    cutEpochs = epochs.iloc[:,possibleEpochs].applymap(lambda x: x[getBins[0]:getBins[1]] )
    return np.swapaxes(np.array([np.vstack(cutEpochs.iloc[i]) for i in range(cutEpochs.shape[0])]),1,2)

def normRows(k):
    return np.array([k[i,:]/(k.max(axis=1)[i]) for i in range(k.shape[0])])
