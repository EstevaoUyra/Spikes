import pandas as pd
from scipy.io import loadmat
import numpy as np
import pickle
from spikeHelper.filters import convHist
from spikeHelper.dataOrganization import XyTfromEpoch

def loadSpikeBehav(fileName):
    data = loadmat(fileName)

    spikes = data['dados'][0,0][1]
    behavior = data['dados'][0,0][0]

    spikes = pd.DataFrame([[ spikes[0,i][0][:,0], spikes[0,i][0][:,1]] for i in range(spikes.shape[1]) if spikes[0,i][0].shape[1]==2], columns=['times','trial'])
    behavior = pd.DataFrame(np.transpose(behavior[0,0][0]), columns=['one','onset','offset','zero','duration','sortIdx','sortLabel'])
    spikes['trialTime'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))

    return spikes, behavior

def normEpoch(spikes,behavior,ratNumber,sigma=20):
    epochs = np.array([[spikes.trialTime[iunit][spikes.trial[iunit]==itrial] for itrial in range(1,behavior.shape[0]+1)] for iunit in range(spikes.shape[0]) ] )
    epochs = pd.DataFrame(epochs, index = ['unit'+str(i) for i in range(spikes.shape[0])], columns = ['trial '+str(i) for i in range(1,behavior.shape[0]+1)])

    # Make sure spike times are consistent with trial durations
    assert ((1000*epochs.apply(lambda x: x.apply(mymax)).max()//1  - np.array([int(1000*behavior.duration[itrial]) for itrial in range(behavior.shape[0])]))>0).sum() == 0

    epochs = filterEpochs(epochs, method='premade', rat=ratNumber)

    for itrial in range(1,behavior.shape[0]+1):
        binSize = int(np.floor(1000*behavior.duration[itrial-1])/10)
        func = lambda x: precisionConvBin(x, int(np.floor(1000*behavior.duration[itrial-1])), sigma, binSize)
        if behavior.duration[itrial-1] > 1
        epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(func)

    # Make sure duration is consistent
    #assert all(np.floor((.5+ behavior.duration)*(1000/binSize))==epochs.applymap(len).iloc[0].values)
    return epochs


def epochData(spikes,behavior,ratNumber,sigma=20,binSize=50):
    epochs = np.array([[spikes.trialTime[iunit][spikes.trial[iunit]==itrial] for itrial in range(1,behavior.shape[0]+1)] for iunit in range(spikes.shape[0]) ] )
    epochs = pd.DataFrame(epochs, index = ['unit'+str(i) for i in range(spikes.shape[0])], columns = ['trial '+str(i) for i in range(1,behavior.shape[0]+1)])

    # Make sure spike times are consistent with trial durations
    assert ((1000*epochs.apply(lambda x: x.apply(mymax)).max()//1  - np.array([int(1000*behavior.duration[itrial]) for itrial in range(behavior.shape[0])]))>0).sum() == 0

    epochs = filterEpochs(epochs, method='premade', rat=ratNumber)

    for itrial in range(1,behavior.shape[0]+1):
        func = lambda x: precisionConvBin(x, int(np.floor(1000*behavior.duration[itrial-1])), sigma, binSize)
        epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(func)

    # Make sure duration is consistent
    assert all(np.floor((.5+ behavior.duration)*(1000/binSize))==epochs.applymap(len).iloc[0].values)
    return epochs

def precisionConvBin(spikeTimes,tmax,sigma,binSize):
    spikeVec = serializeSpikes(spikeTimes,tmax)
    spikeProb = convHist(spikeVec, sigma=sigma,binSize=binSize)
    return spikeProb

def serializeSpikes(spikeTimes, tmax, tmin=-500, dt=1):
    spikes = np.zeros(tmax-tmin)
    spikeTimeInMS = np.floor(spikeTimes*1000/dt) - tmin -1
    spikes[np.array(spikeTimeInMS,dtype=int)] = 1
    return spikes

def filterEpochs(epochs, minFiringRate= 4, minISI=5,method = 'premade', rat = None):
    if method == 'blind':
        goodFR = (epochs.applymap(lambda x: np.sum(x[500:])).mean(axis=1)>minFiringRate).values
        # Dump all neurons that have mean of 1 TOO FAST SPIKE per trial
        goodISI = (epochs.applymap(lambda x: np.histogram(np.diff(np.nonzero(x)),bins = minISI, range = (0,minISI))[0].sum() ).mean(axis=1) < 2).values
        return epochs.iloc[np.logical_and(goodFR,goodISI),:]
    elif method == 'premade':
        assert rat is not None
        goodNeurons = pickle.load(open('Data/goodCellsR'+str(rat)+'_withBaseDiff','rb'))==1
        return epochs.iloc[goodNeurons,:]
    else:
        raise('Nao existe metodo '+ str(method))

def mymax(x, tmin = -.5):
    try:
        return x.max()
    except:
        return tmin


class trialTransform(): #Not being used for long time
    def __init__(self):
        pass

    def fit(self,X,y):
        pass

    def transform(self,X,y = None):
        return np.transpose(X).reshape(-1,X.shape[0])

    def fit_transform(self,X,y=None):
        return self.transform(X,y)

    def get_params(self,deep=True):
        return {}

    def set_params(self):
        return self
