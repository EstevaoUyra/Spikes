import scipy.stats as st
import numpy as np
import pickle

def convHist(spikeVec, sigma=100, bins = 30):
    smoothed = kernelSmooth(spikeVec, sigma)
    binned = binarize(smoothed, bins)
    return binned

def kernelSmooth(spikeVec, sigma = 100):
    normKernel = st.norm(0,sigma).pdf(np.linspace(-3*sigma,3*sigma,6*sigma))
    smoothed = np.convolve(spikeVec, normKernel, 'same')
    return smoothed

def binarize(smoothed, bins=30):
    times = np.arange(len(smoothed))
    binSize = times[-1]/bins
    binned = np.histogram(times, bins = bins, range = (times[0],times[-1]), weights = smoothed[times])[0]*(1000/binSize)
    return binned

def filterEpochs(epochs, minFiringRate= 4, minISI=5,method = 'gabi', rat = None):
    if method == 'blind':
        goodFR = (epochs.applymap(lambda x: np.sum(x[500:])).mean(axis=1)>minFiringRate).values
        # Dump all neurons that have mean of 1 TOO FAST SPIKE per trial
        goodISI = (epochs.applymap(lambda x: np.histogram(np.diff(np.nonzero(x)),bins = minISI, range = (0,minISI))[0].sum() ).mean(axis=1) < 2).values
        return epochs.iloc[np.logical_and(goodFR,goodISI),:]
    elif method == 'gabi':
        assert rat is not None
        goodNeurons = pickle.load(open('Data/goodCellsR'+str(rat)+'_withBaseDiff','rb'))==1
        return epochs.iloc[goodNeurons,:]
    else:
        raise('Nao existe metodo '+ str(method))
