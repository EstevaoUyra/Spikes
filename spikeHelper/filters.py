import scipy.stats as st
import numpy as np
import pickle

def convHist(spikeVec, sigma, binSize):
    smoothed = kernelSmooth(spikeVec, sigma)
    binned = binarize(smoothed, binSize)
    return binned

def kernelSmooth(spikeVec, sigma):
    normKernel = st.norm(0,sigma).pdf(np.linspace(-3*sigma,3*sigma,6*sigma))
    smoothed = np.convolve(spikeVec, normKernel, 'same')
    return smoothed

def binarize(smoothed, binSize,norm=False):
    times = np.arange(len(smoothed))
    nbins = np.array(np.floor(len(smoothed)/binSize),dtype=int)
    binned = np.histogram(times, bins = nbins, range = (times[0],times[-1]), weights = smoothed[times])[0]*(1000/binSize)
    return binned
