import scipy.stats as st
import numpy as np

def convHist(spikeVec, sigma=100, bins = 30, window = (0,1500)):
    smoothed = kernelSmooth(spikeVec, sigma)
    binned = binarize(smoothed, bins, window)
    return binned

def kernelSmooth(spikeVec, sigma = 100):
    normKernel = st.norm(0,sigma).pdf(np.linspace(-3*sigma,3*sigma,6*sigma))
    smoothed = np.convolve(spikeVec, normKernel, 'same')
    return smoothed

def binarize(smoothed, bins=30, window = (0,1500)):
    times = np.arange(window[0],window[1])
    binned = np.histogram(times, bins = bins, range = window, weights = smoothed[times])[0]
    return binned
