import scipy.stats as st
import numpy as np

def convHist(spikeVec, returnSmoothed = False, bins = 30):
    normKernel = st.norm(0,100).pdf(np.linspace(-300,300,600))
    smoothed = np.convolve(spikeVec, normKernel, 'same')
    times = np.arange(1500)
    binned = np.histogram(times,bins, range=[0,1500], weights = smoothed[times])[0]
    if returnSmoothed:
        return binned, smoothed
    else:
        return binned
