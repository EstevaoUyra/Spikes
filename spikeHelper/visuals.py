import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from spikeHelper.filters import convHist

def trialNeuronPlot(epochs, neuron,trial,xmax = None):
    spikeVec = epochs.iloc[neuron,trial-1]*1000
    times = np.nonzero(spikeVec)[0] -500
    sns.rugplot(times)
    plt.plot(np.arange(len(spikeVec))-500,convHist(spikeVec, returnSmoothed = True)[1])
    plt.bar(np.arange(30)*50 - 500,convHist(spikeVec)/50,width= .8*50,alpha=.3)
    plt.title('Activity of neuron %d at trial %d' % (neuron,trial) )
    plt.ylabel('Firing rate (spikes/s)')
    plt.xlabel('Time from nosepoke')
    if xmax != None:
        plt.xlim([-500,xmax])
    

