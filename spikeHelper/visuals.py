import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from spikeHelper.filters import convHist, kernelSmooth
import pandas as pd

def trialNeuronPlot(epochs, neuron, trial, xmax = None):
    spikeVec = epochs.iloc[neuron,trial-1]*1000
    times = np.nonzero(spikeVec)[0] -500
    sns.rugplot(times)
    plt.plot(np.arange(len(spikeVec))-500,kernelSmooth(spikeVec))
    plt.bar(np.arange(30)*50 - 500,convHist(spikeVec)/50,width= .8*50,alpha=.3)
    plt.title('Activity of neuron %d at trial %d' % (neuron,trial) )
    plt.ylabel('Firing rate (spikes/s)')
    plt.xlabel('Time from nosepoke')
    if xmax != None:
        plt.xlim([-500,xmax])

def evolutionPlot(results):
    for j in [1,2,3]:
        ij=[[],0,1,3]
        ratos = ['Rato 7', 'Rato 8', 'Rato 10']
        i = ij[j]
        corrDF = results.iloc[i,-3]
        kappDF = results.iloc[i,-2]
        corrDF['trial']= np.arange(corrDF.shape[0])
        kappDF['trial']= np.arange(kappDF.shape[0])
        corrDF['type']= 'corr'
        kappDF['type']= 'kappa'
        corrTS = pd.melt(corrDF, id_vars=corrDF.columns[-2:])
        kappTS = pd.melt(kappDF, id_vars=kappDF.columns[-2:])
        TS = pd.concat((corrTS,kappTS))
        fig = plt.figure(figsize=(16, 6), dpi= 80, facecolor='w', edgecolor='k')
        plt.subplot(3,1,j)
        sns.tsplot(data=TS, time="trial",condition = 'type', unit="variable", value="value",ci='sd')
        plt.title(ratos[j-1])
