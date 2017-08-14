import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from spikeHelper.filters import convHist, kernelSmooth
from spikeHelper.dataOrganization import trialToXyT, getX, normRows, trialNumber
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis, euclidean
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
    for j in range(4):
        ratos = ['Rato 7', 'Rato 8','Rato 9', 'Rato 10']
        corrDF = results.iloc[j,-3]
        kappDF = results.iloc[j,-2]
        corrDF['trial']= np.arange(corrDF.shape[0])
        kappDF['trial']= np.arange(kappDF.shape[0])
        corrDF['type']= 'corr'
        kappDF['type']= 'kappa'
        corrTS = pd.melt(corrDF, id_vars=corrDF.columns[-2:])
        kappTS = pd.melt(kappDF, id_vars=kappDF.columns[-2:])
        TS = pd.concat((corrTS,kappTS))
        fig = plt.figure(figsize=(16, 6), dpi= 80, facecolor='w', edgecolor='k')
        plt.subplot(4,1,j+1)
        sns.tsplot(data=TS, time="trial",condition = 'type', unit="variable", value="value",ci='sd')
        plt.ylim([0,1])
        plt.title(ratos[j])

def firingRateEvo(epochs):
    meanFR = epochs.applymap(lambda x: np.sum(x[500:]))
    avgFiringRate = pd.DataFrame(meanFR.mean(axis=0),columns = ['Average firing rate'])
    avgFiringRate['trial'] = trialNumber(epochs.columns)
    xlims = np.sort(trialNumber(epochs.columns))[[0,-1]]
    sns.jointplot(y='Average firing rate', x='trial', data=avgFiringRate,kind='reg',ratio=10,size=5,xlim=xlims)

def similarityMatrix(X,y,W=None,z=None,method='greek',compare=False):
    if compare:
        assert W.shape[0] > 1
        assert z.shape[0] > 1
    else:
        assert W == None
        assert z == None
        W = X; z = y;

    if method == 'mah':
        empCov = EmpiricalCovariance()
        precision = empCov.fit(X).get_precision()
        dist = lambda u,v : mahalanobis(u,v,precision)
    elif method == 'greek':
        dist = euclidean

    times = np.unique(y)
    meanActivity = [ W[z==ti].mean(axis=0) for ti in np.unique(z)]
    distances = []
    for ti in times:
        distances.append([])
        distances[-1] = np.array([np.array([dist(u,v) for u in X[y==ti]]).mean() for v in meanActivity])
    return 1/np.array(distances)

def motorPlot(epochs, totalTime, bins=50,sigma=50):
    endSync = epochs.iloc[:,(epochs.applymap(len).iloc[0]>totalTime).values].applymap(lambda x: x[-totalTime:])

    endSync['unit'] = endSync.index

    endSync = endSync.melt(id_vars = 'unit',value_name = 'activity',var_name='trial')

    endSync['activity'] = endSync['activity'].apply(lambda x: convHist(x, sigma=sigma, bins=bins))
    times = -totalTime + totalTime/bins * (np.arange(nbins)+1)

    motor = pd.concat([endSync,pd.DataFrame(np.vstack(endSync['activity'].as_matrix()), columns=times) ],axis=1 ).drop('activity',axis=1)
    motor = motor.melt(id_vars=['unit','trial'],var_name = 'Time to leave',value_name = "Firing Rate" )

    sns.tsplot(motor,ci=[68,95], time= 'Time to leave', value="Firing Rate",unit='trial',condition='unit')
    plt.legend().set_visible(False)

def compareSimilarities(data,title,nTrials=50):

    beg = trialToXyT(data[:,:,:nTrials])
    end = trialToXyT(data[:,:,-nTrials:])
    allt = trialToXyT(data)


    fig, ((ax1, ax2),(ax3,ax4), ( ax5,ax6)) = plt.subplots(3, 2, sharex=True, sharey=True,figsize=(10, 8))
    fig.suptitle(title,fontsize=16)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    sim = similarityMatrix(getX(beg), beg['y'],method='mah')
    sns.heatmap(normRows(sim),ax=ax1,cbar=0)
    ax1.plot(sim.argmax(axis=1)+.5, np.arange(20))
    ax1.set_title(str(nTrials) +' first trials mahalanobis')

    sim = similarityMatrix(getX(end), end['y'],method='mah')
    sns.heatmap(normRows(sim),ax=ax3,cbar_ax=cbar_ax)
    ax3.plot(sim.argmax(axis=1)+.5, np.arange(20))
    ax3.set_title(str(nTrials) +' last trials mahalanobis')

    sim = similarityMatrix(getX(allt), allt['y'],method='mah')
    sns.heatmap(normRows(sim),ax=ax5,cbar=0)
    ax5.plot(sim.argmax(axis=1)+.5, np.arange(20))
    ax5.set_title('All trials mahalanobis')

    sim = similarityMatrix(getX(beg), beg['y'],method='greek')
    sns.heatmap(normRows(sim),ax=ax2,cbar=0)
    ax2.plot(sim.argmax(axis=1)+.5, np.arange(20))
    ax2.set_title(str(nTrials) +' first trials euclidean')

    sim = similarityMatrix(getX(end), end['y'],method='greek')
    sns.heatmap(normRows(sim),ax=ax4,cbar=0)
    ax4.plot(sim.argmax(axis=1)+.5, np.arange(20))
    ax4.set_title(str(nTrials) +' last trials euclidean')

    sim = similarityMatrix(getX(allt), allt['y'],method='greek')
    sns.heatmap(normRows(sim),ax=ax6,cbar=0)
    ax6.plot(sim.argmax(axis=1)+.5, np.arange(20))
    ax6.set_title('All trials euclidean')
