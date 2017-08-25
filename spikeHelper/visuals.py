import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from spikeHelper.filters import convHist, kernelSmooth, oneToOneDist
from spikeHelper.dataOrganization import trialToXyT, getX, normRows, trialNumber
from spikeHelper.similarities import distanceGeneralization, similarityMatrix
import pandas as pd
from matplotlib import gridspec

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


def motorPlot(epochs, totalTime, bins=50,sigma=50,error = [68,95]):
    endSync = epochs.iloc[:,(epochs.applymap(len).iloc[0]>totalTime).values].applymap(lambda x: x[-totalTime:])

    endSync['unit'] = endSync.index

    endSync = endSync.melt(id_vars = 'unit',value_name = 'activity',var_name='trial')

    endSync['activity'] = endSync['activity'].apply(lambda x: convHist(x, sigma=sigma, bins=bins))
    times = -totalTime + totalTime/bins * (np.arange(bins)+1)

    motor = pd.concat([endSync,pd.DataFrame(np.vstack(endSync['activity'].as_matrix()), columns=times) ],axis=1 ).drop('activity',axis=1)
    motor = motor.melt(id_vars=['unit','trial'],var_name = 'Time to leave',value_name = "Firing Rate" )
    sns.tsplot(motor, time= 'Time to leave', value="Firing Rate",unit='trial',condition='unit',ci=error)
    plt.legend().set_visible(False)

def compareSimilarities(data,title,nTrials=50,normalize=True):

    beg = trialToXyT(data[:,:,:nTrials])
    end = trialToXyT(data[:,:,-nTrials:])
    allt = trialToXyT(data)


    fig, ((ax1, ax2),(ax3,ax4), ( ax5,ax6)) = plt.subplots(3, 2, sharex=True, sharey=True,figsize=(10, 8))
    fig.suptitle(title,fontsize=16)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])

    sim = distanceGeneralization(getX(beg), beg['y'], beg['trial'], method='mah')
    sns.heatmap(sim,ax=ax1,cbar=0)
    ax1.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    if normalize:
        sim = normRows(sim)
    ax1.set_title(str(nTrials) +' first trials mahalanobis')

    sim = distanceGeneralization(getX(end), end['y'], end['trial'], method='mah')
    sns.heatmap(sim,ax=ax3,cbar_ax=cbar_ax)
    ax3.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    if normalize:
        sim = normRows(sim)
    ax3.set_title(str(nTrials) +' last trials mahalanobis')

    sim = distanceGeneralization(getX(allt), allt['y'], allt['trial'], method='mah')
    if normalize:
        sim = normRows(sim)
    sns.heatmap(sim,ax=ax5,cbar=0)
    ax5.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax5.set_title('All trials mahalanobis')

    sim = distanceGeneralization(getX(beg), beg['y'], beg['trial'], method='greek')
    if normalize:
        sim = normRows(sim)
    sns.heatmap(sim,ax=ax2,cbar=0)
    ax2.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax2.set_title(str(nTrials) +' first trials euclidean')

    sim = distanceGeneralization(getX(end), end['y'], end['trial'], method='greek')
    if normalize:
        sim = normRows(sim)
    sns.heatmap(sim,ax=ax4,cbar=0)
    ax4.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax4.set_title(str(nTrials) +' last trials euclidean')

    sim = distanceGeneralization(getX(allt), allt['y'], allt['trial'], method='greek')
    if normalize:
        sim = normRows(sim)
    sns.heatmap(sim,ax=ax6,cbar=0)
    ax6.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax6.set_title('All trials euclidean')

def crossSimilarities(shorter, longer, title, normalize=True):

    shorter = trialToXyT(shorter)
    longer = trialToXyT(longer)

    #fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=2,rowspan=2)
    ax2 = plt.subplot2grid((3, 6), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 6), (0, 2))

    fig.suptitle('Mahalanobis | Euclidean\n'+title,fontsize=16)
    #cbar_ax = fig.add_axes([.91, .3, .03, .4])
    cbar_ax = fig.add_axes([.91, .2, .02, .6])

    #Mahalanobis plots
    sim = similarityMatrix(getX(shorter), shorter['y'],getX(longer), longer['y'],method='mah',compare=True,normalize=normalize)
    sns.heatmap(sim,ax=ax1,cbar=0)
    ax1.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax1.set_title('Long as template')
    ax1.set_xticks([])

    sim = similarityMatrix(getX(longer), longer['y'],method='mah',normalize=normalize)
    sns.heatmap(sim,ax=ax3,cbar_ax=cbar_ax)
    ax3.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax3.set_title('Long with itself')

    sim = similarityMatrix(getX(longer), longer['y'],getX(shorter), shorter['y'],method='mah',compare=True,normalize=normalize)
    sns.heatmap(sim,ax=ax2,cbar=0)
    ax2.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax2.set_title('Short as template')
    ax2.set_yticks([])

    sim = similarityMatrix(getX(shorter), shorter['y'],method='mah',normalize=normalize)
    sns.heatmap(sim,ax=ax4,cbar=0)
    ax4.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax4.set_title('Short with itself')
    ax4.set_xticks([]);ax4.set_yticks([])

    ax5 = plt.subplot2grid((3, 6), (0, 3), colspan=2)
    ax7 = plt.subplot2grid((3, 6), (1, 3), colspan=2,rowspan=2)
    ax6 = plt.subplot2grid((3, 6), (1, 5), rowspan=2)
    ax8 = plt.subplot2grid((3, 6), (0, 5))

    # Euclidean plots
    sim = similarityMatrix(getX(shorter), shorter['y'],getX(longer), longer['y'],method='greek',compare=True,normalize=normalize)
    sns.heatmap(sim,ax=ax5,cbar=0)
    ax5.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax5.set_title('Long as template')
    ax5.set_xticks([]); ax5.set_yticks([])

    sim = similarityMatrix(getX(longer), longer['y'],method='greek',normalize=normalize)
    sns.heatmap(sim,ax=ax7,cbar=0)
    ax7.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax7.set_title('Long with itself')
    ax7.set_yticks([])

    sim = similarityMatrix(getX(longer), longer['y'],getX(shorter), shorter['y'],method='greek',compare=True,normalize=normalize)
    sns.heatmap(sim,ax=ax6,cbar=0)
    ax6.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax6.set_title('Short as template')
    ax6.set_yticks([])

    sim = similarityMatrix(getX(shorter), shorter['y'],method='greek',normalize=normalize)
    sns.heatmap(sim,ax=ax8,cbar=0)
    ax8.plot(sim.argmax(axis=1)+.5, np.arange(sim.shape[0])+.5)
    ax8.set_title('Short with itself')
    ax8.set_xticks([]);ax8.set_yticks([])

def heatAct(data):
    meanAct = normRows(np.array([np.array([data[unit][data['y']==i] for i in range(len(np.unique(data['y'])))]).mean(axis=1) for unit in data.columns[:-4]]))
    order = np.argsort(np.nonzero(meanAct==1)[1])
    sns.heatmap(meanAct[order,:])

def plotPredResults(results,kappa=0):
    res = results.iloc[:,:8].applymap(lambda x: x.reshape(-1))
    fig = plt.figure(figsize=(16, 8))
    plt.subplot(2,2,1)

    if kappa==0:
        plt.suptitle('Correlation coefficient')
        lim = [0,1]
    else:
        assert kappa==1
        lim=[0,.5]
        plt.suptitle('Kappa coefficient')
    sns.barplot(data = pd.DataFrame(np.vstack(res.iloc[:,0+kappa].values),index=['Rat 7','Rat 8','Rat 9','Rat 10']).transpose())
    plt.title('Late'); plt.ylim(lim)
    plt.subplot(2,2,2)
    sns.barplot(data = pd.DataFrame(np.vstack(res.iloc[:,2+kappa].values),index=['Rat 7','Rat 8','Rat 9','Rat 10']).transpose())
    plt.title('Early');  plt.ylim(lim)
    plt.subplot(2,2,4)
    sns.barplot(data = pd.DataFrame(np.vstack(res.iloc[:,4+kappa].values),index=['Rat 7','Rat 8','Rat 9','Rat 10']).transpose())
    plt.title('Trained late'); plt.ylim(lim)
    plt.subplot(2,2,3)
    sns.barplot(data = pd.DataFrame(np.vstack(res.iloc[:,6+kappa].values),index=['Rat 7','Rat 8','Rat 9','Rat 10']).transpose())
    plt.title('Trained early');plt.ylim(lim)
