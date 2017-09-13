from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np
import matplotlib.pyplot as plt

def XyTfromEpoch(epochs, getBins=False, minBins=False, maxBins=False,verbose=False):
    trialBins = epochs.applymap(len).iloc[0,:].values

    if getBins == False:
        nBins = trialBins.min()
        getBins = [0,nBins]
        if verbose: print('Number of bins not defined, getting first',nBins)

    if minBins == False:
        minBins = trialBins.min()
        if verbose: print('Minimum size not restricted. Using all up from ',minBins)
    else:
        if verbose: print('Minimum size restricted. Using all up from ',minBins)

    if maxBins == False:
        maxBins = trialBins.max()
        if verbose: print('Maximum size not restricted. Using all up to ',maxBins)
    else:
        if verbose: print('Maximum size restricted. Using all up to ',maxBins)

    possibleEpochs = np.logical_and(np.logical_and(trialBins <= maxBins, trialBins >= minBins),trialBins >=getBins[1])
    cutEpochs = epochs.iloc[:,possibleEpochs].applymap(lambda x: x[getBins[0]:getBins[1]] )
    return np.swapaxes(np.array([np.vstack(cutEpochs.iloc[i]) for i in range(cutEpochs.shape[0])]),1,2)


scoreSurr = {}
score = {}
for rat in []:
    data = pickle.load(open('Data/r'+str(rat)+'_bin50_sigma100.pickle2','rb'))
    data = XyTfromEpoch(data,getBins=[0,36],minBins=40,maxBins=75,verbose=True)[:,0:,:]
    cov = Covariances('oas')

    allCovs = cov.transform(data.transpose().swapaxes(1,2))
    testCovs = np.vstack((allCovs[:100,:,:],allCovs[-100:,:,:]))
    y = 100*[0]+100*[1]
    clf = MDM()
    score[rat] = cross_val_score(clf,testCovs,y,cv=10)

    yrand =  np.random.rand(200) < .5
    scoreSurr[rat] = cross_val_score(clf,testCovs,yrand,cv=10)


    plt.subplot(4,1,rat-6)
    plt.plot(score[rat])
    plt.axhline(scoreSurr[rat].mean(),color='r')
    plt.axhline(scoreSurr[rat].mean()-scoreSurr[rat].std(),color='r',linestyle='--')
    plt.axhline(scoreSurr[rat].mean()+scoreSurr[rat].std(),color='r',linestyle='--')
    plt.suptitle('Decodificacao: trial vem do comeco ou do fim')
#plt.show()

scoreSurr = {}
score = {}
for rat in [7,8,9,10]:
    data = pickle.load(open('Data/r'+str(rat)+'_bin50_sigma100.pickle2','rb'))
    cov = Covariances('oas')

    short = XyTfromEpoch(data,getBins=[0,10],maxBins=20,verbose=True)
    long = XyTfromEpoch(data,getBins=[0,10],minBins=30,verbose=True)

    shortCov = cov.transform(short.transpose().swapaxes(1,2))
    longCov = cov.transform(long.transpose().swapaxes(1,2))

    cov = Covariances('oas')
    y = shortCov.shape[0]*[0]+shortCov.shape[0]*[1]
    clf = MDM()

    score[rat] = []
    scoreSurr[rat] = []
    for i in range((longCov.shape[0]-shortCov.shape[0]-1)//30):
        X = np.vstack((shortCov, longCov[30*i:30*i+shortCov.shape[0],:,:]))
        score[rat] += list(cross_val_score(clf,X,y,cv=5))

        yrand =  np.random.rand(2*shortCov.shape[0]) < .5
        scoreSurr[rat] += list(cross_val_score(clf,X,yrand,cv=2))

    scoreSurr[rat] = np.array(scoreSurr[rat])
    score[rat] = np.array(score[rat])
    plt.subplot(4,1,rat-6)
    plt.plot(score[rat].reshape(5,-1).mean(axis=0))
    plt.axhline(scoreSurr[rat].mean(),color='r')
    plt.axhline(scoreSurr[rat].mean()-scoreSurr[rat].std(),color='r',linestyle='--')
    plt.axhline(scoreSurr[rat].mean()+scoreSurr[rat].std(),color='r',linestyle='--')
plt.show()
