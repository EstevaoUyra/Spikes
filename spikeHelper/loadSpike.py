import pandas as pd
from scipy.io import loadmat
import numpy as np

def loadSpikeBehav(fileName):
    data = loadmat(fileName)

    spikes = data['dados'][0,0][1]
    behavior = data['dados'][0,0][0]

    spikes = pd.DataFrame([[ spikes[0,i][0][:,0], spikes[0,i][0][:,1]] for i in range(spikes.shape[1]) if spikes[0,i][0].shape[1]==2], columns=['times','trial'])
    behavior = pd.DataFrame(np.transpose(behavior[0,0][0]), columns=['one','onset','offset','zero','duration','sortIdx','sortLabel'])
    spikes['trialTime'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))

    epochs = np.array([[spikes.trialTime[iunit][spikes.trial[iunit]==itrial] for itrial in range(1,behavior.shape[0]+1)] for iunit in range(spikes.shape[0]) ] )
    epochs = pd.DataFrame(epochs, index = ['unit'+str(i) for i in range(spikes.shape[0])], columns = ['trial '+str(i) for i in range(1,behavior.shape[0]+1)])

    # Make sure spike times are consistent with trial durations
    assert ((1000*epochs.apply(lambda x: x.apply(mymax)).max()//1  - np.array([int(1000*behavior.duration[itrial]) for itrial in range(behavior.shape[0])]))>0).sum() == 0

    for itrial in range(1,behavior.shape[0]+1):
        epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: serializeSpikes(x, int(1000*behavior.duration[itrial-1])))

    return spikes, behavior, epochs

def serializeSpikes(spikeTimes, tmax, tmin=-500, dt=1):
    spikes = np.zeros(tmax-tmin+1)
    spikes[np.array((spikeTimes*1000/dt)//1 - tmin,dtype=int)] = 1
    return spikes

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
