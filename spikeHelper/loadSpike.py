import pandas as pd
from scipy.io import loadmat
import numpy as np
import pickle
from spikeHelper.filters import convHist, kernelSmooth, binarize
from spikeHelper.dataOrganization import XyTfromEpoch, padNans
import h5py

def loadSpikeBehav(fileName):
    data = loadmat(fileName)

    spikes = data['dados'][0,0][1]
    behavior = data['dados'][0,0][0]

    spikes = pd.DataFrame([[ spikes[0,i][0][:,0], spikes[0,i][0][:,1]] for i in range(spikes.shape[1]) if spikes[0,i][0].shape[1]==2], columns=['times','trial'])
    behavior = pd.DataFrame(np.transpose(behavior[0,0][0]), columns=['one','onset','offset','zero','duration','sortIdx','sortLabel'])
    spikes['trialTime'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))

    return spikes, behavior

def epochData(spikes, behavior, ratNumber, sigma, binSize):
    epochs = np.array([[spikes.trialTime[iunit][spikes.trial[iunit]==itrial] for itrial in range(1,behavior.shape[0]+1)] for iunit in range(spikes.shape[0]) ] )
    epochs = pd.DataFrame(epochs, index = ['unit'+str(i) for i in range(spikes.shape[0])], columns = ['trial '+str(i) for i in range(1,behavior.shape[0]+1)])

    # Make sure spike times are consistent with trial durations
    assert ((1000*epochs.apply(lambda x: x.apply(mymax)).max()//1  - np.array([int(1000*behavior.duration[itrial]) for itrial in range(behavior.shape[0])]))>0).sum() == 0

    # Get only selected neurons
    epochs = filterEpochs(epochs, method='premade', rat=ratNumber)


    if binSize == 'norm':
        for itrial in range(1,behavior.shape[0]+1):
            trialSize = int(1000*behavior.duration[itrial-1])
            binSize = int(np.floor((trialSize)/10))
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: serializeSpikes(x, trialSize))
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: kernelSmooth(x, sigma))
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: x[500:])
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: binarize(x, binSize))
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: x*100/binSize)
    else:
        for itrial in range(1,behavior.shape[0]+1):
            func = lambda x: precisionConvBin(x, int(np.floor(1000*behavior.duration[itrial-1])), sigma, binSize)
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(func)
        # Make sure duration is consistent
        assert all(np.floor((.5+ behavior.duration)*(1000/binSize))==epochs.applymap(len).iloc[0].values)

    return epochs

def precisionConvBin(spikeTimes,tmax,sigma,binSize):
    spikeVec = serializeSpikes(spikeTimes,tmax)
    spikeProb = convHist(spikeVec, sigma=sigma,binSize=binSize)
    return spikeProb

def serializeSpikes(spikeTimes, tmax, tmin=-500, dt=1):
    spikes = np.zeros(tmax-tmin)
    spikeTimeInMS = np.floor(spikeTimes*1000/dt) - tmin -1
    spikes[np.array(spikeTimeInMS,dtype=int)] = 1
    return spikes

def filterEpochs(epochs, minFiringRate= 4, minISI=5,method = 'premade', rat = None):
    if method == 'blind':
        goodFR = (epochs.applymap(lambda x: np.sum(x[500:])).mean(axis=1)>minFiringRate).values
        # Dump all neurons that have mean of 1 TOO FAST SPIKE per trial
        goodISI = (epochs.applymap(lambda x: np.histogram(np.diff(np.nonzero(x)),bins = minISI, range = (0,minISI))[0].sum() ).mean(axis=1) < 2).values
        return epochs.iloc[np.logical_and(goodFR,goodISI),:]
    elif method == 'premade':
        assert rat is not None
        goodNeurons = pickle.load(open('Data/goodCellsR'+str(rat)+'_withBaseDiff','rb'))==1
        return epochs.iloc[goodNeurons,:]
    else:
        raise('Nao existe metodo '+ str(method))

def mymax(x, tmin = -.5):
    try:
        return x.max()
    except:
        return tmin


class rat():
    def __init__(self,ratNumber, sigma, binSize, label = None):
        self.ratNumber = ratNumber
        self.sigma = sigma
        self.binSize = binSize
        if label is None:
            self.label = 'Rat '+ str(ratNumber)
        else:
            self.label = label
        self.dset = self._loadFile()

        self.trialsToUse = None
        self.X = None
        self.y = None
        self.trial = None


    def _loadFile(self):
        f = h5py.File("spikeData.hdf5", "r+")
        dsetName = 'bin'+str(self.binSize)+'_sigma'+str(self.sigma)

        if self.label not in f:
            print(self.label + ' not contained in database. Creating group')
            f.create_group(self.label)
            f[self.label].create_group('epochData')

        if dsetName not in f[self.label]['epochData']:
            print(dsetName + ' not contained in database. Creating dataset.')
            self._create(dsetName)

        return f[self.label]['epochData'][dsetName]


    def _create(self, dsetName):
        f = h5py.File("spikeData.hdf5", "r+")
        spikes, behavior = loadSpikeBehav('Data/rato'+str(self.ratNumber)+'tudo.mat')
        epochs = epochData(spikes,behavior,ratNumber = self.ratNumber, sigma=self.sigma, binSize=self.binSize)
        eachLen = epochs.applymap(len)
        trialLen = eachLen.min(axis=0).values
        epochs = epochs.applymap(lambda x: padNans(x, eachLen.max().max() ) )
        cubicData, trialN = XyTfromEpoch(epochs,returnTrialN = True)
        iti = np.hstack((0,behavior.onset[1:].values - behavior.offset.iloc[:-1].values ) )

        f[self.label]['epochData'].create_dataset(dsetName, data = cubicData)
        f[self.label]['epochData'][dsetName].attrs.create('Trial number', trialN)
        f[self.label]['epochData'][dsetName].attrs.create('Trial length', trialLen)
        f[self.label]['epochData'][dsetName].attrs.create('Trial duration', behavior.duration.values*1000)
        f[self.label]['epochData'][dsetName].attrs.create('Trial start', behavior.onset.values*1000)
        f[self.label]['epochData'][dsetName].attrs.create('Trial end', behavior.offset.values*1000)
        f[self.label]['epochData'][dsetName].attrs.create('Intertrial interval', iti*1000)

    def _trialHas(self,property, value):
        if property == 'minDuration':
            return self.dset.attrs['Trial duration'] > value

        elif property == 'maxDuration':
            return self.dset.attrs['Trial duration'] < value

        elif property == 'trialMin':
            return self.dset.attrs['Trial number'] > value

        elif property == 'trialMax':
            return self.dset.attrs['Trial number'] < value

        elif property == 'ntrials':
            hasProperty = np.cumsum(self.trialsToUse)<= value
            if hasProperty.shape[0] < value:
                print('Could not get %d trials, got %d instead'%(value, trialsToUse.shape[0]))

            return hasProperty

        else:
            raise ValueError('There is no type %s'%selecType)

    def selecTrials(self, restrictions):
        '''Select trialsToUse via a restrictions dict of the form {property: value}
        Accepted properties are:
        'minDuration', 'maxDuration' (in milisseconds),
        'trialMin','trialMax' (Absolute number of the trial)
        'ntrials' (turns off last trials to get a total of ntrials)
        '''
        self.trialsToUse = np.ones(self.dset.attrs['Trial number'].shape[0])
        for propertyi in restrictions:
            self.trialsToUse = np.logical_and(self.trialsToUse, self._trialHas(propertyi, restrictions[propertyi]) )

    def manualSelectTrials(self, trialsOfInterest):
        ''' Give an array of booleans of the size of total trials, or an array of integers for the trials of interest'''

        if trialsOfInterest.unique().shape[0] == 2:
            assert trialsOfInterest.shape[0] == self.dset.shape[2]
            self.trialsToUse = trialsOfInterest
        else:
            assert np.sum(np.diff(np.sort(trialsOfInterest))==0) ==0 #there are no repetitions
            self.trialsToUse = np.array([trial in trialsOfInterest for trial in self.dset.attrs['Trial number'] ])

    def prepareXyT(self, tmin, tmax=None):
        if self.trialsToUse is None:
            self.trialsToUse = np.ones(self.dset.attrs['Trial number'].shape[0])

        X = np.transpose(self.dset[:,:,self.trialsToUse]).reshape(-1,self.dset.shape[0])
        y = np.arange(X.shape[0])%self.dset.shape[1]
        trial = np.array([self.dset.shape[1]*[t] for t in self.dset.attrs['Trial number'][self.trialsToUse]]).reshape(-1,1)

        if tmax is None:
            tmax = self.dset.attrs['Trial duration'][self.trialsToUse].min()

        if self.binSize == 'norm': # Lets consider the smaller bin, to correctly remove motor activity
            binSize = self.dset.attrs['Trial duration'][self.trialsToUse].min()/10
            leftMostBin = np.ceil(tmin/binSize)
            rightMostBin = np.floor(tmax/binSize)
        else: # in this case, account for the baseline which is 500ms
            leftMostBin = np.ceil((tmin+500)/self.binSize)
            rightMostBin = np.floor((tmax+500)/self.binSize)

        toUseTimes = np.logical_and(y >= leftMostBin, y < rightMostBin)

        print(leftMostBin)
        print(rightMostBin)
        self.X = X[toUseTimes,:]
        if self.binSize == 'norm':
            self.y = y[toUseTimes]
        else:
            self.y = y[toUseTimes] - int(500/self.binSize)
        self.trial = trial[toUseTimes]


    def describe(self):
        print('Label: ', self.label)
        print('Bin size:' , self.binSize)
        print('Sigma: ', self.sigma)
        if self.trialsToUse is not None:
            print('Using %d trials '%self.trialsToUse.sum(), self.dset.attrs['Trial number'][self.trialsToUse])
        else:
            print('No selected trials.')
