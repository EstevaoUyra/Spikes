import pandas as pd
from scipy.io import loadmat
import numpy as np
import pickle
from spikeHelper.filters import convHist, kernelSmooth, binarize
from spikeHelper.dataOrganization import XyTfromEpoch, padNans
import h5py
from spikeHelper.visuals import interactWithActivity

def loadSpikeBehav(fileName):
    data = loadmat(fileName)

    spikes = data['dados'][0,0][1]
    behavior = data['dados'][0,0][0]

    spikes = pd.DataFrame([[ spikes[0,i][0][:,0], spikes[0,i][0][:,1]] for i in range(spikes.shape[1]) if spikes[0,i][0].shape[1]==2], columns=['times','trial'])
    behavior = pd.DataFrame(np.transpose(behavior[0,0][0]), columns=['one','onset','offset','zero','duration','sortIdx','sortLabel'])
    spikes['trialTime'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))

    return spikes, behavior

def epochData(spikes, behavior, ratNumber, sigma, binSize, shuffleBins = False):
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
    elif shuffleBins:
        for itrial in range(1,behavior.shape[0]+1):
            trialSize = int(np.floor(1000*behavior.duration[itrial-1]))
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply(lambda x: serializeSpikes(x, trialSize))
            epochs['trial '+str(itrial)] = epochs['trial '+str(itrial)].apply( np.random.permutation ) #shuffling before smoothi
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

def changeArrayFromTo(arr, fromValues, toValues):
    newArr = np.copy(arr)
    for from_i, to_i in zip(fromValues, toValues):
        newArr[arr==from_i] = to_i
    return newArr

class Rat():

    # PRIVATE METHODS
    def __init__(self,ratNumber, sigma=100, binSize=100, label = None):
        self._ratNumber = ratNumber
        self._sigma = sigma
        self._binSize = binSize
        self._isShuffled = None

        self._trialSpecs = {}
        self._trialRestrictions = {'minDuration':None, 'maxDuration':None,
                                    'trialMin':None,'trialMax':None,
                                    'ntrials':None}


        if label is None:
            self.label = 'Rat '+ str(ratNumber)
        else:
            self.label = label
        self._dset = self._loadFile()
        self.trialsToUse = None

        self.X = None
        self.y = None
        self.trial = None

    def _sessionBestMoment(self):
        if self._ratNumber == 7:
            return 500
        elif self._ratNumber == 8:
            return 500
        elif self._ratNumber == 9:
            return 350
        elif self._ratNumber == 10:
            return 900
        else:
            raise ValueError('I dont know this rat')

    def _loadFile(self):
        f = h5py.File("spikeData.hdf5", "r+")
        dsetName = 'bin'+str(self._binSize)+'_sigma'+str(self._sigma)

        if self.label not in f:
            print(self.label + ' not contained in database. Creating group')
            f.create_group(self.label)
            #f[self.label].create_group('epochData')

        if dsetName not in f[self.label]:
            print(dsetName + ' not contained in database. Creating dataset.')
            self._create(dsetName,f)

        for atribute in list(f[self.label][dsetName].attrs):
            self._trialSpecs[atribute] = f[self.label][dsetName].attrs[atribute]

        neuronTimeTrial = f[self.label][dsetName][:,:,:]
        f.close()
        return neuronTimeTrial

    def _create(self, dsetName, f, shuffleBins=False):
        spikes, behavior = loadSpikeBehav('Data/rato'+str(self._ratNumber)+'tudo.mat')
        epochs = epochData(spikes,behavior,ratNumber = self._ratNumber, sigma=self._sigma, binSize=self._binSize,shuffleBins = shuffleBins)
        eachLen = epochs.applymap(len)
        trialLen = eachLen.min(axis=0).values
        epochs = epochs.applymap(lambda x: padNans(x, eachLen.max().max() ) )
        cubicData, trialN = XyTfromEpoch(epochs,returnTrialN = True)
        iti = np.hstack((0,behavior.onset[1:].values - behavior.offset.iloc[:-1].values ) )

        f[self.label].create_dataset(dsetName, data = cubicData,shuffle=True, compression='lzf')
        f[self.label][dsetName].attrs.create('Trial number', trialN)
        f[self.label][dsetName].attrs.create('Trial length', trialLen)
        f[self.label][dsetName].attrs.create('Trial duration', behavior.duration.values*1000)
        f[self.label][dsetName].attrs.create('Trial start', behavior.onset.values*1000)
        f[self.label][dsetName].attrs.create('Trial end', behavior.offset.values*1000)
        f[self.label][dsetName].attrs.create('Intertrial interval', iti*1000)

    def _trialHas(self,property, value):
        if value is None:
            return self.trialsToUse

        if property == 'minDuration':
            return self._trialSpecs['Trial duration'] > value

        elif property == 'maxDuration':
            return self._trialSpecs['Trial duration'] < value

        elif property == 'trialMin':
            if value =='best':
                value = self._sessionBestMoment()
            return self._trialSpecs['Trial number'] > value

        elif property == 'trialMax':
            return self._trialSpecs['Trial number'] < value

        elif property == 'ntrials':
            hasProperty = np.cumsum(self.trialsToUse)<= value
            if hasProperty.shape[0] < value:
                print('Could not get %d trials, got %d instead'%(value, trialsToUse.shape[0]))
            return hasProperty

        else:
            raise ValueError('There is no type %s'%selecType)


    # PUBLIC METHODS

    def selecTrials(self, restrictions):

        '''Select trialsToUse via a restrictions dict of the form {property : value}
        Accepted properties are:
        'minDuration', 'maxDuration' (in milisseconds),
        'trialMin', 'trialMax' (Absolute number of the trial)
        'ntrialsBeg' (turns off last trials to get a total of ntrials)
        '''
        # Reset restrictions
        self._trialRestrictions = {'minDuration':None, 'maxDuration':None,
                                    'trialMin':None,'trialMax':None,
                                    'ntrials':None}

        for propertyi in restrictions:
            self._trialRestrictions[propertyi] = restrictions[propertyi]

        self.trialsToUse = np.ones(self._trialSpecs['Trial number'].shape[0])
        for propertyi in ['minDuration', 'maxDuration','trialMin','trialMax','ntrials']: #enforcing order, leaving ntrials for last
            self.trialsToUse = np.logical_and(self.trialsToUse, self._trialHas(propertyi, self._trialRestrictions[propertyi]) )

        self.X = None
        self.y = None
        self.trial = None

    def manualSelectTrials(self, trialsOfInterest):
        ''' Give an array of booleans of the size of total trials, or an array of integers for the trials of interest'''

        if trialsOfInterest.unique().shape[0] == 2:
            assert trialsOfInterest.shape[0] == self._dset.shape[2]
            self.trialsToUse = trialsOfInterest
        else:
            assert np.sum(np.diff(np.sort(trialsOfInterest))==0) ==0 #there are no repetitions
            self.trialsToUse = np.array([trial in trialsOfInterest for trial in self._trialSpecs['Trial number'] ])


    def selecTimes(self, tmin, tmax=None, shuffle = False):
        if self.trialsToUse is None:
            self.trialsToUse = np.ones(self._trialSpecs['Trial number'].shape[0])


        X = np.transpose(self._dset[:,:,self.trialsToUse]).reshape(-1,self._dset.shape[0])
        y = np.arange(X.shape[0])%self._dset.shape[1]
        trial = np.array([self._dset.shape[1]*[t] for t in self._trialSpecs['Trial number'][self.trialsToUse]]).reshape(-1,1)

        if tmax is None:
            tmax = self._trialSpecs['Trial duration'][self.trialsToUse].min()
        if self._binSize == 'norm': # Lets consider the smaller bin, to correctly remove motor activity
            binSize = self._trialSpecs['Trial duration'][self.trialsToUse].min()/10
            leftMostBin = np.ceil(tmin/binSize)
            rightMostBin = np.floor(tmax/binSize)
        else: # in this case, account for the baseline which is 500ms
            leftMostBin = np.ceil((tmin+500)/self._binSize)
            rightMostBin = np.floor((tmax+500)/self._binSize)

        toUseTimes = np.logical_and(y >= leftMostBin, y < rightMostBin)
        self.X = X[toUseTimes,:]

        if self._binSize == 'norm':
            self.y = y[toUseTimes]
        else:
            self.y = y[toUseTimes] - int(500/self._binSize)

        if shuffle==True:
            self.y = changeArrayFromTo(self.y, np.unique(self.y), np.random.permutation(np.unique(self.y)))
            print('Bins have been shuffled')
            self._isShuffled = True
        else:
            self._isShuffled = False

        self.trial = trial[toUseTimes].reshape(-1)

    def cubicNeuronTimeTrial(self):
        if self._binSize == 'norm':
            useTimes = np.unique(self.y)
        else:
            useTimes = np.unique(self.y) + int(500/self._binSize)

        return self._dset[:,:,self.trialsToUse][:,useTimes,:]

    def describe(self):
        print('Label: %s'%self.label)
        print('Bin size: %s'%self._binSize)
        print('Sigma: %d'%self._sigma)

        if self.trialsToUse is not None:
            print('\nUsing %d trials, according to following restrictions:'%self.trialsToUse.sum())
        else:
            print('\nNo selected trials.')
        for restr in [k +': '+ str(self._trialRestrictions[k]) for k in self._trialRestrictions if self._trialRestrictions[k] is not None]:
            print(restr)


        if self.y is not None and self._isShuffled == False:
            bins = np.unique(self.y)
            if self._binSize == 'norm':
                print('Using normalized bins ', bins)
            else:
                print('\nUsing %d time bins:'%len(bins))

            def printBins(bins):
                for b in bins:
                    print('From %d to %dms'%(b*self._binSize, (b+1)*self._binSize ) )
            if len(bins) <= 5:
                printBins(bins)
            else:
                printBins(bins[:3])
                print('.\n.\n.')
                printBins(bins[-2:])
        elif self._isShuffled == True:
            print('The %d bins have been shuffled'%len(np.unique(self.y)))

        else:
            print('\nTime bins not selected.')

    def interact(self,realTime=False):
        interactWithActivity(self.cubicNeuronTimeTrial(),realTime=realTime)
