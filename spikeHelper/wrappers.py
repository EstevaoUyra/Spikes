class dataWrapper():
    def __init__(self,fileName):


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
