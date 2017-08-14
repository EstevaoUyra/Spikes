from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from spikeHelper.visuals import trialNeuronPlot, firingRateEvo
from spikeHelper.loadSpike import loadSpikeBehav
from spikeHelper.dataOrganization import trialToXyT, getX
from spikeHelper.filters import convHist, filterEpochs
from sklearn.metrics import cohen_kappa_score
import scipy.stats as st
from sklearn.base import clone
from scipy.io import loadmat
import pickle

def expAllRats():
    results = pd.DataFrame(index = ['rat 7','rat 8','rat 9','rat 10'], columns = ['Late corr', 'Late kappa', 'Early corr', 'Early kappa', 'Cross late train corr','Cross late train kappa','Cross early train corr','Cross early train kappa',  'perTrialCorr', 'perTrialKappa','best Params'])

    data = trialToXyT(pickle.load(open('Data/50ms_r7_1000msPlus.pickle','rb')))
    rat = 'rat 7'
    results = experiment1rat(data,rat,results)

    data = trialToXyT(pickle.load(open('Data/50ms_r8_1000msPlus.pickle','rb')))
    rat = 'rat 8'
    results = experiment1rat(data,rat,results)

    data = trialToXyT(pickle.load(open('Data/50ms_r9_1000msPlus.pickle','rb')))
    rat = 'rat 9'
    results = experiment1rat(data,rat,results)

    data = trialToXyT(pickle.load(open('Data/50ms_r10_1000msPlus.pickle','rb')))
    rat = 'rat 10'
    results = experiment1rat(data,rat,results)

    return results

def experiment1rat(data,rat,results,n_splits=[10, 30]):
    print('Calculating for', rat)
    print('Grid Searching')

    assert len(n_splits)==2

    C_grid = range(10,17); Gamma_grid = np.linspace(.1,.4,10)*1./20;
    params = {'C':C_grid,'gamma':Gamma_grid}
    grid = GridSearchCV(SVC(), params, cv=2)
    grid.fit(getX(data[data['end']]),data[data['end']]['y'])
    results['best Params'][rat] = grid.best_params_

    print('Direct training')
    directResults = bothSame(grid.best_estimator_, data, n_splits = n_splits[0])
    results['Late corr'][rat] = directResults['late']['corr']
    results['Late kappa'][rat] = directResults['late']['kappa']
    results['Early corr'][rat] = directResults['early']['corr']
    results['Early kappa'][rat] = directResults['early']['kappa']

    print('Cross training')
    crossResults = bothCross(grid.best_estimator_, data, n_splits=n_splits[1])
    results['Cross late train corr'][rat] = crossResults['lateTrain']['corr']
    results['Cross late train kappa'][rat] = crossResults['lateTrain']['kappa']
    results['Cross early train corr'][rat] = crossResults['begTrain']['corr']
    results['Cross early train kappa'][rat] = crossResults['begTrain']['kappa']

    print('Analyzing evolution')
    corrEvo, kappaEvo = expEvolution(grid.best_estimator_, data, n_splits=n_splits[1])
    results['perTrialCorr'][rat] = results['perTrialCorr'].astype(object)
    results['perTrialKappa'][rat] = results['perTrialKappa'].astype(object)
    results['perTrialCorr'][rat] = corrEvo
    results['perTrialKappa'][rat] = kappaEvo
    return results

def expEvolution(clf, data, n_splits = 30):
    predictions = eachCross(clf, data[data['end'] ], data, n_splits)
    evolutionRes = {}
    evolutionRes['corr'] = [predictions.apply(lambda x: st.pearsonr( x[predictions['trial']==i], predictions['y'][predictions['trial']==i] )[0] ) for i in range(data['trial'].max()) ]
    evolutionRes['kappa'] = [predictions.apply(lambda x: cohen_kappa_score( x[predictions['trial']==i], predictions['y'][predictions['trial']==i] ) ) for i in range(data['trial'].max()) ]
    return (pd.DataFrame(evolutionRes['corr'])).iloc[:,:n_splits], (pd.DataFrame(evolutionRes['kappa'])).iloc[:,:n_splits]

def bothSame(clf, data, n_splits):
    results = {}

    ypred, ytrue = eachSame(clf, data[data['end']], n_splits)
    results['late'] = {}
    results['late']['corr'] = np.array([[st.pearsonr(ypred[ti, clfi, :], ytrue[ti,:])[0] for ti in range(ypred.shape[0])] for clfi in range(ypred.shape[1])])
    results['late']['kappa'] = np.array([[cohen_kappa_score(ypred[ti, clfi, :], ytrue[ti,:]) for ti in range(ypred.shape[0])] for clfi in range(ypred.shape[1])])


    ypred, ytrue = eachSame(clf, data[data['beg']], n_splits)
    results['early'] = {}
    results['early']['corr'] = np.array([[st.pearsonr(ypred[ti, clfi, :], ytrue[ti,:])[0] for ti in range(ypred.shape[0])] for clfi in range(ypred.shape[1])])
    results['early']['kappa'] = np.array([[cohen_kappa_score(ypred[ti, clfi, :], ytrue[ti,:]) for ti in range(ypred.shape[0])] for clfi in range(ypred.shape[1])])

    return results

def eachSame(clf, data, n_splits):
    sh = StratifiedShuffleSplit(n_splits = n_splits, test_size=None, train_size = .9)
    trial = data['trial']
    ypred = [];
    for ti in np.unique(trial):
        Xi = getX(data)[trial!=ti,:]
        yi = data['y'][trial!=ti]
        shuffle = sh.split(Xi,yi)
        ypred.append([])
        if ti % 10 == 0:  print('Analyzing trial ' +str(ti))
        for idx,_ in shuffle:
            clf = clone(clf)
            clf.fit(Xi[idx,:],yi.as_matrix()[idx])
            ypred[-1].append(clf.predict(getX(data)[trial==ti,:]))

    return np.array(ypred), data['y'].values.reshape(-1,np.array(ypred).shape[2])

def bothCross(clf, data, n_splits=30):
    crossResults = {}

    train = data[data['beg']]; test = data[data['end']];
    trainBegTestEnd = eachCross(clf, train, test, n_splits);
    crossResults['begTrain'] = {}
    crossResults['begTrain']['corr'] = trainBegTestEnd.apply(lambda x: st.pearsonr(x, trainBegTestEnd['y'])[0] )[:n_splits].as_matrix()
    crossResults['begTrain']['kappa'] = trainBegTestEnd.apply(lambda x: cohen_kappa_score(x, trainBegTestEnd['y'], weights='linear') )[:n_splits].as_matrix()

    train = data[data['end']]; test = data[data['beg']];
    trainEndTestBeg = eachCross(clf, train, test, n_splits);
    crossResults['lateTrain'] = {}
    crossResults['lateTrain']['corr'] = trainEndTestBeg.apply(lambda x: st.pearsonr(x, trainEndTestBeg['y'])[0] )[:n_splits].as_matrix()
    crossResults['lateTrain']['kappa'] = trainEndTestBeg.apply(lambda x: cohen_kappa_score(x, trainEndTestBeg['y'], weights='linear') )[:n_splits].as_matrix()

    return crossResults

def eachCross(clf, train, test, n_splits=30):
    sh = StratifiedShuffleSplit(n_splits = n_splits,test_size=None, train_size = .9)
    shuffle = sh.split(getX(train),train['y'])
    ypred = []
    nshuf = 0
    for idx,_ in shuffle:
        print('Working on shuffle #' +str(nshuf)); nshuf=nshuf+1;
        clfaux = clone(clf)
        clfaux.fit(getX(train)[idx,:], train['y'].as_matrix()[idx] )
        ypred.append(clfaux.predict(getX(test)))
    results = pd.DataFrame(np.transpose(np.array(ypred)), columns = ['shuffle'+str(i) for i in range(n_splits)])
    results['y'] = test['y'].tolist()
    results['trial'] = test['trial'].tolist()
    return results
