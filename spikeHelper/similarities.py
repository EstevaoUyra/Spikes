from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import numpy as np
from spikeHelper.dataOrganization import normRows, getX, loadBestParams
from sklearn.base import clone

def oneToOneDist(Us, Vs, distfunc):
    eachDist = []
    for ui in Us:
        for vi in Vs:
            eachDist.append(distfunc(ui,vi))
    return np.array(eachDist).mean()

def similarityMatrix(X,y,W=None,z=None,method='greek',compare=False,trials=False, oneToOne=False):
    empCov = EmpiricalCovariance()
    if compare:
        assert W.shape[0] > 1
        assert z.shape[0] > 1
        precision = empCov.fit(np.vstack((X,W))).get_precision()
    else:
        assert W == None
        assert z == None
        assert trials is not False
        W = X; z = y;
        precision = empCov.fit(X).get_precision()

    if method == 'mah':
        dist = lambda u,v : mahalanobis(u,v,precision)
    elif method == 'greek':
        dist = euclidean

    times1 = np.unique(y)
    times2 = np.unique(z)
    template = [ np.mean(W[z==ti],axis=0) for ti in np.unique(z)]
    distances = np.full((len(times1),len(times2)),np.nan)
    if not oneToOne:
        for i, ti in enumerate(times1):
            distances[i,:] = np.array([np.array([dist(u,v) for u in X[y==ti]]).mean() for v in template])
    else:
        for ti in times1:
            for tj in times2:
                distances[ti,tj] = oneToOneDist(X[y==ti], W[z==tj], dist)
    return 1/np.array(distances)


class MahalanobisClassifier():
    def __init__(self,warm_start = False):
        self.warm = warm_start
        self.warmed = False

    def fit(self,X,y):
        self.center_ = [X[y==yi].mean(axis=0) for yi in np.sort(np.unique(y))]
        if self.warmed == False:
            empCov = EmpiricalCovariance()
            self.precision_ = empCov.fit(X).get_precision()
        if self.warm:
            self.warmed = True

        assert np.max(y) == (len(np.unique(y))-1)

    def _predictOne(self, x):
        return np.argmin(self._transformOne(x))

    def predict(self, X):
        return np.array([self._predictOne(x) for x in X])

    def transform(self,X,y = None):
        return np.array([self._transformOne(x) for x in X])

    def _transformOne(self, x):
        return np.array([mahalanobis(x, center,self.precision_) for center in self.center_])

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)

    def get_params(self,deep=True):
        return {'warm_start':self.warm}

    def set_params(self):
        return self

class EuclideanClassifier():
    def __init__(self):
        pass

    def fit(self,X,y):
        self.center_ = [X[y==yi].mean(axis=0) for yi in np.sort(np.unique(y))]
        assert np.max(y) == (len(np.unique(y))-1)

    def _predictOne(self, x):
        return np.argmin(self._transformOne(x))

    def predict(self, X):
        return np.array([self._predictOne(x) for x in X])

    def transform(self,X,y = None):
        return np.array([self._transformOne(x) for x in X])

    def _transformOne(self, x):
        return np.array([euclidean(x, center) for center in self.center_])

    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X,y)

    def get_params(self,deep=True):
        return {}

    def set_params(self):
        return self

def temporalGeneralization(X,y,ytrial,clf,returnCubic=False,transform=False):
    n_classes = len(np.unique(y))

    if not isinstance(clf,SVC):
        clf.fit(X,y)

    times = np.unique(y)
    trials = np.unique(ytrial)

    if transform:
        confusionPerTrial = np.full((n_classes,n_classes,trials.shape[0]),np.nan)
        for i,testTrial in enumerate(trials):
            clfi = clone(clf)
            clfi.fit(X[ytrial != testTrial,:],y[ytrial != testTrial])
            try:
                confusionPerTrial[:,:,i] = 1/clfi.transform(X[ytrial==testTrial])
            except:
                confusionPerTrial[:,:,i] = clfi.decision_function(X[ytrial==testTrial])
    else:
        confusionPerTrial = np.full((n_classes,n_classes,trials.shape[0]),np.nan)
        for i,testTrial in enumerate(trials):
            clfi = clone(clf)
            clfi.fit(X[ytrial != testTrial,:],y[ytrial != testTrial])
            confusionPerTrial[:,:,i] = confusion_matrix(y[ytrial==testTrial], clfi.predict(X[ytrial==testTrial]))

    if returnCubic:
        return confusionPerTrial
    else:
        return np.mean(confusionPerTrial,axis=2)



def crossGeneralization(epochsTrain,epochsTest,rat):
    '''Trains on all of the first input, and tests in each of the second'''
    Xtrain, ytrain = getX(epochsTrain), epochsTrain['y']
    Xtest, ytest = getX(epochsTest), epochsTest['y']
    ytrial = epochsTest['trial']


    parameters = loadBestParams(rat)
    clf = SVC(C=parameters['C'], gamma=10**parameters['logGamma'], decision_function_shape='ovr')
    clf.fit(Xtrain,ytrain)

    n_classes = max(len(np.unique(ytrain)), len(np.unique(ytest)))
    trials = np.unique(epochsTest['trial'])
    confusionPerTrial = np.full((n_classes,n_classes,trials.shape[0]),np.nan)
    for i,testTrial in enumerate(np.unique(ytrial)):
        confusionPerTrial[:,:,i] = confusion_matrix(ytest[ytrial==testTrial], clf.predict(Xtest[ytrial==testTrial]))

    ntrain, ntest = len(np.unique(ytrain)), len(np.unique(ytest))
    tempGen = np.full((ntrain,ntest),np.nan)
    for real_y in np.unique(ytest):
        tempGen[:,real_y] = clf.decision_function(Xtest[ytest==real_y]).mean(axis=0)

    return confusionPerTrial,tempGen


def readout(vec):
    n_classes = int((1 + np.sqrt(1+ 8*len(vec)))/2)
    trueReadout = np.full((n_classes,n_classes),np.nan)
    nread = 0
    for i in range(n_classes):
        trueReadout[i,:] = np.hstack((np.zeros(i+1),vec[nread:nread+n_classes-i-1]))
        nread = nread+n_classes-i-1
    return (trueReadout- trueReadout.transpose()).sum(axis=1)

def meanReadout(X):
    return np.array([readout(x) for x in X]).sum(axis=0)

def generalizationOvO(clf,X,y):
    n = len(np.unique(y))
    tempGen = np.full((n,n),np.nan)
    for real_y in np.unique(y):
        tempGen[real_y,:] = meanReadout(clf.decision_function(X[y==real_y]))
    return tempGen
