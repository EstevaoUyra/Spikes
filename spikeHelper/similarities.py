from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import numpy as np
from spikeHelper.dataOrganization import normRows

def similarityMatrix(X,y,W=None,z=None,method='greek',compare=False, oneToOne=False, normalize=True):
    empCov = EmpiricalCovariance()
    if compare:
        assert W.shape[0] > 1
        assert z.shape[0] > 1
        precision = empCov.fit(np.vstack((X,W))).get_precision()
    else:
        assert W == None
        assert z == None
        W = X; z = y;
        precision = empCov.fit(X).get_precision()

    if method == 'mah':
        dist = lambda u,v : mahalanobis(u,v,precision)
    elif method == 'greek':
        dist = euclidean

    times1 = np.unique(y)
    times2 = np.unique(z)
    template = [ np.median(W[z==ti],axis=0) for ti in np.unique(z)]
    distances = np.full((len(times1),len(times2)),np.nan)
    if not oneToOne:
        for i, ti in enumerate(times1):
            distances[i,:] = np.array([np.array([dist(u,v) for u in X[y==ti]]).mean() for v in template])
    else:
        for ti in times1:
            for tj in times2:
                distances[ti,tj] = oneToOneDist(X[y==ti], W[z==tj], dist)
    if normalize:
        return normRows(1/np.array(distances))
    else:
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
        return {}

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
        return {'warm':self.warm}

    def set_params(self):
        return self

def distanceGeneralization(X,y,ytrial,method='greek'):
    n_classes = len(np.unique(y))

    if method == 'mah':
        clf = MahalanobisClassifier(warm_start=True)
    elif method == 'greek':
        clf = EuclideanClassifier()

    clf.fit(X,y)

    times = np.unique(y)
    trials = np.unique(ytrial)

    loo = LeaveOneOut()
    confusionPerTrial = np.full((times.shape[0],times.shape[0],trials.shape[0]),np.nan)
    for i,testTrial in enumerate(trials):
        clf.fit(X[ytrial != testTrial,:],y[ytrial != testTrial])
        confusionPerTrial[:,:,i] = confusion_matrix(y[ytrial==testTrial], clf.predict(X[ytrial==testTrial]))

    return np.mean(confusionPerTrial,axis=2)

def crossGeneralization(X,y,W,z,method='greek', normalize=True):
    n_classes = len(np.unique(y))

    if method == 'mah':
        clf = MahalanobisClassifier(warm_start=True)
    elif method == 'greek':
        clf = EuclideanClassifier()

    clf.fit(np.vstack(X,W),np.vstack(y,z))

    if normalize:
        return normRows(confusion_matrix(y,ypred))
    else:
        return confusion_matrix(y,ypred)
