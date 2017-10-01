import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score,r2_score,mean_squared_error

def performanceFromConfusion(cubicConfusion,kind='corr',weights = 'linear',returnP = False):

    '''Accepts 'corr', for pearsonr, and 'kappa' for cohen's linear weighted kappa score'''
    allYtrue = np.array([])
    allYpred = np.array([])
    cubicConfusion = cubicConfusion[:,cubicConfusion.sum(axis=2).sum(axis=1)>0,:]
    for trial in range(cubicConfusion.shape[2]):
        real,predicted = np.nonzero(cubicConfusion[:,:,trial])
        allYtrue = np.hstack((allYtrue,real))
        allYpred = np.hstack((allYpred,predicted))

    if kind == 'corr':
        if returnP:
            return pearsonr(allYtrue,allYpred)
        return pearsonr(allYtrue,allYpred)[0]
    elif kind == 'kappa':
        return cohen_kappa_score(allYtrue,allYpred,weights = weights)
    elif kind == 'r2':
        return r2_score(allYtrue,allYpred)
    elif kind == 'mse':
        return mean_squared_error(allYtrue,allYpred)
