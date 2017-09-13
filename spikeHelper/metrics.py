import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

def performanceFromConfusion(cubicConfusion,kind='corr'):
    allYtrue = np.array([])
    allYpred = np.array([])
    for trial in range(cubicConfusion.shape[2]):
        real,predicted = np.nonzero(cubicConfusion[:,:,trial])
        allYtrue = np.hstack((allYtrue,real))
        allYpred = np.hstack((allYpred,predicted))

    if kind == 'corr':
        return pearsonr(allYpred,allYtrue)[0]
    elif kind == 'kappa':
        return cohen_kappa_score(allYpred,allYtrue,weights = 'linear')
