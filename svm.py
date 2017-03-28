from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.base import clone

iris = datasets.load_iris()

# Funcao abaixo testada



def pre_processing():
    return spike.data, spike.target
def binShuffle(data):
    return data
def trialShuffle(data):
    return data


class Animal():
    def __init__(self, animalData, trainSize, Nclf_ForEachTrial, Nunits, TotalUnits):
        self.trialsToUse = range(150)
        self.classifiers = {itrial:[] for itrial in self.trialsToUse}
        self.predictions = {itrial:[] for itrial in self.trialsToUse}
        self.trainSize = trainSize
        self.bestClassifier = None
        self.Nclf_ForEachTrial = Nclf_ForEachTrial
        self.Data = animalData
        self.Nunits = Nunits
        self.TotalUnits = TotalUnits

    def __generateClassifier(self,testTrial):
        trainIndexes = self.__getTrainIndexes(testTrial)
        units = self.__getTrainUnits()
        clf = self.__trainOneClassifier(trainIndexes, units)

        self.__addClassifierToAll(trainIndexes, clf, units)

    def __addClassifierToAll(self,trainIndexes, clf, units):
        possibleTestTrials = list( set(self.trialsToUse)-set(trainIndexes) )
        for iTestTrial in possibleTestTrials:
            if len(self.classifiers[iTestTrial]) < self.Nclf_ForEachTrial:
                self.classifiers[iTestTrial].append({'clf':clf,'units':units})

    def __trainOneClassifier(self,trainIndexes, units):
        if self.bestClassifier is None:
            self.__gridSearch()
        clf = clone(self.bestClassifier)
        X, y = self.__getTrialsData(trainIndexes, units)
        clf.fit(X,y)
        return clf

    def __gridSearch(self):
        grid_trials = self.trialsToUse
        units = range(self.TotalUnits)
        X, y = self.__getTrialsData(grid_trials, units)
        clf = SVC( kernel='rbf' )
        grid = GridSearchCV(clf, self.__gridParams(), cv=5)
        grid.fit(X, y)
        self.bestClassifier = grid.best_estimator_

    def __gridParams(self):
        C_grid = range(1,17);
        Gamma_grid = np.linspace(0.0156,0.25,20)
        return {'C':C_grid,'gamma':Gamma_grid}


    def __getTrainIndexes(self,testTrial):
        nonTestTrials = self.trialsToUse[:]
        nonTestTrials.remove(testTrial)
        trainIndexes = np.random.choice(nonTestTrials, self.trainSize, replace=False)
        return trainIndexes

    def __getTrialsData(self,trialIndexes, units):
        try: X = iris.data[trialIndexes][:, units]
        except: X = iris.data[trialIndexes,units]

        y = iris.target[trialIndexes]
        return X, y

    def __getTrainUnits(self):
        return np.random.choice(range(self.TotalUnits), self.Nunits, replace=False)
        #return [0,1,2,3]

    def generateClassifiers(self):
        for iTestTrial in self.trialsToUse:
            print(iTestTrial)
            while len(self.classifiers[iTestTrial]) < self.Nclf_ForEachTrial:
                self.__generateClassifier(iTestTrial)


    def predictAllResults(self):
        for iTestTrial in self.trialsToUse:
            for iclf in self.classifiers[iTestTrial]:
                X, y = self.__getTrialsData(iTestTrial, iclf['units'])
                self.predictions[iTestTrial].append(iclf['clf'].predict(X))
            #self.pearsonTest(iTestTrial)

    def pearsonTest(self,iTestTrial):
        self.predictions['corr'+str(iTestTrial)] = [scipy.stats.pearsonr(yp,y) for yp in self.predictions[iTestTrial] ]
