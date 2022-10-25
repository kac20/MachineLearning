import numpy as np
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics import r2_score


class KNNClassifier:
    def __init__(self,k):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def getNN(self, testEntry):
        distances = {}

        # get the distance between the test Entry and every training entry
        for trainEntryIdx in range(len(self.X_train)):
            dist = distance.euclidean(self.X_train[trainEntryIdx], testEntry)
            distances[trainEntryIdx] = dist

        #list of training entry index to that entry's distance to the test entry
        sorted_distance_map = list(sorted(distances.items(), key=lambda item: item[1], reverse=False))
        sorted_distance_map = sorted_distance_map[0:self.k]
        return [self.y_train[idx] for idx,distance in sorted_distance_map]

    def predict(self, X_test): # returns a vector of the predictions y_predict
        numTestEntries = len(X_test)
        #initialize y_predict
        y_predict = [0 for i in range(numTestEntries)]

        for testEntryIdx in range(numTestEntries):
            #get the values  of the K nearest neighbors
            nn = self.getNN(X_test[testEntryIdx])

            nnCounts = Counter(nn)
            val, count = nnCounts.most_common()[0]
            y_predict[testEntryIdx] = val
        return y_predict

    def score(self, X_test, y_test): 
        y_predict = self.predict(X_test)
        meanAccuracy = 0
        for i in range(len(y_test)):
            if y_test[i] == y_predict[i]:
                meanAccuracy += 1
        meanAccuracy = meanAccuracy/len(y_test)
        return meanAccuracy

    


class KNNRegression:
    def __init__(self,k):
        self.k = k
    
    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def getNN(self, testEntry):
        distances = {}

        # get the distance between the test Entry and every training entry
        for trainEntryIdx in range(len(self.X_train)):
            dist = distance.euclidean(self.X_train[trainEntryIdx], testEntry)
            distances[trainEntryIdx] = dist

        #list of training entry index to that entry's distance to the test entry
        sorted_distance_map = list(sorted(distances.items(), key=lambda item: item[1], reverse=False))
        sorted_distance_map = sorted_distance_map[0:self.k]
        return [self.y_train[idx] for idx,distance in sorted_distance_map]


    def predict(self, X_test):
        numTestEntries = len(X_test)
        #initialize y_predict
        y_predict = [0 for i in range(numTestEntries)]

        for testEntryIdx in range(numTestEntries):
            #get the values  of the K nearest neighbors
            nn = self.getNN(X_test[testEntryIdx])

            nnCounts = Counter(nn)
            val, count = nnCounts.most_common()[0]
            y_predict[testEntryIdx] = sum(nnCounts.values()) / len(nnCounts)
        return y_predict

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        error = r2_score(y_test, y_predict)
        return error

    
