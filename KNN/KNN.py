from KNNClassifier import KNNClassifier
from KNNClassifier import KNNRegression
from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


# Create 3 datasets  (taken from lab)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1, n_samples=200)

linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0, n_samples=200),
            make_circles(noise=0.2, factor=0.5, random_state=1, n_samples=200),
            linearly_separable]


# HELPER FUNCTIONS
def split_data(X, y, training_percent):
    """
    randomly split the data X,y so that 
    """

    #1. determine how many rows to take by doing Math.floor(training_percent*len(X))
    training_size = math.floor(training_percent*len(X))

    #2. randomly choose that many integers from 0, len(X) and store indices in a list
    selected_entries = random.sample(range(0, len(X)), training_size)

    #3. run through those indices. add rows to X train an y train and put others in test
    X_train, y_train, X_test, y_test = [], [], [], []
    for entryIdx in range(len(X)):
        if entryIdx in selected_entries:
            X_train.append(X[entryIdx])
            y_train.append(y[entryIdx])
        else:
            X_test.append(X[entryIdx])
            y_test.append(y[entryIdx])
    
    return X_train, y_train, X_test, y_test

for dataSetNum, dataSet in enumerate(datasets):
    X, y = dataSet

    #2. split the data
    X_train, y_train, X_test, y_test = split_data(X, y, training_percent = 0.8)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Store the prediction error for each k value for this data set
    trainingError = {}
    testingError = {}

    cm_bright=ListedColormap(['#FF0000', '#0000FF'])
    
    for subplotNum, k in enumerate([1,3,20]):
        
        #3. construct the classifier based on k
        classifier = KNNClassifier(k)
        # answerClassifier = KNeighborsRegressor(k)

        #4. train the classifier
        classifier.train(X_train, y_train)
        # answerClassifier.fit(X_train, y_train)

        #5. use the model to create a prediction vector
        testingPrediction = classifier.predict(X_test, y_test)
        trainingPrediction = classifier.predict(X_train, y_train)

        #store the error of the prediction
        trainingError[k] = trainingScore
        testingError[k] = testingScore

        # plot the classification Results
        fig = plt.figure()
        ax = plt.subplot(1,3,subplotNum+1)
        ax.scatter(X_train[:,0], X_train[:,1], c = classifier.predict(X_train), cmap=cm_bright, s=200, edgecolors='k', alpha=0.5)
        fig.savefig("k = " + str(k) + "Data set num:", str(dataSetNum))


    
    # # Plot the error vs K
    # fig = plt.figure()
    # plt.plot(trainingError.keys(), trainingError.values(), label = str(dataSetNum) + "Training Error")
    # plt.plot(testingError.keys(), testingError.values(), label = str(dataSetNum) + "Testing Error")
    # plt.xlabel("Model complexity in terms of k")
    # fig.axes[0].invert_xaxis()
    # plt.ylabel("Model Error")
    # plt.title("Data set " + str(dataSetNum) + " KNN Error vs. Model Complexity")
    # plt.legend()
    # plt.savefig(str(dataSetNum) + " correct KNN train vs. test error results.png")
    


