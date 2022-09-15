from json.encoder import INFINITY
import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from operator import itemgetter

from sklearn.feature_selection import RFE
import itertools
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

"""
This file containes all of the models used in homework 1
"""


class LeastSquares:
    def __init__(self):
        self.model = linear_model.LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class RidgeRegression:
    def __init__(self):
        self.model = linear_model.Ridge()
        self.param_range = (0.001,20)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def set_param(self, param_val):
        self.model.set_params(alpha = param_val)



class BestSubsets:
    def __init__(self):
        self.model = linear_model.LinearRegression()
        self.best_subset = []

    def fit(self, X_train, y_train, X_val, y_val):
        """
        train the model with the inputted training data: 
        do this by trying seeing which out of all subsets of features give the lowest error.
        """
        #keep track of the min error / best model associated with that
        min_mse = 1
        best_model = None

        # evaluate error for every subset size
        for k in range(1, len(X_train)):
            # evaluate the error on all possible feature sets of subset size k
            for feature_set in itertools.combinations(X_train.columns, k):
                # Fit model on feature_set and calculate error
                model = linear_model.LinearRegression().fit(X_train[list(feature_set)], y_train)
                error = mean_squared_error(model.predict(X_val[list(feature_set)]), y_val)

                #if this is better than our current best, then replace it
                if error < min_mse:
                    best_model = model
                    min_mse = error
                    self.best_subset = feature_set
        
        # set the model
        self.model = best_model
        print("best subsets model : ", self.model.feature_names_in_)

    def predict(self, X_test):
        return self.model.predict(X_test[list(self.best_subset)])


class Lasso:
    def __init__(self):
        self.model = linear_model.Lasso()
        self.param_range = (0.001,10)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def set_param(self, param_val):
        self.model.set_params(alpha = param_val)

    

class RecursiveFeature:
    def __init__(self):
        self.model = RFE(estimator = SVR(kernel="linear"), n_features_to_select=1)
   

    def fit(self, X_train, y_train, X_val, y_val):
        min_error = 1
        best_k = 1
        best_model = None
        for k in range(1, 13):
             model = RFE(estimator = SVR(kernel="linear"), n_features_to_select=k)
             model.fit(X_train, y_train)
             error = mean_squared_error(model.predict(X_val), y_val)

             if error < min_error:
                min_error = error
                best_k = k
                best_model = model


        self.model = best_model
        print( best_k)

    def predict(self, X_test):
        return self.model.predict(X_test)

class ElasticNet:
    def __init__(self):
        self.model = linear_model.ElasticNet()
        self.param1_range = (.0001, 20)
        self.param2_range = (0.0001,0.99999)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def set_params(self, param1_val, param2_val):
        self.model.set_params(alpha = param1_val, l1_ratio= param2_val)

class AdaptiveLasso:
    def __init__(self):
        self.model = linear_model.Lasso(fit_intercept=True)
        self.param_range = (.001, 1)
        self.alpha = .1

    def fit(self, X_train, y_train):
        alpha = self.alpha
        gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

        n_samples, n_features = X_train.shape

        weights = np.ones(n_features)
        n_lasso_iterations = 10

        for k in range(n_lasso_iterations):
            X_w = X_train / weights[np.newaxis, :]
            self.model = linear_model.Lasso(alpha = alpha, fit_intercept=True)
            self.model.fit(X_w, y_train)
            coef_ = self.model.coef_ / weights
            weights = gprime(coef_)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def set_param(self, param_val):
        self.alpha = param_val
     