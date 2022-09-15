from json.encoder import INFINITY
from sys import int_info
import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
import itertools
from sklearn.ensemble import RandomForestRegressor
from operator import itemgetter
import asgl
from sklearn.svm import SVR
from collections import defaultdict
from sklearn.metrics import mean_squared_error

from ML_models import LeastSquares, RidgeRegression, BestSubsets, Lasso, RecursiveFeature, ElasticNet, AdaptiveLasso
from sklearn.preprocessing import StandardScaler


#Load in and Process the Data
labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Read the data file
Data = pd.read_csv('./housing.csv.xls', header=None, delimiter=r"\s+", names=labels)

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
response = "MEDV"

X, y= Data.loc[:,features], Data[response]
y = np.log(y)

def tuneParameters(model, X_train, y_train, X_val, y_val):
    """
    tunes the model's hyper paremter using Grid Search by testing different
    parameters on the validation set X_val, y_val
    """

    param_vals = np.linspace(model.param_range[0], model.param_range[1], 100)

    param_star = 0
    min_error = 1

    for param_val in param_vals:
        model.set_param(param_val)
        model.fit(X_train, y_train)
        mse = mean_squared_error(model.predict(X_val), y_val)
        # errors.append(mse)
        if mse < min_error:
            min_error = mse
            param_star = param_val

    model.set_param(param_star)
    model.fit(X_train,y_train)
    # print(param_star)

def tune2Parameters(model, X_train, y_train, X_val, y_val):
    """
    tunes the model's 2 hyper paremters using Grid Search by testing different
    parameter combinations on the validation set X_val, y_val
    """
    param1_vals = np.linspace(model.param1_range[0], model.param1_range[1], 100)
    param2_vals = np.linspace(model.param2_range[0], model.param2_range[1], 100)

    param1_star = 0
    param2_star = 0
    min_error = 1

    for param1_val in param1_vals:
        for param2_val in param2_vals:
            model.set_params(param1_val, param2_val)
            model.fit(X_train, y_train)
            mse = mean_squared_error(model.predict(X_val), y_val)

            if mse < min_error:
                min_error = mse
                param1_star = param1_val
                param2_star = param2_val

    model.set_params(param1_star, param2_star)
    model.fit(X_train,y_train)
    # print(param1_star, param2_star)
    
def getAverageMSE(X,y, model, tuningParams = []):
    """
    Inputs:
    • X, a data frame of n entries where each column is centered/normalized
    • y, a vector of n entries representing the response of X.
    • model, a regression model with the following:
            - a fit method -> trains the model
            - a predict method -> takes a data set and returns a y_predict using the trained model
            - a set_params method -> sets the parameters of the model if any
            - a param_range field -> a range of values that to be considered 
                                    for the hyper parameter of this model

    Effects:
    1. Splits the data X, y into training, validation, and test sets.
    2. Tune the model's hyper parameters (if any) using the validation set
    3. Calculate the MSE of the trained model on the test set
    Repeat the process 10 times and report the average MSE.

    * additionally, prints the feature coefficient sums to get an indication of features selected by models
    

    """
    top_features = defaultdict(float)
    numTrials = 10
    mse_scores = []
    for i in range(numTrials):
        #Split the data (60% train, 20% validate, 20% test)
        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=.4)
        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=.5)

        # Standardize the data
        list_numerical = X_train.columns
        scaler = StandardScaler().fit(X_train[list_numerical]) 
        
        X_train[list_numerical]= scaler.transform(X_train[list_numerical])
        X_val[list_numerical]= scaler.transform(X_val[list_numerical])
        X_test[list_numerical]= scaler.transform(X_test[list_numerical])


        # Fit the model / tune the response
        if "subs" in tuningParams or "k" in tuningParams:
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)

            #Tune the model parameters if it has any on validation data
            if len(tuningParams) == 1:
                tuneParameters(model, X_train, y_train, X_val, y_val)
            if len(tuningParams) == 2:
                tune2Parameters(model, X_train, y_train, X_val, y_val)

        
        #Determine the MSE of the model on the testing data
        error = mean_squared_error(model.predict(X_test), y_test)
        
        mse_scores.append(mean_squared_error(model.predict(X_test), y_test))

        # Keep track of feature coefficients
        for (coeficient, feature) in zip(model.model.coef_ , list(X_train.columns)):
            top_features[feature] += abs(coeficient)
        

    #Print results
    print("\n"+ (str(model.model)).split("(")[0] + " MSE Average ", round(sum(mse_scores)/numTrials,4))

    # Display the distribution of selected features
    fig = plt.figure()
    plt.bar(top_features.keys(), top_features.values())
    plt.title((str(model.model)).split("(")[0] + " features")
    plt.ylabel("ranking across 10 trials")
    plt.xlabel("features")
    plt.savefig("features " + (str(model.model)).split("(")[0] + ".jpg")
    return mse_scores

def visualize_linear_model_mses():
    trials = list(range(1,11))

    #Linear Regression
    lr_mses = getAverageMSE(X,y, LeastSquares())


    # #Ridge Regression (tuning parameter alpha)
    rr_mses = getAverageMSE(X,y, RidgeRegression(), tuningParams = ["lambda"])

    #Lasso
    lasso = linear_model.Lasso()
    lasso_mses = getAverageMSE(X,y, Lasso(), tuningParams = ["lambda"])

    # RFE
    # rfe_mses = getAverageMSE(X, y, RecursiveFeature(), tuningParams = ["k"])

    # Elastic Net
    en_mses= getAverageMSE(X,y, ElasticNet(), tuningParams = ["lambda", "l1_ratio"])

    # Best subsets
    bsubs_mses = getAverageMSE(X, y, BestSubsets(), tuningParams = ["subs"])

    # Adaptive Lasso
    al_mses = getAverageMSE(X, y, AdaptiveLasso(), tuningParams = ["lambda"])

    

visualize_linear_model_mses()


